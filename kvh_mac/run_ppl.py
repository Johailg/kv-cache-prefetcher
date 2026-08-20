"""
kvh.run_ppl -- multi-turn quality under a cache budget, teacher-forced.

Per policy and budget, on assistant tokens only:

    ppl        perplexity of the reference continuation
    d_ppl      ppl - ppl_full, the compression cost in Zhang's units
    kl         KL(full || policy) on the next-token distribution
    top1       agreement with the full-cache argmax
    residency  measured resident KV fraction -- the x axis, never the knob
    recovery   raw mass on the resident set, all steps
    esc_rate   escape rate under the measurement lineage
    rec_esc    mass_nosink on the resident set, escape steps only
    h_esc      binary escape hit rate (cluster policies only)

Teacher forcing means every policy sees the same token stream, so the columns
are comparable and the full-cache reference is exact. It is also a LOWER BOUND
on divergence: a policy that free-runs will drift further. Generation is a
separate run that does not exist yet.

THE FEEDING INVARIANT
---------------------
Every token is fed to the model exactly once, in order. `pending` always holds
the logits that predict the token at `pos`. The first version of this file
violated that -- it fed the boundary token twice, once to score and once to
advance -- and every number downstream would have been computed against a cache
containing a duplicated position. If you change this loop, check the invariant
with --assert-feeding.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

# Works both as a package (`python -m kvh.run_ppl`) and as a plain script
# (`python run_ppl.py` from inside the directory). The relative form is tried
# first so the package layout stays canonical.
try:
    from . import attn as A
    from . import policies as P
    from .sessions import Session, build_sessions
except ImportError:
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import attn as A
    import policies as P
    from sessions import Session, build_sessions

TOPK = 64          # size of the reference distribution kept for the KL estimate


# ==========================================================================
# one pass over one session
# ==========================================================================

class Feeder:
    """Wraps the model so position bookkeeping is in one place.

    fed_upto is the number of tokens already pushed through. Every call must
    start exactly there; the assert is what catches a double-feed.
    """

    def __init__(self, model, ids, assert_feeding=True):
        self.model = model
        self.ids = ids
        self.past = None
        self.fed_upto = 0
        self.assert_feeding = assert_feeding

    def feed(self, start, stop):
        """Push ids[start:stop]. -> logits [1, stop-start, V], where row i
        predicts the token at position start+i+1."""
        if self.assert_feeding and start != self.fed_upto:
            raise AssertionError(
                f"feeding invariant violated: next unfed position is "
                f"{self.fed_upto} but got start={start}")
        out = self.model(self.ids[:, start:stop], past_key_values=self.past,
                         use_cache=True)
        self.past = out.past_key_values
        self.fed_upto = stop
        return out.logits


@torch.no_grad()
def run_session(model, sess: Session, state: Optional[A.PolicyState], shape,
                device, dtype, decode_chunk: int, reference: Optional[dict],
                assert_feeding: bool = True):
    """Returns (record, collected_reference).

    reference is None on the full-cache pass, in which case we collect one;
    otherwise we score against the one passed in.
    """
    if state is not None:
        state.reset_sequence(*shape, device, dtype)

    f = Feeder(model, sess.ids, assert_feeding)
    collecting = reference is None
    ref_ids: List[torch.Tensor] = []
    ref_probs: List[torch.Tensor] = []

    nll_total = 0.0
    n_scored = 0
    kl_total = 0.0
    top1_hits = 0
    by_turn: Dict[int, List[float]] = {}
    cursor = 0                       # index into the flat reference arrays

    pending = None                   # logits [1, 1, V] predicting token `pos`
    pos = 0

    for turn, span_start, span_end in sess.spans:
        # ---- prefill everything up to the assistant span ------------------
        if span_start > pos:
            logits = f.feed(pos, span_start)
            pending = logits[:, -1:]                 # predicts token span_start
            pos = span_start
        if pending is None:
            raise AssertionError("span starts at position 0; nothing predicts it")

        # ---- teacher-force the assistant span -----------------------------
        while pos < span_end:
            n = min(decode_chunk, span_end - pos)

            # We need logits predicting tokens pos .. pos+n-1.
            #   pending covers pos.
            #   feeding tokens pos .. pos+n-2 covers pos+1 .. pos+n-1.
            if n > 1:
                rest = f.feed(pos, pos + n - 1)
                logits = torch.cat([pending, rest], dim=1)     # [1, n, V]
            else:
                logits = pending                               # [1, 1, V]

            logprobs = F.log_softmax(logits[0].float(), dim=-1)   # [n, V]
            gold = sess.ids[0, pos:pos + n]                      # [n]
            nll = -logprobs[torch.arange(n, device=device), gold]

            nll_total += float(nll.sum())
            n_scored += n
            slot = by_turn.setdefault(turn, [0.0, 0.0])
            slot[0] += float(nll.sum())
            slot[1] += n

            if collecting:
                p, i = logprobs.exp().topk(TOPK, dim=-1)
                ref_probs.append(p.cpu())
                ref_ids.append(i.cpu())
            else:
                kl, hits = _score_against_reference(
                    logprobs, reference, cursor, n, device)
                kl_total += kl
                top1_hits += hits
            cursor += n

            # advance past the last token of the chunk; this is the ONLY place
            # it gets fed, which is what keeps the invariant
            pending = f.feed(pos + n - 1, pos + n)[:, -1:]
            pos += n

    record = dict(n_tok=n_scored, nll=nll_total, kl=kl_total,
                  top1=top1_hits, by_turn=by_turn)
    if state is not None:
        record.update(residency=state.acct.residency,
                      resid_spread=state.acct.residency_spread,
                      recovery=state.acct.recovery,
                      esc_rate=state.acct.escape_rate,
                      rec_esc=state.acct.recovery_esc,
                      h_esc=state.acct.h_esc)
    collected = (dict(ids=torch.cat(ref_ids), probs=torch.cat(ref_probs))
                 if collecting else None)
    return record, collected


def _score_against_reference(logprobs, reference, cursor, n, device):
    """KL(full || policy) over the reference's top-TOKP support plus one lumped
    tail bucket, and top-1 agreement.

    The tail term keeps this an honest estimate rather than a truncated one:
    whatever mass the full model put outside its own top-64 is compared against
    whatever the policy put there.
    """
    ref_i = reference["ids"][cursor:cursor + n].to(device)        # [n, TOPK]
    ref_p = reference["probs"][cursor:cursor + n].to(device)      # [n, TOPK]
    pol_p = logprobs.gather(1, ref_i).exp()                       # [n, TOPK]

    kl = (ref_p * (ref_p.clamp_min(1e-12).log() - pol_p.clamp_min(1e-12).log())).sum(1)
    ref_tail = (1 - ref_p.sum(1)).clamp_min(0)
    pol_tail = (1 - pol_p.sum(1)).clamp_min(1e-12)
    kl = kl + ref_tail * (ref_tail.clamp_min(1e-12).log() - pol_tail.log())

    hits = int((logprobs.argmax(-1) == ref_i[:, 0]).sum())
    return float(kl.sum()), hits


# ==========================================================================
# aggregation
# ==========================================================================

def _mean(records, key):
    vals = [r[key] for r in records if key in r and r[key] == r[key]]   # drops nan
    return sum(vals) / len(vals) if vals else float("nan")


def aggregate(label, knob, knob_kind, records, baseline):
    n = sum(r["n_tok"] for r in records)
    ppl = math.exp(sum(r["nll"] for r in records) / max(n, 1))

    turns: Dict[int, List[float]] = {}
    for r in records:
        for t, (s, c) in r["by_turn"].items():
            slot = turns.setdefault(t, [0.0, 0.0])
            slot[0] += s
            slot[1] += c

    row = dict(
        policy=label, knob=knob, knob_kind=knob_kind, n_tok=n,
        ppl=round(ppl, 4),
        d_ppl=round(ppl - baseline["ppl"], 4) if baseline else 0.0,
        kl=round(sum(r["kl"] for r in records) / max(n, 1), 6),
        top1=round(sum(r["top1"] for r in records) / max(n, 1), 4) if baseline else 1.0,
        residency=round(_mean(records, "residency"), 4),
        resid_spread=round(_mean(records, "resid_spread"), 4),
        recovery=round(_mean(records, "recovery"), 4),
        esc_rate=round(_mean(records, "esc_rate"), 4),
        rec_esc=round(_mean(records, "rec_esc"), 4),
        h_esc=round(_mean(records, "h_esc"), 4),
    )
    for t in sorted(turns)[:8]:
        s, c = turns[t]
        row[f"ppl_t{t}"] = round(math.exp(s / max(c, 1)), 3)
    return row


def show(row, seconds=None):
    tail = f"  [{seconds:.0f}s]" if seconds else ""
    print(f"  {row['policy']:14s} {row['knob_kind']}={row['knob']:<7} "
          f"resid {row['residency']:.4f}(+-{row['resid_spread']:.3f})  "
          f"rec {row['recovery']:.4f}  "
          f"ppl {row['ppl']:.3f}  dppl {row['d_ppl']:+.3f}  kl {row['kl']:.4f}  "
          f"top1 {row['top1']:.4f}  esc {row['esc_rate']:.3f}  "
          f"rec|esc {row['rec_esc']:.4f}  h_esc {row['h_esc']:.4f}{tail}", flush=True)


# ==========================================================================
# driver
# ==========================================================================

def load_index(path):
    if not path:
        return None
    import numpy as np
    z = np.load(path)
    arrays = {k: torch.from_numpy(z[k]) for k in z.files if z[k].dtype != object}
    return P.ClusterIndex(arrays["centroids"], arrays["hubs"],
                          arrays.get("flow"), arrays.get("freq"),
                          arrays.get("local_frac"))


def plan_jobs(policy_names, budgets, ks, flow_slots):
    """-> list of (name, budget, k, flow_slots). One row of the table each."""
    jobs = []
    for name in policy_names:
        if name == "full":
            continue
        if name in P.CLUSTER_POLICIES:
            for k in ks:
                if name == "hybrid":
                    jobs += [(name, float("nan"), k, f) for f in flow_slots]
                else:
                    jobs.append((name, float("nan"), k, None))
        elif name == "sink_only":
            jobs.append((name, float("nan"), None, None))
        else:
            jobs += [(name, b, None, None) for b in budgets]
    return jobs


def _flush(rows, out):
    """Write the CSV after EVERY job. The final write below is then just a
    no-op repeat -- but a walltime kill at hour 15 of 16 no longer costs the
    whole run."""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--policies",
                    default="full,sink_only,random,streaming,h2o,snapkv,oracle")
    ap.add_argument("--budgets", default="0.03,0.06,0.125,0.25")
    ap.add_argument("--ks", default="", help="cluster budgets k")
    ap.add_argument("--flow-slots", default="", help="hybrid allocation sweep")
    ap.add_argument("--index", default="", help="lineage the cluster POLICIES use")
    ap.add_argument("--escape-index", default="",
                    help="lineage used ONLY to measure escapes. Defaults to --index. "
                         "Set it on token runs too: it is how h2o and snapkv get a "
                         "number on the offline ladder's axis.")
    ap.add_argument("--n-sink", type=int, default=4)
    ap.add_argument("--lf-threshold", type=float, default=0.15,
                    help="routed: heads with local_frac below this use flow")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--min-turns", type=int, default=4)
    ap.add_argument("--max-sessions", type=int, default=100)
    ap.add_argument("--decode-chunk", type=int, default=1,
                    help="1 is exact; larger recomputes the mask every N tokens")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--out", default="results/ppl.csv")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype,
        attn_implementation="eager").to(args.device).eval()
    shape = A.model_shape(model)
    print(f"{args.model}: layers {shape[0]}, kv heads {shape[1]}, head_dim {shape[2]}")

    sessions = build_sessions(args.sessions, tok, args.max_len, args.min_turns,
                              args.max_sessions, args.device)
    print(f"{len(sessions)} sessions, "
          f"{sum(s.ids.shape[1] for s in sessions)} tokens, "
          f"{sum(len(s.spans) for s in sessions)} assistant turns")
    if not sessions:
        raise SystemExit("no sessions survived templating -- check the jsonl schema")

    index = load_index(args.index)
    escape_index = load_index(args.escape_index) or index
    if escape_index is not None:
        print(f"escape readout on (C={escape_index.n_clusters}, "
              f"thresh={A.ESCAPE_THRESHOLD})")

    state = A.PolicyState(P.FullCache(), n_sink=args.n_sink,
                          escape_index=escape_index)
    restore = A.install(model, state)
    rows = []
    try:
        print("reference pass (full cache) ...", flush=True)
        references, full_records = [], []
        for sess in sessions:
            state.policy = P.FullCache()
            rec, ref = run_session(model, sess, state, shape, args.device, dtype,
                                   args.decode_chunk, reference=None)
            references.append(ref)
            full_records.append(rec)
        rows.append(aggregate("full", 1.0, "budget", full_records, baseline=None))
        show(rows[-1])
        _flush(rows, args.out)

        jobs = plan_jobs(
            [p for p in args.policies.split(",") if p],
            [float(x) for x in args.budgets.split(",") if x],
            [int(x) for x in args.ks.split(",") if x],
            [int(x) for x in args.flow_slots.split(",") if x != ""] or [None])

        for name, budget, k, fslots in jobs:
            t0 = time.time()
            records = []
            for sess, ref in zip(sessions, references):
                state.policy = P.build(name, budget, index, args.n_sink, k,
                                       fslots, args.lf_threshold)
                rec, _ = run_session(model, sess, state, shape, args.device, dtype,
                                     args.decode_chunk, reference=ref)
                records.append(rec)
            label = name if fslots is None else f"{name}_f{fslots}"
            rows.append(aggregate(label, budget if k is None else k,
                                  "budget" if k is None else "k",
                                  records, baseline=rows[0]))
            show(rows[-1], time.time() - t0)
            _flush(rows, args.out)
    finally:
        restore()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")
    print("plot d_ppl and kl against RESIDENCY, never against knob.")


if __name__ == "__main__":
    main()