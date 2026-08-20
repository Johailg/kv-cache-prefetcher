"""
kvh.gate_baselines -- certify the published baselines against reference
semantics, the way gate_canon certifies the cluster policies against
mt_common.

    python gate_baselines.py                     # everything, ~seconds, CPU
    python gate_baselines.py --policy h2o
    python -m kvh.gate_baselines --draws 8

No model, no GPU, no transformers. Runs on a login node while the allocation
is dry.

WHAT IT PROVES AND WHAT IT DOES NOT
-----------------------------------
It proves the SELECTION is the published one: given the same attention
stream, kvh's select() returns the same resident set as the reference
algorithm, index for index. Everything else about the run -- the model, the
sessions, the ppl arithmetic -- is out of scope; G1 already covers the claim
that a full-cache patched forward reproduces stock logits.

The reference side is a transcription, not a copy, of

    H2O     FMInference/H2O   h2o_hf/utils_real_drop/modify_llama.py
                              class H2OKVCache_LayerWise (__call__,
                              _update_hh_score)
    SnapKV  FasterDecoding/SnapKV
                              snapkv/monkeypatch/snapkv_utils.py
                              class SnapKVCluster.update_kv

Their code is entangled with real cache surgery -- it gathers surviving keys
into a new contiguous tensor, which shifts positions and is exactly why it
cannot be imported into a masking harness. What is transcribed here is the
selection rule only, expressed on original (unshifted) indices. Check the
upstream licences before redistributing this file with a repo.

CALIBRATION, WHICH IS PART OF THE RESULT
----------------------------------------
Three kvh conventions have to be neutralised for the diff to mean anything.
Each is a real difference from the papers; the gate makes each one a measured
quantity rather than a footnote.

  n_sink      the sink pin has no counterpart in either paper. Run with
              --n-sink 0 for the exact diff; --sink-sweep reports how many
              index positions the pin moves at n_sink = 4.
  budget      kvh's `keep` counts OLD columns and the newly written key is
              free on top, so kvh's resident total is keep + n_new while the
              papers' cache_size includes the new key. The gate drives kvh
              with an ABSOLUTE budget equal to the reference cache_size, which
              makes the two agree exactly; under a fractional knob the same
              convention is a constant +n_new offset, in the same class as the
              sink offset and equally harmless on a residency axis.
  GQA         kvh collapses the G query heads sharing a KV head into one
              keep/drop decision (group_sum). Neither reference does. The gate
              group-sums the reference too for the exact diff, and --gqa
              reports the disagreement between collapsed and per-head
              selection as its own number.
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from . import attn as A
    from . import policies as P
except ImportError:
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import attn as A
    import policies as P


# ==========================================================================
# driving the REAL kvh path
# ==========================================================================

def kvh_call(policy, layer, n_old, q, k, v, attn_mask, n_kv, groups, chunk=64):
    """One forward call through the actual attn.py machinery.

    Mirrors attn.make_attention_fn's step order exactly -- select, then the
    masked attention, then observe -- so the rolling observation-window buffer
    in _masked_attention is exercised rather than re-implemented here.

    q [1, Hq, Q, D], k/v [1, Hkv, K, D] with K = n_old + Q.
    -> resident mask over the OLD columns, [Hkv, n_old] bool (or None).
    """
    k_len = k.shape[2]
    key_rep, value_rep = A.repeat_kv(k, groups), A.repeat_kv(v, groups)

    policy_mask = policy.select(layer, n_old, k_len - n_old)
    policy.note_keys(layer, k[0, :, n_old:, :])

    resident = A.build_resident_mask(policy_mask, n_kv, k_len, n_old, q.device)
    score_mask = (None if resident is None
                  else resident.repeat_interleave(groups, dim=0)[None, :, None, :])

    obs_rows = int(getattr(policy, "obs_rows", 0) or 0)
    _, stats = A._masked_attention(
        q, key_rep, value_rep, attn_mask, q.shape[-1] ** -0.5, score_mask,
        n_kv, groups, chunk, track_recovery=False, obs_rows=obs_rows)

    policy.observe(P.Obs(
        layer=layer,
        col_mass_resident=stats["mass_resident"],
        col_mass_true=stats["mass_true"],
        col_mass_nosink=stats["mass_nosink"],
        col_mass_resident_lastw=stats["mass_resident_lastw"],
        n_old=n_old, n_new=k_len - n_old, is_prefill=(q.shape[2] > 1)))
    return policy_mask


def resident_sets(policy_mask, n_old, k_len, n_kv) -> List[frozenset]:
    """[Hkv, n_old] bool (or None) -> one frozenset of ORIGINAL indices per KV
    head, including the always-resident new columns."""
    out = []
    for h in range(n_kv):
        if policy_mask is None:
            out.append(frozenset(range(k_len)))
        else:
            idx = torch.nonzero(policy_mask[h], as_tuple=False).flatten().tolist()
            out.append(frozenset(idx) | set(range(n_old, k_len)))
    return out


def causal_additive(q_len, k_len, device, dtype):
    """The additive mask transformers materialises on the eager path."""
    i = torch.arange(k_len - q_len, k_len, device=device).unsqueeze(1)
    j = torch.arange(k_len, device=device).unsqueeze(0)
    m = torch.zeros(q_len, k_len, device=device, dtype=dtype)
    m.masked_fill_(j > i, torch.finfo(dtype).min)
    return m[None, None]


# ==========================================================================
# reference: H2O
# ==========================================================================

class RefH2O:
    """H2OKVCache_LayerWise, on original indices.

    Faithful to three properties of the released class:

      _update_hh_score   hh_score accumulates the attention of the CURRENT
                         (already compressed) cache, summed over query rows.
      __call__           heavy hitters come from hh_score[:, :seq_len -
                         recent_size], i.e. the prefix only; the recent window
                         is concatenated afterwards and never contested.
      compaction         `self.hh_score = self.hh_score[mask]` physically drops
                         the evicted entries, so eviction is IRREVERSIBLE.

    Not reproduced, because it is the part that cannot live in a masking
    harness: the gather that rebuilds key_states contiguously and renumbers
    every surviving position.
    """

    def __init__(self, n_heads, hh_size, recent_size, n_positions):
        self.hh_size, self.recent_size = hh_size, recent_size
        self.cache = hh_size + recent_size
        self.resident = [[] for _ in range(n_heads)]
        # float32, matching kvh's own _score accumulator. Held at full width
        # rather than compacted -- an evicted entry is never in `cols` again,
        # so it neither accumulates nor becomes a candidate, which is what the
        # released compaction achieves by deleting it.
        self.hh = torch.zeros(n_heads, n_positions, dtype=torch.float32)

    def step(self, rows: List[torch.Tensor], new_index: int) -> List[frozenset]:
        """rows[h] is head h's attention over its own resident set INCLUDING
        the new key, summed over query rows. -> the attention set used."""
        used = []
        for h, row in enumerate(rows):
            self.resident[h].append(new_index)
            cols = list(self.resident[h])
            used.append(frozenset(cols))
            idx = torch.tensor(cols, dtype=torch.long)
            self.hh[h].index_add_(0, idx, row.float())
            if len(cols) > self.cache:
                cut = len(cols) - self.recent_size
                recent = cols[cut:] if self.recent_size > 0 else []
                prefix = cols[:cut]
                order = sorted(prefix, key=lambda c: -float(self.hh[h, c]))
                surv = set(order[:self.hh_size]) | set(recent)
                self.resident[h] = sorted(surv)
        return used


def gate_h2o(args, device, dtype) -> Tuple[int, int]:
    torch.manual_seed(args.seed)
    n_kv, groups, D = args.kv_heads, args.groups, args.head_dim
    n_q, T = n_kv * groups, args.steps
    bad = tot = 0

    for draw in range(args.draws):
        q_all = torch.randn(1, n_q, T, D, device=device, dtype=dtype)
        k_all = torch.randn(1, n_kv, T, D, device=device, dtype=dtype)
        v_all = torch.randn(1, n_kv, T, D, device=device, dtype=dtype)

        pol = P.H2O(float(args.cache), heavy_ratio=args.heavy_ratio,
                    n_sink=args.n_sink)
        pol.n_sink = args.n_sink
        pol.reset(1, n_kv, D, device, dtype)

        # the reference's (hh_size, recent_size) is read off the split kvh uses
        # in STEADY STATE -- once n_old >= cache, kvh's keep is pinned at
        # `cache` and the split stops moving. Below that both sides keep
        # everything, so the fill phase agrees trivially.
        free = max(0, args.cache - (args.n_sink if P.CHARGE_SINKS else 0))
        n_recent = int(round(free * (1.0 - args.heavy_ratio)))
        ref = RefH2O(n_kv, free - n_recent, n_recent, T)

        for t in range(T):
            mask = kvh_call(pol, 0, t, q_all[:, :, t:t + 1], k_all[:, :, :t + 1],
                            v_all[:, :, :t + 1], None, n_kv, groups, args.chunk)
            kvh_sets = resident_sets(mask, t, t + 1, n_kv)

            rows = []
            for h in range(n_kv):
                cols = sorted(ref.resident[h] + [t])
                sc = torch.zeros(len(cols), dtype=torch.float32)
                for g in range(groups):
                    qq = q_all[0, h * groups + g, t]
                    kk = k_all[0, h, cols]
                    sc += F.softmax(((kk @ qq) * (D ** -0.5)).float(), dim=-1)
                rows.append(sc)
            ref_sets = [frozenset(s) for s in ref.step(rows, t)]

            for h in range(n_kv):
                tot += 1
                if kvh_sets[h] != ref_sets[h]:
                    bad += 1
                    if bad <= args.show:
                        print(f"    draw {draw} t={t} h={h}: "
                              f"ref-only {sorted(ref_sets[h] - kvh_sets[h])[:8]}  "
                              f"kvh-only {sorted(kvh_sets[h] - ref_sets[h])[:8]}")
    return bad, tot


# ==========================================================================
# reference: SnapKV
# ==========================================================================

def ref_snapkv(q, k, cap, window, kernel, pooling, n_kv, groups,
               collapse=True) -> List[frozenset]:
    """SnapKVCluster.update_kv, returning kept ORIGINAL indices per head.

        attn = softmax(Q[-W:] K^T / sqrt(d))          causal on the WxW tail
        vote = attn[..., -W:, :-W].sum(dim=-2)        PREFIX columns only
        pool = {max,avg}_pool1d(vote, kernel, pad=kernel//2, stride=1)
        keep = topk(pool, cap - W)  U  [P-W, P)

    collapse=True sums the G query heads sharing a KV head before pooling,
    which is what kvh does; False scores each query head on its own, which is
    what the reference does. The difference is reported by --gqa.
    """
    P_ = k.shape[2]
    D = k.shape[-1]
    if P_ < cap:
        return [frozenset(range(P_)) for _ in range(n_kv if collapse else q.shape[1])]

    W = min(window, P_)
    prefix = P_ - W
    heads = n_kv if collapse else q.shape[1]
    votes = torch.zeros(heads, prefix, dtype=torch.float64, device=k.device)
    for hq in range(q.shape[1]):
        h = hq // groups
        w = (q[0, hq, -W:] @ k[0, h].transpose(0, 1)) * (D ** -0.5)   # [W, P]
        keep_mask = torch.ones(W, P_, dtype=torch.bool, device=k.device)
        for i in range(W):
            keep_mask[i, prefix + i + 1:] = False
        w = w.masked_fill(~keep_mask, float("-inf"))
        p = F.softmax(w.float(), dim=-1).double()
        votes[h if collapse else hq] += p[:, :prefix].sum(dim=0)

    op = F.max_pool1d if pooling == "max" else F.avg_pool1d
    pooled = op(votes.unsqueeze(1).float(), kernel_size=kernel, stride=1,
                padding=kernel // 2).squeeze(1)[:, :prefix]
    take = min(cap - W, prefix)
    top = pooled.topk(take, dim=1).indices
    return [frozenset(top[h].tolist()) | set(range(prefix, P_))
            for h in range(heads)]


def gate_snapkv(args, device, dtype) -> Tuple[int, int]:
    torch.manual_seed(args.seed + 1)
    n_kv, groups, D = args.kv_heads, args.groups, args.head_dim
    n_q, P_ = n_kv * groups, args.prefill
    bad = tot = 0

    for pooling, kernel in [("max", 7), ("max", 5), ("avg", 5), ("avg", 7)]:
        sub_bad = 0
        for draw in range(args.draws):
            q = torch.randn(1, n_q, P_, D, device=device, dtype=dtype)
            k = torch.randn(1, n_kv, P_, D, device=device, dtype=dtype)
            v = torch.randn(1, n_kv, P_, D, device=device, dtype=dtype)

            pol = P.SnapKV(float(args.cache), obs_window=args.window,
                           pool=kernel, pooling=pooling, n_sink=args.n_sink)
            pol.n_sink = args.n_sink
            pol.reset(1, n_kv, D, device, dtype)

            m = causal_additive(P_, P_, device, dtype)
            kvh_call(pol, 0, 0, q, k, v, m, n_kv, groups, args.chunk)  # the prefill
            mask = pol.select(0, P_, 1)                          # first decode
            kvh_sets = resident_sets(mask, P_, P_, n_kv)

            ref_sets = ref_snapkv(q, k, args.cache, args.window, kernel,
                                  pooling, n_kv, groups, collapse=True)
            for h in range(n_kv):
                tot += 1
                if kvh_sets[h] != ref_sets[h]:
                    bad += 1
                    sub_bad += 1
                    if bad <= args.show:
                        print(f"    {pooling}/{kernel} draw {draw} h={h}: "
                              f"ref-only {sorted(ref_sets[h] - kvh_sets[h])[:8]}  "
                              f"kvh-only {sorted(kvh_sets[h] - ref_sets[h])[:8]}")
        flag = "ok " if sub_bad == 0 else "DIFF"
        print(f"    {flag} pooling={pooling} kernel={kernel}: "
              f"{sub_bad}/{args.draws * n_kv} heads mismatched")
    return bad, tot


# ==========================================================================
# side measurements: what the deviations actually cost
# ==========================================================================

def measure_gqa(args, device, dtype):
    """How much does collapsing G query heads into one decision change the
    selected set? Reported as Jaccard between the collapsed choice and each
    query head's own choice. This is a SHARED constraint -- H2O and the cluster
    policies pay it too -- so it is context for the memo, not a defect."""
    torch.manual_seed(args.seed + 2)
    n_kv, groups, D, P_ = args.kv_heads, args.groups, args.head_dim, args.prefill
    if groups == 1:
        print("    groups=1, nothing to collapse")
        return
    sims = []
    for _ in range(args.draws):
        q = torch.randn(1, n_kv * groups, P_, D, device=device, dtype=dtype)
        k = torch.randn(1, n_kv, P_, D, device=device, dtype=dtype)
        col = ref_snapkv(q, k, args.cache, args.window, 7, "max", n_kv, groups, True)
        per = ref_snapkv(q, k, args.cache, args.window, 7, "max", n_kv, groups, False)
        for hq in range(n_kv * groups):
            a, b = col[hq // groups], per[hq]
            sims.append(len(a & b) / max(len(a | b), 1))
    mean = sum(sims) / len(sims)
    print(f"    collapsed-vs-per-query-head Jaccard: mean {mean:.4f} "
          f"min {min(sims):.4f} over {len(sims)} head-draws")


def measure_sink(args, device, dtype):
    """How many index positions does the sink pin move, at n_sink = 4?
    The pin is kvh's addition; neither paper has one."""
    torch.manual_seed(args.seed + 3)
    n_kv, groups, D, P_ = args.kv_heads, args.groups, args.head_dim, args.prefill
    moved = []
    for _ in range(args.draws):
        q = torch.randn(1, n_kv * groups, P_, D, device=device, dtype=dtype)
        k = torch.randn(1, n_kv, P_, D, device=device, dtype=dtype)
        v = torch.randn(1, n_kv, P_, D, device=device, dtype=dtype)
        sets = {}
        for ns in (0, args.n_sink_probe):
            pol = P.SnapKV(float(args.cache), obs_window=args.window, pool=7,
                           pooling="max", n_sink=ns)
            pol.n_sink = ns
            pol.reset(1, n_kv, D, device, dtype)
            m = causal_additive(P_, P_, device, dtype)
            kvh_call(pol, 0, 0, q, k, v, m, n_kv, groups, args.chunk)
            sets[ns] = resident_sets(pol.select(0, P_, 1), P_, P_, n_kv)
        for h in range(n_kv):
            a, b = sets[0][h], sets[args.n_sink_probe][h]
            moved.append(len(a ^ b) / 2.0)
    print(f"    n_sink 0 -> {args.n_sink_probe}: "
          f"{sum(moved)/len(moved):.2f} index positions swapped per head "
          f"(cap {args.cache}), max {max(moved):.0f}")


# ==========================================================================
# driver
# ==========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="all", choices=["all", "h2o", "snapkv"])
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--steps", type=int, default=60, help="h2o: decode steps")
    ap.add_argument("--prefill", type=int, default=192, help="snapkv: prompt length")
    ap.add_argument("--cache", type=int, default=64,
                    help="absolute cache size; the papers' knob")
    ap.add_argument("--window", type=int, default=32, help="snapkv obs window")
    ap.add_argument("--heavy-ratio", type=float, default=0.5)
    ap.add_argument("--n-sink", type=int, default=0,
                    help="0 for the exact diff; neither paper has sinks")
    ap.add_argument("--n-sink-probe", type=int, default=4,
                    help="value used by --sink-sweep")
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--groups", type=int, default=2, help="G, the GQA factor")
    ap.add_argument("--head-dim", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=64,
                    help="query-chunk size; keep it BELOW --prefill so the "
                         "rolling observation-window buffer spans chunks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=6, help="max mismatches printed")
    ap.add_argument("--gqa", action="store_true", help="measure the group collapse")
    ap.add_argument("--sink-sweep", action="store_true", help="measure the sink pin")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    print(f"gate_baselines  Hkv={args.kv_heads} G={args.groups} D={args.head_dim} "
          f"cache={args.cache} n_sink={args.n_sink} "
          f"CHARGE_SINKS={P.CHARGE_SINKS} {args.dtype}/{args.device}")
    if args.n_sink != 0:
        print("  NOTE: n_sink != 0. Neither reference has a sink concept, so "
              "mismatches here are expected and are not a fidelity failure.")

    fails = []
    if args.policy in ("all", "h2o"):
        print(f"\nH2O vs H2OKVCache_LayerWise  ({args.draws} draws x "
              f"{args.steps} steps x {args.kv_heads} heads)")
        bad, tot = gate_h2o(args, device, dtype)
        print(f"  {'PASS' if bad == 0 else 'FAIL'}  {bad}/{tot} "
              f"({100.0 * bad / max(tot, 1):.3f}%) step-head resident sets differ")
        if bad:
            fails.append("h2o")

    if args.policy in ("all", "snapkv"):
        print(f"\nSnapKV vs SnapKVCluster.update_kv  ({args.draws} draws x "
              f"{args.kv_heads} heads, prefill {args.prefill})")
        bad, tot = gate_snapkv(args, device, dtype)
        print(f"  {'PASS' if bad == 0 else 'FAIL'}  {bad}/{tot} "
              f"({100.0 * bad / max(tot, 1):.3f}%) head selections differ")
        if bad:
            fails.append("snapkv")

    if args.gqa:
        print("\nGQA group collapse (a kvh/paper difference, not a bug)")
        measure_gqa(args, device, dtype)
    if args.sink_sweep:
        print("\nSink pin (a kvh addition, no counterpart in either paper)")
        measure_sink(args, device, dtype)

    print("\n" + ("ALL PASS" if not fails else "FAILED: " + ", ".join(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
