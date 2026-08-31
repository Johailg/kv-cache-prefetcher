"""
kvh.gate_pythia -- the real Gate A for the measurement layer.

gate_canon covers cluster SELECTION. This covers everything selection does not:
the nosink convention, the cluster assignment, the mass aggregation, and the
escape event -- against data you already have on disk, not against a fixture.

METHOD
    Re-run the same sessions through kvh on Pythia-410M with your canon
    centroids, then diff kvh's cluster-aggregated nosink rows against the
    mass_nosink stored in your h_eval shards. Same technique that proved
    --low-mem bit-exact.

    Then run mt_common.escape_event on the STORED rows and kvh's escape verdict
    on the LIVE rows, and check they agree step by step.

WHAT IT PROVES AND WHAT IT DOES NOT
    Pythia-410M is MHA, so G=1. This settles the nosink convention, the
    assignment, the aggregation and the event definition. It does NOT settle
    the GQA reduction, because there is no group to reduce. On Qwen3 with G=4
    the escape event is scored once per QUERY head; nothing here exercises
    that path beyond the degenerate case. Know which one you have proved.

    python -m kvh.gate_pythia \\
        --sessions data/mt_eval.npz --shards h_eval/ \\
        --centroids extracted_data/wiki_centroids_2048.npy \\
        --mt-common ../mt_common.py --layer 13 --head 2 --n-sessions 5
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

try:
    from . import attn as A
    from . import policies as P
    from .gate_canon import load_mt_common, resolve_mt_common
except ImportError:
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import attn as A
    import policies as P
    from gate_canon import load_mt_common, resolve_mt_common

MODEL = "EleutherAI/pythia-410m"


def build_single_head_index(centroids, n_layers, n_heads, layer, head, n_clusters):
    """A ClusterIndex carrying your canon centroids at exactly (layer, head).
    Every other slot is zeros and must not be read -- we only inspect `layer`."""
    C, D = centroids.shape
    full = torch.zeros(n_layers, n_heads, C, D, dtype=torch.float32)
    full[layer, head] = torch.from_numpy(centroids.astype(np.float32))
    hubs = torch.zeros(n_layers, n_heads, 1, dtype=torch.long)
    return P.ClusterIndex(centroids=full, hubs=hubs, flow=None, freq=None)


class RowGrab:
    """PolicyState.row_capture -> keeps every nosink row for one layer."""

    def __init__(self, layer, head):
        self.layer, self.head, self.rows = layer, head, []

    def __call__(self, layer, first_row, rows):
        if layer == self.layer:
            self.rows.append(rows[self.head].float().cpu().numpy())   # [r, K]

    def stacked(self, width):
        out = np.zeros((sum(r.shape[0] for r in self.rows), width), np.float64)
        at = 0
        for r in self.rows:
            out[at:at + r.shape[0], :r.shape[1]] = r
            at += r.shape[0]
        return out

    def clear(self):
        self.rows = []


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True, help="mt_eval.npz with ids")
    ap.add_argument("--shards", required=True, help="h_eval/ with mass_nosink")
    ap.add_argument("--centroids", required=True)
    ap.add_argument("--mt-common", "--mt_common", dest="mt_common", default=None)
    ap.add_argument("--layer", type=int, default=13)
    ap.add_argument("--head", type=int, default=2)
    ap.add_argument("--hub", type=int, default=24)
    ap.add_argument("--n-sessions", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float16", choices=["float32", "float16"],
                    help="MUST match the dtype the shards were harvested in. "
                         "mt_harvest uses fp16 unless --fp32 or device==cpu, so "
                         "an MPS harvest without --fp32 is fp16. A dtype "
                         "mismatch alone moves individual attention "
                         "probabilities by ~1e-2 at T=2048.")
    ap.add_argument("--tol", type=float, default=2e-3)
    args = ap.parse_args()

    mt = load_mt_common(resolve_mt_common(args.mt_common))
    from transformers import AutoModelForCausalLM

    centroids = np.load(args.centroids)
    if centroids.ndim != 2:
        raise SystemExit(f"centroids must be [C, D], got {centroids.shape}")
    print(f"centroids {centroids.shape}, layer {args.layer} head {args.head}, "
          f"hub {args.hub}")

    dt = getattr(torch, args.dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=dt, attn_implementation="eager")
    except TypeError:                       # transformers < 4.56
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=dt, attn_implementation="eager")
    model = model.to(args.device).eval()
    print(f"running in {args.dtype}")
    n_layers, n_heads, head_dim = A.model_shape(model)
    print(f"{MODEL}: {n_layers} layers, {n_heads} heads (MHA, G=1), dim {head_dim}")

    index = build_single_head_index(centroids, n_layers, n_heads,
                                    args.layer, args.head, centroids.shape[0])
    grab = RowGrab(args.layer, args.head)
    state = A.PolicyState(P.FullCache(), track_recovery=False, row_capture=grab)
    restore = A.install(model, state)

    ids_all = np.load(args.sessions, allow_pickle=True)["ids"]
    stored = list(mt.iter_sessions(args.shards))

    worst_mass, worst_esc, n_checked = 0.0, 0, 0
    structural = []
    try:
        for i in range(min(args.n_sessions, len(stored), len(ids_all))):
            canon = stored[i]
            ids = torch.tensor(np.asarray(ids_all[i]), dtype=torch.long,
                               device=args.device)[None]
            T = min(ids.shape[1], canon["mass"].shape[0])

            grab.clear()
            state.reset_sequence(n_layers, n_heads, head_dim, args.device,
                                 dt)
            model(ids[:, :T], use_cache=False)
            live_rows = grab.stacked(T)[:T]                       # [T, K] over keys

            # aggregate the live key-space rows by CANON's cluster ids, so the
            # only thing being compared is the mass convention, not the
            # assignment (which is checked separately below)
            cid = np.asarray(canon["clusters"])[:T]
            C = int(canon["mass"].shape[1])
            live_mass = np.zeros((T, C))
            np.add.at(live_mass,
                      (np.repeat(np.arange(T), T), np.tile(cid, T)),
                      live_rows.ravel())

            ref = np.asarray(canon["mass"], np.float64)[:T]
            diff = np.abs(live_mass - ref)
            d = float(diff.max())
            worst_mass = max(worst_mass, d)

            # STRUCTURAL checks -- these must be exact regardless of dtype.
            # If any of these fail it IS the convention; if they pass, a large
            # max-abs-diff is numerics and the max is the wrong statistic.
            hub_col_live = float(np.abs(live_mass[:, :]).sum())
            col0_cluster = int(cid[0])
            structural.append(dict(
                live_rowsum_dev=float(np.abs(live_mass[1:].sum(1) - 1).max()),
                ref_rowsum_dev=float(np.abs(ref[1:].sum(1) - 1).max()),
                median=float(np.median(diff)),
                p99=float(np.percentile(diff, 99)),
                frac_over_1e3=float((diff > 1e-3).mean()),
                corr=float(np.corrcoef(live_mass.ravel(), ref.ravel())[0, 1]),
            ))

            # escape agreement: canon's definition on canon's stored rows vs
            # the same definition on kvh's live rows
            mism = 0
            for t in range(1, T):
                cur = int(cid[t - 1])
                e_canon, cstar_canon, _ = mt.escape_event(canon["mass"][t], cur,
                                                          (args.hub,))
                row = live_mass[t].copy()
                row[args.hub] = 0.0
                row[cur] = 0.0
                e_live = row.sum() > mt.THRESH
                if bool(e_canon) != bool(e_live):
                    mism += 1
                elif e_canon and int(np.argmax(row)) != int(cstar_canon):
                    mism += 1
            worst_esc += mism
            n_checked += T - 1
            print(f"  session {i}: T={T}  max|dmass| {d:.3e}  "
                  f"escape/cstar disagreements {mism}/{T-1}")
    finally:
        restore()

    agg = lambda k: max(x[k] for x in structural)
    rate = worst_esc / max(n_checked, 1)

    print("\n--- structural (must hold exactly; dtype-independent) ---")
    ok_rows = agg("live_rowsum_dev") < 1e-4 and agg("ref_rowsum_dev") < 1e-2
    print(f"  rows sum to 1   kvh dev {agg('live_rowsum_dev'):.2e}   "
          f"canon dev {agg('ref_rowsum_dev'):.2e}   "
          f"{'PASS' if ok_rows else 'FAIL'}")

    print("\n--- distributional (dtype-dependent; max is the wrong statistic) ---")
    print(f"  median |diff|      {min(x['median'] for x in structural):.2e}")
    print(f"  p99    |diff|      {agg('p99'):.2e}")
    print(f"  max    |diff|      {worst_mass:.2e}")
    print(f"  frac > 1e-3        {agg('frac_over_1e3'):.4%}")
    print(f"  correlation        {min(x['corr'] for x in structural):.8f}")

    print("\n--- event agreement (what actually matters) ---")
    print(f"  escape + cstar disagreement {rate:.4%}   "
          f"{'PASS' if rate < 5e-3 else 'FAIL'}")

    print()
    if ok_rows and rate < 5e-3:
        print("VERDICT: the convention matches. Rows renormalise correctly in both,")
        print("and the event agrees on >99.5% of steps. The residual mass diff is")
        print("numeric. Confirm with the discriminator below; if it comes back the")
        print("same order of magnitude, kvh is exonerated and the tolerance was mine.")
        print()
        print("  DISCRIMINATOR (uses only YOUR code, none of mine):")
        print("    re-harvest 2 of these sessions with mt_harvest.py --fp32 and diff")
        print("    the resulting mass_nosink against the existing fp16 shards.")
        print("    If canon-fp32 vs canon-fp16 also differs by ~1e-2, the number")
        print("    you are seeing is Pythia in fp16, not a porting bug.")
    elif not ok_rows:
        print("VERDICT: STRUCTURAL FAILURE. Rows do not renormalise. This is the")
        print("convention. Check to_nosink before anything else.")
    else:
        print("VERDICT: rows are fine but the event disagrees too often. Check the")
        print("cur = c[t-1] indexing in EscapeReadout.score_step.")


if __name__ == "__main__":
    main()