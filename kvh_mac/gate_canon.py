"""
kvh.gate_canon -- the missing Gate A, for the cluster layer only.

WHAT THIS IS FOR
----------------
Reading 2400 lines to check whether my cluster policies match yours is the
expensive way. This is the cheap way: drive mt_common.topk / recency_set /
hybrid_set and the corresponding kvh policy on the SAME synthetic step, and
assert the chosen cluster sets are identical. On divergence it prints the step,
both sets, and the state that produced them.

It needs no model, no GPU, no harvest. It runs in seconds on a laptop.

IT ALSO WORKS ON CODE YOU WRITE YOURSELF. If you rewrite _ClusterPolicy from
mt_common rather than reading mine, point this at your version and it tells you
whether you got it right. That is the whole argument for rewriting only the
~200 lines that carry semantics: the check is mechanical either way.

    python -m kvh.gate_canon --mt-common /path/to/mt_common.py --trials 2000

WHAT IT DOES NOT COVER
----------------------
The mass convention, the escape event on real attention, GQA group reduction,
and everything in attn.py. Those need G1 and a real model. This covers exactly
one thing: given a history of cluster ids and a flow row, do the two
implementations stage the same clusters.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys

import numpy as np
import torch

# Works both as a package (`python -m kvh.gate_canon`) and as a plain script
# (`python gate_canon.py` from inside the directory). The relative form is tried
# first so the package layout stays canonical.
try:
    from . import policies as P
except ImportError:
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import policies as P


def resolve_mt_common(given):
    """Find mt_common.py. The file lives in the research repo, not next to the
    harness, so a bare relative path from the wrong cwd is the usual failure."""
    import os
    candidates = [given] if given else []
    here = os.path.dirname(os.path.abspath(__file__))
    candidates += ["mt_common.py",
                   os.path.join("..", "mt_common.py"),
                   os.path.join(here, "mt_common.py"),
                   os.path.join(here, "..", "mt_common.py")]
    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    raise SystemExit(
        "could not find mt_common.py. Tried:\n  " + "\n  ".join(
            os.path.abspath(c) for c in candidates if c) +
        "\nPass it explicitly:  --mt-common /full/path/to/mt_common.py")


def load_mt_common(path):
    spec = importlib.util.spec_from_file_location("mt_common", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mt_common"] = mod
    spec.loader.exec_module(mod)
    return mod


def make_index(flow_row_for_cur, freq, hub, n_clusters, head_dim=8):
    """A ClusterIndex whose flow matrix has the same row for every state, so the
    policy's row lookup is pinned to the fixture's row. Centroids are unused
    here because we inject the assignment directly."""
    C = n_clusters
    flow = torch.zeros(1, 1, C, C)
    flow[0, 0] = torch.tensor(flow_row_for_cur, dtype=torch.float32)[None, :].repeat(C, 1)
    return P.ClusterIndex(
        centroids=torch.zeros(1, 1, C, head_dim),
        hubs=torch.tensor([[[hub]]], dtype=torch.long),
        flow=flow,
        freq=torch.tensor(freq, dtype=torch.float32).view(1, 1, C))


def drive_policy(policy, history, n_clusters):
    """Run one select() against an injected history, return the chosen set.

    history: list[int] of cluster ids for positions 0..t-1, so history[-1] is
    c[t-1] -- exactly what select() sees when attn.py calls it before
    note_keys(). Bypassing note_keys is the point: it isolates the SELECTION
    logic from the key-assignment logic.
    """
    policy.reset(1, 1, 8, torch.device("cpu"), torch.float32)
    ids = torch.tensor(history, dtype=torch.long).view(1, -1)      # [1, n_old]
    policy._assign[0] = ids

    last = torch.full((1, n_clusters), -1.0)
    for step, c in enumerate(history):
        last[0, c] = float(step)
    policy._last[0] = last
    policy._step[0] = len(history)

    policy.select(layer=0, n_old=len(history), n_new=1)
    chosen = policy.chosen_clusters(0)
    return {int(c) for c in torch.nonzero(chosen[0]).flatten()}


def compare(mt, trials, n_clusters, seed, verbose_on_fail=True):
    rng = np.random.default_rng(seed)
    hubs = (mt.HUBS[0],)
    results = {}

    cases = ["flow", "static", "recency_raw", "recency_fair"] + \
            [f"hybrid_f{f}" for f in (0, 1, 2)]
    for case in cases:
        mismatches, examples = 0, []
        for _ in range(trials):
            k = int(rng.integers(2, 9))
            hist_len = int(rng.integers(k + 3, 300))
            history = list(rng.integers(0, n_clusters, size=hist_len))
            cur = int(history[-1])
            flow_row = rng.random(n_clusters)
            freq = rng.random(n_clusters)

            # ---- canon ----
            if case == "flow":
                want = mt.topk(flow_row, cur, k, hubs)
            elif case == "static":
                want = mt.topk(freq, cur, k, hubs)
            elif case == "recency_raw":
                want = mt.recency_set(np.array(history), k)
            elif case == "recency_fair":
                want = mt.hybrid_set(cur, np.argsort(flow_row)[::-1],
                                     np.array(history), k, hubs, 0)
            else:
                f = int(case.split("_f")[1])
                want = mt.hybrid_set(cur, np.argsort(flow_row)[::-1],
                                     np.array(history), k, hubs, f)

            # ---- kvh ----
            index = make_index(flow_row, freq, hubs[0], n_clusters)
            name = "hybrid" if case.startswith("hybrid") else case
            fslots = int(case.split("_f")[1]) if case.startswith("hybrid") else None
            policy = P.build(name, float("nan"), index, n_sink=0, k=k,
                             flow_slots=fslots)
            got = drive_policy(policy, history, n_clusters)

            # kvh always has the hub physically resident on top of whatever
            # canon staged, for EVERY policy including recency_raw (where canon
            # may already have the hub among its k). So the identity to test is
            #     kvh_set == canon_set | hubs
            # Subtracting the hub from kvh instead, as the first version of this
            # file did, wrongly failed recency_raw whenever the hub happened to
            # be recent -- which is most of the time, since the hub carries
            # ~0.6 of the mass.
            expected = set(int(x) for x in want) | set(hubs)
            if got != expected:
                mismatches += 1
                if len(examples) < 3:
                    examples.append((k, cur, sorted(expected), sorted(got),
                                     sorted(expected ^ got)))
        rate = mismatches / trials
        results[case] = rate
        flag = "PASS" if rate == 0 else ("~" if rate < 0.02 else "FAIL")
        print(f"  {case:14s} mismatch {rate:6.3%}   {flag}")
        if examples and verbose_on_fail:
            for k, cur, want, got, diff in examples:
                print(f"      k={k} cur={cur}\n"
                      f"        canon {want}\n"
                      f"        kvh   {got}\n"
                      f"        symmetric difference {diff}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mt-common", "--mt_common", dest="mt_common", default=None,
                    help="path to your mt_common.py; searched for automatically "
                         "in . and .. if omitted")
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = resolve_mt_common(args.mt_common)
    mt = load_mt_common(path)
    print(f"loaded {path}")
    print(f"mt_common: K={mt.K} HUBS={mt.HUBS} ALPHA={mt.ALPHA} THRESH={mt.THRESH}")
    print(f"{args.trials} random (k, history, flow row) draws per policy\n")
    results = compare(mt, args.trials, mt.K, args.seed)

    worst = max(results.values())
    print()
    if worst == 0:
        print("EXACT MATCH on every policy. The cluster layer is a faithful port.")
    elif worst < 0.02:
        print("Small mismatch rate. Most likely tie-breaking: torch.topk and "
              "np.argsort order ties differently. Check whether the differing "
              "clusters have equal scores; if so this is cosmetic. If not, it "
              "is a real semantic difference and the numbers are not comparable "
              "to canon.")
    else:
        print("REAL DIVERGENCE. Do not run the e2e sweep until this is zero -- "
              "any cluster number it produces is measuring a different policy "
              "than the offline ladder did.")


if __name__ == "__main__":
    main()