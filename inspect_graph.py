"""
Is the flow graph trivial? Diagnostics on BOTH builds.

  raw  : P_flow with hubs included. Johail's prediction lives here, on
         record: many rows share identical (top-1, top-2) pairs pointing at
         the hub(s); row entropy low-to-moderate.
  excl : canonical HUBS=(16,) build -- the object behind every post-pin
         result. The hub prediction CANNOT apply here (column 16 is
         zeroed); the live question for this build is row diversity.

Claude's counter-prediction, on record: the excl build shows >= 40 distinct
top-4 staged sets across the 62 non-hub rows. Full degeneracy is already
excluded by Test A (flow >> static -- identical rows would make flow a
static predictor) and by the nonzero control-stratum penalty (0.069).

Prints per build:
  - top-1 destination histogram over rows (mode share)
  - most common ordered (top-1, top-2) pairs over rows
  - number of distinct top-4 sets (non-hub rows only, seedless -- pure row
    content, no current-cluster freebie)
  - row entropy: median / p10 / p90, vs entropy of the global row g
  - top-5 spectral energy share (squared singular values)
  - saves flowgraph_raw.png / flowgraph_excl.png, rows sorted by train
    occupancy (most-visited state at the top)

Offline, instant. Requires harvest_wiki.npz.
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

K = 64
HUBS = (16,)


def build(clusters, mass, train_idx, alpha=0.1, exclude=()):
    A = np.full((K, K), alpha)
    g = np.zeros(K)
    occ = np.zeros(K)
    n = 0
    for w in train_idx:
        c, m = clusters[w], mass[w]
        for t in range(1, len(c)):
            A[c[t - 1]] += m[t]
            g += m[t]
            occ[c[t - 1]] += 1
            n += 1
    for j in exclude:
        A[:, j] = 0.0
        g[j] = 0.0
    P = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
    return P, g / max(n, 1), occ


def entropy(p):
    p = p[p > 1e-12]
    return float(-(p * np.log2(p)).sum())


def report(P, g, occ, label, drop_hubs):
    rows = [i for i in range(K) if not (drop_hubs and i in HUBS)]
    order = np.argsort(P, axis=1)[:, ::-1]

    top1 = Counter(int(order[i, 0]) for i in rows)
    top2 = Counter((int(order[i, 0]), int(order[i, 1])) for i in rows)
    sets4 = {frozenset(int(x) for x in order[i, :4]) for i in rows}
    H = np.array([entropy(P[i]) for i in rows])
    s = np.linalg.svd(P[rows], compute_uv=False)
    e = s ** 2

    print(f"\n===== {label} ({len(rows)} rows) =====")
    print("top-1 destinations: " + ", ".join(
        f"c{c}x{n}" for c, n in top1.most_common(6)))
    print(f"  mode share: {top1.most_common(1)[0][1] / len(rows):.2f}")
    print("top (top-1, top-2) pairs: " + ", ".join(
        f"{p}x{n}" for p, n in top2.most_common(5)))
    print(f"distinct top-4 sets: {len(sets4)} / {len(rows)}")
    print(f"row entropy bits: median {np.median(H):.2f}  "
          f"p10 {np.percentile(H, 10):.2f}  p90 {np.percentile(H, 90):.2f}  "
          f"| H(g) = {entropy(g):.2f}")
    print(f"top-5 spectral energy share: {e[:5].sum() / e.sum():.3f}")

    srt = sorted(rows, key=lambda i: -occ[i])
    plt.figure(figsize=(7, 6))
    plt.imshow(P[srt], aspect="auto", cmap="viridis")
    plt.colorbar(label="P_flow[row, col]")
    plt.xlabel("destination cluster")
    plt.ylabel("state cluster (sorted by train occupancy)")
    plt.title(label)
    fname = f"flowgraph_{'excl' if drop_hubs else 'raw'}.png"
    plt.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"saved {fname}")


if __name__ == "__main__":
    z = np.load("extracted_data/harvest_wiki.npz")
    clu, mass = z["clusters"], z["mass_nosink"]
    train_idx = list(range(0, len(clu), 2))

    P_raw, g_raw, occ = build(clu, mass, train_idx)
    P_exc, g_exc, _ = build(clu, mass, train_idx, exclude=set(HUBS))

    report(P_raw, g_raw, occ, "RAW build (hubs included)", drop_hubs=False)
    report(P_exc, g_exc, occ, "EXCLUDED build, HUBS=(16,)", drop_hubs=True)