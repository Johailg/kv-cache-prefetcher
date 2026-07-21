"""
rowmap_curve.py -- THE ISOMORPHISM TEST (pre-registered in memo Section 6).

On record before first run, verbatim from the memo: "I predict that the
curves will agree for smaller perturbations, but for larger ones the
similarity will decay faster than coverage."

Two ways to hand the prefetcher a WRONG pointer p in place of the true state
c* = c_{t-1}:
  time  : p = c_{t-1-lag}, lag in LAGS         (staleness -- moves along
                                                 graph EDGES)
  noise : p = runner-up centroid of the key at  (ambiguity -- moves along
          t-1, from runnerup_{tag}.npz           METRIC proximity)

Both reduce to "consult P_flow[p] at metric displacement x = d(mu_p, mu_c*)".
Per perturbed step we record:
  x     centroid distance d(p, c*)
  jac   J(S(p), S(c*)) between the two staged sets
        [structural ceiling for identical rows: (k-1)/(k+1) = 0.6 at k=4]
  cov   true non-hub attention mass at t captured by S(p)
  cov0  same mass captured by S(c*)  (unperturbed reference)

Output: per-distance-bin means split by perturbation origin, plus a per-lag
table (connects directly to the Test F staleness sweep).

Read-out:
  same y at same x for both origins -> smoothness is a function of metric
      distance alone; time and noise are the same probe (isomorphism holds)
  time above noise at same x        -> graph edges carry structure beyond
      metric proximity; temporal locality is its own mechanism
Self-transitions (p == c*) are excluded: they are trivially jac = 1 and
would swamp the small-lag strata.

Requires harvest_{tag}.npz, runnerup_{tag}.npz, wiki_centroids.npy.
"""
import numpy as np
from scipy.spatial.distance import cdist

K = 64
HUBS = (16,)
K_SET = 4
LAGS = (1, 2, 4, 8, 16, 32)
NBINS = 8


def load(tag):
    z = np.load(f"extracted_data/harvest_{tag}.npz")
    r = np.load(f"extracted_data/runnerup_{tag}.npz")
    return z["clusters"], z["mass_nosink"], r["top2"]


def build_flow(clusters, mass, train_idx, alpha=0.1, exclude=()):
    A = np.full((K, K), alpha)
    for w in train_idx:
        c, m = clusters[w], mass[w]
        for t in range(1, len(c)):
            A[c[t - 1]] += m[t]
    for j in exclude:
        A[:, j] = 0.0
    return A / A.sum(1, keepdims=True)


def topk(row, cur, k, exclude=()):
    s = [cur] if cur not in exclude else []
    for j in np.argsort(row)[::-1]:
        if j != cur and j not in exclude:
            s.append(int(j))
        if len(s) == k:
            break
    return frozenset(s[:k])


def collect(clusters, mass, top2, idx, sets, Dc):
    """rows: (origin: 0=time / 1=noise, lag, x, jac, cov, cov0)"""
    hubs = list(HUBS)
    out = []
    for w in idx:
        c, mm, t2 = clusters[w], mass[w], top2[w]
        for t in range(1, len(c)):
            cstar = int(c[t - 1])
            S0 = sets[cstar]
            row = mm[t].copy()
            row[hubs] = 0.0
            cov0 = row[list(S0)].sum()

            c2 = int(t2[t - 1, 1])                       # noise origin
            if c2 != cstar:
                Sp = sets[c2]
                out.append((1, 0, Dc[c2, cstar],
                            len(Sp & S0) / len(Sp | S0),
                            row[list(Sp)].sum(), cov0))

            for lag in LAGS:                             # time origins
                if t - 1 - lag >= 0:
                    p = int(c[t - 1 - lag])
                    if p != cstar:
                        Sp = sets[p]
                        out.append((0, lag, Dc[p, cstar],
                                    len(Sp & S0) / len(Sp | S0),
                                    row[list(Sp)].sum(), cov0))
    return np.array(out)


def report(rows, label):
    x = rows[:, 2]
    edges = np.quantile(x, np.linspace(0, 1, NBINS + 1))
    edges[-1] += 1e-9

    def f(mask, col):
        return rows[mask, col].mean() if mask.any() else float("nan")

    print(f"\n--- {label}: binned by centroid distance d(p, c*) ---")
    print(f"{'bin':>14} | {'n_time':>7} {'n_noise':>7} | "
          f"{'jac_t':>6} {'jac_n':>6} | {'cov_t':>6} {'cov_n':>6} | {'cov0':>6}")
    for b in range(NBINS):
        m = (x >= edges[b]) & (x < edges[b + 1])
        tm = m & (rows[:, 0] == 0)
        nm = m & (rows[:, 0] == 1)
        print(f"{edges[b]:>6.3f}-{edges[b + 1]:>6.3f} | "
              f"{int(tm.sum()):>7} {int(nm.sum()):>7} | "
              f"{f(tm, 3):>6.3f} {f(nm, 3):>6.3f} | "
              f"{f(tm, 4):>6.3f} {f(nm, 4):>6.3f} | {f(m, 5):>6.3f}")

    print(f"\n{label}: per-lag view (time origin), noise origin last")
    print(f"{'lag':>5} | {'n':>7} | {'mean_x':>6} | {'jac':>6} | "
          f"{'cov':>6} | {'cov0':>6}")
    for lag in LAGS:
        m = (rows[:, 0] == 0) & (rows[:, 1] == lag)
        if m.any():
            print(f"{lag:>5} | {int(m.sum()):>7} | {rows[m, 2].mean():>6.3f} | "
                  f"{rows[m, 3].mean():>6.3f} | {rows[m, 4].mean():>6.3f} | "
                  f"{rows[m, 5].mean():>6.3f}")
    m = rows[:, 0] == 1
    print(f"{'nz':>5} | {int(m.sum()):>7} | {rows[m, 2].mean():>6.3f} | "
          f"{rows[m, 3].mean():>6.3f} | {rows[m, 4].mean():>6.3f} | "
          f"{rows[m, 5].mean():>6.3f}")


if __name__ == "__main__":
    w_clu, w_mass, w_top2 = load("wiki")
    c_clu, c_mass, c_top2 = load("code")
    W = len(w_clu)
    train_idx = list(range(0, W, 2))
    test_idx = list(range(1, W, 2))

    P_flow = build_flow(w_clu, w_mass, train_idx, exclude=set(HUBS))
    centroids = np.load("extracted_data/wiki_centroids.npy")
    Dc = cdist(centroids, centroids)
    sets = {c: topk(P_flow[c], c, K_SET, set(HUBS)) for c in range(K)}

    report(collect(w_clu, w_mass, w_top2, test_idx, sets, Dc), "wiki held-out")
    report(collect(c_clu, c_mass, c_top2, range(len(c_clu)), sets, Dc),
           "code OOD")