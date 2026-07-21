"""
Extensions to analyze_next.py. Two tests. Both are cheap and offline.

  E. HUB-PINNED RESIDUAL COVERAGE
     Pin the top-h global-mass clusters (train-derived) as permanently
     resident. Zero their columns out of the mass matrix, renormalize,
     and re-run coverage on what's left. This answers: is flow_graph's
     win over static real destination structure, or inherited from one
     hub cluster everyone would pin anyway?
     Also prints the hub cluster ids + their mass share so you can go
     inspect their modal tokens against your residual-spike keys.

  F. STALENESS SWEEP  (do not run until you've written your prediction)
     flow_graph coverage at fixed k, but the staged set is only
     recomputed every m steps and frozen in between. Coverage vs m tells
     you how much lead time H the topology actually supports.

Requires harvest_{wiki,code}.npz from harvest_attention.py.
"""
import numpy as np

K = 64
KS = (2, 4, 8)          # k values for hub-pinned test (k counts NON-hub residents)
MS = (1, 2, 4, 8, 16, 32)  # refresh intervals for staleness sweep
K_STALE = 4             # fixed budget for the staleness sweep
N_HUBS = (1, 2)         # try pinning top-1 and top-2 hubs


def load(tag):
    z = np.load(f"extracted_data/harvest_{tag}.npz")
    return z["clusters"], z["mass_nosink"]


def build_predictors(clusters, mass, train_idx, alpha=0.1, exclude=()):
    A_flow = np.full((K, K), alpha)
    g = np.zeros(K)
    n = 0
    for w in train_idx:
        c, m = clusters[w], mass[w]
        for t in range(1, len(c)):
            A_flow[c[t - 1]] += m[t]
            g += m[t]
            n += 1
    for j in exclude:                      # hubs never occupy a predicted slot
        A_flow[:, j] = 0.0
        g[j] = 0.0
    P_flow = A_flow / np.maximum(A_flow.sum(1, keepdims=True), 1e-12)
    return P_flow, g / n


def topk_set(row, cur, k, exclude=()):
    s = [cur] if cur not in exclude else []
    for j in np.argsort(row)[::-1]:
        if j != cur and j not in exclude:
            s.append(j)
        if len(s) == k:
            break
    return np.array(s[:k], dtype=int)


def residual_mass(mass, hubs):
    m = mass.copy()
    m[:, :, list(hubs)] = 0.0
    return m  # NOT renormalized: coverage below reports fraction of TOTAL
              # non-sink mass captured by (hubs pinned + k predicted), which
              # keeps numbers comparable to the original tables.


# ---------------- E. hub-pinned coverage ----------------

def hub_pinned(w_clu, w_mass, c_clu, c_mass, train_idx, test_idx):
    # global mass on train windows decides the hubs
    _, g_full = build_predictors(w_clu, w_mass, train_idx)
    order = np.argsort(g_full)[::-1]
    for h in N_HUBS:
        hubs = set(int(x) for x in order[:h])
        share = g_full[list(hubs)].sum()
        print(f"\n===== pinning {h} hub(s): clusters {sorted(hubs)} "
              f"(train mass share {share:.3f}) =====")
        P_flow, g = build_predictors(w_clu, w_mass, train_idx, exclude=hubs)
        for label, clus, mass, idx in (
            ("wiki held-out", w_clu, w_mass, test_idx),
            ("code OOD", c_clu, c_mass, range(len(c_clu))),
        ):
            mres = residual_mass(mass, hubs)
            names = ["oracle", "flow_graph", "static"]
            res = {nm: {k: 0.0 for k in KS} for nm in names}
            hubmass = 0.0
            cnt = 0
            for w in idx:
                c, m, mr = clus[w], mass[w], mres[w]
                for t in range(1, len(c)):
                    row = mr[t]
                    hubmass += m[t, list(hubs)].sum()
                    desc = np.argsort(row)[::-1]
                    cur = c[t - 1]
                    for k in KS:
                        res["oracle"][k] += row[desc[:k]].sum()
                        res["flow_graph"][k] += row[
                            topk_set(P_flow[cur], cur, k, hubs)].sum()
                        res["static"][k] += row[
                            topk_set(g, cur, k, hubs)].sum()
                    cnt += 1
            print(f"\n--- {label}: hub mass alone {hubmass / cnt:.3f}; "
                  f"below = ADDITIONAL mass from k non-hub residents ---")
            print(f"{'k':>4} | " + " | ".join(f"{nm:>10}" for nm in names))
            for k in KS:
                print(f"{k:>4} | " + " | ".join(
                    f"{res[nm][k] / cnt:>10.3f}" for nm in names))


# ---------------- F. staleness sweep ----------------

def staleness(w_clu, w_mass, c_clu, c_mass, train_idx, test_idx):
    P_flow, _ = build_predictors(w_clu, w_mass, train_idx)
    print(f"\n===== staleness sweep: flow_graph, k={K_STALE}, "
          f"staged set refreshed every m steps =====")
    for label, clus, mass, idx in (
        ("wiki held-out", w_clu, w_mass, test_idx),
        ("code OOD", c_clu, c_mass, range(len(c_clu))),
    ):
        line = []
        for m_int in MS:
            tot, cnt = 0.0, 0
            for w in idx:
                c, m = clus[w], mass[w]
                staged = None
                for t in range(1, len(c)):
                    if staged is None or (t - 1) % m_int == 0:
                        ref = c[t - 1]
                        staged = topk_set(P_flow[ref], ref, K_STALE)
                    tot += m[t][staged].sum()
                    cnt += 1
            line.append(f"m={m_int}: {tot / cnt:.3f}")
        print(f"{label}:  " + "   ".join(line))


if __name__ == "__main__":
    w_clu, w_mass = load("wiki")
    c_clu, c_mass = load("code")
    W = len(w_clu)
    train_idx = list(range(0, W, 2))
    test_idx = list(range(1, W, 2))

    hub_pinned(w_clu, w_mass, c_clu, c_mass, train_idx, test_idx)

    # -- Test F is gated: write your predicted coverage-vs-m shape and the
    # -- knee location down FIRST, then uncomment.
    staleness(w_clu, w_mass, c_clu, c_mass, train_idx, test_idx)