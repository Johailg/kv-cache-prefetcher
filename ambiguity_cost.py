"""
GATED: before running, write down (1) your predicted %-identical staged sets
on the ambiguous stratum and (2) the one-line smoothness argument behind it.

Is argmax ambiguity FREE or COSTLY?
Population : steps whose STATE key (at t-1, the row index) has d1/d2 > R_AMB.
Compare    : staged set from P_flow[c1] (argmax row) vs P_flow[c2] (runner-up
             row) vs a soft inverse-distance 2-row mixture.
Score      : % identical sets, mean Jaccard, and true non-hub attention mass
             captured at t by each set. Unambiguous steps are the control.
Stringent  : hubs excluded from rows and from scored mass, so overlap can't
             be inflated by the freebie cluster every set would contain.

Requires runnerup_{tag}.npz from harvest_runnerup.py (run it first if the
file is missing -- it asserts alignment against the main harvest).
"""
import numpy as np

K = 64
HUBS = (16,)
R_AMB = 0.90
K_SET = 4


def load(tag):
    z = np.load(f"extracted_data/harvest_{tag}.npz")
    r = np.load(f"extracted_data/runnerup_{tag}.npz")
    return z["clusters"], z["top5"], z["mass_nosink"], r["top2"]


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
    return set(s[:k])


def run(clusters, top5, mass, top2, idx, P_flow, label):
    hubs = set(HUBS)
    strata = {"ambiguous": [], "control": []}
    for w in idx:
        c, d5, mm, t2 = clusters[w], top5[w], mass[w], top2[w]
        for t in range(1, len(c)):
            d1, d2 = float(d5[t - 1, 0]), float(d5[t - 1, 1])
            c1, c2 = int(t2[t - 1, 0]), int(t2[t - 1, 1])
            S1 = topk(P_flow[c1], c1, K_SET, hubs)
            S2 = topk(P_flow[c2], c2, K_SET, hubs)
            w1 = (1 / d1) / (1 / d1 + 1 / d2)
            soft_row = w1 * P_flow[c1] + (1 - w1) * P_flow[c2]
            Ss = topk(soft_row, c1, K_SET, hubs)
            row = mm[t].copy()
            row[list(hubs)] = 0.0
            rec = (S1 == S2,
                   len(S1 & S2) / len(S1 | S2),
                   row[list(S1)].sum(), row[list(S2)].sum(),
                   row[list(Ss)].sum())
            strata["ambiguous" if d1 / d2 > R_AMB else "control"].append(rec)

    print(f"\n--- {label} ---")
    for name, recs in strata.items():
        a = np.array(recs, dtype=float)
        print(f"{name:>10} (n={len(a)}): identical {100 * a[:, 0].mean():.1f}%  "
              f"jaccard {a[:, 1].mean():.3f}  |  mass: argmax {a[:, 2].mean():.3f}  "
              f"runner-up {a[:, 3].mean():.3f}  soft {a[:, 4].mean():.3f}")


if __name__ == "__main__":
    w_clu, w_top5, w_mass, w_top2 = load("wiki")
    c_clu, c_top5, c_mass, c_top2 = load("code")
    W = len(w_clu)
    train_idx = list(range(0, W, 2))
    test_idx = list(range(1, W, 2))

    P_flow = build_flow(w_clu, w_mass, train_idx, exclude=set(HUBS))
    run(w_clu, w_top5, w_mass, w_top2, test_idx, P_flow, "wiki held-out")
    run(c_clu, c_top5, c_mass, c_top2, range(len(c_clu)), P_flow, "code OOD")