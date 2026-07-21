"""
Escape-conditioned simulator (v2 of escape_sim.py).

EVENT (policy-free, per your definition): at step t, take the non-sink mass
row, zero the hubs AND the live current cluster c[t-1]. Escape iff the
remaining mass exceeds THRESH. The needed cluster cstar is the argmax of
that residual row. Both policies are scored against the SAME event and the
SAME cstar, so h_esc and h_st_esc are finally the conditional quantities
your break-even inequality is written in.

POLICIES:
  staged : hubs pinned + k clusters from P_flow[state pointer].
           Refresh per MS mode; "ev" = event-driven, refresh only when the
           live state cluster differs from the pointer used at last refresh.
  static : hubs pinned + live current free + top-k global mass clusters.

CHURN INSTRUMENTATION: on every refresh, count |new_set - old_set| = clusters
actually fetched. Reported per refresh and per step. Bandwidth per graph is
then  B = (f/step) * tok_per_sec * bytes_per_cluster  -- your arithmetic.

Columns:
  esc%     fraction of steps that are escapes at THRESH
  h_esc    P(cstar in staged | escape)     <- the h in the inequality
  h_st_esc P(cstar in static | escape)     <- the h_static in the inequality
  h_all    unconditional hit rate on cstar (continuity with the old table;
           note cstar now excludes current, so this is NOT the old h_staged)
  f/ref    mean clusters fetched per refresh
  f/step   mean clusters fetched per decode satep

Requires harvest_{wiki,code}.npz.
"""
import numpy as np

K = 64
HUBS = (16,17)                 # add 17 to test two-hub pinning
KS = (2, 4, 8)
MS = (1, 8, 16, "ev")        # "ev" = refresh only when the state pointer moves
THRESH = 0.25                # primary escape threshold (matches old CONSEQ)
SWEEP = (0.10, 0.25, 0.50)   # sensitivity: full table printed per threshold


def load(tag):
    z = np.load(f"extracted_data/harvest_{tag}.npz")
    return z["clusters"], z["mass_nosink"]


def build_flow(clusters, mass, train_idx, alpha=0.1, exclude=()):
    A = np.full((K, K), alpha)
    g = np.zeros(K)
    n = 0
    for w in train_idx:
        c, m = clusters[w], mass[w]
        for t in range(1, len(c)):
            A[c[t - 1]] += m[t]
            g += m[t]
            n += 1
    for j in exclude:
        A[:, j] = 0.0
        g[j] = 0.0
    return A / A.sum(1, keepdims=True), g / n


def topk(row, cur, k, exclude=()):
    s = [cur] if (cur >= 0 and cur not in exclude) else []
    for j in np.argsort(row)[::-1]:
        if j != cur and j not in exclude:
            s.append(int(j))
        if len(s) == k:
            break
    return set(s[:k])


def simulate(clusters, mass, idx, P_flow, g, label):
    hubs = set(HUBS)
    hub_list = list(hubs)
    static_cache = {}

    print(f"\n--- {label} ---")
    for thr in SWEEP:
        print(f"\n  [THRESH = {thr:.2f}]")
        print(f"  {'k':>3} {'m':>4} | {'esc%':>6} | {'h_esc':>7} {'h_st_esc':>8} "
              f"| {'h_all':>7} {'h_st_all':>8} | {'f/ref':>6} {'f/step':>7}")
        for k in KS:
            static_cache.clear()
            for m_int in MS:
                n = n_esc = 0
                hit_es = hit_eg = hit_as = hit_ag = 0
                fetched = refreshes = 0
                for w in idx:
                    c, mm = clusters[w], mass[w]
                    staged, ref = None, None
                    for t in range(1, len(c)):
                        cur = int(c[t - 1])
                        need = (staged is None
                                or (m_int == "ev" and cur != ref)
                                or (m_int != "ev" and (t - 1) % m_int == 0))
                        if need:
                            new = topk(P_flow[cur], cur, k, hubs)
                            fetched += len(new - staged) if staged else len(new)
                            refreshes += 1
                            staged, ref = new, cur
                        if cur not in static_cache:
                            static_cache[cur] = topk(g, cur, k, hubs)
                        static_set = static_cache[cur]

                        row = mm[t].copy()
                        row[hub_list] = 0.0
                        row[cur] = 0.0
                        esc_mass = row.sum()
                        cstar = int(np.argmax(row))
                        in_s = cstar in staged
                        in_g = cstar in static_set

                        hit_as += in_s
                        hit_ag += in_g
                        n += 1
                        if esc_mass > thr:
                            hit_es += in_s
                            hit_eg += in_g
                            n_esc += 1
                ne = max(n_esc, 1)
                print(f"  {k:>3} {str(m_int):>4} | {100 * n_esc / n:>5.1f}% | "
                      f"{hit_es / ne:>7.3f} {hit_eg / ne:>8.3f} | "
                      f"{hit_as / n:>7.3f} {hit_ag / n:>8.3f} | "
                      f"{fetched / max(refreshes, 1):>6.2f} "
                      f"{fetched / n:>7.3f}")


if __name__ == "__main__":
    w_clu, w_mass = load("wiki")
    c_clu, c_mass = load("code")
    W = len(w_clu)
    train_idx = list(range(0, W, 2))
    test_idx = list(range(1, W, 2))

    P_flow, g = build_flow(w_clu, w_mass, train_idx, exclude=set(HUBS))
    simulate(w_clu, w_mass, test_idx, P_flow, g, "wiki held-out")
    simulate(c_clu, c_mass, range(len(c_clu)), P_flow, g, "code OOD")