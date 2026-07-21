"""
Offline analysis on harvest_{wiki,code}.npz. Four tests, no model needed.

  A. PREFETCH COVERAGE  -- the load-bearing one.
     Train key-transition graph + attention-flow graph on wiki-train windows.
     On held-out wiki AND on code (OOD), measure: fraction of attention mass
     captured with k clusters resident, vs oracle / static / recency baselines.
       - flow_graph ~ static        -> topology adds nothing over frequency
       - recency wins               -> "attention is recency-local", diff paper
       - key_graph ~ flow_graph     -> key trajectory is a valid state variable
       - flow_graph -> oracle, small k -> the prefetch story has legs

  B. HELD-OUT CROSS-ENTROPY, orders 0-3, move-only.
     Replaces the biased in-sample plug-in numbers; answers history depth.
     Honest "info from order" = CE(order 0) - CE(order 1) on held-out data.

  C. ARGMAX MARGIN STATS from top-5 centroid distances (soft-vs-argmax evidence).

  D. BOOTSTRAP CIs (resample windows) for self-loop rate + off-diag asymmetry.
"""
import numpy as np
from collections import defaultdict, Counter

K = 64
KS = (1, 2, 4, 8, 16)
RNG = np.random.default_rng(0)


def load(tag):
    z = np.load(f"extracted_data/harvest_{tag}.npz")
    return z["clusters"], z["top5"], z["mass"], z["mass_nosink"]


# ---------------- A. coverage ----------------

def build_predictors(clusters, mass, train_idx, alpha=0.1):
    A_key = np.full((K, K), alpha)   # c_{t-1} -> c_t   (key trajectory)
    A_flow = np.full((K, K), alpha)  # c_{t-1} -> attn mass by cluster at t
    g = np.zeros(K)
    n = 0
    for w in train_idx:
        c, m = clusters[w], mass[w]
        for t in range(1, len(c)):
            A_key[c[t - 1], c[t]] += 1
            A_flow[c[t - 1]] += m[t]
            g += m[t]
            n += 1
    return (A_key / A_key.sum(1, keepdims=True),
            A_flow / A_flow.sum(1, keepdims=True),
            g / n)


def topk_set(row, cur, k):
    """current cluster resident for free + top predicted others, k total."""
    s = [cur]
    for j in np.argsort(row)[::-1]:
        if j != cur:
            s.append(j)
        if len(s) == k:
            break
    return np.array(s[:k])


def recency_set(hist, k):
    s, seen = [], set()
    for x in hist[::-1]:
        if x not in seen:
            s.append(x)
            seen.add(int(x))
        if len(s) == k:
            break
    return np.array(s)


def coverage(clusters, mass, idx, P_key, P_flow, g, label):
    names = ["oracle", "flow_graph", "key_graph", "recency", "static"]
    res = {nm: {k: 0.0 for k in KS} for nm in names}
    cnt = 0
    for w in idx:
        c, m = clusters[w], mass[w]
        for t in range(1, len(c)):
            row = m[t]
            desc = np.argsort(row)[::-1]
            cur = c[t - 1]
            for k in KS:
                res["oracle"][k] += row[desc[:k]].sum()
                res["flow_graph"][k] += row[topk_set(P_flow[cur], cur, k)].sum()
                res["key_graph"][k] += row[topk_set(P_key[cur], cur, k)].sum()
                res["recency"][k] += row[recency_set(c[:t], k)].sum()
                res["static"][k] += row[topk_set(g, cur, k)].sum()
            cnt += 1
    print(f"\n--- coverage: {label}  (mean attn mass captured, {cnt} steps) ---")
    print(f"{'k':>4} | " + " | ".join(f"{nm:>10}" for nm in names))
    for k in KS:
        print(f"{k:>4} | " + " | ".join(f"{res[nm][k] / cnt:>10.3f}" for nm in names))


# ---------------- B. held-out CE ----------------

def move_seqs(clusters):
    out = []
    for c in clusters:
        keep = np.concatenate(([True], c[1:] != c[:-1]))
        out.append([int(x) for x in c[keep]])
    return out


class BackoffLM:
    """Interpolated backoff: P_k = lam*ML_k + (1-lam)*P_{k-1}; P_0 = add-1 unigram."""

    def __init__(self, order, lam):
        self.order, self.lam = order, lam
        self.ctx = [defaultdict(Counter) for _ in range(order + 1)]

    def train(self, seqs):
        for s in seqs:
            for i in range(len(s)):
                for k in range(self.order + 1):
                    if i >= k:
                        self.ctx[k][tuple(s[i - k:i])][s[i]] += 1
        self._u_tot = sum(self.ctx[0][()].values())

    def prob(self, hist, x):
        p = (self.ctx[0][()][x] + 1) / (self._u_tot + K)
        for k in range(1, self.order + 1):
            if len(hist) >= k:
                cc = self.ctx[k].get(tuple(hist[-k:]))
                if cc:
                    n = sum(cc.values())
                    p = self.lam * (cc[x] / n) + (1 - self.lam) * p
        return p

    def bits(self, seqs):
        tot, n = 0.0, 0
        for s in seqs:
            for i in range(1, len(s)):
                h = tuple(s[max(0, i - self.order):i])
                tot += -np.log2(self.prob(h, s[i]))
                n += 1
        return tot / n


def ce_sweep(train_seqs, test_seqs, label):
    # carve dev split off train to pick lambda honestly
    cut = max(1, int(0.8 * len(train_seqs)))
    tr, dev = train_seqs[:cut], train_seqs[cut:]
    print(f"\n--- held-out CE (bits/move): {label} ---")
    for order in (0, 1, 2, 3):
        if order == 0:
            lm = BackoffLM(0, 0.0)
            lm.train(train_seqs)
            print(f"order {order}: {lm.bits(test_seqs):.3f}")
            continue
        best = (None, np.inf)
        for lam in (0.3, 0.5, 0.7, 0.9):
            lm = BackoffLM(order, lam)
            lm.train(tr)
            b = lm.bits(dev)
            if b < best[1]:
                best = (lam, b)
        lm = BackoffLM(order, best[0])
        lm.train(train_seqs)
        print(f"order {order}: {lm.bits(test_seqs):.3f}   (lam={best[0]})")


# ---------------- C. margins ----------------

def margin_report(top5, tag):
    d = top5.reshape(-1, 5)
    r = d[:, 0] / d[:, 1]
    gap = d[:, 1] - d[:, 0]
    print(f"\n--- margins: {tag} ---")
    print(f"d1/d2 ratio: median {np.median(r):.3f}  "
          f"p10 {np.percentile(r, 10):.3f}  p90 {np.percentile(r, 90):.3f}")
    print(f"frac ambiguous  r>0.90: {(r > 0.90).mean() * 100:.1f}%   "
          f"r>0.95: {(r > 0.95).mean() * 100:.1f}%")
    print(f"gap d2-d1: median {np.median(gap):.4f}  "
          f"(vs median d1 {np.median(d[:, 0]):.4f})")


# ---------------- D. bootstrap ----------------

def bootstrap(clusters, B=1000):
    W = len(clusters)
    sl, asym = [], []
    for _ in range(B):
        idx = RNG.integers(0, W, W)
        A = np.zeros((K, K))
        for w in idx:
            c = clusters[w].astype(int)
            np.add.at(A, (c[:-1], c[1:]), 1)
        sl.append(np.trace(A) / A.sum())
        Ao = A.copy()
        np.fill_diagonal(Ao, 0)
        asym.append(np.linalg.norm(Ao - Ao.T) / np.linalg.norm(Ao))
    for name, v in (("self-loop rate", np.array(sl)),
                    ("off-diag asymmetry", np.array(asym))):
        print(f"{name}: mean {v.mean():.3f}  "
              f"95% CI [{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]")


# ---------------- main ----------------

if __name__ == "__main__":
    w_clu, w_top5, w_mass, w_mass_ns = load("wiki")
    c_clu, c_top5, c_mass, c_mass_ns = load("code")

    W = len(w_clu)
    train_idx = list(range(0, W, 2))   # even windows train
    test_idx = list(range(1, W, 2))    # odd windows test

    # ---- A: sink-EXCLUDED is the primary read (sink keys are always resident
    #         anyway -- attending to them costs a prefetcher nothing) ----
    P_key, P_flow, g = build_predictors(w_clu, w_mass_ns, train_idx)
    coverage(w_clu, w_mass_ns, test_idx, P_key, P_flow, g,
             "wiki held-out, sink excluded")
    coverage(c_clu, c_mass_ns, range(len(c_clu)), P_key, P_flow, g,
             "CODE (OOD, wiki-trained predictors), sink excluded")

    # sink-included for reference, to see how much the sink distorts things
    P_key2, P_flow2, g2 = build_predictors(w_clu, w_mass, train_idx)
    coverage(w_clu, w_mass, test_idx, P_key2, P_flow2, g2,
             "wiki held-out, sink INCLUDED (reference)")

    # ---- B: held-out CE, move-only ----
    w_moves = move_seqs(w_clu)
    c_moves = move_seqs(c_clu)
    tr_moves = [w_moves[i] for i in train_idx]
    te_moves = [w_moves[i] for i in test_idx]
    ce_sweep(tr_moves, te_moves, "wiki (in-domain)")
    ce_sweep(tr_moves, c_moves, "code (OOD, wiki-trained)")

    # ---- C ----
    margin_report(w_top5, "wiki")
    margin_report(c_top5, "code")

    # ---- D ----
    print("\n--- bootstrap over wiki windows ---")
    bootstrap(w_clu)