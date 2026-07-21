import pickle, numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# ---- load saved artifacts (no model, no forward pass) ----
g = np.load(r"extracted_data\gate1.npz")
wiki_res, code_res = g["wiki_res"], g["code_res"]
d_val, d_code, D = float(g["d_val"]), float(g["d_code"]), float(g["D"])

with open(r"extracted_data\assigned.pkl", "rb") as f:
    assigned = pickle.load(f)
tok = np.concatenate([t for t, _ in assigned])
clu = np.concatenate([c for _, c in assigned])

# ---- 1. Gate 1 ratios ----
print("="*50, "\nGATE 1: VOCABULARY TRANSFER\n", "="*50, sep="")
print(f"d_val {d_val:.3f}  d_code {d_code:.3f}  D {D:.3f}")
print(f"d_val/D   {d_val/D:.3f}   (the real floor; 0.40 line is invalid because this trips it)")
print(f"d_code/D  {d_code/D:.3f}")
print(f"d_code/d_val {d_code/d_val:.3f}   (kill > 1.5)  -> {'PASS' if d_code/d_val < 1.5 else 'KILL'}")

# ---- 2. permutation null for lexical pull ----
def mean_modal_purity(tok, clu, min_n=10):
    m = defaultdict(list)
    for t, c in zip(tok, clu):
        m[int(t)].append(int(c))
    purs = [max(Counter(cs).values())/len(cs) for cs in m.values() if len(cs) >= min_n]
    return float(np.mean(purs)), len(purs)

obs, n_tok = mean_modal_purity(tok, clu)
rng = np.random.default_rng(0)
nullv = np.array([mean_modal_purity(tok, rng.permutation(clu))[0] for _ in range(300)])
print("\n" + "="*50, "\nLEXICAL PULL (permutation null)\n", "="*50, sep="")
print(f"tokens(n>=10) {n_tok}  observed {obs*100:.2f}%  null {nullv.mean()*100:.2f}%"
      f"  excess {(obs-nullv.mean())*100:+.2f}pts  z={(obs-nullv.mean())/nullv.std():.1f}")
pos = np.concatenate([np.arange(len(c)) for _, c in assigned])
print(f"pos vs cluster-id corr (the 'climb'): {np.corrcoef(pos, clu)[0,1]:+.3f}")

# ---- 3. THE FORK: residual distribution shape ----
for name, r in [("wiki", wiki_res), ("code", code_res)]:
    print(f"\n{name}_res: mean {r.mean():.3f}  median {np.median(r):.3f}"
          f"  std {r.std():.3f}  p5 {np.percentile(r,5):.3f}  p95 {np.percentile(r,95):.3f}")

plt.hist(wiki_res, bins=60, alpha=.6, label="wiki", density=True)
plt.hist(code_res, bins=60, alpha=.6, label="code", density=True)
plt.axvline(wiki_res.mean(), color="r", ls="--")
plt.xlabel("residual (key → nearest centroid)"); plt.legend(); plt.show()

import numpy as np
from collections import defaultdict, Counter

def cond_entropy(seq, k=1):
    """H(c_t | previous k clusters), in bits, in-sample plug-in estimate."""
    ctx = defaultdict(Counter)
    for i in range(k, len(seq)):
        ctx[tuple(seq[i-k:i])][seq[i]] += 1
    total = sum(sum(c.values()) for c in ctx.values())
    H = 0.0
    for c in ctx.values():
        n = sum(c.values())
        p_ctx = n / total
        h = -sum((v/n) * np.log2(v/n) for v in c.values())
        H += p_ctx * h
    return H

# clu = np.concatenate([c for _, c in assigned])   # you already have this
H0 = np.log2(len(np.unique(clu)))          # uniform-over-active-clusters ceiling
H1 = cond_entropy(clu, k=1)                # real, first-order

rng = np.random.default_rng(0)
shuf = rng.permutation(clu)
H1_shuf = cond_entropy(shuf, k=1)          # null: same marginals, no order

print(f"H(c_t) ceiling (log2 active):   {H0:.3f} bits")
print(f"H(c_t | c_t-1) real:            {H1:.3f} bits")
print(f"H(c_t | c_t-1) shuffled null:   {H1_shuf:.3f} bits")
print(f"information from order:         {H1_shuf - H1:+.3f} bits")

tight_pos = pos[wiki_res < np.percentile(wiki_res, 5)]

# Check if they cluster at a specific window index
plt.hist(tight_pos, bins=60, alpha=0.7, edgecolor='black')
plt.title("Window Positions of the Tightest 5% Keys")
plt.xlabel("Position in Sequence Context")
plt.ylabel("Frequency")
plt.show()

import numpy as np

K = 64  # cluster count

def build_transition_matrix(assigned, K=64):
    """Directed cluster-transition counts, accumulated within windows only."""
    A = np.zeros((K, K), dtype=np.float64)
    for _tok, clu in assigned:                 # (1) per-window: never concatenate
        c = np.asarray(clu)
        for a, b in zip(c[:-1], c[1:]):        # (2) within-window pairs only
            A[a, b] += 1                        # (3) directed: a->b, NOT symmetrized
    return A

A = build_transition_matrix(assigned, K)
print(A)
print(np.trace(A) / np.sum(A))

# row-normalize to a transition kernel P(next | current); guard empty rows
row = A.sum(axis=1, keepdims=True)
P = np.divide(A, row, out=np.zeros_like(A), where=row > 0)   # (4) rows = outgoing dist

print(f"transitions counted:      {int(A.sum())}")
print(f"nonzero directed edges:   {int((A > 0).sum())} / {K*K}")
print(f"dead clusters (no out):   {int((row == 0).sum())}")
print(f"asymmetry ||A-Aᵀ||/||A||: {np.linalg.norm(A-A.T)/np.linalg.norm(A):.3f}")

import numpy as np

# --- off-diagonal asymmetry: the arrow of time among ACTUAL moves ---
A_off = A.copy()
np.fill_diagonal(A_off, 0.0)
asym_off = np.linalg.norm(A_off - A_off.T) / np.linalg.norm(A_off)
print(f"moves only:                 {int(A_off.sum())} transitions "
      f"({A_off.sum()/A.sum()*100:.1f}% of all)")
print(f"off-diagonal asymmetry:     {asym_off:.3f}   (was 0.148 with self-loops)")

# --- order test on MOVE-ONLY sequence: does 4.26 survive? ---
# rebuild a transition stream that skips stay-put steps, within windows only
def cond_entropy(seq, k=1):
    from collections import defaultdict, Counter
    ctx = defaultdict(Counter)
    for i in range(k, len(seq)):
        ctx[tuple(seq[i-k:i])][seq[i]] += 1
    total = sum(sum(c.values()) for c in ctx.values())
    H = 0.0
    for c in ctx.values():
        n = sum(c.values()); p = n/total
        H += p * (-sum((v/n)*np.log2(v/n) for v in c.values()))
    return H

# collapse consecutive repeats WITHIN each window, then concatenate the de-looped runs
moves = []
for _tok, clu in assigned:
    c = np.asarray(clu)
    keep = np.concatenate(([True], c[1:] != c[:-1]))   # drop a step if same as previous
    moves.append(c[keep])
move_seq = np.concatenate(moves)

H1_move = cond_entropy(move_seq, k=1)
rng = np.random.default_rng(0)
H1_move_shuf = cond_entropy(rng.permutation(move_seq), k=1)
print(f"\nmove-only H(c_t|c_t-1) real: {H1_move:.3f} bits")
print(f"move-only shuffled null:     {H1_move_shuf:.3f} bits")
print(f"move-only info from order:   {H1_move_shuf - H1_move:+.3f} bits   (was +4.258)")