#!/usr/bin/env python3
"""
head_sweep.py -- per-head prevalence sweep of flow-graph structure.

WHY
  The memo is N=1 (pythia-410m, L13 H2, randomly picked). This measures the
  same battery on EVERY (layer, head) of a target model, so "how common is
  this?" gets a number instead of a caveat. Model-agnostic: run it on 410m
  (prevalence on the original model + harness calibration), then 1.4b/2.8b
  (scale), later Qwen (family/tokenizer -- test #3).

CONVENTIONS (mirrors the memo)
  - 500-token non-overlapping windows, per-window state only
  - K=64 k-means per head, fit on train-split TRAIN-DOMAIN keys
  - sink = window position 0: column zeroed, rows renormalized
  - P_flow[i] = mean attention-over-clusters profile at steps whose state
    cluster (cluster of the key emitted at t-1) is i        [train split]
  - static  = global mean attention-over-clusters profile   [train split]
  - hub = argmax of static profile; escape event = zero hub + state-cluster
    mass, escape iff remainder > THRESH; residents = {hub} U top-k non-hub
  - keys are post-RoPE KV-cache keys (what a real prefetcher would cluster);
    --keys-source qkv gives pre-rotary keys via hook (GPTNeoX only)

MOVE-ONLY GUARDRAILS (the June agreement)
  Raw dH is inflated by self-loops; prev-token heads have razor-sharp but
  FREE structure. So per head we also report:
    dH_move    entropy drop conditioned on c_t != c_{t-1}
    local_mass mean attention mass at relative offsets 0..4 (offset-rule
               discriminator; ~1.0 means the head is a one-line rule)
    esc_pct    escape rate (mechanical heads barely fire escapes)

CALIBRATION GATE -- RUN THIS FIRST, before any big-model run is trusted:
    python head_sweep.py --model EleutherAI/pythia-410m --layers 13 --heads 2 \
        --domain wiki=wiki.txt --domain code=code.txt --train-domain wiki
  L13H2 must land near the canonical numbers (cov@4 flow ~0.95 wiki,
  h_esc ~0.92 / h_st_esc ~0.18 at k=4, THRESH=0.25). This script is a
  reconstruction from the memo, NOT from escape_sim_v2.py -- if calibration
  misses, my conventions differ from yours somewhere (sink renorm, escape
  scoring, smoothing, key source) and we reconcile BEFORE burning a night.
  UPDATE (Jul 20): escape scoring and set construction now mirror
  escape_sim_v2.py exactly -- every set seeds the live state cluster and
  fills to k (topk clone); h_esc = P(argmax of residual row is staged |
  escape), not mass-captured; escape fills exclude the hub; f/step counts
  the initial fill. Remaining dials if calibration still misses: sink
  renormalization (cluster_mass renormalizes rows after zeroing position 0
  -- if your mass_nosink is UNnormalized, esc% will disagree first; flip
  there), add-alpha smoothing vs your back-off, and tight-fit 0.1 being
  absolute (may not transfer across head_dim).

FULL SWEEP (overnight-runner shaped)
    python head_sweep.py --model EleutherAI/pythia-1.4b \
        --domain wiki=wiki.txt --domain code=code.txt --train-domain wiki \
        --max-windows 40 --out sweep_1p4b
  --quick             stride-4 layers/heads, 12 windows (smoke test first!)
  --model ...pythia-2.8b if the card has >=12 GB (5.6 GB weights fp16 +
                      ~1 GB transient attention per window)
  Output: {out}.csv, one row per (layer, q_head, domain), + summary print.

# ---------------------------------------------------------------------------
# PRE-REGISTRATION (fill BEFORE the full run; postdiction doesn't count)
# CLAUDE:
#   structured heads (dH_move >= 1.0 bit AND h_esc >= 2x h_st_esc at k=4):
#     25-45% of head-layers, concentrated mid-to-late layers
#   whitespace hubs (hub_mass >= 0.30, hub_ws_frac >= 0.5, hub_pos_mean < 0.35):
#     10-25% of heads
#   prevalence non-decreasing 410m -> 1.4b; L13H2 above median, not top decile
# JOHAIL:
#   structured-head fraction: ____   layer profile: ____
#   whitespace-hub fraction:  ____   scale direction: ____
# ---------------------------------------------------------------------------
"""

import argparse, csv, math, os, string, sys, time
from collections import defaultdict

import numpy as np
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans
from transformers import AutoModelForCausalLM, AutoTokenizer

# canonical reference (410m L13 H2) printed in calibration mode
CANON = {"cov4_flow_wiki": 0.951, "cov4_static_wiki": 0.815,
         "h_esc_wiki": 0.923, "h_st_esc_wiki": 0.176,
         "h_esc_code": 0.717, "h_st_esc_code": 0.117, "selfloop": 0.697}

WS_PUNCT = set(string.whitespace) | set(string.punctuation)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-1.4b")
    p.add_argument("--domain", action="append", required=True,
                   help="NAME=path.txt (repeatable); plain text files")
    p.add_argument("--train-domain", default="wiki")
    p.add_argument("--max-windows", type=int, default=40)
    p.add_argument("--window", type=int, default=500)
    p.add_argument("--clusters", type=int, default=64)
    p.add_argument("--layers", default=None, help="comma list, default all")
    p.add_argument("--heads", default=None, help="comma list of q-heads")
    p.add_argument("--k-list", default="2,4,8")
    p.add_argument("--escape-k", type=int, default=4)
    p.add_argument("--thresh", type=float, default=0.25)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--alpha", type=float, default=0.5, help="add-alpha smoothing")
    p.add_argument("--tight", type=float, default=0.1, help="tight-fit residual")
    p.add_argument("--keys-source", choices=["cache", "qkv"], default="cache")
    p.add_argument("--minibatch-kmeans", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="sweep")
    return p.parse_args()


def load_windows(tok, path, W, max_win):
    text = open(path, encoding="utf-8", errors="replace").read()
    ids = tok(text, return_tensors=None)["input_ids"]
    wins = [np.array(ids[i:i + W], dtype=np.int64)
            for i in range(0, len(ids) - W + 1, W)][:max_win]
    if len(wins) < 4:
        sys.exit(f"{path}: only {len(wins)} windows of {W} tokens; need >= 4")
    return wins


def get_cache_keys(pkv, l):
    if hasattr(pkv, "key_cache"):
        return pkv.key_cache[l]
    if hasattr(pkv, "layers"):
        return pkv.layers[l].keys
    return pkv[l][0]


class QKVHook:
    """Pre-rotary keys via hook on GPTNeoX attention.query_key_value."""
    def __init__(self, model, layers, H, hd):
        self.buf, self.H, self.hd = {}, H, hd
        self.handles = []
        for l in layers:
            mod = model.gpt_neox.layers[l].attention.query_key_value
            self.handles.append(mod.register_forward_hook(self._mk(l)))

    def _mk(self, l):
        def hook(_m, _i, out):
            b, s, _ = out.shape
            qkv = out.view(b, s, self.H, 3 * self.hd)
            self.buf[l] = qkv[..., self.hd:2 * self.hd].permute(0, 2, 1, 3)
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def main():
    a = parse_args()
    rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    klist = [int(x) for x in a.k_list.split(",")]
    kmax, ke, C, W = max(klist), a.escape_k, a.clusters, a.window

    domains = dict(d.split("=", 1) for d in a.domain)
    assert a.train_domain in domains, "--train-domain must be one of --domain"

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=dt, attn_implementation="eager").to(dev).eval()
    cfg = model.config
    L = cfg.num_hidden_layers
    Hq = cfg.num_attention_heads
    Hkv = getattr(cfg, "num_key_value_heads", None) or Hq
    grp = Hq // Hkv
    hd = cfg.hidden_size // Hq

    layers = ([int(x) for x in a.layers.split(",")] if a.layers
              else list(range(0, L, 4 if a.quick else 1)))
    qheads = ([int(x) for x in a.heads.split(",")] if a.heads
              else list(range(0, Hq, 4 if a.quick else 1)))
    kvheads = sorted({qh // grp for qh in qheads})
    max_win = 12 if a.quick else a.max_windows
    print(f"[cfg] {a.model}: {L}L x {Hq}H (kv={Hkv}, hd={hd}) | sweeping "
          f"{len(layers)} layers x {len(qheads)} q-heads | {max_win} win/domain")
    print(open(__file__, encoding="utf-8").read().split("PRE-REGISTRATION")[1]
          .split('"""')[0])  # echo prereg block into the log

    # token-level whitespace/punct flag, per domain computed lazily
    ws_cache = {}
    def is_ws(tid):
        if tid not in ws_cache:
            s = tok.decode([tid])
            ws_cache[tid] = (s.strip() == "") or all(c in WS_PUNCT for c in s)
        return ws_cache[tid]

    # ---------------- pass 1: harvest keys (no attentions) -----------------
    wins = {d: load_windows(tok, p, W, max_win) for d, p in domains.items()}
    n_tr_dom = len(wins[a.train_domain])
    order = rng.permutation(n_tr_dom)
    n_tr = max(2, int(round(a.train_frac * n_tr_dom)))
    tr_idx = set(order[:n_tr].tolist())         # train-split window indices
    print({d: len(w) for d, w in wins.items()},
          f"| train split {n_tr}/{n_tr_dom}")

    keys = {d: {l: [] for l in layers} for d in domains}   # fp16 arrays
    hook = None
    if a.keys_source == "qkv":
        assert hasattr(model, "gpt_neox"), "--keys-source qkv is GPTNeoX-only"
        hook = QKVHook(model, layers, Hq, hd)
    with torch.no_grad():
        for d, wlist in wins.items():
            for w in wlist:
                inp = torch.tensor(w[None], device=dev)
                out = model(inp, use_cache=(a.keys_source == "cache"),
                            output_attentions=False)
                for l in layers:
                    kt = (hook.buf[l] if hook else
                          get_cache_keys(out.past_key_values, l))
                    keys[d][l].append(kt[0].to(torch.float16).cpu().numpy())
    if hook:
        hook.close()
    print("[pass1] keys harvested")

    # ------------- stage 2: per-head kmeans, labels, geometry --------------
    KM = MiniBatchKMeans if a.minibatch_kmeans else KMeans
    cent, labels, geo, cstats, ent = {}, {}, {}, {}, {}
    t0 = time.time()
    for l in layers:
        kd = {d: np.stack(keys[d][l]).astype(np.float32) for d in domains}
        for kvh in kvheads:
            tr_keys = np.concatenate(
                [kd[a.train_domain][i, kvh] for i in sorted(tr_idx)])
            km = KM(n_clusters=C, n_init=3, random_state=a.seed).fit(tr_keys)
            cent[(l, kvh)] = km.cluster_centers_
            cc = km.cluster_centers_
            D = float(np.mean(np.linalg.norm(
                cc[:, None] - cc[None], axis=-1)[np.triu_indices(C, 1)]))
            for d in domains:
                flat = kd[d][:, kvh].reshape(-1, hd)
                dist = np.linalg.norm(
                    flat[:, None] - cc[None], axis=-1)  # (n, C)
                lab = dist.argmin(1)
                mind = dist[np.arange(len(lab)), lab]
                labels[(l, kvh, d)] = lab.reshape(-1, W).astype(np.int16)
                tight = mind < a.tight
                st = {"cnt": np.bincount(lab, minlength=C),
                      "ws": np.zeros(C), "pos": np.zeros(C),
                      "tightc": np.bincount(lab[tight], minlength=C)}
                toks = np.stack(wins[d]).reshape(-1)
                wsf = np.array([is_ws(t) for t in
                                np.unique(toks)])  # per unique id
                ws_map = dict(zip(np.unique(toks), wsf))
                wsv = np.array([ws_map[t] for t in toks], dtype=np.float64)
                pos = np.tile(np.arange(W) / W, len(wins[d]))
                np.add.at(st["ws"], lab, wsv)
                m = pos > 0
                np.add.at(st["pos"], lab[m], pos[m])
                st["posn"] = np.bincount(lab[m], minlength=C)
                cstats[(l, kvh, d)] = st
                which = ("val" if d == a.train_domain else d)
                gmask = (np.array([i not in tr_idx
                                   for i in range(len(wins[d]))]).repeat(W)
                         if d == a.train_domain else np.ones(len(lab), bool))
                geo[(l, kvh, which)] = (float(mind[gmask].mean()), D,
                                        float(tight.mean()))
            # entropy from train-domain labels
            lt = labels[(l, kvh, a.train_domain)]
            T = np.zeros((C, C))
            frq = np.zeros(C)
            for i in sorted(tr_idx):
                np.add.at(T, (lt[i, :-1], lt[i, 1:]), 1)
                frq += np.bincount(lt[i], minlength=C)
            p0 = (frq + a.alpha) / (frq.sum() + a.alpha * C)
            P1 = (T + a.alpha) / (T.sum(1, keepdims=True) + a.alpha * C)
            Tm = T.copy(); np.fill_diagonal(Tm, 0)
            p0m = (Tm.sum(0) + a.alpha) / (Tm.sum() + a.alpha * C)
            P1m = (Tm + a.alpha) / (Tm.sum(1, keepdims=True) + a.alpha * C)
            h0 = h1 = h0m = h1m = n = nm = 0.0
            for i in range(len(lt)):
                if i in tr_idx:
                    continue
                cp, ct = lt[i, :-1], lt[i, 1:]
                h0 -= np.log2(p0[ct]).sum(); h1 -= np.log2(P1[cp, ct]).sum()
                n += len(ct)
                mv = ct != cp
                h0m -= np.log2(p0m[ct[mv]]).sum()
                h1m -= np.log2(P1m[cp[mv], ct[mv]]).sum()
                nm += mv.sum()
            ent[(l, kvh)] = {"selfloop": float(np.trace(T) / max(T.sum(), 1)),
                             "H0": h0 / n, "H1": h1 / n,
                             "dH": (h0 - h1) / n,
                             "dH_move": (h0m - h1m) / max(nm, 1)}
        del kd
        print(f"[stage2] layer {l} done ({time.time()-t0:.0f}s)")
    del keys

    # ---------------- pass 3a: fit P_flow/static on train windows ----------
    P_acc = defaultdict(lambda: np.zeros((C, C)))
    P_cnt = defaultdict(lambda: np.zeros(C))
    S_acc = defaultdict(lambda: np.zeros(C))
    S_n = defaultdict(float)

    def cluster_mass(attn_lh, lab):
        A = attn_lh.astype(np.float32).copy()
        A[:, 0] = 0.0
        rs = A.sum(1, keepdims=True)
        valid = rs[:, 0] > 1e-8
        A[valid] /= rs[valid]
        one = np.zeros((W, C), np.float32)
        one[np.arange(W), lab] = 1.0
        return A[1:] @ one, valid[1:], A            # M (W-1,C), valid, A_z

    def fwd_attn(w):
        with torch.no_grad():
            out = model(torch.tensor(w[None], device=dev),
                        use_cache=False, output_attentions=True)
        return out.attentions

    for i in sorted(tr_idx):
        att = fwd_attn(wins[a.train_domain][i])
        for l in layers:
            al = att[l][0].float().cpu().numpy()
            for qh in qheads:
                lab = labels[(l, qh // grp, a.train_domain)][i]
                M, vr, _ = cluster_mass(al[qh], lab)
                st = lab[:-1]
                np.add.at(P_acc[(l, qh)], st[vr], M[vr])
                P_cnt[(l, qh)] += np.bincount(st[vr], minlength=C)
                S_acc[(l, qh)] += M[vr].sum(0)
                S_n[(l, qh)] += vr.sum()
    print("[pass3a] flow graphs fit")

    def seeded(order_row, cur, k, banned=()):
        # escape_sim_v2.topk clone: seed the live state cluster (unless
        # banned, i.e. cur is a hub), fill to k from the ranking, skipping
        # cur and banned clusters.
        s = [cur] if cur not in banned else []
        for j in order_row:
            if len(s) == k:
                break
            j = int(j)
            if j != cur and j not in banned:
                s.append(j)
        return s

    flow_cov, stat_cov, flow_stg, stat_stg, hubs = {}, {}, {}, {}, {}
    for key_ in P_acc:
        P = P_acc[key_] / np.maximum(P_cnt[key_], 1)[:, None]
        P[P_cnt[key_] == 0] = 1.0 / C
        sp = S_acc[key_] / max(S_n[key_], 1)
        hub = int(sp.argmax()); hubs[key_] = hub
        oP = np.argsort(-P, axis=1)
        oG = np.argsort(-sp)
        Pz = P.copy(); Pz[:, hub] = -np.inf
        oPz = np.argsort(-Pz, axis=1)
        spz = sp.copy(); spz[hub] = -np.inf
        oGz = np.argsort(-spz)
        # coverage sets (S1-style, hub predictable): seeded, per state cluster
        flow_cov[key_] = {k: np.array([seeded(oP[c], c, k)
                                       for c in range(C)]) for k in klist}
        stat_cov[key_] = {k: np.array([seeded(oG, c, k)
                                       for c in range(C)]) for k in klist}
        # escape sets (S3-style, hub pinned/excluded): seeded, hub banned
        flow_stg[key_] = np.array([seeded(oPz[c], c, ke, banned=(hub,))
                                   for c in range(C)])
        stat_stg[key_] = np.array([seeded(oGz, c, ke, banned=(hub,))
                                   for c in range(C)])
        hubs[(key_, "share")] = float(sp[hub])

    # ---------------- pass 3b: score val + OOD windows ---------------------
    acc = defaultdict(lambda: defaultdict(float))
    score_list = ([(a.train_domain, i) for i in range(n_tr_dom)
                   if i not in tr_idx] +
                  [(d, i) for d in domains if d != a.train_domain
                   for i in range(len(wins[d]))])
    for d, i in score_list:
        att = fwd_attn(wins[d][i])
        dn = "val" if d == a.train_domain else d
        for l in layers:
            al = att[l][0].float().cpu().numpy()
            for qh in qheads:
                key_ = (l, qh)
                lab = labels[(l, qh // grp, d)][i]
                M, vr, Az = cluster_mass(al[qh], lab)
                st = lab[:-1]
                A_ = acc[(l, qh, dn)]
                n = vr.sum(); A_["n"] += n
                rows = np.arange(W - 1)
                for k in klist:
                    top = np.argpartition(M, -k, axis=1)[:, -k:]
                    A_[f"cov{k}_oracle"] += np.take_along_axis(
                        M, top, 1).sum(1)[vr].sum()
                    fs = flow_cov[key_][k][st]
                    A_[f"cov{k}_flow"] += M[rows[:, None], fs].sum(1)[vr].sum()
                    ss = stat_cov[key_][k][st]
                    A_[f"cov{k}_static"] += M[rows[:, None],
                                              ss].sum(1)[vr].sum()
                seen, covr = [], np.zeros((W - 1, len(klist)))
                for t in range(1, W):
                    c = int(lab[t - 1])
                    if c in seen:
                        seen.remove(c)
                    seen.insert(0, c)
                    for j, k in enumerate(klist):
                        covr[t - 1, j] = M[t - 1, seen[:k]].sum()
                for j, k in enumerate(klist):
                    A_[f"cov{k}_recency"] += covr[vr, j].sum()
                # escape-conditioned, escape_sim_v2 conventions:
                # hit = P(argmax of residual row is in the staged set)
                hub = hubs[key_]
                rem = M.copy(); rem[:, hub] = 0
                rem[rows, st] = 0
                rsum = rem.sum(1)
                esc = (rsum > a.thresh) & vr
                A_["n_esc"] += esc.sum()
                if esc.any():
                    cstar = rem.argmax(1)
                    hit_f = (flow_stg[key_][st] == cstar[:, None]).any(1)
                    hit_s = (stat_stg[key_][st] == cstar[:, None]).any(1)
                    A_["h_esc"] += (hit_f & esc).sum()
                    A_["h_st"] += (hit_s & esc).sum()
                # f/step at m=1: staged-set churn, initial fill counted
                # (matches escape_sim_v2's fetched accounting)
                res, fetch = None, 0
                for t in range(1, W):
                    new = set(flow_stg[key_][lab[t - 1]].tolist())
                    fetch += len(new) if res is None else len(new - res)
                    res = new
                A_["fetch"] += fetch
                # offset-rule discriminator
                lm = np.zeros(W)
                for off in range(5):
                    idx = np.arange(off, W)
                    lm[idx] += Az[idx, idx - off]
                A_["local"] += lm[1:][vr].sum()
    print("[pass3b] scoring done")

    # ---------------- write CSV + summary ----------------------------------
    cols = (["layer", "qhead", "domain", "n_steps", "d_mean", "d_ratio", "D",
             "tight_frac", "selfloop", "H0", "H1", "dH", "dH_move",
             "hub_id", "hub_mass", "hub_key_frac", "hub_ws_frac",
             "hub_pos_mean", "tight_in_hub", "esc_pct", "h_esc", "h_st_esc",
             "f_step", "local_mass"]
            + [f"cov{k}_{m}" for k in klist
               for m in ("oracle", "flow", "static", "recency")])
    rows_out = []
    for (l, qh, dn), A_ in sorted(acc.items()):
        kvh = qh // grp
        key_ = (l, qh)
        d_mean, D, tightf = geo[(l, kvh, dn)]
        d_val = geo[(l, kvh, "val")][0]
        dom_name = a.train_domain if dn == "val" else dn
        st = cstats[(l, kvh, dom_name)]
        hub = hubs[key_]
        n = max(A_["n"], 1)
        r = {"layer": l, "qhead": qh, "domain": dn, "n_steps": int(n),
             "d_mean": round(d_mean, 4), "d_ratio": round(d_mean / d_val, 3),
             "D": round(D, 4), "tight_frac": round(tightf, 4),
             **{k: round(v, 4) for k, v in ent[(l, kvh)].items()},
             "hub_id": hub, "hub_mass": round(hubs[(key_, "share")], 3),
             "hub_key_frac": round(st["cnt"][hub] / st["cnt"].sum(), 4),
             "hub_ws_frac": round(st["ws"][hub] / max(st["cnt"][hub], 1), 3),
             "hub_pos_mean": round(st["pos"][hub] / max(st["posn"][hub], 1), 3),
             "tight_in_hub": round(st["tightc"][hub]
                                   / max(st["tightc"].sum(), 1), 3),
             "esc_pct": round(A_["n_esc"] / n, 3),
             "h_esc": round(A_["h_esc"] / max(A_["n_esc"], 1), 3),
             "h_st_esc": round(A_["h_st"] / max(A_["n_esc"], 1), 3),
             "f_step": round(A_["fetch"] / n, 3),
             "local_mass": round(A_["local"] / n, 3)}
        for k in klist:
            for m_ in ("oracle", "flow", "static", "recency"):
                r[f"cov{k}_{m_}"] = round(A_[f"cov{k}_{m_}"] / n, 3)
        rows_out.append(r)
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows_out)

    val = [r for r in rows_out if r["domain"] == "val"]
    if val:
        struct = [r for r in val if r["dH_move"] >= 1.0
                  and r["h_esc"] >= 2 * max(r["h_st_esc"], 1e-9)]
        wshub = [r for r in val if r["hub_mass"] >= 0.30
                 and r["hub_ws_frac"] >= 0.5 and r["hub_pos_mean"] < 0.35]
        mech = [r for r in val if r["local_mass"] > 0.9]
        print(f"\n===== SUMMARY ({a.model}, {len(val)} heads, train-val) =====")
        print(f"structured (dH_move>=1 & h_esc>=2x static): "
              f"{len(struct)}/{len(val)} = {len(struct)/len(val):.1%}")
        print(f"whitespace hubs (mass>=.3, ws>=.5, front-loaded): "
              f"{len(wshub)}/{len(val)} = {len(wshub)/len(val):.1%}")
        print(f"offset-rule heads (local_mass>0.9, LRU-tier these): "
              f"{len(mech)}/{len(val)} = {len(mech)/len(val):.1%}")
        by_layer = defaultdict(list)
        for r in val:
            by_layer[r["layer"]].append(r["dH_move"])
        prof = " ".join(f"L{l}:{np.mean(v):.2f}"
                        for l, v in sorted(by_layer.items()))
        print(f"mean dH_move by layer: {prof}")
    if "410m" in a.model and 13 in layers and 2 in qheads:
        print("\n[calibration] canonical L13H2 references:", CANON)
        print("compare against the layer=13 qhead=2 rows above; if outside "
              "~+/-0.03 on h_esc/h_st_esc or ~+/-0.02 on cov4, reconcile "
              "conventions with escape_sim_v2 before the big run.")
    print(f"\nwrote {a.out}.csv ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()