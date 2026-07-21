"""
Overnight multi-domain transfer run.

  python overnight_multidomain.py fetch   # download new domain text files
  python overnight_multidomain.py run     # harvest + analysis, all domains

fetch writes datasets/{fiction,dialogue,math,news}_val.txt plus wiki_big
(wiki_val.txt at up to 1000 lines -- the 10x diversity fix). Every domain is
just a plain text file: if any download fails (HF datasets drift), paste any
large text into datasets/<tag>_val.txt and the runner picks it up
automatically. fiction comes straight from Project Gutenberg, no HF needed.

run: for every datasets/*_val.txt (wiki and code use their existing
harvests as reference rows) --
  harvest attention with the FROZEN wiki centroids; predictors train on
  wiki-train windows ONLY, so every other domain is pure OOD. Saves
  harvest_<tag>.npz (with token ids), then prints per domain:
    geometry    d_dom, d_dom/d_val, d_dom/D
    hub         key share + mass share of cluster 16, top-5 hub tokens
    coverage    oracle / flow / static at k = 2, 4, 8 (sink excluded)
    break-even  h_esc vs h_st_esc (k=4, ev refresh, THRESH=0.25, hub pinned)
  and a final cross-domain summary table.

PREDICTIONS GO ON RECORD BEFORE `run`. Claude's, written 2026-07-18:
  C1. hub mass share tracks newline/whitespace frequency: dialogue and news
      land clearly below wiki's 0.57; fiction and math nearer to it.
  C2. d_dom/d_val stays within 1.0-1.4 for the English prose domains; math
      is the worst of the new set.
  C3. flow beats static at k=4 on every domain, with the margin shrinking
      roughly monotonically in d_dom/d_val.
JOHAIL: add J1..J3 here before running.
"""
import os
import sys
import glob
import numpy as np
import torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.spatial.distance import cdist

LAYER_IDX, HEAD_IDX, K, SEQ_LEN = 13, 2, 64, 500
OUT = "extracted_data"
MODEL_DIR = r"models\pythia-410m"
HUBS = (16,)
MAX_WINDOWS = 40           # per-domain cap so the run fits in a night
KS = (2, 4, 8)
K_ESC, THRESH = 4, 0.25


# ---------------- fetch ----------------

def _write(tag, lines, cap=4000):
    p = f"datasets/{tag}_val.txt"
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines[:cap]))
    print(f"[fetch] {p}: {min(len(lines), cap)} lines")


def fetch():
    os.makedirs("datasets", exist_ok=True)
    import requests

    try:  # fiction -- direct download, no HF dependency
        lines = []
        for url in ("https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
                    "https://www.gutenberg.org/cache/epub/2701/pg2701.txt"):
            txt = requests.get(url, timeout=60).text
            body = txt.split("*** START", 1)[-1].split("*** END", 1)[0]
            lines += [ln.strip() for ln in body.split("\n") if ln.strip()]
        _write("fiction", lines)
    except Exception as e:
        print(f"[fetch] fiction FAILED: {e}")

    try:  # dialogue
        from datasets import load_dataset
        ds = load_dataset("knkarthick/dialogsum", split="validation")
        lines = []
        for ex in ds:
            lines += [ln.strip() for ln in ex["dialogue"].split("\n")
                      if ln.strip()]
        _write("dialogue", lines)
    except Exception as e:
        print(f"[fetch] dialogue FAILED: {e} -- paste any transcript into "
              f"datasets/dialogue_val.txt instead")

    try:  # math -- streaming, take a slice
        from datasets import load_dataset
        ds = load_dataset("open-web-math/open-web-math", split="train",
                          streaming=True)
        lines = []
        for i, ex in enumerate(ds):
            if i >= 150:
                break
            lines += [ln for ln in ex["text"].split("\n") if ln.strip()]
        _write("math", lines)
    except Exception as e:
        print(f"[fetch] math FAILED: {e}")

    try:  # news
        from datasets import load_dataset
        ds = load_dataset("ag_news", split="test")
        lines = [ex["text"].strip() for ex in ds if ex["text"].strip()]
        _write("news", lines)
    except Exception as e:
        print(f"[fetch] news FAILED: {e}")

    src = "datasets/wiki_val.txt"          # wiki_big: the 10x diversity fix
    if os.path.exists(src):
        with open(src, encoding="utf-8") as f:
            lines = [ln for ln in f.read().split("\n") if ln.strip()]
        if len(lines) > 100:
            _write("wiki_big", lines, cap=1000)
        else:
            print(f"[fetch] wiki_val.txt has only {len(lines)} lines; "
                  f"wiki_big skipped -- point it at a bigger dump if you have one")


# ---------------- harvest ----------------

def windows(tokenizer, lines, L=SEQ_LEN):
    ids = tokenizer("\n".join(lines), return_tensors="pt").input_ids[0]
    return [ids[i:i + L].unsqueeze(0) for i in range(0, len(ids) - L + 1, L)]


def harvest(model, chunks, centroids, tag):
    clus, top5, mass, mass_ns, toks = [], [], [], [], []
    eye = np.eye(K, dtype=np.float32)
    for n, input_ids in enumerate(chunks):
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True,
                        output_attentions=True)
        keys = out.past_key_values.layers[LAYER_IDX].keys[0, HEAD_IDX].cpu().numpy()
        keys = keys / np.linalg.norm(keys, axis=1, keepdims=True)
        d = cdist(keys, centroids)
        idx5 = np.argsort(d, axis=1)[:, :5]
        c = idx5[:, 0]
        attn = out.attentions[LAYER_IDX][0, HEAD_IDX].float().cpu().numpy()
        oh = eye[c]
        m = attn @ oh
        a2 = attn.copy()
        a2[:, 0] = 0.0
        rs = a2.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        m2 = (a2 / rs) @ oh
        clus.append(c.astype(np.int16))
        top5.append(np.take_along_axis(d, idx5, axis=1).astype(np.float32))
        mass.append(m.astype(np.float32))
        mass_ns.append(m2.astype(np.float32))
        toks.append(input_ids[0].cpu().numpy().astype(np.int32))
        print(f"[{tag}] window {n + 1}/{len(chunks)}", end="\r")
    np.savez_compressed(
        os.path.join(OUT, f"harvest_{tag}.npz"),
        clusters=np.stack(clus), top5=np.stack(top5), mass=np.stack(mass),
        mass_nosink=np.stack(mass_ns), tokens=np.stack(toks))
    print(f"\n[{tag}] saved {len(chunks)} windows -> harvest_{tag}.npz")
    return (np.stack(clus), np.stack(top5), np.stack(mass_ns),
            np.stack(toks))


# ---------------- offline battery ----------------

def build_predictors(clusters, mass, train_idx, alpha=0.1, exclude=()):
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
    return A / np.maximum(A.sum(1, keepdims=True), 1e-12), g / max(n, 1)


def topk(row, cur, k, exclude=()):
    s = [cur] if cur not in exclude else []
    for j in np.argsort(row)[::-1]:
        if j != cur and j not in exclude:
            s.append(int(j))
        if len(s) == k:
            break
    return s[:k]


def domain_report(tag, clu, top5, mass_ns, toks, idx, P0, g0, Px, gx,
                  d_val, D, tokenizer):
    d_dom = float(top5[:, :, 0].mean())
    hub_keys = float((clu == HUBS[0]).mean())
    hub_mass = float(mass_ns[:, 1:, HUBS[0]].mean())
    cnt = Counter()
    for w in range(len(clu)):
        sel = np.asarray(clu[w]) == HUBS[0]
        cnt.update(np.asarray(toks[w])[sel].tolist())
    top_toks = ", ".join(repr(tokenizer.decode([t])) for t, _ in
                         cnt.most_common(5)) if cnt else "(none)"

    hubs = set(HUBS)
    cov = {nm: {k: 0.0 for k in KS} for nm in ("oracle", "flow", "static")}
    n = n_esc = hit_es = hit_eg = 0
    for w in idx:
        c, mm = clu[w], mass_ns[w]
        for t in range(1, len(c)):
            row = mm[t]
            cur = int(c[t - 1])
            desc = np.argsort(row)[::-1]
            for k in KS:                       # Test-A convention, unpinned
                cov["oracle"][k] += row[desc[:k]].sum()
                cov["flow"][k] += row[topk(P0[cur], cur, k)].sum()
                cov["static"][k] += row[topk(g0, cur, k)].sum()
            r2 = row.copy()                    # escape battery, hub pinned
            r2[list(hubs)] = 0.0
            r2[cur] = 0.0
            cstar = int(np.argmax(r2))
            staged = topk(Px[cur], cur, K_ESC, hubs)   # ev == m=1
            stat = topk(gx, cur, K_ESC, hubs)
            if r2.sum() > THRESH:
                hit_es += cstar in staged
                hit_eg += cstar in stat
                n_esc += 1
            n += 1

    print(f"\n===== {tag} ({n} steps) =====")
    print(f"geometry : d_dom {d_dom:.3f}   d_dom/d_val {d_dom / d_val:.2f}"
          f"   d_dom/D {d_dom / D:.2f}")
    print(f"hub 16   : key share {100 * hub_keys:.1f}%   "
          f"mass share {hub_mass:.3f}   top tokens: {top_toks}")
    print(f"{'k':>4} | {'oracle':>7} {'flow':>7} {'static':>7}")
    for k in KS:
        print(f"{k:>4} | " + " ".join(f"{cov[nm][k] / n:>7.3f}"
                                      for nm in ("oracle", "flow", "static")))
    ne = max(n_esc, 1)
    print(f"escape   : esc% {100 * n_esc / n:.1f}   "
          f"h_esc {hit_es / ne:.3f}   h_st_esc {hit_eg / ne:.3f}")
    return dict(tag=tag, ratio=d_dom / d_val, hub_mass=hub_mass,
                flow4=cov["flow"][4] / n, static4=cov["static"][4] / n,
                h_esc=hit_es / ne, h_st=hit_eg / ne)


def run():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model.eval()
    centroids = np.load(os.path.join(OUT, "wiki_centroids.npy"))
    g1 = np.load(os.path.join(OUT, "gate1.npz"))
    d_val, D = float(g1["d_val"]), float(g1["D"])

    zw = np.load(os.path.join(OUT, "harvest_wiki.npz"))
    w_clu, w_mass = zw["clusters"], zw["mass_nosink"]
    train_idx = list(range(0, len(w_clu), 2))
    P0, g0 = build_predictors(w_clu, w_mass, train_idx)
    Px, gx = build_predictors(w_clu, w_mass, train_idx, exclude=set(HUBS))

    rows = []
    for path in sorted(glob.glob("datasets/*_val.txt")):
        tag = os.path.basename(path).replace("_val.txt", "")
        if tag in ("wiki", "code"):
            continue                       # reference rows added below
        with open(path, encoding="utf-8") as f:
            lines = [ln for ln in f.read().split("\n") if ln.strip()]
        chunks = windows(tokenizer, lines)[:MAX_WINDOWS]
        if len(chunks) < 3:
            print(f"[{tag}] not enough text, skipped")
            continue
        clu, top5, mass_ns, toks = harvest(model, chunks, centroids, tag)
        rows.append(domain_report(tag, clu, top5, mass_ns, toks,
                                  range(len(clu)), P0, g0, Px, gx,
                                  d_val, D, tokenizer))

    for tag in ("wiki", "code"):           # reference rows from existing npz
        z = np.load(os.path.join(OUT, f"harvest_{tag}.npz"))
        clu, top5, mass_ns = z["clusters"], z["top5"], z["mass_nosink"]
        toks = z["tokens"] if "tokens" in z else np.zeros_like(clu)
        idx = (list(range(1, len(clu), 2)) if tag == "wiki"
               else range(len(clu)))       # wiki scored on held-out half only
        rows.append(domain_report(tag, clu, top5, mass_ns, toks, idx,
                                  P0, g0, Px, gx, d_val, D, tokenizer))

    print("\n===== cross-domain summary (wiki-trained everything) =====")
    print(f"{'domain':>10} | {'d/dval':>6} {'hubmass':>7} | "
          f"{'flow@4':>7} {'stat@4':>7} | {'h_esc':>6} {'h_st':>6}")
    for r in rows:
        print(f"{r['tag']:>10} | {r['ratio']:>6.2f} {r['hub_mass']:>7.3f} | "
              f"{r['flow4']:>7.3f} {r['static4']:>7.3f} | "
              f"{r['h_esc']:>6.3f} {r['h_st']:>6.3f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "fetch":
        fetch()
    else:
        run()