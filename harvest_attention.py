"""
Harvest everything that needs forward passes, in ONE pass per domain:
  - cluster id per key                      (same as before)
  - top-5 centroid distances per key        (Test C: soft-vs-argmax evidence)
  - per-step attention mass aggregated by
    the cluster of each ATTENDED key        (Test A: prefetch-relevant signal)

Saves extracted_data/harvest_{wiki,code}.npz. Everything downstream is offline.

NOTE: attn_implementation="eager" is required -- sdpa/flash paths don't return
attention weights. Expect this to run ~2-3x slower than your extract.py.
"""
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.spatial.distance import cdist

LAYER_IDX, HEAD_IDX, K, SEQ_LEN = 13, 2, 64, 500
OUT = "extracted_data"
MODEL_DIR = r"models\pythia-410m"


def load_text(path):
    with open(path, encoding="utf-8") as f:
        return [ln for ln in f.read().split("\n") if ln.strip()]


def windows(tokenizer, lines, L=SEQ_LEN):
    ids = tokenizer("\n".join(lines), return_tensors="pt").input_ids[0]
    return [ids[i:i + L].unsqueeze(0) for i in range(0, len(ids) - L + 1, L)]


def harvest(model, chunks, centroids, tag):
    clus, top5, mass, mass_ns = [], [], [], []
    eye = np.eye(K, dtype=np.float32)
    for n, input_ids in enumerate(chunks):
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True, output_attentions=True)

        keys = out.past_key_values.layers[LAYER_IDX].keys[0, HEAD_IDX].cpu().numpy()
        keys = keys / np.linalg.norm(keys, axis=1, keepdims=True)

        d = cdist(keys, centroids)                    # replicates kmeans.transform
        idx5 = np.argsort(d, axis=1)[:, :5]
        c = idx5[:, 0]

        # (T, T) causal attention for L13/H2; row t = query t's distribution
        attn = out.attentions[LAYER_IDX][0, HEAD_IDX].float().cpu().numpy()

        oh = eye[c]                                   # (T, K) one-hot of key clusters
        m = attn @ oh                                 # (T, K) mass on each cluster

        # sink-excluded variant: zero attention to position 0, renormalize rows
        a2 = attn.copy()
        a2[:, 0] = 0.0
        rs = a2.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        m2 = (a2 / rs) @ oh

        clus.append(c.astype(np.int16))
        top5.append(np.take_along_axis(d, idx5, axis=1).astype(np.float32))
        mass.append(m.astype(np.float32))
        mass_ns.append(m2.astype(np.float32))
        print(f"[{tag}] window {n + 1}/{len(chunks)}", end="\r")

    np.savez_compressed(
        os.path.join(OUT, f"harvest_{tag}.npz"),
        clusters=np.stack(clus),          # (W, 500) int16
        top5=np.stack(top5),              # (W, 500, 5) float32
        mass=np.stack(mass),              # (W, 500, 64) float32
        mass_nosink=np.stack(mass_ns),    # (W, 500, 64) float32
    )
    print(f"\n[{tag}] saved {len(chunks)} windows -> harvest_{tag}.npz")


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model.eval()

    centroids = np.load(os.path.join(OUT, "wiki_centroids.npy"))

    harvest(model, windows(tokenizer, load_text(r"datasets\wiki_val.txt")[:100]),
            centroids, "wiki")
    harvest(model, windows(tokenizer, load_text(r"datasets\code_val.txt")[:700]),
            centroids, "code")