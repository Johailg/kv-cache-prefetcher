"""
kvh.harvest -- build the ClusterIndex for a new model. Pay-once, per mt_harvest.

STAGE 1  `harvest`, one GPU pass
    Store, per (layer, kv_head): post-rotary L2-normalised keys as fp16, and
    the top-N of each DECODE row of mass_nosink (values fp16, indices int32,
    tail scalar). Nothing about clustering is decided here, so any later change
    to C, hub rule or graph is offline numpy against the same shards. This is
    what mt_remass.py does on the Pythia side.

STAGE 2  `fit`, CPU
    Per (layer, kv_head):
      centroids  KMeans over the normalised keys, centres kept RAW
      assignment argmin ||x - c||, computed as argmax(2 x.c - ||c||^2) since
                 ||x|| = 1. Algebraically identical to mt_harvest.dists_to and
                 avoids a [T, C, D] temporary.
      mass       rebuilt from the stored top-N, aggregated by attended cluster
      g          mean mass row over all steps
      hub        argmax of g
      flow       A = full(alpha); A[c[t-1]] += m[t]; hub COLUMNS zeroed;
                 rows normalised.   MASS ROWS, NOT TRANSITION COUNTS.
      freq       g with the hub entry zeroed
      local_frac mean mass at offsets 0..4 -- the head_sweep guardrail

STORAGE, work it out before launching
    bytes/session ~= (head-layers) x T x topk x 6
    36 layers x 8 kv heads x 4096 x top-64  =  ~450 MB per session.
    Use --layers. Three layers x 8 heads = 24 head-layers, ~38 MB/session,
    ~8 GB for 200 sessions. That is also n=24 on the per-head question, which
    is the 16-head expansion, on the model Zhang asked about.

    python -m kvh.harvest harvest --model Qwen/Qwen3-4B \
        --sessions data/wildchat_train.jsonl --out h_qwen/ \
        --layers 8,17,26 --max-sessions 200
    python -m kvh.harvest fit --shards h_qwen/ --clusters 64 --alpha 0.1 \
        --out index_qwen.npz
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

# Works both as a package (`python -m kvh.harvest`) and as a plain script
# (`python harvest.py` from inside the directory). The relative form is tried
# first so the package layout stays canonical.
try:
    from . import attn as A
    from . import policies as P
except ImportError:
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import attn as A
    import policies as P


class HarvestPolicy(P.Policy):
    """Evicts nothing. Records the L2-normalised keys.

    The mass rows are captured separately, through PolicyState.row_capture, so
    that EVERY row is stored -- prefill rows included. mt_harvest takes one
    forward pass over the whole sequence and keeps the top-64 of every row of
    the [T, T] attention matrix; mt_common.build_flow then loops over all
    t >= 1. Capturing only decode rows would train the graph on a different
    support, which can move the hub, which is a different lineage.
    """
    name = "harvest"

    def __init__(self, layers):
        self.layers = layers

    def reset(self, *a, **kw):
        super().reset(*a, **kw)
        self.keys = {}

    def wanted(self, layer):
        return self.layers is None or layer in self.layers

    def note_keys(self, layer, keys_new):
        if not self.wanted(layer):
            return
        unit = torch.nn.functional.normalize(keys_new.float(), dim=-1)
        block = unit.to(torch.float16).cpu()                  # [Hkv, n_new, D]
        seen = self.keys.get(layer)
        self.keys[layer] = block if seen is None else torch.cat([seen, block], 1)


class RowStore:
    """Collects the top-N of every nosink row, as PolicyState.row_capture.

    rows arrive as [Hkv, n_rows, K] with K growing across calls, so each chunk
    is padded out to the final width at save time. Row r of the store is
    absolute sequence position r -- no prefill offset to remember later.
    """

    def __init__(self, policy: HarvestPolicy, topk: int):
        self.policy = policy
        self.topk = topk
        self.val, self.idx, self.tail = {}, {}, {}

    def __call__(self, layer, first_row, rows):
        if not self.policy.wanted(layer):
            return
        n = min(self.topk, rows.shape[2])
        val, idx = torch.topk(rows, n, dim=2)                 # [Hkv, r, n]
        self.val.setdefault(layer, []).append(val.half().cpu().numpy())
        self.idx.setdefault(layer, []).append(idx.to(torch.int32).cpu().numpy())
        self.tail.setdefault(layer, []).append(
            (1.0 - val.sum(2)).clamp_min(0).half().cpu().numpy())

    def stack(self, layer, topk):
        """-> (val, idx, tail) with row index == absolute position."""
        def pad(chunks):
            out = []
            for c in chunks:                                  # [Hkv, r, n]
                if c.shape[2] < topk:
                    w = np.zeros((c.shape[0], c.shape[1], topk), c.dtype)
                    w[:, :, :c.shape[2]] = c
                    c = w
                out.append(c)
            return np.concatenate(out, axis=1).transpose(1, 0, 2)   # [T, Hkv, topk]
        val = pad(self.val[layer])
        idx = pad(self.idx[layer])
        tail = np.concatenate(self.tail[layer], axis=1).T           # [T, Hkv]
        return val, idx, tail

    def clear(self):
        self.val, self.idx, self.tail = {}, {}, {}


@torch.no_grad()
def harvest(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from .sessions import build_sessions
    except ImportError:
        from sessions import build_sessions

    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype,
        attn_implementation="eager").to(args.device).eval()
    n_layers, n_kv, head_dim = A.model_shape(model)
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(range(n_layers)))
    print(f"{args.model}: {n_layers} layers, {n_kv} kv heads, dim {head_dim}; "
          f"harvesting {len(layers) * n_kv} head-layers {layers}")

    sessions = build_sessions(args.sessions, tok, args.max_len, args.min_turns,
                              args.max_sessions, args.device)
    if not sessions:
        raise SystemExit(
                f"build_sessions returned 0 sessions from {args.sessions}"
                f"(max_len = {args.max_len}, min_turns = {args.min_turns}). nothign to harvest.")

    print(f"{len(sessions)} sessions")
    os.makedirs(args.out, exist_ok=True)

    policy = HarvestPolicy(set(layers))
    store = RowStore(policy, args.topk)
    state = A.PolicyState(policy, track_recovery=False, query_chunk=args.chunk,
                          row_capture=store)
    restore = A.install(model, state)
    try:
        for i, sess in enumerate(sessions):
            state.reset_sequence(n_layers, n_kv, head_dim, args.device, dtype)
            store.clear()
            # ONE forward over the whole sequence, chunked inside the attention
            # function. Same object canon builds from output_attentions, without
            # ever materialising the full [T, T] matrix.
            model(sess.ids, use_cache=False)

            blob = {}
            for L in layers:
                val, idx, tail = store.stack(L, args.topk)
                blob[f"keys_{L}"] = policy.keys[L].numpy()      # [Hkv, T, D]
                blob[f"val_{L}"] = val                          # [T, Hkv, topk]
                blob[f"idx_{L}"] = idx                          # [T, Hkv, topk]
                blob[f"tail_{L}"] = tail                        # [T, Hkv]
            np.savez_compressed(
                os.path.join(args.out, f"shard_{i:05d}.npz"), **blob,
                n_layers=n_layers, n_kv=n_kv, head_dim=head_dim,
                layers=np.array(layers),
                config=json.dumps(dict(model=args.model, topk=args.topk)))
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(sessions)}", flush=True)
    finally:
        restore()
    print(f"wrote shards to {args.out}")


def assign_clusters(unit_keys, centres):
    """unit_keys [T, D] with ||x|| = 1, centres [C, D] raw -> [T] int64.

    argmin ||x - c||^2 = argmin (||x||^2 - 2 x.c + ||c||^2)
                       = argmax (2 x.c - ||c||^2)          since ||x||^2 = 1

    Same answer as torch.cdist / mt_harvest.dists_to, without materialising a
    [T, C, D] difference tensor. The ||c||^2 term is what makes this different
    from plain cosine -- do not drop it.
    """
    return (2.0 * unit_keys @ centres.T
            - (centres ** 2).sum(1)[None, :]).argmax(1)


def fit(args):
    from sklearn.cluster import KMeans

    files = sorted(glob.glob(os.path.join(args.shards, "shard_*.npz")))
    if not files:
        raise SystemExit(f"no shards in {args.shards}")
    head = np.load(files[0], allow_pickle=True)
    n_layers = int(head["n_layers"])
    n_kv = int(head["n_kv"])
    head_dim = int(head["head_dim"])
    layers = [int(x) for x in head["layers"]]
    C = args.clusters
    print(f"{len(files)} shards, layers {layers}, {n_kv} kv heads, dim {head_dim}")

    centroids = np.zeros((n_layers, n_kv, C, head_dim), np.float32)
    hubs = np.zeros((n_layers, n_kv, 1), np.int64)
    flow = np.zeros((n_layers, n_kv, C, C), np.float32)
    freq = np.zeros((n_layers, n_kv, C), np.float32)
    local = np.zeros((n_layers, n_kv), np.float32)
    rng = np.random.default_rng(args.seed)

    for L in layers:
        # ---- centroids ---------------------------------------------------
        for h in range(n_kv):
            X = np.concatenate([np.load(f)[f"keys_{L}"][h] for f in files],
                               axis=0).astype(np.float32)
            X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
            if X.shape[0] > args.pool_cap:
                X = X[rng.choice(X.shape[0], args.pool_cap, replace=False)]
            centroids[L, h] = KMeans(C, n_init="auto",
                                     random_state=args.seed).fit(X).cluster_centers_

        # ---- mass, g, flow -----------------------------------------------
        transitions = np.full((n_kv, C, C), float(args.alpha), np.float64)
        g = np.zeros((n_kv, C), np.float64)
        local_acc = np.zeros(n_kv, np.float64)
        n_steps = 0

        for f in files:
            z = np.load(f)
            keys = z[f"keys_{L}"].astype(np.float32)       # [Hkv, T, D]
            val = z[f"val_{L}"].astype(np.float64)         # [T, Hkv, topk]
            idx = z[f"idx_{L}"].astype(np.int64)           # [T, Hkv, topk]
            T = val.shape[0]

            for h in range(n_kv):
                X = keys[h]
                X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
                cid = assign_clusters(X, centroids[L, h])  # [T]

                attended = cid[idx[:, h, :]]               # [T, topk] cluster per hit
                rows = np.repeat(np.arange(T), attended.shape[1])
                mass = np.zeros((T, C))
                np.add.at(mass, (rows, attended.ravel()), val[:, h, :].ravel())

                # row index IS the absolute position, so c[t-1] is cid[t-1] and
                # the loop runs over t = 1..T-1, matching build_flow exactly.
                np.add.at(transitions[h], cid[:T - 1], mass[1:])
                g[h] += mass[1:].sum(0)

                offsets = np.arange(T)[:, None] - idx[:, h, :]
                local_acc[h] += float(val[:, h, :][(offsets >= 0) & (offsets < 5)].sum())
            n_steps += T - 1

        for h in range(n_kv):
            hub = int(g[h].argmax())
            share = float(g[h][hub] / max(g[h].sum(), 1e-9))
            hubs[L, h, 0] = hub
            transitions[h][:, hub] = 0.0                   # hub COLUMNS, after counting
            g[h][hub] = 0.0
            row_sums = np.maximum(transitions[h].sum(1, keepdims=True), 1e-9)
            flow[L, h] = (transitions[h] / row_sums).astype(np.float32)
            freq[L, h] = (g[h] / max(n_steps, 1)).astype(np.float32)
            local[L, h] = local_acc[h] / max(n_steps, 1)
            print(f"  L{L} h{h}: hub {hub}  mass share {share:.3f}  "
                  f"local_frac {local[L, h]:.3f}", flush=True)

    np.savez_compressed(args.out, centroids=centroids, hubs=hubs, flow=flow,
                        freq=freq, local_frac=local)
    print(f"wrote {args.out}")
    print("READ THE TWO NUMBERS ABOVE BEFORE GOING FURTHER:")
    print("  mass share near 0.6  -> the hub mechanism survived (Pythia L13H2 was 0.62)")
    print("  mass share near 1/C  -> no dominant cluster; cluster policies are moot here")
    print("  local_frac near 1.0  -> offset-rule head; the cluster machinery does")
    print("                          not apply and it should be routed, not clustered")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest")
    h.add_argument("--model", required=True)
    h.add_argument("--sessions", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--layers", default="", help="comma list; default all (large)")
    h.add_argument("--topk", type=int, default=64)
    h.add_argument("--chunk", type=int, default=512,
                   help="query rows per attention chunk; caps the [q, K] matrix")
    h.add_argument("--max-len", type=int, default=4096)
    h.add_argument("--min-turns", type=int, default=4)
    h.add_argument("--max-sessions", type=int, default=200)
    h.add_argument("--device", default="cuda")
    h.add_argument("--dtype", default="bfloat16")
    h.set_defaults(fn=harvest)

    f = sub.add_parser("fit")
    f.add_argument("--shards", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--clusters", type=int, default=64)
    f.add_argument("--alpha", type=float, default=0.1)
    f.add_argument("--pool-cap", type=int, default=50_000)
    f.add_argument("--seed", type=int, default=42)
    f.set_defaults(fn=fit)

    a = ap.parse_args()
    a.fn(a)
