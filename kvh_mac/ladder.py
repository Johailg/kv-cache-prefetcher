import argparse, glob, sys, time
import numpy as np

# Scoring space is HUB-EXCLUDED: M has the hub column zeroed and rows
# renormalized, and the hub is banned as a candidate for every policy.
def select_oracle(k, ctx):
    return np.argsort(-ctx["m_true"])[:k]

def select_flow(k, ctx):
    return np.argsort(-ctx["flow_row"])[:k]

def select_static(k, ctx):
    return np.argsort(-ctx["freq_row"])[:k]

def select_recency(k, ctx):
    ls = ctx["last_seen"]
    S = np.argsort(-ls)[:k]
    return S[ls[S] >= 0]

POLICIES = {"oracle": select_oracle, "flow": select_flow,
            "static": select_static, "recency": select_recency}

def assign(X, C):
    return np.argmax(2.0 * X @ C.T - (C * C).sum(1), axis=1)

def dense_rows(idx, val, c, nc):
    T = idx.shape[0]
    off = np.arange(T)[:, None] * nc + c[idx]
    M = np.bincount(off.ravel(), weights=val.ravel(), minlength=T*nc).reshape(T, nc)
    return M / np.maximum(M.sum(1, keepdims=True), 1e-12)

def hw(arr, h, nh):
    return arr[h] if arr.shape[0] == nh else arr[:, h]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect")
    ap.add_argument("--fit"); ap.add_argument("--shards")
    ap.add_argument("--test-from", type=int, default=200)
    ap.add_argument("--layers", default="3,13,24")
    ap.add_argument("--ks", default="1,2,4,8,16")
    ap.add_argument("--warmup", type=int, default=64)
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--route-lf", type=float, default=0.15,
                    help="route heads with local_frac below this to flow, else recency")
    ap.add_argument("--per-head", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if a.inspect:
        z = np.load(a.inspect, allow_pickle=True)
        for k in z.files:
            print(f"  {k:16s} {str(getattr(z[k],'shape','-')):26s} {getattr(z[k],'dtype','-')}")
        return

    f = np.load(a.fit)
    for need in ("centroids","flow","freq","hubs","local_frac"):
        if need not in f.files: sys.exit(f"ERROR: no '{need}' in {a.fit}; has {f.files}")
    cent, flow, freq, hubs, lf = (f["centroids"], f["flow"], f["freq"],
                                  f["hubs"], f["local_frac"])
    layers = [int(x) for x in a.layers.split(",")]
    ks = [int(x) for x in a.ks.split(",")]
    nh, nc = flow.shape[1], flow.shape[-1]
    print(f"--- fit: C={nc} clusters, {nh} kv heads ---")
    for L in layers:
        print(f"  L{L} hub {[int(hubs[L,h,0]) for h in range(nh)]}")
        print(f"  L{L} local_frac " + " ".join(f"{lf[L,h]:.3f}" for h in range(nh)))
        print(f"  L{L} routed->flow " +
              " ".join(("F" if lf[L,h] < a.route_lf else ".") for h in range(nh)))

    allsh = sorted(glob.glob(f"{a.shards}/shard_*.npz"))
    if not allsh: sys.exit(f"ERROR: no shard_*.npz in {a.shards}")
    if len(allsh) <= a.test_from:
        sys.exit(f"ERROR: --test-from {a.test_from} but only {len(allsh)} shards")
    shards = allsh[a.test_from:]
    if a.max_shards: shards = shards[:a.max_shards]
    print(f"\n{len(allsh)} shards, testing on {len(shards)}"
          f" ({shards[0].split('/')[-1]} .. {shards[-1].split('/')[-1]})")

    acc = {(L,h,k,p): [0.0,0] for L in layers for h in range(nh)
           for k in ks for p in POLICIES}
    hubacc = {(L,h): [0.0,0] for L in layers for h in range(nh)}
    scount = {(L,h): np.zeros(nc) for L in layers for h in range(nh)}
    t0 = time.time()

    for si, sp in enumerate(shards):
        z = np.load(sp)
        for L in layers:
            keys, val, idx = z[f"keys_{L}"], z[f"val_{L}"], z[f"idx_{L}"]
            for h in range(nh):
                hub = int(hubs[L, h, 0])
                K, V, I = hw(keys,h,nh), hw(val,h,nh), hw(idx,h,nh)
                c = assign(K.astype(np.float64), cent[L,h].astype(np.float64))
                M = dense_rows(I, V.astype(np.float64), c, nc)

                hub_mass = M[:, hub].copy()
                M2 = M.copy(); M2[:, hub] = 0.0
                s = M2.sum(1, keepdims=True)
                live = s[:,0] > 1e-9
                M2 = M2 / np.maximum(s, 1e-12)
                Msel = M2.copy(); Msel[:, hub] = -np.inf

                FR = flow[L,h].astype(np.float64).copy(); FR[:, hub] = -np.inf
                GR = freq[L,h].astype(np.float64).copy(); GR[hub]    = -np.inf

                last = np.full(nc, -1)
                for t in range(M.shape[0]):
                    if t > a.warmup and live[t]:
                        scount[(L,h)][c[t-1]] += 1
                        e = hubacc[(L,h)]; e[0] += float(hub_mass[t]); e[1] += 1
                        ctx = {"m_true": Msel[t], "flow_row": FR[c[t-1]],
                               "freq_row": GR, "hist": c[:t], "last_seen": last}
                        for k in ks:
                            for pn, fn in POLICIES.items():
                                S = np.asarray(fn(k, ctx), dtype=int)
                                e = acc[(L,h,k,pn)]
                                e[0] += float(M2[t, S].sum()); e[1] += 1
                    if c[t] != hub: last[c[t]] = t
        if (si+1) % 10 == 0: print(f"  {si+1}/{len(shards)}  ({time.time()-t0:.0f}s)")

    def agg(L,k,p,heads):
        s = sum(acc[(L,h,k,p)][0] for h in heads)
        n = sum(acc[(L,h,k,p)][1] for h in heads)
        return s / max(n,1)

    def routed(L,k):
        s = n = 0.0
        for h in range(nh):
            p = "flow" if lf[L,h] < a.route_lf else "recency"
            s += acc[(L,h,k,p)][0]; n += acc[(L,h,k,p)][1]
        return s / max(n,1)

    for L in layers:
        hm = sum(hubacc[(L,h)][0] for h in range(nh)) / \
             max(sum(hubacc[(L,h)][1] for h in range(nh)), 1)
        print(f"\n--- L{L} HELD OUT, hub-excluded (C={nc}, hub mass {hm:.4f}) ---")
        print("   k    k/C | " + " | ".join(f"{p:>9s}" for p in POLICIES) + " |    routed")
        for k in ks:
            print(f"{k:4d} {k/nc:6.4f} | " +
                  " | ".join(f"{agg(L,k,p,range(nh)):9.4f}" for p in POLICIES) +
                  f" | {routed(L,k):9.4f}")
        if a.per_head:
            print("  per-head flow-minus-recency, by local_frac:")
            for h in sorted(range(nh), key=lambda x: lf[L,x]):
                d = " ".join(f"k{k}:{agg(L,k,'flow',[h])-agg(L,k,'recency',[h]):+.3f}"
                             for k in ks)
                print(f"    h{h} lf={lf[L,h]:.3f}  {d}")

    print("\n--- fitted flow rows (in-sample), reweighted by TEST state frequency ---")
    for L in layers:
        num = {k:0.0 for k in ks}; tot = 0.0
        for h in range(nh):
            n = scount[(L,h)].sum()
            if n == 0: continue
            p = scount[(L,h)] / n
            cs = np.cumsum(np.sort(flow[L,h], axis=1)[:, ::-1], axis=1)
            for k in ks: num[k] += n * float(p @ cs[:, min(k,nc)-1])
            tot += n
        print(f"  L{L}: " + "  ".join(f"k={k}:{num[k]/max(tot,1):.4f}" for k in ks))

    print(f"\nelapsed {time.time()-t0:.0f}s")
    if a.out:
        np.savez(a.out,
            sums=np.array([[[[acc[(L,h,k,p)][0] for p in POLICIES] for k in ks]
                            for h in range(nh)] for L in layers]),
            counts=np.array([[[[acc[(L,h,k,p)][1] for p in POLICIES] for k in ks]
                              for h in range(nh)] for L in layers]),
            local_frac=np.array([[lf[L,h] for h in range(nh)] for L in layers]),
            n_clusters=nc, layers=np.array(layers), ks=np.array(ks),
            policies=np.array(list(POLICIES)))
        print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
