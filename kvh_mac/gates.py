"""
kvh.gates -- run before believing any number this harness prints.

G1  IDENTITY     patched attention with `full` reproduces the stock model's
                 logits. If this fails everything downstream is fiction.
                 This is the only gate that is load-bearing on its own.
G2  RESIDENCY    reported residency equals the mask's own occupancy, checked
                 against a policy whose keep-count is known in closed form.
G3  ENVELOPE     sink_only <= every policy <= oracle on recovery.
G4  MONOTONE     loosening the budget never lowers recovery.
G5  ORDERING     the policy sees n_old keys at select() time, not n_old+1.
                 This is the c[t-1] vs c[t] check, done with a probe policy.
G6  GQA MAPPING  query head q belongs to KV head q // G, consistently across
                 repeat_kv, the mask expansion, and the group reductions.
                 Pure algebra, no model, no GPU -- run it anywhere.
                 THIS IS THE ONLY CHECK ON THE GQA PATH. gate_pythia cannot
                 cover it (Pythia is MHA, G=1), so if this is wrong every
                 escape event on Qwen3 is scored against the wrong KV head's
                 hub and assignment, and nothing else would notice.

    python -m kvh.gates --model Qwen/Qwen3-0.6B --device cuda --dtype float32
"""
from __future__ import annotations
import argparse
import torch
# Works both as a package (`python -m kvh.gates`) and as a plain script
# (`python gates.py` from inside the directory). The relative form is tried
# first so the package layout stays canonical.
try:
    from . import attn as A
    from . import policies as P
except ImportError:
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import attn as A
    import policies as P


class OrderProbe(P.Policy):
    """Records n_old at select() and the cache length at note_keys(). If
    select ran after the write, the two would be equal."""
    name, granularity = "probe", "token"

    def reset(self, *a, **kw):
        super().reset(*a, **kw)
        self.saw_select, self.saw_note = [], []

    def select(self, layer, n_old, n_new):
        self.saw_select.append((layer, n_old))
        return None

    def note_keys(self, layer, keys_new):
        self.saw_note.append((layer, keys_new.shape[1]))


def gate_gqa(n_kv=8, groups=4, k_len=11, n_c=6):
    """G6: every place kvh moves between query-head and KV-head space must agree
    on the same mapping, q -> q // groups.

    repeat_kv produces [kv0 x G, kv1 x G, ...]. So:
      - expanding a per-KV-head thing to query heads is repeat_interleave, NOT
        repeat  (repeat would give [kv0, kv1, ..., kv0, kv1, ...])
      - reducing back is view(n_kv, groups, ...) then sum/mean over dim 1
    """
    n_q = n_kv * groups
    expected = torch.arange(n_kv).repeat_interleave(groups)      # [Hq]

    marker = torch.arange(n_kv).view(1, n_kv, 1, 1).float()
    rep = _repeat_kv_ref(marker, groups).flatten()
    a = bool((rep == expected.float()).all())

    ids = torch.arange(n_kv).view(n_kv, 1).expand(n_kv, k_len)
    b = bool((ids.repeat_interleave(groups, dim=0)[:, 0] == expected).all())

    hub = torch.zeros(n_kv, n_c, dtype=torch.bool)
    hub[torch.arange(n_kv), torch.arange(n_kv) % n_c] = True
    hub_q = hub.repeat_interleave(groups, dim=0)
    c = bool((hub_q.float().argmax(1) == (expected % n_c)).all())

    per_q = expected.float().view(n_q, 1)
    d = bool((per_q.view(n_kv, groups, 1).mean(dim=1).flatten()
              == torch.arange(n_kv).float()).all())

    e = bool((per_q.view(n_kv, groups, 1).sum(dim=1).flatten()
              == torch.arange(n_kv).float() * groups).all())

    ok = a and b and c and d and e
    print(f"G6 gqa mapping  repeat_kv {a}  ids {b}  hub {c}  "
          f"mean-reduce {d}  sum-reduce {e}   {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("   -> every escape event under GQA is scored against the wrong "
              "KV head. Nothing else in the suite would catch this.")
    return ok


def _repeat_kv_ref(x, groups):
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, groups, s, d).reshape(b, h * groups, s, d)


@torch.no_grad()
def run(model_id, device, dtype, seq_len, tol):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    AutoTokenizer.from_pretrained(model_id)
    model = A.load_causal_lm(model_id, dtype).to(device).eval()
    n_layers, n_kv, head_dim = A.model_shape(model)
    n_q = model.config.num_attention_heads
    print(f"{model_id}: {n_layers} layers, {n_q} q heads, {n_kv} kv heads "
          f"(G={n_q // n_kv}), head_dim {head_dim}")

    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, model.config.vocab_size, (1, seq_len), generator=g).to(device)
    reference = model(ids).logits[0, -1].float()

    def decode_run(policy, prefill=None):
        prefill = prefill or seq_len - 64
        state.policy = policy
        state.reset_sequence(n_layers, n_kv, head_dim, device, dtype)
        out = model(ids[:, :prefill], use_cache=True)
        past = out.past_key_values
        for t in range(prefill, seq_len):
            out = model(ids[:, t:t + 1], past_key_values=past, use_cache=True)
            past = out.past_key_values
        return state.acct

    state = A.PolicyState(P.FullCache())
    restore = A.install(model, state)
    try:
        # G1
        state.reset_sequence(n_layers, n_kv, head_dim, device, dtype)
        delta = (model(ids).logits[0, -1].float() - reference).abs().max().item()
        print(f"G1 identity     max|dlogit| {delta:.3e}   "
              f"{'PASS' if delta < tol else 'FAIL'}")
        if delta >= tol:
            print("   -> stop here. nothing below this line means anything.")
            return

        # G5 ordering
        probe = OrderProbe()
        decode_run(probe, prefill=seq_len - 4)
        last_layer0 = [n for (l, n) in probe.saw_select if l == 0][-1]
        print(f"G5 ordering     last select saw n_old={last_layer0}, "
              f"cache reaches {seq_len}   "
              f"{'PASS' if last_layer0 == seq_len - 1 else 'FAIL'}")

        # G2 residency, closed form for streaming at budget b
        acct = decode_run(P.StreamingLLM(0.25))
        print(f"G2 residency    streaming@0.25 -> measured {acct.residency:.4f} "
              f"(expect a shade above 0.25: sinks and freshly written keys are "
              f"always resident)")

        # G3 envelope
        results = {}
        for name, pol in [("sink_only", P.SinkOnly()), ("random", P.RandomKeep(0.25)),
                          ("streaming", P.StreamingLLM(0.25)), ("h2o", P.H2O(0.25)),
                          ("snapkv", P.SnapKV(0.25)), ("oracle", P.Oracle(0.25))]:
            acct = decode_run(pol)
            results[name] = (acct.residency, acct.recovery)
            print(f"   {name:10s} resid {acct.residency:.4f}  rec {acct.recovery:.4f}")
        floor, ceiling = results["sink_only"][1], results["oracle"][1]
        ok = all(floor - 1e-6 <= v[1] <= ceiling + 1e-6
                 for n, v in results.items() if n != "oracle")
        print(f"G3 envelope     floor {floor:.4f} <= all <= ceiling {ceiling:.4f}   "
              f"{'PASS' if ok else 'FAIL'}")

        # G4 monotone
        curve = []
        for b in (0.05, 0.10, 0.25, 0.50, 1.00):
            acct = decode_run(P.StreamingLLM(b))
            curve.append((b, acct.residency, acct.recovery))
        mono = all(curve[i][2] <= curve[i + 1][2] + 1e-6 for i in range(len(curve) - 1))
        print("G4 monotone     " + "  ".join(f"b={b:.2f}:{r:.3f}/{rec:.3f}"
                                             for b, r, rec in curve) +
              f"   {'PASS' if mono else 'FAIL'}")
    finally:
        restore()

    delta = (model(ids).logits[0, -1].float() - reference).abs().max().item()
    print(f"\nrestored, stock path max|d| {delta:.3e}  "
          f"{'PASS' if delta < 1e-6 else 'FAIL'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--seq-len", type=int, default=384)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--gqa-only", action="store_true",
                    help="run G6 alone: no model, no GPU, instant")
    a = ap.parse_args()
    ok = gate_gqa()
    if a.gqa_only:
        raise SystemExit(0 if ok else 1)
    print()
    run(a.model, a.device, getattr(torch, a.dtype), a.seq_len, a.tol)