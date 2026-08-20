import json, argparse, os, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--n", type=int, default=250)
ap.add_argument("--max-len", type=int, default=2048)
ap.add_argument("--min-spans", type=int, default=7)
ap.add_argument("--scan", type=int, default=80000)
ap.add_argument("--source", default="wildchat", choices=["wildchat","ultrachat"])
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
if a.source == "wildchat":
    ds, field = load_dataset("allenai/WildChat-1M", split="train", streaming=True), "conversation"
else:
    ds, field = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft", streaming=True), "messages"

need = 2 * a.min_spans
kept, lens, seen, cheap = [], [], 0, 0
for ex in ds:
    seen += 1
    if seen > a.scan: break
    conv = ex.get(field)
    if not conv: continue
    msgs = [{"role": m["role"], "content": m["content"]} for m in conv
            if m.get("role") in ("user","assistant") and m.get("content")]
    if len(msgs) < need or msgs[0]["role"] != "user": continue
    cheap += 1
    n = len(tok.apply_chat_template(msgs[:need], tokenize=True))
    if n > a.max_len: continue
    kept.append({"messages": msgs}); lens.append(n)
    if len(kept) >= a.n: break

assert kept, "FATAL: zero sessions kept"
with open(a.out, "w") as f:
    for r in kept: f.write(json.dumps(r) + "\n")
L = np.array(lens)
print(f"scanned {seen}, >={need} msgs: {cheap}, kept {len(kept)} -> {a.out}")
print(f"tokens in first {a.min_spans} spans: p10 {np.percentile(L,10):.0f} "
      f"p50 {np.percentile(L,50):.0f} max {L.max()}")
os._exit(0)
