#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"; source .venv/bin/activate
export HF_HOME="$PWD/hf_cache" PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
echo "=== START $(date) ==="
python run_ppl.py --model Qwen/Qwen3-0.6B --sessions chat_turns_test50.jsonl \
  --policies full,oracle,cluster_oracle --budgets 0.1433 --ks 8,2 \
  --index fit_all28.npz --max-sessions 10 --max-len 2048 --min-turns 6 \
  --device mps --dtype float32 --decode-chunk 1 \
  --out results/ppl_gates.csv
echo "=== EXIT $? at $(date) ==="
