#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export HF_HOME="$PWD/hf_cache"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
mkdir -p results
echo "=== START $(date) ==="
python run_ppl.py --model Qwen/Qwen3-0.6B \
  --sessions chat_turns_test50.jsonl \
  --policies full,h2o,snapkv --budgets 0.125,0.06,0.03 \
  --max-sessions 10 --max-len 2048 --min-turns 6 \
  --device mps --dtype float32 --decode-chunk 1 \
  --out results/ppl_tok_fixed.csv
echo "=== EXIT $? at $(date) ==="
