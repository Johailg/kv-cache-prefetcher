#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"; source .venv/bin/activate
export HF_HOME="$PWD/hf_cache" PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
echo "=== START $(date) ==="
python run_ppl.py --model Qwen/Qwen3-0.6B --sessions chat_turns_test50.jsonl \
  --policies full,page_oracle --budgets 9,10 \
  --max-sessions 10 --max-len 2048 --min-turns 6 \
  --device mps --dtype float32 --decode-chunk 1 \
  --out results/e2e_page.csv
echo "=== EXIT $? at $(date) ==="
