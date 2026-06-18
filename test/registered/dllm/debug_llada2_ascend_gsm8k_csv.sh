#!/usr/bin/env bash
set -euo pipefail

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
CSV=${1:-"$SCRIPT_DIR/debug_llada2_mini_gsm8k_results.csv"}

case "$CSV" in
  /*) ;;
  *) CSV="$SCRIPT_DIR/$CSV" ;;
esac

mkdir -p "$(dirname "$CSV")"
cd "$SCRIPT_DIR"
: > "$CSV"
export PYTHONPATH="$REPO_ROOT/python:${PYTHONPATH:-}"
echo "Run dir: $SCRIPT_DIR"
echo "CSV result: $CSV"

# only warmup, not for final result, so we use a small config to save time
env \
  SGLANG_DLLM_CSV="$CSV" \
  SGLANG_DLLM_RUN_NAME="warmup" \
  SGLANG_DLLM_ALGORITHM_CONFIG="$SCRIPT_DIR/joint_threshold.yaml" \
  SGLANG_DLLM_BS=1 \
  SGLANG_DLLM_TP=1 \
  SGLANG_DLLM_EP=1 \
  SGLANG_DLLM_DP=1 \
  SGLANG_DLLM_MOE_A2A_BACKEND=none \
  python3 test_llada2_ascend_gsm8k_bf16.py

echo "Done. CSV result: $CSV"
