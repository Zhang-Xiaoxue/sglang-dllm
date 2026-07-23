#!/usr/bin/env bash
set -euo pipefail

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} #

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
MODEL_SIZE=${SGLANG_DLLM_MODEL_SIZE:-${2:-mini}}
EVAL_NAMES=${SGLANG_DLLM_EVAL_NAMES:-${3:-gpqa,piqa}}
CSV=${1:-"$SCRIPT_DIR/llada2_${MODEL_SIZE}_gpqa_piqa.csv"}

if [[ "$MODEL_SIZE" != "mini" && "$MODEL_SIZE" != "flash" ]]; then
  echo "MODEL_SIZE must be mini or flash, got: $MODEL_SIZE" >&2
  exit 1
fi

case "$CSV" in
  /*) ;;
  *) CSV="$SCRIPT_DIR/$CSV" ;;
esac

mkdir -p "$(dirname "$CSV")"
cd "$SCRIPT_DIR"
: > "$CSV"
export PYTHONPATH="$REPO_ROOT/python:${PYTHONPATH:-}"

echo "Run dir: $SCRIPT_DIR"
echo "Model size: $MODEL_SIZE"
echo "Eval names: $EVAL_NAMES"
echo "CSV result: $CSV"

run_case() {
  local precision=$1
  local bs=$2
  local tp=$3
  local ep=$4
  local dp=$5
  local moe_a2a_backend=${6:-none}
  local run_name=${7:-"llada2_${MODEL_SIZE}_${precision}_eval_bs${bs}_tp${tp}_ep${ep}_dp${dp}"}

  if [[ "$MODEL_SIZE" == "flash" && "$tp" -lt 4 ]]; then
    echo "Skip invalid flash case: tp must be >= 4, got tp=$tp for ${run_name}" >&2
    return 0
  fi

  if [[ "$moe_a2a_backend" != "none" ]]; then
    run_name="${run_name}_${moe_a2a_backend}"
  fi

  echo
  echo "===== ${run_name} -> test_llada2_ascend_gpqa_piqa.py ====="
  env \
    SGLANG_DLLM_CSV="$CSV" \
    SGLANG_DLLM_RUN_NAME="$run_name" \
    SGLANG_DLLM_MODEL_SIZE="$MODEL_SIZE" \
    SGLANG_DLLM_PRECISION="$precision" \
    SGLANG_DLLM_EVAL_NAMES="$EVAL_NAMES" \
    SGLANG_DLLM_ALGORITHM_CONFIG="$SCRIPT_DIR/joint_threshold.yaml" \
    SGLANG_DLLM_BS="$bs" \
    SGLANG_DLLM_TP="$tp" \
    SGLANG_DLLM_EP="$ep" \
    SGLANG_DLLM_DP="$dp" \
    SGLANG_DLLM_MOE_A2A_BACKEND="$moe_a2a_backend" \
    python3 test_llada2_ascend_gpqa_piqa.py
}

# Accuracy evals are slower than throughput probes. Keep the default matrix small
# and add more rows when you need a full tp/bs sweep.
BF16_CASES=(
  "1 1 1 1 none"
  "4 1 1 1 none"
  "8 1 1 1 none"
  "1 4 1 1 none"
  "4 4 1 1 none"
  "8 4 1 1 none"
  # "16 4 1 1 none"
  # "32 4 1 1 none"
  # "1 8 1 1 none"
  # "8 8 1 1 none"
  # "16 8 1 1 none"
  # "32 8 1 1 none"
)

INT8_CASES=(
  "1 1 1 1 none"
  "4 1 1 1 none"
  "8 1 1 1 none"
  "1 4 1 1 none"
  "4 4 1 1 none"
  "8 4 1 1 none"
  # "16 4 1 1 none"
  # "32 4 1 1 none"
  # "1 8 1 1 none"
  # "8 8 1 1 none"
  # "16 8 1 1 none"
  # "32 8 1 1 none"
)

for case_cfg in "${BF16_CASES[@]}"; do
  read -r bs tp ep dp moe_a2a_backend <<< "$case_cfg"
  run_case bf16 "$bs" "$tp" "$ep" "$dp" "$moe_a2a_backend"
done

for case_cfg in "${INT8_CASES[@]}"; do
  read -r bs tp ep dp moe_a2a_backend <<< "$case_cfg"
  run_case int8 "$bs" "$tp" "$ep" "$dp" "$moe_a2a_backend"
done

echo
echo "Done. CSV result: $CSV"
