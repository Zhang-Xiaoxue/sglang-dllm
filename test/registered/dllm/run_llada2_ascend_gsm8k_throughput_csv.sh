#!/usr/bin/env bash
set -euo pipefail

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} #

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
MODEL_SIZE=${SGLANG_DLLM_MODEL_SIZE:-${2:-mini}}
CSV=${1:-"$SCRIPT_DIR/llada2_${MODEL_SIZE}_gsm8k_bf16_int8.csv"}

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
echo "CSV result: $CSV"

run_case() {
  local test_file=$1
  local precision=$2
  local bs=$3
  local tp=$4
  local ep=$5
  local dp=$6
  local moe_a2a_backend=${7:-none}
  local run_name=${8:-"llada2_${MODEL_SIZE}_${precision}_bs${bs}_tp${tp}_ep${ep}_dp${dp}"}

  if [[ "$MODEL_SIZE" == "flash" && "$tp" -lt 4 ]]; then
    echo "Skip invalid flash case: tp must be >= 4, got tp=$tp for ${run_name}" >&2
    return 0
  fi

  if [[ "$moe_a2a_backend" != "none" ]]; then
    run_name="${run_name}_${moe_a2a_backend}"
  fi

  echo
  echo "===== ${run_name} -> ${test_file} ====="
  env \
    SGLANG_DLLM_CSV="$CSV" \
    SGLANG_DLLM_RUN_NAME="$run_name" \
    SGLANG_DLLM_MODEL_SIZE="$MODEL_SIZE" \
    SGLANG_DLLM_ALGORITHM_CONFIG="$SCRIPT_DIR/joint_threshold.yaml" \
    SGLANG_DLLM_BS="$bs" \
    SGLANG_DLLM_TP="$tp" \
    SGLANG_DLLM_EP="$ep" \
    SGLANG_DLLM_DP="$dp" \
    SGLANG_DLLM_MOE_A2A_BACKEND="$moe_a2a_backend" \
    python3 "$test_file"
}

# Warmup, not for final comparison. It still writes a row named warmup_* to the CSV.
run_case test_llada2_ascend_gsm8k_bf16.py bf16 1 4 1 1 none "warmup_${MODEL_SIZE}_bf16"

# # ----------------------------------- bf16 -------------------------------------------
# Format: "bs tp ep dp moe_a2a_backend".
BF16_CASES=(
  "1 1 1 1 none"
  "4 1 1 1 none"
  "8 1 1 1 none"
  "1 4 1 1 none"
  "4 4 1 1 none"
  "8 4 1 1 none"
  "16 4 1 1 none"
  "32 4 1 1 none"
  # "1 8 1 1 none"
  # "4 8 1 1 none"
  # "8 8 1 1 none"
  # "16 8 1 1 none"
  # "32 8 1 1 none"
  # "64 8 1 1 none"  OOM for flash, and mini with some configs, so skip by default. Can be enabled when needed.
)

for case_cfg in "${BF16_CASES[@]}"; do
  read -r bs tp ep dp moe_a2a_backend <<< "$case_cfg"
  run_case test_llada2_ascend_gsm8k_bf16.py bf16 "$bs" "$tp" "$ep" "$dp" "$moe_a2a_backend"
done

# ----------------------------------- int8 -------------------------------------------
# Warmup for int8, not for final comparison. It still writes a row named warmup_* to the CSV.
run_case test_llada2_ascend_gsm8k_int8.py int8 1 4 1 1 none "warmup_${MODEL_SIZE}_int8"

# Format: "bs tp ep dp moe_a2a_backend". Keep this aligned with BF16_CASES by default.
INT8_CASES=(
  "1 1 1 1 none"
  "4 1 1 1 none"
  "8 1 1 1 none"
  "1 4 1 1 none"
  "4 4 1 1 none"
  "8 4 1 1 none"
  "16 4 1 1 none"
  "32 4 1 1 none"
  # "1 8 1 1 none"
  # "4 8 1 1 none"
  # "8 8 1 1 none"
  # "16 8 1 1 none"
  # "32 8 1 1 none"
  # "64 8 1 1 none"  OOM for flash, and mini with some configs, so skip by default. Can be enabled when needed.
)

for case_cfg in "${INT8_CASES[@]}"; do
  read -r bs tp ep dp moe_a2a_backend <<< "$case_cfg"
  run_case test_llada2_ascend_gsm8k_int8.py int8 "$bs" "$tp" "$ep" "$dp" "$moe_a2a_backend"
done

# ----------------------------------- EP example -------------------------------------
# Uncomment the loop below when needed.
# EP_BF16_CASES=(
#   "4 4 4 1 deepep"
# )
# for case_cfg in "${EP_BF16_CASES[@]}"; do
#   read -r bs tp ep dp moe_a2a_backend <<< "$case_cfg"
#   run_case test_llada2_ascend_gsm8k_ep_bf16.py bf16_ep "$bs" "$tp" "$ep" "$dp" "$moe_a2a_backend"
# done

echo

echo "Done. CSV result: $CSV"
