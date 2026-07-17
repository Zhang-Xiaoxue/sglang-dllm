#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
status=0

read -r -a MODEL_SIZES <<< "${SGLANG_DLLM_ALL_MODEL_SIZES:-${SGLANG_DLLM_MODEL_SIZE:-mini}}"
BS_LIST=${SGLANG_DLLM_EP_BS_LIST:-"256"}
TP_LIST=${SGLANG_DLLM_EP_TP_LIST:-"1"}
EP_LIST=${SGLANG_DLLM_EP_SIZE_LIST:-"1"}
PROFILE_BS_LIST=${SGLANG_DLLM_PROFILE_BS_LIST:-$BS_LIST}
PROFILE_TP_LIST=${SGLANG_DLLM_PROFILE_TP_LIST:-$TP_LIST}
PROFILE_EP_LIST=${SGLANG_DLLM_PROFILE_EP_SIZE_LIST:-$EP_LIST}
INCLUDE_ORIGINAL=${SGLANG_DLLM_RUN_ALL_INCLUDE_ORIGINAL:-1}

echo "===== LLaDA2 none+EP all cases ====="
echo "Model sizes: ${MODEL_SIZES[*]}"
echo "BS list: $BS_LIST"
echo "TP list: $TP_LIST"
echo "EP list: $EP_LIST"
echo "Modes: gsm8k_fixed (graph), kernel_profile (graph), original (default graph), gsm8k_variable (eager)"
echo "Profile warmup runs: ${SGLANG_DLLM_PROFILE_WARMUP_RUNS:-1}"
echo "Include original GSM8K: $INCLUDE_ORIGINAL"

run_stage() {
  local stage=$1
  local model_size=$2
  shift 2

  echo
  echo "===== Start stage: ${stage}, model_size=${model_size} ====="
  if env \
      SGLANG_DLLM_MODEL_SIZE="$model_size" \
      SGLANG_DLLM_EP_BS_LIST="$BS_LIST" \
      SGLANG_DLLM_EP_TP_LIST="$TP_LIST" \
      SGLANG_DLLM_EP_SIZE_LIST="$EP_LIST" \
      SGLANG_DLLM_PROFILE_BS_LIST="$PROFILE_BS_LIST" \
      SGLANG_DLLM_PROFILE_TP_LIST="$PROFILE_TP_LIST" \
      SGLANG_DLLM_PROFILE_EP_SIZE_LIST="$PROFILE_EP_LIST" \
      SGLANG_DLLM_PROFILE_WARMUP_RUNS="${SGLANG_DLLM_PROFILE_WARMUP_RUNS:-1}" \
      "$@" \
      bash "$SCRIPT_DIR/run_comp_EP_backend.sh"; then
    echo "===== PASS stage: ${stage}, model_size=${model_size} ====="
  else
    local rc=$?
    status=$rc
    echo "===== FAIL stage: ${stage}, model_size=${model_size} exit=${rc} =====" >&2
  fi
}

for model_size in "${MODEL_SIZES[@]}"; do
  echo
  echo "===== Start all cases: model_size=$model_size ====="

  run_stage "gsm8k_fixed (exact-BS decode graph)" "$model_size" \
    SGLANG_DLLM_RUN_VARIABLE_INPUT_BENCHMARK=0 \
    SGLANG_DLLM_RUN_FIXED_INPUT_BENCHMARK=1 \
    SGLANG_DLLM_RUN_KERNEL_PROFILE=0 \
    SGLANG_DLLM_RUN_ORIGINAL_TESTS=0

  run_stage "kernel_profile (exact-BS decode graph)" "$model_size" \
    SGLANG_DLLM_RUN_VARIABLE_INPUT_BENCHMARK=0 \
    SGLANG_DLLM_RUN_FIXED_INPUT_BENCHMARK=0 \
    SGLANG_DLLM_RUN_KERNEL_PROFILE=1 \
    SGLANG_DLLM_RUN_ORIGINAL_TESTS=0

  if [[ "$INCLUDE_ORIGINAL" == "1" ]]; then
    run_stage "original (default graph policy)" "$model_size" \
      SGLANG_DLLM_RUN_VARIABLE_INPUT_BENCHMARK=0 \
      SGLANG_DLLM_RUN_FIXED_INPUT_BENCHMARK=0 \
      SGLANG_DLLM_RUN_KERNEL_PROFILE=0 \
      SGLANG_DLLM_RUN_ORIGINAL_TESTS=1
  else
    echo "===== Skip stage: original, model_size=${model_size} ====="
  fi

  run_stage "gsm8k_variable (graph disabled)" "$model_size" \
    SGLANG_DLLM_RUN_VARIABLE_INPUT_BENCHMARK=1 \
    SGLANG_DLLM_RUN_FIXED_INPUT_BENCHMARK=0 \
    SGLANG_DLLM_RUN_KERNEL_PROFILE=0 \
    SGLANG_DLLM_RUN_ORIGINAL_TESTS=0
done

exit "$status"
