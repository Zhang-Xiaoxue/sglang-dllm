#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
status=0

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"

run_matrix() {
  local csv=$1
  local model_size=$2
  shift 2

  echo
  echo "===== EP backend matrix: model_size=${model_size}, csv=${csv} ====="
  if "$@"; then
    echo "===== PASS matrix: model_size=${model_size} ====="
  else
    local rc=$?
    echo "===== FAIL matrix: model_size=${model_size} exit=${rc}; continuing =====" >&2
    status=$rc
  fi
}

MODEL_SIZE=${SGLANG_DLLM_MODEL_SIZE:-mini}
BS_LIST=${SGLANG_DLLM_EP_BS_LIST:-"1 4 8 16 32 64 128 256 512"}
TP_LIST=${SGLANG_DLLM_EP_TP_LIST:-"8 4 2 1"}
EP_LIST=${SGLANG_DLLM_EP_SIZE_LIST:-"8 4 2 1"}
MEM_FRACTION=${SGLANG_DLLM_MEM_FRACTION_STATIC:-0.6}
OUTPUT_LEN=${SGLANG_DLLM_BS_OUTPUT_LEN:-512}
BLOCK_SIZE=${SGLANG_DLLM_BLOCK_SIZE:-32}
WARMUP_RUNS=${SGLANG_DLLM_BS_WARMUP_RUNS:-1}
RUN_KERNEL_PROFILE=${SGLANG_DLLM_RUN_KERNEL_PROFILE:-0}
PROFILE_STEPS=${SGLANG_DLLM_PROFILE_STEPS:-5}
PROFILE_OUTPUT_LEN=${SGLANG_DLLM_PROFILE_OUTPUT_TOKENS:-$(((PROFILE_STEPS + 1) * BLOCK_SIZE))}
PROFILE_WARMUP_RUNS=${SGLANG_DLLM_PROFILE_WARMUP_RUNS:-1}

if [[ "$RUN_KERNEL_PROFILE" == "1" ]]; then
  default_run_benchmarks=0
else
  default_run_benchmarks=1
fi

# Default run: performance-only fixed-output benchmarks. No GSM8K accuracy test.
if [[ "${SGLANG_DLLM_RUN_VARIABLE_INPUT_BENCHMARK:-$default_run_benchmarks}" == "1" ]]; then
  variable_csv=${SGLANG_DLLM_VARIABLE_INPUT_CSV:-"llada2_${MODEL_SIZE}_ep_none_gsm8k_variable_fixed512.csv"}
  run_matrix "$variable_csv" "$MODEL_SIZE" \
    env \
      SGLANG_DLLM_EP_BS_LIST="$BS_LIST" \
      SGLANG_DLLM_EP_TP_LIST="$TP_LIST" \
      SGLANG_DLLM_EP_SIZE_LIST="$EP_LIST" \
      SGLANG_DLLM_EP_BACKEND_LIST="none" \
      SGLANG_DLLM_MEM_FRACTION_STATIC="$MEM_FRACTION" \
      SGLANG_DLLM_WORKLOAD_MODE="gsm8k_variable" \
      SGLANG_DLLM_BS_OUTPUT_LEN="$OUTPUT_LEN" \
      SGLANG_DLLM_BLOCK_SIZE="$BLOCK_SIZE" \
      SGLANG_DLLM_BS_WARMUP_RUNS="$WARMUP_RUNS" \
      SGLANG_DLLM_PROFILE_GRAPH="0" \
      SGLANG_DLLM_LOG_DIR="$SCRIPT_DIR/logs/llada2_ep_none_gsm8k_variable" \
      SGLANG_DLLM_TESTS="TestLLaDA2.test_bs_speed" \
      bash "$SCRIPT_DIR/run_llada2_ascend_gsm8k_EP_test_csv.sh" "$variable_csv" "$MODEL_SIZE"
fi

if [[ "${SGLANG_DLLM_RUN_FIXED_INPUT_BENCHMARK:-$default_run_benchmarks}" == "1" ]]; then
  fixed_csv=${SGLANG_DLLM_FIXED_INPUT_CSV:-"llada2_${MODEL_SIZE}_ep_none_gsm8k_fixed32x512.csv"}
  run_matrix "$fixed_csv" "$MODEL_SIZE" \
    env \
      SGLANG_DLLM_EP_BS_LIST="$BS_LIST" \
      SGLANG_DLLM_EP_TP_LIST="$TP_LIST" \
      SGLANG_DLLM_EP_SIZE_LIST="$EP_LIST" \
      SGLANG_DLLM_EP_BACKEND_LIST="none" \
      SGLANG_DLLM_MEM_FRACTION_STATIC="$MEM_FRACTION" \
      SGLANG_DLLM_WORKLOAD_MODE="gsm8k_fixed" \
      SGLANG_DLLM_BS_INPUT_LEN="${SGLANG_DLLM_BS_INPUT_LEN:-32}" \
      SGLANG_DLLM_BS_OUTPUT_LEN="$OUTPUT_LEN" \
      SGLANG_DLLM_BLOCK_SIZE="$BLOCK_SIZE" \
      SGLANG_DLLM_BS_WARMUP_RUNS="$WARMUP_RUNS" \
      SGLANG_DLLM_PROFILE_GRAPH="0" \
      SGLANG_DLLM_LOG_DIR="$SCRIPT_DIR/logs/llada2_ep_none_gsm8k_fixed" \
      SGLANG_DLLM_TESTS="TestLLaDA2.test_bs_speed" \
      bash "$SCRIPT_DIR/run_llada2_ascend_gsm8k_EP_test_csv.sh" "$fixed_csv" "$MODEL_SIZE"
fi

# Opt-in kernel profiling. Each setting sends one short profiled batch, waits
# for completed trace artifacts, and then advances to the next setting.
if [[ "$RUN_KERNEL_PROFILE" == "1" ]]; then
  profile_csv=${SGLANG_DLLM_KERNEL_PROFILE_CSV:-"llada2_${MODEL_SIZE}_ep_none_gsm8k_fixed32_profile${PROFILE_OUTPUT_LEN}_steps${PROFILE_STEPS}_kernel.csv"}
  profile_bs_list=${SGLANG_DLLM_PROFILE_BS_LIST:-$BS_LIST}
  profile_tp_list=${SGLANG_DLLM_PROFILE_TP_LIST:-$TP_LIST}
  profile_ep_list=${SGLANG_DLLM_PROFILE_EP_SIZE_LIST:-$EP_LIST}
  run_matrix "$profile_csv" "$MODEL_SIZE" \
    env \
      SGLANG_DLLM_EP_BS_LIST="$profile_bs_list" \
      SGLANG_DLLM_EP_TP_LIST="$profile_tp_list" \
      SGLANG_DLLM_EP_SIZE_LIST="$profile_ep_list" \
      SGLANG_DLLM_EP_BACKEND_LIST="none" \
      SGLANG_DLLM_MEM_FRACTION_STATIC="$MEM_FRACTION" \
      SGLANG_DLLM_WORKLOAD_MODE="gsm8k_fixed" \
      SGLANG_DLLM_BS_INPUT_LEN="${SGLANG_DLLM_BS_INPUT_LEN:-32}" \
      SGLANG_DLLM_BS_OUTPUT_LEN="$OUTPUT_LEN" \
      SGLANG_DLLM_BLOCK_SIZE="$BLOCK_SIZE" \
      SGLANG_DLLM_BS_WARMUP_RUNS="$PROFILE_WARMUP_RUNS" \
      SGLANG_DLLM_PROFILE_GRAPH="1" \
      SGLANG_DLLM_PROFILE_STEPS="$PROFILE_STEPS" \
      SGLANG_DLLM_PROFILE_OUTPUT_TOKENS="$PROFILE_OUTPUT_LEN" \
      SGLANG_DLLM_PROFILE_DIR="$SCRIPT_DIR/profiles/llada2_ep_none_gsm8k_fixed_kernel" \
      SGLANG_DLLM_LOG_DIR="$SCRIPT_DIR/logs/llada2_ep_none_gsm8k_fixed_kernel" \
      SGLANG_DLLM_TESTS="TestLLaDA2.test_bs_speed" \
      bash "$SCRIPT_DIR/run_llada2_ascend_gsm8k_EP_test_csv.sh" "$profile_csv" "$MODEL_SIZE"
fi

# Explicit opt-in recovery path for the original accuracy + performance tests.
if [[ "${SGLANG_DLLM_RUN_ORIGINAL_TESTS:-0}" == "1" ]]; then
  original_csv=${SGLANG_DLLM_ORIGINAL_CSV:-"llada2_${MODEL_SIZE}_gsm8k_ep_original.csv"}
  run_matrix "$original_csv" "$MODEL_SIZE" \
    env -u SGLANG_DLLM_WORKLOAD_MODE \
      -u SGLANG_DLLM_FIXED_WORKLOAD \
      -u SGLANG_DLLM_TESTS \
      SGLANG_DLLM_EP_BS_LIST="$BS_LIST" \
      SGLANG_DLLM_EP_TP_LIST="$TP_LIST" \
      SGLANG_DLLM_EP_SIZE_LIST="$EP_LIST" \
      SGLANG_DLLM_EP_BACKEND_LIST="none" \
      SGLANG_DLLM_MEM_FRACTION_STATIC="$MEM_FRACTION" \
      SGLANG_DLLM_BS_MAX_NEW_TOKENS="$OUTPUT_LEN" \
      SGLANG_DLLM_GSM8K_NUM_QUESTIONS="${SGLANG_DLLM_GSM8K_NUM_QUESTIONS:-1319}" \
      SGLANG_DLLM_LOG_DIR="$SCRIPT_DIR/logs/llada2_ep_none_original" \
      bash "$SCRIPT_DIR/run_llada2_ascend_gsm8k_EP_test_csv.sh" "$original_csv" "$MODEL_SIZE"
fi

exit "$status"
