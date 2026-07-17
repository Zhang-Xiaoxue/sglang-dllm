#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
MODEL_SIZE=${SGLANG_DLLM_MODEL_SIZE:-${2:-mini}}
CSV=${1:-"$SCRIPT_DIR/llada2_${MODEL_SIZE}_gsm8k_EP_test.csv"}

count_visible_devices() {
  local devices=${ASCEND_RT_VISIBLE_DEVICES:-}
  local count=0
  local device

  IFS=',' read -r -a visible_devices <<< "$devices"
  for device in "${visible_devices[@]}"; do
    device=${device//[[:space:]]/}
    if [[ -n "$device" ]]; then
      ((count += 1))
    fi
  done
  echo "$count"
}

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

if [[ "$MODEL_SIZE" == "flash" ]]; then
  FLASH_MIN_VISIBLE_DEVICES=${SGLANG_DLLM_FLASH_MIN_VISIBLE_DEVICES:-4}
  VISIBLE_DEVICE_COUNT=$(count_visible_devices)
  if (( VISIBLE_DEVICE_COUNT < FLASH_MIN_VISIBLE_DEVICES )); then
    echo "Skip flash matrix: ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-<unset>} exposes ${VISIBLE_DEVICE_COUNT} device(s), but flash requires >=${FLASH_MIN_VISIBLE_DEVICES}."
    echo "CSV result: $CSV"
    exit 0
  fi
fi

if [[ -z "${SGLANG_DLLM_GSM8K_DATA_PATH:-}" && -f /tmp/test.jsonl ]]; then
  export SGLANG_DLLM_GSM8K_DATA_PATH=/tmp/test.jsonl
fi

# Optional unittest selector, for example:
#   SGLANG_DLLM_TESTS="TestLLaDA2.test_bs_speed" bash run_llada2_ascend_gsm8k_EP_test_csv.sh
TESTS=${SGLANG_DLLM_TESTS:-}
TEST_ARGS=()
if [[ -n "$TESTS" ]]; then
  read -r -a TEST_ARGS <<< "$TESTS"
fi

WORKLOAD_MODE=${SGLANG_DLLM_WORKLOAD_MODE:-}
IS_BENCHMARK=0
if [[ -n "$WORKLOAD_MODE" && "$WORKLOAD_MODE" != "original" ]]; then
  IS_BENCHMARK=1
elif [[ "${SGLANG_DLLM_FIXED_WORKLOAD:-0}" == "1" ]]; then
  IS_BENCHMARK=1
fi
LOG_DIR=${SGLANG_DLLM_LOG_DIR:-"$SCRIPT_DIR/logs/llada2_ep"}
mkdir -p "$LOG_DIR"
PROFILE_GRAPH=${SGLANG_DLLM_PROFILE_GRAPH:-0}
PROFILE_ROOT=${SGLANG_DLLM_PROFILE_DIR:-"$SCRIPT_DIR/profiles/llada2_ep"}
PROFILE_STEPS=${SGLANG_DLLM_PROFILE_STEPS:-5}
PROFILE_OUTPUT_TOKENS=${SGLANG_DLLM_PROFILE_OUTPUT_TOKENS:-$(((PROFILE_STEPS + 1) * ${SGLANG_DLLM_BLOCK_SIZE:-32}))}

# Defaults keep the matrix useful but not enormous. Override with env lists as needed.
read -r -a BS_LIST <<< "${SGLANG_DLLM_EP_BS_LIST:-1 16}"
read -r -a TP_LIST <<< "${SGLANG_DLLM_EP_TP_LIST:-1 4}"
read -r -a EP_LIST <<< "${SGLANG_DLLM_EP_SIZE_LIST:-1 2 4}"
if [[ "$IS_BENCHMARK" == "1" ]]; then
  DEFAULT_BACKEND_LIST="none"
else
  DEFAULT_BACKEND_LIST="none deepep"
fi
read -r -a BACKEND_LIST <<< "${SGLANG_DLLM_EP_BACKEND_LIST:-$DEFAULT_BACKEND_LIST}"

DP_SIZE=${SGLANG_DLLM_EP_DP_SIZE:-1}
FAIL_FAST=${SGLANG_DLLM_EP_FAIL_FAST:-0}

TOTAL_CASES=0
PASSED_CASES=0
FAILED_CASE_COUNT=0
SKIPPED_CASES=0
FAILED_CASES=()

# Keep deepep graph capture startup bounded by default. Override when testing larger prefill buckets.
export SGLANG_DLLM_CUDA_GRAPH_BS_PREFILL=${SGLANG_DLLM_CUDA_GRAPH_BS_PREFILL:-32,64,128,256,512}

echo "Run dir: $SCRIPT_DIR"
echo "Model size: $MODEL_SIZE"
echo "CSV result: $CSV"
echo "GSM8K data: ${SGLANG_DLLM_GSM8K_DATA_PATH:-<download>}"
echo "BS list: ${BS_LIST[*]}"
echo "TP list: ${TP_LIST[*]}"
echo "EP list: ${EP_LIST[*]}"
echo "Backends: ${BACKEND_LIST[*]}"
echo "Workload mode: ${WORKLOAD_MODE:-original}"
if [[ "$IS_BENCHMARK" == "1" ]]; then
  echo "Fixed output tokens/request: ${SGLANG_DLLM_BS_OUTPUT_LEN:-512}"
  if [[ "$WORKLOAD_MODE" == "gsm8k_fixed" || "$WORKLOAD_MODE" == "synthetic_fixed" ]]; then
    echo "Fixed input tokens/request: ${SGLANG_DLLM_BS_INPUT_LEN:-32}"
  else
    echo "Input tokens/request: variable"
  fi
  if [[ "$WORKLOAD_MODE" == "gsm8k_fixed" || "$WORKLOAD_MODE" == "synthetic_fixed" ]]; then
    echo "Graph policy: decode=current BS only, prefill=disabled, padding=disabled"
  else
    echo "Graph policy: disabled for variable-input end-to-end benchmark"
  fi
  echo "Kernel profiling: $PROFILE_GRAPH ($PROFILE_STEPS steps)"
  if [[ "$PROFILE_GRAPH" == "1" ]]; then
    echo "Profiler output tokens/request: $PROFILE_OUTPUT_TOKENS"
    echo "Profiler root: $PROFILE_ROOT/<setting>/<timestamp>"
  fi
fi
echo "Case logs: $LOG_DIR"
echo "Note: deepep cases with ep != tp are skipped because server_args forces ep_size=tp_size."
if [[ -n "$TESTS" ]]; then
  echo "Test selector: $TESTS"
fi

run_case() {
  local bs=$1
  local tp=$2
  local ep=$3
  local dp=$4
  local moe_a2a_backend=$5
  local max_running_requests
  if [[ "$IS_BENCHMARK" == "1" ]]; then
    max_running_requests=$bs
  else
    max_running_requests=${SGLANG_DLLM_MAX_RUNNING_REQUESTS:-${SGLANG_DLLM_EP_MAX_RUNNING_REQUESTS:-$bs}}
  fi
  local run_name="llada2_${MODEL_SIZE}_bf16_ep_bs${bs}_tp${tp}_ep${ep}_dp${dp}_${moe_a2a_backend}"
  if [[ -n "$WORKLOAD_MODE" && "$WORKLOAD_MODE" != "original" ]]; then
    run_name="${run_name}_${WORKLOAD_MODE}"
  fi
  local log_path="$LOG_DIR/${run_name}.log"
  local profile_case_dir="$PROFILE_ROOT"
  if [[ "$PROFILE_GRAPH" == "1" ]]; then
    local profile_shape="in${SGLANG_DLLM_BS_INPUT_LEN:-variable}_profileout${PROFILE_OUTPUT_TOKENS}_block${SGLANG_DLLM_BLOCK_SIZE:-32}_steps${PROFILE_STEPS}"
    profile_case_dir="${PROFILE_ROOT%/}/${run_name}_${profile_shape}"
  fi

  if [[ "$MODEL_SIZE" == "flash" && "$tp" -lt 4 ]]; then
    echo "Skip invalid flash case: tp must be >= 4, got tp=$tp for ${run_name}" >&2
    ((SKIPPED_CASES += 1))
    return 0
  fi

  if (( ep > tp )); then
    echo "Skip invalid EP case: ep=$ep > tp=$tp for ${run_name}" >&2
    ((SKIPPED_CASES += 1))
    return 0
  fi

  if [[ "$moe_a2a_backend" == "deepep" ]] && (( ep != tp )); then
    echo "Skip non-equivalent DeepEP case: requested ep=$ep, tp=$tp for ${run_name}; DeepEP forces ep_size=tp_size" >&2
    ((SKIPPED_CASES += 1))
    return 0
  fi

  echo
  echo "===== ${run_name} -> test_llada2_ascend_gsm8k_ep_bf16.py ====="
  if [[ "$PROFILE_GRAPH" == "1" ]]; then
    echo "Profiler setting directory: $profile_case_dir"
  fi
  ((TOTAL_CASES += 1))
  if env \
      SGLANG_DLLM_CSV="$CSV" \
      SGLANG_DLLM_RUN_NAME="$run_name" \
      SGLANG_DLLM_MODEL_SIZE="$MODEL_SIZE" \
      SGLANG_DLLM_ALGORITHM_CONFIG="$SCRIPT_DIR/joint_threshold.yaml" \
      SGLANG_DLLM_BS="$bs" \
      SGLANG_DLLM_TP="$tp" \
      SGLANG_DLLM_EP="$ep" \
      SGLANG_DLLM_DP="$dp" \
      SGLANG_DLLM_MAX_RUNNING_REQUESTS="$max_running_requests" \
      SGLANG_DLLM_MOE_A2A_BACKEND="$moe_a2a_backend" \
      SGLANG_DLLM_PROFILE_DIR="$profile_case_dir" \
      SGLANG_DLLM_PROFILE_STEPS="$PROFILE_STEPS" \
      SGLANG_DLLM_PROFILE_OUTPUT_TOKENS="$PROFILE_OUTPUT_TOKENS" \
      python3 test_llada2_ascend_gsm8k_ep_bf16.py "${TEST_ARGS[@]}" \
      2>&1 | tee "$log_path"; then
    ((PASSED_CASES += 1))
    echo "===== PASS: ${run_name} ====="
  else
    local status=$?
    ((FAILED_CASE_COUNT += 1))
    FAILED_CASES+=("${run_name} (exit=${status})")
    echo "===== FAIL: ${run_name} (exit=${status}); continuing =====" >&2
    if [[ "$FAIL_FAST" == "1" ]]; then
      return "$status"
    fi
  fi
}

if [[ -n "${SGLANG_DLLM_EP_CASES:-}" ]]; then
  IFS=';' read -r -a EP_CASES <<< "$SGLANG_DLLM_EP_CASES"
  for case_cfg in "${EP_CASES[@]}"; do
    read -r bs tp ep dp moe_a2a_backend <<< "$case_cfg"
    if [[ -z "${moe_a2a_backend:-}" ]]; then
      echo "Invalid EP case: '$case_cfg'. Expected: bs tp ep dp moe_a2a_backend" >&2
      exit 1
    fi
    run_case "$bs" "$tp" "$ep" "$dp" "$moe_a2a_backend"
  done
else
  for bs in "${BS_LIST[@]}"; do
    for tp in "${TP_LIST[@]}"; do
      for ep in "${EP_LIST[@]}"; do
        for moe_a2a_backend in "${BACKEND_LIST[@]}"; do
          run_case "$bs" "$tp" "$ep" "$DP_SIZE" "$moe_a2a_backend"
        done
      done
    done
  done
fi

echo
echo "Done. CSV result: $CSV"
echo "Case summary: total=$TOTAL_CASES passed=$PASSED_CASES failed=$FAILED_CASE_COUNT skipped=$SKIPPED_CASES"
if (( FAILED_CASE_COUNT > 0 )); then
  echo "Failed cases:" >&2
  printf '  - %s\n' "${FAILED_CASES[@]}" >&2
  exit 1
fi
