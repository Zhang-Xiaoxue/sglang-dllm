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

# Defaults keep the matrix useful but not enormous. Override with env lists as needed.
read -r -a BS_LIST <<< "${SGLANG_DLLM_EP_BS_LIST:-1 16}"
read -r -a TP_LIST <<< "${SGLANG_DLLM_EP_TP_LIST:-1 4}"
read -r -a EP_LIST <<< "${SGLANG_DLLM_EP_SIZE_LIST:-1 2 4}"
read -r -a BACKEND_LIST <<< "${SGLANG_DLLM_EP_BACKEND_LIST:-none deepep}"

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
  local max_running_requests=${SGLANG_DLLM_MAX_RUNNING_REQUESTS:-${SGLANG_DLLM_EP_MAX_RUNNING_REQUESTS:-$bs}}
  local run_name="llada2_${MODEL_SIZE}_bf16_ep_bs${bs}_tp${tp}_ep${ep}_dp${dp}_${moe_a2a_backend}"

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
      python3 test_llada2_ascend_gsm8k_ep_bf16.py "${TEST_ARGS[@]}"; then
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
