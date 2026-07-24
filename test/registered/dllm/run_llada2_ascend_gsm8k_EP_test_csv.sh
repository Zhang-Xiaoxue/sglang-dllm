#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

MODEL_SIZE=${2:-${SGLANG_DLLM_MODEL_SIZE:-mini}}
CSV=${1:-"${SCRIPT_DIR}/llada2_${MODEL_SIZE}_ep_backend.csv"}
BS_LIST=${SGLANG_DLLM_EP_BS_LIST:-"1 4 8 16 32 64 128 256"}
TP_LIST=${SGLANG_DLLM_EP_TP_LIST:-2}
EP_LIST=${SGLANG_DLLM_EP_SIZE_LIST:-2}
BACKEND_LIST=${SGLANG_DLLM_EP_BACKEND_LIST:-"none deepep"}
DP_SIZE=${SGLANG_DLLM_EP_DP_SIZE:-1}
GSM8K_NUM_QUESTIONS=${SGLANG_DLLM_GSM8K_NUM_QUESTIONS:-512}
DEEPEP_LOW_LATENCY_MAX_BS=${SGLANG_DLLM_DEEPEP_LOW_LATENCY_MAX_BS:-32}
TEST_SELECTOR=${SGLANG_DLLM_TESTS:-}
TEST_MODE=original
[[ -n "${TEST_SELECTOR}" ]] && TEST_MODE="selector:${TEST_SELECTOR}"

if [[ "${MODEL_SIZE}" != "mini" && "${MODEL_SIZE}" != "flash" ]]; then
  echo "MODEL_SIZE must be mini or flash, got: ${MODEL_SIZE}" >&2
  exit 2
fi

if [[ "${CSV}" != /* ]]; then
  CSV="${SCRIPT_DIR}/${CSV}"
fi
mkdir -p "$(dirname "${CSV}")"
: > "${CSV}"

export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"
if [[ -z "${SGLANG_DLLM_GSM8K_DATA_PATH:-}" && -f /tmp/test.jsonl ]]; then
  export SGLANG_DLLM_GSM8K_DATA_PATH=/tmp/test.jsonl
fi

count_visible_devices() {
  local count=0
  local device
  IFS=',' read -r -a devices <<< "${ASCEND_RT_VISIBLE_DEVICES:-}"
  for device in "${devices[@]}"; do
    [[ -n "${device//[[:space:]]/}" ]] && ((count += 1))
  done
  echo "${count}"
}

resolve_mem_fraction_static() {
  local bs=$1

  if [[ -n "${SGLANG_DLLM_MEM_FRACTION_STATIC:-}" ]]; then
    echo "${SGLANG_DLLM_MEM_FRACTION_STATIC}"
  elif (( bs <= 8 )); then
    echo "0.80"
  elif (( bs <= 16 )); then
    echo "0.70"
  elif (( bs <= 32 )); then
    echo "0.60"
  elif (( bs <= 64 )); then
    echo "0.50"
  else
    echo "0.40"
  fi
}

resolve_deepep_mode() {
  local bs=$1

  if [[ -n "${SGLANG_DLLM_DEEPEP_MODE:-}" ]]; then
    echo "${SGLANG_DLLM_DEEPEP_MODE}"
  elif (( bs <= DEEPEP_LOW_LATENCY_MAX_BS )); then
    echo "low_latency"
  else
    echo "normal"
  fi
}

VISIBLE_DEVICE_COUNT=$(count_visible_devices)
TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0
FAILED_CASES=()

echo "===== EP backend matrix ====="
echo "model=${MODEL_SIZE} devices=${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
echo "bs=[${BS_LIST}] tp=[${TP_LIST}] ep=[${EP_LIST}] backend=[${BACKEND_LIST}]"
echo "rules: none=(ep<=tp and tp%ep==0), deepep=(ep==tp and ep>1, mode=low_latency if bs<=${DEEPEP_LOW_LATENCY_MAX_BS} else normal)"
echo "mode=${TEST_MODE}"
echo "questions=${GSM8K_NUM_QUESTIONS}"
echo "mem_fraction_static=${SGLANG_DLLM_MEM_FRACTION_STATIC:-<auto-by-bs>}"
echo "deepep_mode=${SGLANG_DLLM_DEEPEP_MODE:-<auto-by-bs>}"
echo "csv=${CSV}"

run_case() {
  local bs=$1
  local tp=$2
  local ep=$3
  local backend=$4
  local deepep_mode=""
  local mem_fraction_static
  mem_fraction_static=$(resolve_mem_fraction_static "${bs}")
  local name="llada2_${MODEL_SIZE}_bf16_ep_bs${bs}_tp${tp}_ep${ep}_dp${DP_SIZE}_${backend}"

  if (( tp > VISIBLE_DEVICE_COUNT )); then
    echo "SKIP ${name}: tp=${tp}, visible devices=${VISIBLE_DEVICE_COUNT}" >&2
    ((SKIPPED += 1))
    return
  fi
  if [[ "${backend}" != "none" && "${backend}" != "deepep" ]]; then
    echo "SKIP ${name}: unsupported backend=${backend}" >&2
    ((SKIPPED += 1))
    return
  fi
  if [[ "${backend}" == "none" ]] && (( ep <= 0 || ep > tp || tp % ep != 0 )); then
    echo "SKIP ${name}: backend=none requires ep<=tp and tp%ep==0" >&2
    ((SKIPPED += 1))
    return
  fi
  if [[ "${backend}" == "deepep" ]] && (( ep != tp || ep <= 1 )); then
    echo "SKIP ${name}: backend=deepep requires ep==tp and ep>1" >&2
    ((SKIPPED += 1))
    return
  fi
  if [[ "${MODEL_SIZE}" == "flash" && ${tp} -lt 4 ]]; then
    echo "SKIP ${name}: flash requires tp>=4" >&2
    ((SKIPPED += 1))
    return
  fi
  if [[ "${backend}" == "deepep" ]]; then
    deepep_mode=$(resolve_deepep_mode "${bs}")
    name="${name}_${deepep_mode}"
  fi
  local test_args=()
  [[ -n "${TEST_SELECTOR}" ]] && test_args+=("${TEST_SELECTOR}")

  echo
  echo "===== RUN ${name} questions=${GSM8K_NUM_QUESTIONS} mem_fraction_static=${mem_fraction_static} deepep_mode=${deepep_mode:-none} ====="
  ((TOTAL += 1))
  if env \
      SGLANG_DLLM_CSV="${CSV}" \
      SGLANG_DLLM_RUN_NAME="${name}" \
      SGLANG_DLLM_MODEL_SIZE="${MODEL_SIZE}" \
      SGLANG_DLLM_ALGORITHM_CONFIG="${SCRIPT_DIR}/joint_threshold.yaml" \
      SGLANG_DLLM_BS="${bs}" \
      SGLANG_DLLM_TP="${tp}" \
      SGLANG_DLLM_EP="${ep}" \
      SGLANG_DLLM_DP="${DP_SIZE}" \
      SGLANG_DLLM_MOE_A2A_BACKEND="${backend}" \
      SGLANG_DLLM_DEEPEP_MODE="${deepep_mode}" \
      SGLANG_DLLM_GSM8K_NUM_QUESTIONS="${GSM8K_NUM_QUESTIONS}" \
      SGLANG_DLLM_MEM_FRACTION_STATIC="${mem_fraction_static}" \
      python3 "${SCRIPT_DIR}/test_llada2_ascend_gsm8k_ep_bf16.py" "${test_args[@]}"; then
    echo "===== PASS ${name} ====="
    ((PASSED += 1))
  else
    local rc=$?
    echo "===== FAIL ${name}: exit=${rc}; continuing =====" >&2
    ((FAILED += 1))
    FAILED_CASES+=("${name} (exit=${rc})")
  fi

}

for bs in ${BS_LIST}; do
  for tp in ${TP_LIST}; do
    for ep in ${EP_LIST}; do
      for backend in ${BACKEND_LIST}; do
        run_case "${bs}" "${tp}" "${ep}" "${backend}"
      done
    done
  done
done

echo
echo "Done: total=${TOTAL} passed=${PASSED} failed=${FAILED} skipped=${SKIPPED}"
echo "CSV: ${CSV}"
if (( FAILED > 0 )); then
  printf '  failed: %s\n' "${FAILED_CASES[@]}" >&2
  exit 1
fi
