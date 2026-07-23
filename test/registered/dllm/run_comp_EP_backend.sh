#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Comparison matrix. These are the only settings normally worth editing.
DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,5,6,7}
CSV=${SGLANG_DLLM_EP_CSV:-llada2_deepep_vs_none_full_matrix.csv}
MODEL_SIZE=${SGLANG_DLLM_MODEL_SIZE:-mini}
BS_LIST=${SGLANG_DLLM_EP_BS_LIST:-"1 4 8 16 32 64 128 256"}
TP_LIST=${SGLANG_DLLM_EP_TP_LIST:-"4 2 1"}
EP_LIST=${SGLANG_DLLM_EP_SIZE_LIST:-"4 2 1"}
BACKEND_LIST=${SGLANG_DLLM_EP_BACKEND_LIST:-"deepep none"}

export ASCEND_RT_VISIBLE_DEVICES=${DEVICES}

echo "===== LLaDA2 EP backend comparison ====="
echo "devices=${DEVICES} model=${MODEL_SIZE}"
echo "bs=[${BS_LIST}] tp=[${TP_LIST}] ep=[${EP_LIST}]"
echo "backends=[${BACKEND_LIST}] csv=${CSV}"

exec env \
  SGLANG_DLLM_EP_BS_LIST="${BS_LIST}" \
  SGLANG_DLLM_EP_TP_LIST="${TP_LIST}" \
  SGLANG_DLLM_EP_SIZE_LIST="${EP_LIST}" \
  SGLANG_DLLM_EP_BACKEND_LIST="${BACKEND_LIST}" \
  bash "${SCRIPT_DIR}/run_llada2_ascend_gsm8k_EP_test_csv.sh" \
    "${CSV}" "${MODEL_SIZE}"
