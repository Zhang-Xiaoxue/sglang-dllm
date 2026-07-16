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

# run_matrix llada2_gsm8k_ep_test_mini_0716_zxx.csv mini \
#   env \
#     SGLANG_DLLM_EP_BS_LIST="256" \
#     SGLANG_DLLM_EP_TP_LIST="8 4 2 1" \
#     SGLANG_DLLM_EP_SIZE_LIST="8 4 2 1" \
#     SGLANG_DLLM_EP_BACKEND_LIST="none" \
#     SGLANG_DLLM_MEM_FRACTION_STATIC="0.7" \
#     SGLANG_DLLM_BS_MAX_NEW_TOKENS="512" \
#     bash "$SCRIPT_DIR/run_llada2_ascend_gsm8k_EP_test_csv.sh" llada2_gsm8k_ep_test_mini_0716_zxx.csv mini

run_matrix llada2_gsm8k_ep_test_flash_0716_zxx_none.csv flash \
 env \
    SGLANG_DLLM_EP_BS_LIST="1 4 8 16 32 64 128 256" \
    SGLANG_DLLM_EP_TP_LIST="8 4 2 1" \
    SGLANG_DLLM_EP_SIZE_LIST="8 4 2 1" \
    SGLANG_DLLM_EP_BACKEND_LIST="none" \
    SGLANG_DLLM_MEM_FRACTION_STATIC="0.8" \
    SGLANG_DLLM_BS_MAX_NEW_TOKENS="512" \
   bash "$SCRIPT_DIR/run_llada2_ascend_gsm8k_EP_test_csv.sh" llada2_gsm8k_ep_test_flash_0716_zxx_none.csv flash

exit "$status"
