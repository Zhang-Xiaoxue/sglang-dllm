#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LOG_DIR="${SGLANG_LOG_DIR:-${REPO_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${SGLANG_LAUNCH_LOG_FILE:-${LOG_DIR}/llada2_ep_bf16_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "${LOG_FILE}")"
if [[ "${SGLANG_TEE_LOG:-0}" == "1" ]]; then
    exec > >(tee -a "${LOG_FILE}") 2>&1
    echo "[setup_sglang] logging to ${LOG_FILE} (tee enabled)"
else
    echo "[setup_sglang] logging to ${LOG_FILE}"
    exec >> "${LOG_FILE}" 2>&1
    echo "[setup_sglang] logging to ${LOG_FILE}"
fi

export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"
export SGLANG_TEE_LOG=1 # 在屏幕上也打印log
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-1,2,3,5}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1024}
export SGLANG_NPU_DLLM_DEEPEP_PREFILL_GRAPH=${SGLANG_NPU_DLLM_DEEPEP_PREFILL_GRAPH:-1}
export SGLANG_NPU_PIECEWISE_STATIC_INPUT_COPY=${SGLANG_NPU_PIECEWISE_STATIC_INPUT_COPY:-1}

if [[ "${SGLANG_KEEP_DEBUG_ENVS:-0}" != "1" ]]; then
    unset SGLANG_NPU_DEEPEP_DEBUG_GRAPH_LOG
    unset SGLANG_NPU_DEEPEP_EAGER_POST_MOE_GRAPH
    unset SGLANG_NPU_PIECEWISE_EAGER_GRAPH
    unset SGLANG_NPU_PIECEWISE_EAGER_FROM_GRAPH
    unset SGLANG_NPU_PIECEWISE_EAGER_LAST_GRAPH
    unset SGLANG_NPU_PIECEWISE_SYNC_REPLAY
    unset SGLANG_NPU_DEEPEP_DISABLE_MOE_SPLIT
fi

python -m sglang.launch_server \
        --model-path /data/public_models/LLaDA/LLaDA2.1-mini \
	--served-model-name LLaDA2.1-mini \
        --host 0.0.0.0 \
        --port 8001 \
        --device npu \
        --attention-backend ascend \
        --dtype bfloat16 \
        --kv-cache-dtype auto \
	--trust-remote-code \
	--mem-fraction-static 0.90 \
        --max-running-requests 4 \
        --enable-tokenizer-batch-encode \
	--skip-server-warmup \
        --enable-cache-report \
        --tp 4 \
        --ep 4 \
        --dp-size 1 \
        --moe-dp-size 1 \
        --moe-a2a-backend deepep \
        --deepep-mode auto \
        --cuda-graph-backend-prefill tc_piecewise \
        --cuda-graph-bs-prefill 32 \
        --cuda-graph-tc-compiler eager \
        --disable-radix-cache \
        --dllm-algorithm "JointThreshold" \
	--dllm-algorithm-config /data/home/z84301856/proj_sglang/sglang_dllm_own_0706/sglang-dllm/test/registered/dllm/joint_threshold.yaml


# # LLADA2.1-flash
# python -m sglang.launch_server \
#         --model-path /home/ma-user/work/z84301856/models/LLaDA2.1-flash \
# 	--served-model-name LLaDA2.1-flash \
#         --host 0.0.0.0 \
#         --port 8001 \
#         --device npu \
#         --attention-backend ascend \
#         --dtype bfloat16 \
#         --kv-cache-dtype auto \
# 	--trust-remote-code \
# 	--mem-fraction-static 0.90 \
#         --max-running-requests 1 \
#         --dllm-algorithm "JointThreshold" \
#         --dllm-algorithm-config /data/home/z84301856/proj_sglang/sglang_dllm_own_0706/sglang-dllm/test/registered/dllm/joint_threshold.yaml \
#         --enable-tokenizer-batch-encode \
# 	--skip-server-warmup \
#         --enable-cache-report \
#         --tp 4 \
#         --disable-radix-cache 