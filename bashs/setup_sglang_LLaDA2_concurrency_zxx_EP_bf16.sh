#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LOG_DIR="${SGLANG_LOG_DIR:-${REPO_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${SGLANG_LAUNCH_LOG_FILE:-${LOG_DIR}/llada2_ep_bf16_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "${LOG_FILE}")"
export SGLANG_TEE_LOG=${SGLANG_TEE_LOG:-1}
if [[ "${SGLANG_TEE_LOG}" == "1" ]]; then
    exec > >(tee -a "${LOG_FILE}") 2>&1
    echo "[setup_sglang] logging to ${LOG_FILE} (tee enabled)"
else
    echo "[setup_sglang] logging to ${LOG_FILE}"
    exec >> "${LOG_FILE}" 2>&1
    echo "[setup_sglang] logging to ${LOG_FILE}"
fi

export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,5,6,7}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1024}
export SGLANG_NPU_DLLM_DEEPEP_PREFILL_GRAPH=0
export SGLANG_NPU_PIECEWISE_STATIC_INPUT_COPY=${SGLANG_NPU_PIECEWISE_STATIC_INPUT_COPY:-1}

MODEL_PATH=${SGLANG_MODEL_PATH:-/data/public_models/LLaDA/LLaDA2.1-mini}
SERVED_MODEL_NAME=${SGLANG_SERVED_MODEL_NAME:-LLaDA2.1-mini}
PORT=${SGLANG_PORT:-8001}
TP_SIZE=${SGLANG_TP_SIZE:-4}
EP_SIZE=${SGLANG_EP_SIZE:-${TP_SIZE}}
MOE_DP_SIZE=${SGLANG_MOE_DP_SIZE:-1}
MOE_A2A_BACKEND=${SGLANG_MOE_A2A_BACKEND:-deepep}
MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-4}
ENABLE_DP_ATTENTION=${SGLANG_ENABLE_DP_ATTENTION:-1}
MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.90}
DEEPEP_MODE=${SGLANG_DEEPEP_MODE:-low_latency}
# Keep BF16 model weights/experts while reducing DeepEP dispatch traffic.
# Set SGLANG_DEEPEP_DISPATCH_DTYPE=auto to restore BF16 communication.
DEEPEP_DISPATCH_DTYPE=${SGLANG_DEEPEP_DISPATCH_DTYPE:-int8}
read -r -a CUDA_GRAPH_BS_DECODE <<< "${SGLANG_CUDA_GRAPH_BS_DECODE:-1 2 4}"

DP_ATTN_ARGS=()
TOKENIZER_ARGS=(--enable-tokenizer-batch-encode)
if [[ "${ENABLE_DP_ATTENTION}" == "1" ]]; then
    DP_SIZE=${SGLANG_DP_SIZE:-${TP_SIZE}}
    DP_ATTN_ARGS=(--enable-dp-attention --enable-dp-lm-head)
    TOKENIZER_ARGS=()
else
    DP_SIZE=${SGLANG_DP_SIZE:-1}
fi

DEEPEP_ARGS=()
if [[ "${MOE_A2A_BACKEND}" == "deepep" ]]; then
    DEEPEP_ARGS=(
        --deepep-mode "${DEEPEP_MODE}"
        --deepep-dispatcher-output-dtype "${DEEPEP_DISPATCH_DTYPE}"
    )
fi

IFS=',' read -r -a VISIBLE_DEVICES <<< "${ASCEND_RT_VISIBLE_DEVICES}"
VISIBLE_DEVICE_COUNT=0
for VISIBLE_DEVICE in "${VISIBLE_DEVICES[@]}"; do
    VISIBLE_DEVICE=${VISIBLE_DEVICE//[[:space:]]/}
    [[ -n "${VISIBLE_DEVICE}" ]] && ((VISIBLE_DEVICE_COUNT += 1))
done
if (( VISIBLE_DEVICE_COUNT < TP_SIZE )); then
    echo "[setup_sglang] TP=${TP_SIZE}, but only ${VISIBLE_DEVICE_COUNT} NPU(s) are visible" >&2
    exit 2
fi
if (( TP_SIZE < 1 || EP_SIZE < 1 || DP_SIZE < 1 )); then
    echo "[setup_sglang] TP, EP, and DP sizes must be positive" >&2
    exit 2
fi
if (( EP_SIZE > TP_SIZE || TP_SIZE % EP_SIZE != 0 )); then
    echo "[setup_sglang] EP=${EP_SIZE} must divide TP=${TP_SIZE}" >&2
    exit 2
fi
if [[ "${MOE_A2A_BACKEND}" == "deepep" && "${EP_SIZE}" != "${TP_SIZE}" ]]; then
    echo "[setup_sglang] DeepEP requires EP_SIZE == TP_SIZE in this codebase" >&2
    exit 2
fi
if [[ "${ENABLE_DP_ATTENTION}" == "1" && "${DP_SIZE}" != "${TP_SIZE}" ]]; then
    echo "[setup_sglang] LLaDA2 DP attention currently requires DP_SIZE == TP_SIZE" >&2
    exit 2
fi

MAX_GRAPH_BS=0
for GRAPH_BS in "${CUDA_GRAPH_BS_DECODE[@]}"; do
    (( GRAPH_BS > MAX_GRAPH_BS )) && MAX_GRAPH_BS=${GRAPH_BS}
done
MAX_GLOBAL_BS=${MAX_RUNNING_REQUESTS}
(( MAX_GRAPH_BS > MAX_GLOBAL_BS )) && MAX_GLOBAL_BS=${MAX_GRAPH_BS}
MAX_LOCAL_BS=${MAX_GLOBAL_BS}
if [[ "${ENABLE_DP_ATTENTION}" == "1" ]]; then
    MAX_LOCAL_BS=$(((MAX_GLOBAL_BS + DP_SIZE - 1) / DP_SIZE))
fi
MIN_DISPATCH_TOKENS=$((MAX_LOCAL_BS * 32))
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-${MIN_DISPATCH_TOKENS}}
if [[ "${MOE_A2A_BACKEND}" == "deepep" ]] && \
   (( SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK < MIN_DISPATCH_TOKENS )); then
    echo "[setup_sglang] DeepEP dispatch capacity must be at least ${MIN_DISPATCH_TOKENS} for max requests and graph buckets" >&2
    exit 2
fi

if [[ "${SGLANG_KEEP_DEBUG_ENVS:-0}" != "1" ]]; then
    unset SGLANG_NPU_DEEPEP_DEBUG_GRAPH_LOG
    unset SGLANG_NPU_DEEPEP_EAGER_POST_MOE_GRAPH
    unset SGLANG_NPU_PIECEWISE_EAGER_GRAPH
    unset SGLANG_NPU_PIECEWISE_EAGER_FROM_GRAPH
    unset SGLANG_NPU_PIECEWISE_EAGER_LAST_GRAPH
    unset SGLANG_NPU_PIECEWISE_SYNC_REPLAY
    unset SGLANG_NPU_DEEPEP_DISABLE_MOE_SPLIT
fi

echo "[setup_sglang] model=${MODEL_PATH} devices=${ASCEND_RT_VISIBLE_DEVICES}"
echo "[setup_sglang] TP=${TP_SIZE} EP=${EP_SIZE} DP=${DP_SIZE} MOE_DP=${MOE_DP_SIZE} backend=${MOE_A2A_BACKEND}"
echo "[setup_sglang] dp_attention=${ENABLE_DP_ATTENTION} max_running_requests=${MAX_RUNNING_REQUESTS} mem_fraction_static=${MEM_FRACTION_STATIC}"
echo "[setup_sglang] decode_graph=full decode_bs=${CUDA_GRAPH_BS_DECODE[*]} prefill_graph=disabled"
if [[ "${MOE_A2A_BACKEND}" == "deepep" ]]; then
    echo "[setup_sglang] deepep_mode=${DEEPEP_MODE} dispatch_dtype=${DEEPEP_DISPATCH_DTYPE} max_dispatch_tokens_per_rank=${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK}"
fi

python -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
	--served-model-name "${SERVED_MODEL_NAME}" \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --device npu \
        --attention-backend ascend \
        --dtype bfloat16 \
        --kv-cache-dtype auto \
	--trust-remote-code \
	--mem-fraction-static "${MEM_FRACTION_STATIC}" \
        --max-running-requests "${MAX_RUNNING_REQUESTS}" \
        --random-seed 0 \
	"${TOKENIZER_ARGS[@]}" \
	--skip-server-warmup \
        --enable-cache-report \
        --tp "${TP_SIZE}" \
        --ep "${EP_SIZE}" \
        --dp-size "${DP_SIZE}" \
        --moe-dp-size "${MOE_DP_SIZE}" \
        --moe-a2a-backend "${MOE_A2A_BACKEND}" \
        "${DEEPEP_ARGS[@]}" \
        "${DP_ATTN_ARGS[@]}" \
        --cuda-graph-backend-decode full \
        --cuda-graph-bs-decode "${CUDA_GRAPH_BS_DECODE[@]}" \
        --cuda-graph-backend-prefill disabled \
        --disable-radix-cache \
        --dllm-algorithm "JointThreshold" \
	--dllm-algorithm-config "${REPO_ROOT}/test/registered/dllm/joint_threshold.yaml"


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
