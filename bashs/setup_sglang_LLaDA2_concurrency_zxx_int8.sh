#!/bin/bash

# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

#export ASCEND_LAUNCH_BLOCKING=1
# export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
# export SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1
# unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
# export AUTOFUSE_FLAGS="--enable_autofuse=true;--autofuse_enable_pass=reduce,concat"


#export ASCEND_SLOG_PRINT_TO_STDOUT=1
#export ASCEND_GLOBAL_LOG_LEVEL=3
#export ASCEND_PROCESS_LOG_PATH=./tmp/ascend_log
## 可选：把 runtime 异步错误尽量提前暴露（不同版本变量名可能不同）
#export TORCH_NPU_LOG_LEVEL=INFO

#export SGLANG_NPU_GRAPH_DEBUG=1

# quant: llada2p1_w8a8c16_quant_SparseMoe_final , llada2p1_flash_quant_sparseMoe_final

MODEL_ROOT=${LLADA_MODEL_ROOT:-/data/home/z84301856/proj_sglang/models/LLaDA}

# 默认保留原先脚本里的较短量化模型；需要切换时传 static/static-more 或完整模型目录。
DEFAULT_QUANT_MODEL=${LLADA_DEFAULT_QUANT_MODEL:-${MODEL_ROOT}/llada2.1-mini-modelslim-w8a8c16-moe-dynamic-act3}
STATIC_QUANT_MODEL=${LLADA_STATIC_QUANT_MODEL:-${MODEL_ROOT}/llada2.1-mini-modelslim-w8a8c16-moe-static}
STATIC_MORE_CALIB_QUANT_MODEL=${LLADA_STATIC_MORE_CALIB_QUANT_MODEL:-${MODEL_ROOT}/llada2.1-mini-modelslim-w8a8c16-moe-static-moreCalib}
QUANT_MODEL=${LLADA_QUANT_MODEL_PATH:-${LLADA_QUANT_MODEL:-${DEFAULT_QUANT_MODEL}}}

print_usage() {
    echo "Usage: bash $0 [default|short|dynamic|static|static-more|/path/to/quant_model]"
    echo "       bash $0 --quant-model-path /path/to/quant_model"
    echo "       LLADA_QUANT_MODEL=static bash $0"
}

resolve_quant_model() {
    case "$1" in
        default|short|dynamic|dynamic-act3)
            echo "${DEFAULT_QUANT_MODEL}"
            ;;
        static)
            echo "${STATIC_QUANT_MODEL}"
            ;;
        static-more|static-moreCalib|static-morecalib|moreCalib|morecalib|more)
            echo "${STATIC_MORE_CALIB_QUANT_MODEL}"
            ;;
        *)
            echo "$1"
            ;;
    esac
}

if [[ $# -gt 0 ]]; then
    case "$1" in
        -h|--help)
            print_usage
            exit 0
            ;;
        --quant-model|--quant-model-path|--model-path)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] Missing value for $1"
                print_usage
                exit 1
            fi
            QUANT_MODEL=$2
            shift 2
            ;;
        *)
            QUANT_MODEL=$1
            shift
            ;;
    esac
fi

if [[ $# -gt 0 ]]; then
    echo "[ERROR] Unknown extra arguments: $*"
    print_usage
    exit 1
fi

QUANT_MODEL=$(resolve_quant_model "${QUANT_MODEL}")
if [[ ! -f "${QUANT_MODEL}/config.json" ]]; then
    echo "[ERROR] Quant model path is not ready: ${QUANT_MODEL}"
    echo "        Expected config.json under this model directory."
    print_usage
    exit 1
fi

echo "[INFO] Using quant model: ${QUANT_MODEL}"


################################### cann-recipe quantization ############################################################
# # ------------------------------ LLaDA2.1 Mini -------------------------------------
# python -m sglang.launch_server \
#         --model-path /data/home/z84301856/proj_sglang/models/LLaDA/llada2.1-mini-cannrecipe-w8a8c16-moe-attn-v1 \
# 	--served-model-name LLaDA2.1-mini \
#         --host 0.0.0.0 \
#         --port 8000 \
#         --device npu \
#         --attention-backend ascend \
#         --dtype bfloat16 \
#         --kv-cache-dtype auto \
# 	--trust-remote-code \
# 	--disable-radix-cache \
#         --mem-fraction-static 0.90 \
#         --max-running-requests 1 \
#         --dllm-algorithm "JointThreshold" \
#         --dllm-algorithm-config /data/home/z84301856/proj_sglang/sglang_dllm_yuan/sglang-dllm/test/registered/dllm/joint_threshold.yaml \
#         --enable-tokenizer-batch-encode \
# 	--skip-server-warmup \
#         --enable-cache-report \
#         --tp 1 \
#         --ep 1 \
#         --quantization compressed-tensors \
#         # --disable-cuda-graph

#         # --dllm-algorithm "LowConfidence" \
# 	# --dllm-algorithm-config dllm_config_zxx.yaml \

# ------------------------------ LLaDA2.1 Flash -------------------------------------
# python -m sglang.launch_server \
#         --model-path /data/home/z84301856/proj_sglang/models/LLaDA/llada2.1-flash-cannrecipe-w8a8c16-moe \
# 	--served-model-name LLaDA2.1-flash \
#         --host 0.0.0.0 \
#         --port 8000 \
#         --device npu \
#         --attention-backend ascend \
#         --dtype bfloat16 \
#         --kv-cache-dtype auto \
# 	--trust-remote-code \
# 	--disable-radix-cache \
#         --mem-fraction-static 0.90 \
#         --max-running-requests 1 \
#         --dllm-algorithm "JointThreshold" \
#         --dllm-algorithm-config /data/home/z84301856/proj_sglang/sglang_dllm_yuan/sglang-dllm/test/registered/dllm/joint_threshold.yaml \
#         --enable-tokenizer-batch-encode \
# 	--skip-server-warmup \
#         --enable-cache-report \
#         --tp 4 \
#         --quantization compressed-tensors \
#         # --disable-cuda-graph

#         # --dllm-algorithm "LowConfidence" \
# 	# --dllm-algorithm-config dllm_config_zxx.yaml \

##################################### modelslim quantization #####################################
# ------------------------------ LLaDA2.1 Mini -------------------------------------
python -m sglang.launch_server \
        --model-path "${QUANT_MODEL}" \
	--served-model-name LLaDA2.1-mini \
        --host 0.0.0.0 \
        --port 8000 \
        --device npu \
        --attention-backend ascend \
        --dtype bfloat16 \
        --kv-cache-dtype auto \
	--trust-remote-code \
	--disable-radix-cache \
        --mem-fraction-static 0.90 \
        --max-running-requests 1 \
        --dllm-algorithm "JointThreshold" \
        --dllm-algorithm-config /data/home/z84301856/proj_sglang/sglang_dllm_yuan/sglang-dllm/test/registered/dllm/joint_threshold.yaml \
        --enable-tokenizer-batch-encode \
	--skip-server-warmup \
        --enable-cache-report \
        --tp 1 \
        --ep 1 \
        --quantization modelslim
        # --disable-cuda-graph

        # --dllm-algorithm "LowConfidence" \
	# --dllm-algorithm-config dllm_config_zxx.yaml \
