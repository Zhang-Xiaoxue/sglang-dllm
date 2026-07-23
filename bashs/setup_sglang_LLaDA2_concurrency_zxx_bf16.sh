#!/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

#export ASCEND_LAUNCH_BLOCKING=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export AUTOFUSE_FLAGS="--enable_autofuse=true;--autofuse_enable_pass=reduce,concat"


#export ASCEND_SLOG_PRINT_TO_STDOUT=1
#export ASCEND_GLOBAL_LOG_LEVEL=3
#export ASCEND_PROCESS_LOG_PATH=./tmp/ascend_log
## 可选：把 runtime 异步错误尽量提前暴露（不同版本变量名可能不同）
#export TORCH_NPU_LOG_LEVEL=INFO

#export SGLANG_NPU_GRAPH_DEBUG=1

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
        --max-running-requests 1 \
        --dllm-algorithm "JointThreshold" \
	--dllm-algorithm-config /data/home/z84301856/proj_sglang/sglang_dllm_own_0706/sglang-dllm/test/registered/dllm/joint_threshold.yaml \
        --enable-tokenizer-batch-encode \
	--skip-server-warmup \
        --enable-cache-report \
        --tp 1 \
        --disable-radix-cache

        # --dllm-algorithm "LowConfidence" \
	#  --dllm-algorithm-config dllm_config_zxx.yaml \

        # --disable-cuda-graph \
        # --model-loader-extra-config '{"weights_path":"/workspace/sglang/sglang_zxx/cann-recipes-infer/models/llada/quant/w8a8c16_layer_all_mlp_false","int8_dequant":true,"int8_scale_suffix":"_scale"}'

        # --model-path /workspace/sglang/sglang_zxx/cann-recipes-infer/models/llada/quant/w8a8c16_quantALL \

# LLADA2.1-flash
# python -m sglang.launch_server \
#         --model-path /data/public_models/LLaDA/LLaDA2.1-flash \
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
