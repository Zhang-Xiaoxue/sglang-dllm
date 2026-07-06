#!/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=1,2,3,5

# #export ASCEND_LAUNCH_BLOCKING=1
# export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
# export SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1
# unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
# export AUTOFUSE_FLAGS="--enable_autofuse=true;--autofuse_enable_pass=reduce,concat"

export HCCL_BUFFSIZE=1024
# export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
# export SGLANG_DEEPEP_BF16_DISPATCH=1
# export SGLANG_DEBUG_GRAPH_CAN_RUN=1

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
        --disable-radix-cache \
        --dllm-algorithm "JointThreshold" \
        --dllm-algorithm-config /data/home/z84301856/proj_sglang/sglang_dllm_yuan/sglang-dllm/test/registered/dllm/joint_threshold.yaml
        # --cuda-graph-max-bs 4 \
        # --enforce-piecewise-cuda-graph \
        # --piecewise-cuda-graph-tokens 32 64 96 128 \

        # --disable-cuda-graph
        # --dllm-algorithm "LowConfidence" \
        # --dllm-algorithm-config "/home/ma-user/work/z84301856/sglang-dllm/bashs/dllm_config_zxx.yaml" \        】

        # --moe-a2a-backend none or deepep \
        # --disable-cuda-graph \
        # --enforce-piecewise-cuda-graph \
        # --piecewise-cuda-graph-compiler eager \

        

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
#         --dllm-algorithm-config /home/ma-user/work/z84301856/sglang-dllm/test/registered/dllm/joint_threshold.yaml \
#         --enable-tokenizer-batch-encode \
# 	--skip-server-warmup \
#         --enable-cache-report \
#         --tp 4 \
#         --disable-radix-cache 