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

################################### cann-recipe quantization ############################################################
# ------------------------------ LLaDA2.1 Mini -------------------------------------
# python -m sglang.launch_server \
#         --model-path /data/home/z84301856/proj_sglang/models/LLaDA/llada2.1-mini-cannrecipe-w8a8c16-moe \
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
#         --tp 2 \
#         --ep 2 \
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
        --model-path /data/home/z84301856/proj_sglang/models/LLaDA/llada2.1-mini-modelslim-w8a8c16-moe \
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
        --quantization modelslim \
        # --disable-cuda-graph
        
        # --dllm-algorithm "LowConfidence" \
	# --dllm-algorithm-config dllm_config_zxx.yaml \