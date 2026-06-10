# MODEL_PATH=/data/public_models/LLaDA/LLaDA2.1-mini
MODEL_PATH=/data/public_models/LLaDA/LLaDA2.1-flash

python3 -m sglang.launch_server \
    --model-path=${MODEL_PATH} \
    --host 127.0.0.1 \
    --port 8188 \
    --device npu \
    --attention-backend ascend \
    --dtype bfloat16 \
    --kv-cache-dtype auto \
    --dllm-algorithm JointThreshold \
    --tp-size 8 \
    --max-running-requests 4 \
    --enable-tokenizer-batch-encode \
    --trust-remote-code \
    --disable-radix-cache \
    --disable-overlap-schedule \
    --mem-fraction-static 0.8 \
    --cuda-graph-bs 1 2 3 4