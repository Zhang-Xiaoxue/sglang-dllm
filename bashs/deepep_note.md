[npu_piecewise_backend.py (line 10)](/data/home/z84301856/proj_sglang/sglang_dllm_own_0706/sglang-dllm/python/sglang/srt/compilation/npu_piecewise_backend.py:10) 引入 get_pcg_capture_stream()，NPU graph 捕获时显式使用 runner 的 capture stream，并启用 auto_dispatch_capture=True。
[llada2.py (line 372)](/data/home/z84301856/proj_sglang/sglang_dllm_own_0706/sglang-dllm/python/sglang/srt/models/llada2.py:372) 补了 NPU dual-stream 分支，避免 capture 模式下误用 CUDA stream API。
上一轮 NPU piecewise graph 的 capture stream 适配：[npu_piecewise_backend.py (line 77)](/data/home/z84301856/proj_sglang/sglang_dllm_own_0706/sglang-dllm/python/sglang/srt/compilation/npu_piecewise_backend.py:77)。

# moe-a2a none
1. 修改python/sglang/srt/layers/quantization/unquant.py. 
    使用layer.num_local_experts计算expert_num
    对 topk_ids < 0 ：topk_ids 临时改成 0，避免 NPU routing 非法索引；topk_weights 置 0，保证该 rank 对非本地 expert 不贡献输出


# moe-a2a deepep
1. set HCCL_BUFFSIZE=1024
2. 修复deepep
[forward_batch_info.py (line 1284)](/data/home/z84301856/proj_sglang/sglang_dllm_own_0706/sglang-dllm/python/sglang/srt/model_executor/forward_batch_info.py:1284)：对 ForwardMode.DLLM_EXTEND 单独处理。按真实 token 数裁剪 full_logits。同步还原 input_ids / positions / out_cache_loc / extend_num_tokens，避免 padding 后 DLLM 循环继续用错 shape。只有 next_token_logits 存在时才裁剪它。 原因：--moe-a2a-backend deepep 会触发 require_mlp_sync()，forward 后会进 post_forward_mlp_sync_batch()；但 LLaDA2 的 DLLM 路径只返回 full_logits，next_token_logits=None。原逻辑按普通 extend 去裁 next_token_logits，所以报：TypeError: 'NoneType' object is not subscriptable
>>> prepare_mlp_sync_batch() 会把本地 batch 扩成全局 token 排布，post_forward_mlp_sync_batch() 再裁回来。现在 DeepEP 路径返回的对象没有 logits，所以需要确认是哪一个模型 forward 分支在这个模式下只返回 hidden states。post_forward_mlp_sync_batch() 的实现确实假设 next_token_logits 一定存在。none+EP 没撞到，deepep+EP 撞到，说明 deepep 触发了 require_mlp_sync/全局 padding，而 LLaDA2 的某个 decode 轮次只产 hidden states。现在我去看 LLaDA2 顶层 forward 和 logits processor 的返回条件。找到了一个关键点：DLLM 的 forward mode 是 DLLM_EXTEND，而 is_extend() 也把它视为 extend。DeepEP 触发 MLP sync 后，post 走了普通 extend 裁剪逻辑，但 DLLM logits 输出结构可能不同。现在具体看 _get_dllm_logits。确认了：DLLM 的 _get_dllm_logits() 只填 full_logits，不会填 next_token_logits。所以这次错误不是 DeepEP kernel 计算失败，而是 deepep 打开 MLP sync 后，通用 post-process 硬裁了一个不存在的字段。需要给 DLLM/full_logits 加对应裁剪逻辑，不能只跳过，否则 padding token 会污染 DLLM 的 block logits。
>>> -在 post_forward_mlp_sync_batch() 里加 DLLM 专用裁剪：对 full_logits 按真实 token 数裁剪；对 next_token_logits 只有存在时才裁剪。这样不会影响普通自回归模型，也能保住 DLLM 用的完整 block logits
>>> -ForwardMode.DLLM_EXTEND 在 MLP sync 后只裁 full_logits/hidden states，并且不再强制索引 next_token_logits=None。这和普通 extend 分开处理，避免影响自回归路径。
2.1. `bashs/setup_sglang_LLaDA2_concurrency_zxx_EP_bf16.sh` 显式打开 prefill tc_piecewise graph：`--cuda-graph-backend-prefill tc_piecewise --cuda-graph-bs-prefill 32 64 96 128 --cuda-graph-tc-compiler eager`。原因：默认兼容性逻辑会因为 NPU / MoE A2A / DLLM 自动禁用 prefill graph；显式设置 backend 会跳过 auto-disable。
2.2. `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`：DLLM 场景下 capture 使用 `ForwardMode.DLLM_EXTEND`，replay 返回时保留并裁剪 `full_logits`，只在 `next_token_logits` 存在时裁剪。原因：LLaDA2 DLLM 的 logits_processor 只返回 `full_logits`，graph replay 原逻辑假设 `next_token_logits` 一定存在。
2.3. `prefill_cuda_graph_runner.py`：capture forward 内 `set_is_extend_in_batch(forward_batch.forward_mode.is_extend())`，让 DeepEP `auto` 在 DLLM/EXTEND prefill graph capture 时选择 normal dispatch/combine，而不是误走 low_latency。
>>> 重启后预期启动日志出现 `Capture piecewise CUDA graph begin` 和 `Capture cuda graph num tokens [32, 64, 96, 128]`；请求命中 32/64/96/128 token bucket 时，metrics 中 `npu graph` 应变为 `True`。
（这里已经可以跑通，但是graph是False）

3. deepep+ep+NPU Graph 适配
3.1 解决Graph capture
历史尝试：曾在 [setup_sglang_LLaDA2_concurrency_zxx_EP_bf16.sh](/data/home/z84301856/proj_sglang/sglang_dllm_own_0706/sglang-dllm/bashs/setup_sglang_LLaDA2_concurrency_zxx_EP_bf16.sh) 显式开启 prefill graph：--cuda-graph-backend-prefill tc_piecewise --cuda-graph-bs-prefill 32 64 96 128 --cuda-graph-tc-compiler eager；后续因 DeepEP + DLLM_EXTEND replay 输出错误，已改为 --cuda-graph-backend-prefill disabled。
在 [prefill_cuda_graph_runner.py (line 135)](/data/home/z84301856/proj_sglang/sglang_dllm_own_0706/sglang-dllm/python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py:135) 让 DLLM capture 使用 ForwardMode.DLLM_EXTEND，否则 graph replay 会走普通 extend logits 分支。
在 [prefill_cuda_graph_runner.py (line 299)](/data/home/z84301856/proj_sglang/sglang_dllm_own_0706/sglang-dllm/python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py:299) 让 DeepEP auto 在 prefill graph capture 时走 normal 分支，并在 replay 输出里保留 full_logits，避免 DLLM 再遇到 next_token_logits=None 问题。
---- 至此， 可以capture graph，但是compile报错
3.2. `python/sglang/srt/models/llada2.py`：tc_piecewise graph 编译/捕获时跳过 `split_qkv_rmsnorm_rope_pos_cache_half_npu`，走普通 qkv split + q/k norm + rope 路径。原因：该 fused helper 内部调用 `triton.runtime.driver.active.utils.get_device_properties()`，Dynamo 在 piecewise compile 阶段会报 `Attempted to call function marked as skipped: NPUUtils.get_device_properties`。普通 eager/NPU 非 piecewise 路径仍继续使用 fused helper。
5. `python/sglang/srt/compilation/npu_piecewise_backend.py`：补齐 `is_in_torch_compile_warmup()` 判断，compile warmup 阶段直接运行 `entry.runnable(*args)`，不进入 `torch.npu.graph(...)` 捕获。原因：tc_piecewise 的 `Compiling num tokens` 阶段只应触发 torch.compile warmup，不应捕获 graph；之前 NPU backend 少了 CUDA backend 的 warmup 短路，导致还没进入真正 capture session 时读取 `get_pcg_capture_stream()` 为空并报 `PCG capture stream is not set`。

6. 历史尝试：曾放宽 `PrefillCudaGraphRunner.can_run()`，让 tc_piecewise 不再受 `can_run_dp_breakable_cuda_graph` 阻挡，运行时可显示 `npu graph: True`；但实际输出出现全 `!`，说明该 DeepEP + DLLM_EXTEND replay 路径不保真。该放宽已回滚，当前保留 `global_num_tokens_cpu` / `can_run_dp_breakable_cuda_graph` 保护。

7. 性能优化实验结论：尝试让 DLLM 的 32-token prefill graph 走 DeepEP low_latency，但在 `Capturing num tokens=128` 时触发 NPU AICore exception / `AclrtSynchronizeDeviceWithTimeout 507015`。结论：当前 NPU piecewise graph capture 与 DeepEP low_latency 不兼容，已回退到稳定的 auto->normal 路径；脚本中保留一行注释说明，不再默认设置该实验环境变量。

8. Correctness 回滚：`npu graph=True` 后出现输出全是 `!`，说明 DeepEP + DLLM_EXTEND 的 prefill tc_piecewise replay 产生了错误 `full_logits`。处理：恢复 `PrefillCudaGraphRunner.can_run()` 中对 `global_num_tokens_cpu` / `can_run_dp_breakable_cuda_graph` 的保护，让 DLLM_EXTEND 在 DeepEP/MLP-sync 下回退 eager；同时脚本改为 `--cuda-graph-backend-prefill disabled`，避免启动阶段白白 capture。当前优先保证 deepep+ep 输出正确，prefill graph 不再强开。

9. 重新开启 graph 的保守同步实验：新增 `SGLANG_NPU_PIECEWISE_SYNC_REPLAY=1`。开启后，`NPUPiecewiseBackend` 会在 NPU graph replay 前后执行 `torch.npu.synchronize()`，用于验证之前 `npu graph=True` 但输出全 `!` 是否由 tc_piecewise graph segment 与 eager DeepEP dispatch/combine 的顺序问题导致。`PrefillCudaGraphRunner.can_run()` 只在 NPU + 该开关 + tc_piecewise backend 下绕过 DLLM_EXTEND 的 `can_run_dp_breakable_cuda_graph` 保护。脚本当前开启 `--cuda-graph-backend-prefill tc_piecewise --cuda-graph-bs-prefill 32 --cuda-graph-tc-compiler eager`，先只 capture 当前测试命中的 32-token bucket，减少启动开销。验证重点：日志应出现 `npu graph: True`，且 `test_in_docker_zxx_bf16.sh` 输出不能再退化成全 `!`。

10. 同步实验结论与 replay metadata 修复：`SGLANG_NPU_PIECEWISE_SYNC_REPLAY=1` 能让 replay 前后同步，但实测仍输出全 `!`，说明问题不是单纯 graph segment 与 eager DeepEP 的 stream 顺序。进一步检查发现 capture 路径 `_run_forward()` 会设置 `set_is_extend_in_batch(True)`，但 prefill graph replay 路径没有重新设置 `set_dp_buffer_len()` / `set_is_extend_in_batch()`；DeepEP `auto` replay 时可能解析成与 capture 不一致的模式。修复：在 `PrefillCudaGraphRunner.replay()` 的 `replay_prepare()` 后补齐 DP buffer metadata 和 `set_is_extend_in_batch(static_forward_batch.forward_mode.is_extend())`。脚本改用 `SGLANG_NPU_DLLM_DEEPEP_PREFILL_GRAPH=1` 作为显式 opt-in；`SGLANG_NPU_PIECEWISE_SYNC_REPLAY=1` 只保留为 debug 注释，默认不开，避免每个 graph segment 都同步导致极慢。

11. 分步排查开关：新增 `SGLANG_NPU_DEEPEP_DEBUG_GRAPH_LOG=1`，用于打印 prefill graph replay metadata 和 DeepEP `auto` 实际解析模式；新增 `SGLANG_NPU_DEEPEP_DISABLE_MOE_SPLIT=1`，用于在 NPU+DeepEP+tc_piecewise 下绕过 `moe_forward_piecewise_cuda_graph_impl`，直接走 `forward_impl`，以判断错误是否来自 MoE split-op。建议顺序：Case A 保持当前脚本只开 debug log，看 `resolved_mode` 是否始终为 `DeepEPMode.NORMAL`；Case B 若仍全 `!`，再打开 `SGLANG_NPU_DEEPEP_DISABLE_MOE_SPLIT=1`，观察是 compile/capture 报错、graph false、还是输出恢复正确。

12. Case 2 结果：设置 `SGLANG_NPU_DLLM_DEEPEP_PREFILL_GRAPH=0` 后，运行日志显示 `npu graph: False`，输出恢复正确。结论：DeepEP eager + DLLM + EP 本身正确；错误只在 prefill graph replay 路径出现。Case 1 中 `npu graph: True` 且全 `!`，Case 2 中 `npu graph: False` 且正确，因此下一步优先定位 tc_piecewise graph replay 内部的 MoE split-op / captured subgraph 交互。

13. Case 3 结果与新修复：设置 `SGLANG_NPU_DEEPEP_DISABLE_MOE_SPLIT=1` 后，tc_piecewise 编译直接追进 DeepEP dispatcher，报 `pybind11_object.__new__` / `deep_ep.buffer.Buffer.capture()` 无法被 Dynamo trace。因此 MoE split-op 必须保留，用于把 DeepEP 从 torch.compile 图里切出去。日志还显示绕过 split-op 时 `is_extend_in_batch=False`，DeepEP `auto` 解析成 `LOW_LATENCY`。新增修复：在 `moe_forward_piecewise_cuda_graph_impl()` 内根据 `forward_context.forward_batch.forward_mode.is_extend()` 重新设置 `set_is_extend_in_batch(...)`，确保 prefill split-op 执行 DeepEP 前看到 DLLM_EXTEND/extend 状态，避免 prefill graph split-op 误走 low_latency。
