# Codex Handoff: LLaDA2 Ascend EP/DeepEP

Last updated: 2026-07-21

## 1. Final goal

Make LLaDA2.1-mini BF16 inference on Ascend work correctly with EP for both
`--moe-a2a-backend none` and `deepep`, retain a correct decode NPU graph, and
understand/close the DeepEP performance gap. Model weights and expert GEMMs stay
BF16; INT8 is used only for DeepEP dispatch traffic.

The production target is TP=EP=4. The user currently permits tests only on physical
NPUs 5 and 7, so current validation is TP=EP=DP=2 with DP attention.

## 2. Original problem and reproduction

Production launch:

```bash
ASCEND_RT_VISIBLE_DEVICES=5,7 \
SGLANG_TP_SIZE=2 SGLANG_EP_SIZE=2 SGLANG_DP_SIZE=2 \
SGLANG_ENABLE_DP_ATTENTION=1 \
SGLANG_MOE_A2A_BACKEND=deepep \
bash bashs/setup_sglang_LLaDA2_concurrency_zxx_EP_bf16.sh
```

The failure progression was:

1. `none+EP` initially errored or generated wrong text.
2. `deepep+EP` failed during compile/graph capture.
3. After capture started working, graph replay generated all `!` or garbled text.
4. Eager execution was correct but slow.
5. Decode full graph became correct, but DeepEP remained slower than `none`.

The deterministic debug prompt asks for Eliza's weekly earnings. Correct output
contains `$460`.

## 3. Current code and repository state

Main repository:

```text
path:   /data/home/z84301856/proj_sglang/sglang_dllm_own_0706/sglang-dllm
branch: EP_deploy_zxx_final
HEAD:   bc00a482833d7f45072363399d6e03b8ec514efb
```

Modified/untracked main files:

```text
M  bashs/debug_sglang_LLaDA2_deepep_perf.sh
M  bashs/setup_sglang_LLaDA2_concurrency_zxx_EP_bf16.sh
M  python/sglang/srt/distributed/parallel_state.py
M  python/sglang/srt/dllm/algorithm/joint_threshold.py
M  python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py
M  python/sglang/srt/layers/moe/token_dispatcher/deepep.py
M  python/sglang/srt/layers/quantization/unquant.py
M  python/sglang/srt/managers/scheduler_components/dp_attn.py
M  python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
M  python/sglang/srt/models/llada2.py
?? CODEX_HANDOFF.md
?? bashs/joint_threshold_perf_fixed.yaml
```

Operator repository:

```text
path: /data/home/z84301856/proj_sglang/sgl-kernel-npu
HEAD: f451a5964fe11c596e4b96d92e16af0b1b4a75b4 (origin/main)
```

It is intentionally dirty. Do not reset or discard its changes. Relevant local
changes are in `csrc/deepep/deep_ep.cpp`, `event.hpp`, and
`pytorch_npu_helper.hpp`; there are also user/operator changes in `ops2` and
untracked build/profile files.

Installed packages:

```text
deep_ep       1.0.0+f451a596.cann.9.0.0.b250
sgl_kernel_npu 2026.6.0
CANN          9.0.0
npu-smi       25.5.1
hardware      Ascend 910B3
```

## 4. Modified files and purpose

### Main repository

- `parallel_state.py`: NPU/HCCL `reduce_scatterv` implementation; avoids using a
  missing CUDA/PyNCCL communicator.
- `joint_threshold.py`: synchronizes adaptive dLLM forward-loop activity across DP
  ranks so every rank enters the same number of EP collectives; optional forward
  count logging.
- `dp_attn.py`: allows `DLLM_EXTEND` through the DP decode graph path.
- `decode_cuda_graph_runner.py`: derives graph buckets correctly for fixed-size dLLM
  replay.
- `deepep.py`: resolves NPU dLLM AUTO mode to LOW_LATENCY; detects the current
  DeepEP API and passes `quant_mode="int8"` when INT8 dispatch is requested. The
  older `use_fp8=True` argument is not sufficient with the f451 API.
- `fused_moe_method_npu.py`: Triton per-token INT8 dequant before BF16 expert GEMM.
- `unquant.py`: propagates DeepEP dispatch scales to the NPU unquantized expert path.
- `llada2.py`: fixes TP1 shared-expert accumulation order so shared output is not
  multiplied by a routed TP all-reduce; keeps helper use consistent.
- `debug_sglang_LLaDA2_deepep_perf.sh`: reproducible matrix/run/profile driver,
  graph-hit and forward-count reporting, capacity validation, NPU readiness polling,
  logs and CSV summaries. A failed `npu-smi` query now waits/fails instead of
  incorrectly starting the case.
- `joint_threshold_perf_fixed.yaml`: performance-only fixed two-forward/block config.
  Its generated text is intentionally not an accuracy signal.
- `setup_sglang_LLaDA2_concurrency_zxx_EP_bf16.sh`: production-style DP-attention,
  DP-LM-head, decode-full-graph launch. Prefill graph is disabled. DeepEP dispatch
  now defaults to INT8; set `SGLANG_DEEPEP_DISPATCH_DTYPE=auto` to restore BF16
  communication.

### Operator repository

- `deep_ep.cpp`: constructs completion events after async low-latency
  dispatch/combine calls.
- `event.hpp`: real NPU event record/wait instead of an empty placeholder.
- `pytorch_npu_helper.hpp`: captures the ACLNN workspace tensor in deferred lambdas
  so its lifetime covers execution.

## 5. Failed attempts and reasons

1. **tc-piecewise prefill graph with DeepEP**: all `!`, garbled output, AICore
   exceptions, or capture errors. Replay synchronization did not make it both fast
   and correct. Stable configuration disables prefill graph.
2. **Tracing DeepEP pybind through Dynamo by disabling the MoE split**: failed on
   `Buffer.capture()` / `pybind11_object.__new__`; DeepEP remains outside that
   compiled segment.
3. **DeepEP NORMAL mode**: `server_args.py` disables decode graph for NORMAL, so it
   is unsuitable for this graph performance target.
4. **Hunting skipped QKV/RMSNorm/RoPE subgraphs**: the fused helper is about 11 us
   per layer, roughly 0.2 ms/model-forward, and is not the DeepEP bottleneck.
5. **Shared-expert secondary stream / dispatch-hook overlap**: earlier variants
   regressed. A fresh NPU graph experiment at BS8 also fell from 3452.83 to 3392.95
   token/s and was fully removed.
6. **DeepEP graph with local graph BS8 / capacity 256**: BF16 graph capture failed
   in `npuSynchronizeDevice` with error 507057. The same shape runs eager, so this is
   a graph/collective limit, not a generic operator shape rejection.
7. **Operator microbenchmark with default `HCCL_BUFFSIZE=200`**: tiling rejected the
   LLaDA2 shape because it needs 297 MB. Rerunning with the SGLang setting
   `HCCL_BUFFSIZE=1024` passed.
8. **Old "INT8" tests**: before the f451 API adaptation, `use_fp8=True` was ignored
   and the C++ operator received quant mode `none`. Those results were BF16 and must
   not be cited as INT8.

## 6. Confirmed and uncertain information

### Confirmed

- `none+EP` and `deepep+EP` are correct in the current stable paths.
- DeepEP INT8 plus decode full graph is correct: exact answer `$460`, 22 graph hits,
  zero graph misses, no degeneration.
- The f451 trace shows DeepEP dispatch output dtype `INT8`; this is true INT8 traffic.
- INT8 reduces the BS8 dispatch device kernel from about 279 us to 203 us (27%) and
  the non-outlier AICPU dispatch portion from about 540 us to 446 us (17%). Combine
  stays BF16 and is essentially unchanged.
- INT8 dequant costs around 18-24 us per sparse layer, so the dispatch saving remains
  net-positive.
- Decode graph is active. Prefill graph is deliberately disabled.
- Shared expert TP1 is required with DP attention because ranks own different token
  subsets; naive TP sharding would require another token collective.
- Dispatch/combine have partial overlap with surrounding work, but remain a
  per-layer critical chain. There is no useful inter-layer overlap.
- NPUs 5 and 7 are linked by HCCS; their pairing is not a PCIe topology penalty.
- The skipped `split_qkv_rmsnorm_rope_pos_cache_half_npu` graph break is not the
  present bottleneck.

### Still uncertain

- The true crossover on TP=EP=4 or EP>=8 has not been measured after the f451 and
  true-INT8 fixes. The user currently permits only NPUs 5 and 7.
- Full GSM8K accuracy impact of INT8 dispatch has not yet been measured; one exact
  reasoning sample and the operator cosine check pass.
- The f451 operator benchmark shows substantial rank/outlier variance, so its simple
  average bandwidth is not a reliable end-to-end estimator.
- The source of graph capture failure at local BS8/capacity256 is not yet fixed.

## 7. Key errors, logs, and test results

### Fair fixed BS8, TP=EP=DP=2, NPUs 5 and 7

`logs/deepep_perf_debug/f451_fixed_dp_ab_bs8_2npu57_0721_codex/summary.csv`

| Backend | Dispatch | token/s | forwards | ms/forward | graph |
|---|---|---:|---:|---:|---|
| none | BF16 | 3931.52 | 12 | 21.125 | 23 true / 0 false |
| deepep | BF16 | 3253.29 | 12 | 25.637 | 23 true / 0 false |
| deepep | INT8 | 3452.83 | 12 | 24.124 | 23 true / 0 false |

True INT8 is 6.1% faster than DeepEP BF16, but DeepEP remains 12.2% slower than
`none` on EP2.

INT8 result:
`logs/deepep_perf_debug/f451_true_int8_fixed_dp_bs8_2npu57_0721_codex/`

Accuracy result:
`logs/deepep_perf_debug/f451_true_int8_accuracy_bs1_2npu57_0721_retry/`

```text
answer contains $460
graph_true=22 graph_false=0
model_forwards=66
throughput=461.47 token/s
```

Same-shape profiler directories:

```text
logs/deepep_perf_debug/f451_true_int8_profile_fixed_bs8_2npu57_0721_codex/
logs/deepep_perf_debug/f451_bf16_profile_fixed_bs8_2npu57_0721_codex/
```

### Key stack traces

NPU communicator misuse fixed in `parallel_state.py`:

```text
AttributeError: 'NoneType' object has no attribute 'change_state'
  parallel_state.py:824 in reduce_scatterv
```

Insufficient DeepEP capacity:

```text
num_max_dispatch_tokens_per_rank >= x.size(0)
```

Insufficient HCCL buffer in the standalone operator benchmark:

```text
HCCL_BUFFSIZE is too SMALL ... NEEDED_HCCL_BUFFSIZE=297MB,
HCCL_BUFFSIZE=200MB
RuntimeError: call aclnnMoeLowLatencyDispatchV2 failed
```

Large graph bucket failure:

```text
RuntimeError: npuSynchronizeDevice ... SUSPECT REMOTE ERROR, error code 507057
```

## 8. Environment and commands

Use `login:false` for tool commands; login shells can hang in this environment.
Hardware commands and NPU tests need access outside the filesystem sandbox.

Fixed performance run:

```bash
ASCEND_RT_VISIBLE_DEVICES=5,7 \
SGLANG_DEEPEP_DEBUG_TP=2 \
SGLANG_DEEPEP_DEBUG_BATCH_SIZE=8 \
SGLANG_DEEPEP_DEBUG_MAX_RUNNING_REQUESTS=8 \
SGLANG_DEEPEP_DEBUG_CUDA_GRAPH_BS_DECODE="1 2 4" \
SGLANG_DEEPEP_DEBUG_MAX_NEW_TOKENS=128 \
SGLANG_DEEPEP_DEBUG_DISPATCH_DTYPE=int8 \
SGLANG_DEEPEP_DEBUG_ALGORITHM_CONFIG=bashs/joint_threshold_perf_fixed.yaml \
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128 \
bash bashs/debug_sglang_LLaDA2_deepep_perf.sh deepep_dp_attention run
```

Accuracy run: remove `SGLANG_DEEPEP_DEBUG_ALGORITHM_CONFIG`, set batch 1,
max-new-tokens 512, graph buckets `1 2`, capacity 32, and expected text 460.

Standalone operator correctness/performance:

```bash
ASCEND_RT_VISIBLE_DEVICES=5,7 HCCL_BUFFSIZE=1024 \
python3 tests/python/deepep/test_low_latency.py \
  --num-processes 2 --num-tokens 128 --hidden 2048 \
  --num-topk 8 --num-experts 256 --quant-type int8
```

Run that command from `/data/home/z84301856/proj_sglang/sgl-kernel-npu`.

## 9. Most likely root cause

The remaining EP2 gap is expected from this model's routing geometry, not a decode
graph miss:

- 256 experts, top-k 8, 8 routing groups, top-k groups 4.
- With EP2, each rank owns four groups. Selecting all four groups from only one rank
  has probability `2 * C(4,4) / C(8,4) = 2/70`, about 2.86%.
- Therefore roughly 97% of tokens select groups spanning both ranks. DeepEP sends
  nearly every token to both ranks, so it saves little versus replication/all-gather
  while adding dispatch, combine, AICPU control, event, and dequant work 19 times.
- EP4 still spans at least two ranks per token. DeepEP's intended advantage should
  become clearer near EP8, where selecting four of eight groups reaches about half
  the ranks instead of all ranks.

This explains why official DeepEP speedups do not apply automatically to EP2 for a
top-k-8 model. The implementation is now correct and faster with INT8, but the
parallel geometry does not provide enough communication sparsity.

## 10. Recommended next steps

1. Run full/representative GSM8K A/B for BF16-dispatch versus true INT8-dispatch;
   keep INT8 as production default only if accuracy remains within tolerance.
2. When four or eight cards are available, run normalized TP=EP=DP=4 and EP8 tests
   with identical fixed forward counts. This is the decisive crossover test.
3. Investigate the local-BS8/capacity256 graph-capture error separately using the
   operator graph test; ordinary eager execution already proves the shape works.
4. Optimize the operator critical chain, prioritizing AICPU setup and combine. A
   fused dispatch+expert-FFN+combine path has the largest upside, but current BF16
   experts are not supported by the available fused operator.
5. Add the dispatch dtype to the registered EP CSV test output so BF16 and INT8
   communication runs cannot be confused.

## 11. Do not repeat unchanged

- Do not re-enable tc-piecewise prefill graph without a new correctness fix.
- Do not disable MoE splitting to trace the DeepEP pybind object through Dynamo.
- Do not use DeepEP NORMAL mode when decode graph is required.
- Do not reapply the shared-expert secondary-stream/hook overlap experiments.
- Do not cite pre-f451 `use_fp8=True` runs as INT8.
- Do not compare adaptive JointThreshold request throughput without normalizing by
  model-forward count.
- Do not run the operator benchmark with HCCL buffer below 297 MB for this shape.
- Do not reset NPUs; this HCCS configuration can reset all cards.

## 12. Files to read first

1. `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`
2. `python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`
3. `python/sglang/srt/layers/quantization/unquant.py`
4. `python/sglang/srt/models/llada2.py`
5. `python/sglang/srt/dllm/algorithm/joint_threshold.py`
6. `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`
7. `python/sglang/srt/managers/scheduler_components/dp_attn.py`
8. `python/sglang/srt/distributed/parallel_state.py`
9. `bashs/debug_sglang_LLaDA2_deepep_perf.sh`
10. `bashs/setup_sglang_LLaDA2_concurrency_zxx_EP_bf16.sh`
11. `/data/home/z84301856/proj_sglang/sgl-kernel-npu/csrc/deepep/deep_ep.cpp`
12. `/data/home/z84301856/proj_sglang/sgl-kernel-npu/csrc/deepep/event.hpp`
