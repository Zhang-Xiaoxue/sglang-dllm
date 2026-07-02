# LLaDA2 BF16 vs INT8 Quant Benefit Overview

- Root: `/data/home/z84301856/proj_sglang/sglang_dllm_yuan/sglang-dllm/test/registered/dllm/results_bf16_int8_cannrecipeinfer_moeattn`
- Quant model: cannrecipeinfer moe-attn results
- Each subdirectory contains its own `analysis/` folder with CSV summaries, SVG charts, and markdown reports.

## Summary Table

| model | task | matched cases | int8 wins | mean eval speedup | median eval speedup | mean bs speedup | mean acc/score delta | worst eval speedup | best eval speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mini | gsm8k | 6 | 5/6 | 1.1938x | 1.1842x | 1.253x | 0.0133 | 0.9988x | 1.4152x |
| flash | gsm8k | 5 | 5/5 | 1.2855x | 1.2514x | 1.2242x | -0.004 | 1.1188x | 1.4713x |
| mini | gpqa | 6 | 5/6 | 1.1782x | 1.1517x | n/a | -0.0025 | 0.9909x | 1.3564x |
| flash | gpqa | 3 | 3/3 | 1.2762x | 1.298x | n/a | -0.0051 | 1.1951x | 1.3356x |

## Per-Task Conclusions

### mini gsm8k

- Report: [mini_bf16_int8_quant_analysis_gsm8k/analysis/quant_benefit_report.md](mini_bf16_int8_quant_analysis_gsm8k/analysis/quant_benefit_report.md)
- INT8 wins: 5/6
- Mean eval speedup: 1.1938x; median: 1.1842x
- Mean bs throughput speedup: 1.253x; median: 1.1113x
- Mean accuracy/score delta: 0.0133; median: 0.01

### flash gsm8k

- Report: [flash_bf16_int8_quant_analysis_gsm8k/analysis/quant_benefit_report.md](flash_bf16_int8_quant_analysis_gsm8k/analysis/quant_benefit_report.md)
- INT8 wins: 5/5
- Mean eval speedup: 1.2855x; median: 1.2514x
- Mean bs throughput speedup: 1.2242x; median: 1.2051x
- Mean accuracy/score delta: -0.004; median: 0

### mini gpqa

- Report: [mini_bf16_int8_quant_analysis_gpqa/analysis/quant_benefit_report.md](mini_bf16_int8_quant_analysis_gpqa/analysis/quant_benefit_report.md)
- INT8 wins: 5/6
- Mean eval speedup: 1.1782x; median: 1.1517x
- Mean accuracy/score delta: -0.0025; median: 0.0101

### flash gpqa

- Report: [flash_bf16_int8_quant_analysis_gpqa/analysis/quant_benefit_report.md](flash_bf16_int8_quant_analysis_gpqa/analysis/quant_benefit_report.md)
- INT8 wins: 3/3
- Mean eval speedup: 1.2762x; median: 1.298x
- Mean accuracy/score delta: -0.0051; median: 0

## Why INT8 Is Below Theoretical Expectations

1. End-to-end eval latency includes scheduling, HTTP/client overhead, prompt prefill, decode control, answer parsing, and synchronization. INT8 only accelerates part of the model compute path.
2. TP communication and synchronization do not shrink like INT8 weights. This is most visible on flash, where tp>=4 is required, and on small-batch cases where fixed communication overhead dominates.
3. W8A8C16/compressed-tensors often adds quant/dequant/scale/cast kernels. If these are not fully fused, they eat into GEMM savings.
4. BF16 kernels on Ascend can already be efficient at larger batch/concurrency. Once BF16 is saturated, INT8 relative speedup naturally shrinks.
5. Accuracy/score deltas are single-run measurements. GPQA/GSM8K generation and parsing can move by a few questions between configs; repeat critical configs before treating small deltas as real regressions.

## Suggested Next Checks

- Repeat the strongest and weakest configs three times to separate quantization effect from eval variance.
- For speed, compare both `bs_speedup` and task throughput speedup; if only one improves, the bottleneck is outside pure model throughput.
- For slow TP cases, profile communication/sync and quant/dequant kernels separately from GEMM.
- For GPQA/GSM8K accuracy, inspect invalid/formatting failures and answer extraction behavior in low-score cases.
