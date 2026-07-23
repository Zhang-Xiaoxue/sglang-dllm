# LLaDA2 Eval INT8 Quant Benefit Report

- Source CSV: `/data/home/z84301856/proj_sglang/sglang_dllm_yuan/sglang-dllm/test/registered/dllm/results/flash_bf16_int8_quant_analysis_gpqa/llada2_flash_gpqa.csv`
- Matched bf16/int8 cases: 3

## Summary

- INT8 output throughput wins: 3/3
- Mean throughput speedup: 1.2955x
- Median throughput speedup: 1.3219x
- Mean latency speedup: 1.2838x
- Median latency speedup: 1.3218x
- Mean score delta: +0.0017
- Median score delta: +0.0152

## Charts

### Score by tp

![Score by tp](score_by_tp.svg)

### Output throughput by tp

![Output throughput by tp](output_throughput_by_tp.svg)

### Latency by tp

![Latency by tp](latency_by_tp.svg)

### INT8 / BF16 throughput speedup

![INT8 / BF16 throughput speedup](throughput_speedup_heatmap.svg)

### INT8 - BF16 score delta

![INT8 - BF16 score delta](score_delta_heatmap.svg)

## Per-Case Table

| eval | bs | tp | bf16 score | int8 score | score delta | bf16 tok/s | int8 tok/s | speedup | bf16 latency s | int8 latency s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpqa | 1 | 4 | 0.2929 | 0.3081 | +0.0152 | 127.72 | 153.84 | 1.2045x | 2505.32 | 2119.08 |
| gpqa | 4 | 4 | 0.3384 | 0.3081 | -0.0303 | 176.32 | 239.81 | 1.3601x | 1778.84 | 1345.73 |
| gpqa | 8 | 4 | 0.3182 | 0.3384 | +0.0202 | 233.05 | 308.08 | 1.3219x | 1396.09 | 1036.25 |

## Notes

- `throughput_speedup = int8_output_throughput / bf16_output_throughput`.
- `latency_speedup = bf16_latency / int8_latency`; larger than 1 means INT8 is faster.
- GPQA/PIQA scores can move by one or more questions between runs; repeat key configs before treating small deltas as accuracy regressions.
