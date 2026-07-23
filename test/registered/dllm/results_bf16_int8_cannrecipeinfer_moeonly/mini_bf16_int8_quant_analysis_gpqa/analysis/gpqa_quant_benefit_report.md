# LLaDA2 Mini GPQA INT8 Quant Benefit Report

- Source CSV: `/data/home/z84301856/proj_sglang/sglang_dllm_yuan/sglang-dllm/test/registered/dllm/results/mini_bf16_int8_quant_analysis_gpqa/llada2_mini_gpqa.csv`
- Matched bf16/int8 cases: 5
- Passed rows: 10/10
- Eval: GPQA, API: chat, threads: 128, max_tokens: 2048

## Summary

- INT8 output throughput wins: 4/5
- Mean throughput speedup: 1.1729x
- Median throughput speedup: 1.2365x
- Mean latency speedup: 1.1652x
- Median latency speedup: 1.1948x
- Mean score delta: -0.0061
- Median score delta: +0.0051
- Score improved / dropped / equal: 3/2/0

## Notable Cases

- Best throughput speedup: bs=4, tp=1, 1.3649x, score delta -0.0051
- Worst throughput speedup: bs=1, tp=4, 0.7870x, score delta +0.0051
- Best score delta: bs=8, tp=1, +0.0202, throughput speedup 1.3604x
- Worst score delta: bs=1, tp=1, -0.0606, throughput speedup 1.2365x

## Per-Case Table

| bs | tp | bf16 score | int8 score | score delta | bf16 tok/s | int8 tok/s | speedup | bf16 latency s | int8 latency s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0.2828 | 0.2222 | -0.0606 | 215.29 | 266.22 | 1.2365x | 1448.54 | 1212.34 |
| 4 | 1 | 0.2475 | 0.2424 | -0.0051 | 309.72 | 422.73 | 1.3649x | 1031.77 | 753.69 |
| 8 | 1 | 0.2323 | 0.2525 | +0.0202 | 418.30 | 569.04 | 1.3604x | 762.13 | 559.63 |
| 1 | 4 | 0.2273 | 0.2323 | +0.0051 | 268.84 | 211.56 | 0.7870x | 1169.41 | 1495.35 |
| 4 | 4 | 0.2323 | 0.2424 | +0.0101 | 438.17 | 488.85 | 1.1157x | 723.16 | 646.73 |

## Interpretation

- INT8 has clear throughput benefit for tp=1 and bs=4,tp=4, but bs=1,tp=4 is slower than bf16. This points to TP communication/dequant/cast overhead dominating at small batch or higher TP.
- GPQA score changes are mixed and the sample count is small enough that one-question movement is about 0.0051 score. Treat single-run score deltas as noisy unless repeated runs confirm the trend.
- The largest accuracy drop appears at bs=1,tp=1; because the same model should not conceptually depend on bs, this is likely affected by decoding/runtime nondeterminism or output parsing variance. Repeating the run is recommended before drawing an accuracy conclusion.
