# LLaDA2 BF16 vs INT8 Quantization Analysis

Source CSV: `/data/home/z84301856/proj_sglang/sglang_dllm_yuan/sglang-dllm/test/registered/dllm/results_bf16_int8_cannrecipeinfer_moeattn/mini_bf16_int8_quant_analysis_gsm8k/llada2_mini_gsm8k.csv`
Matched bf16/int8 cases: **6**
INT8 wins on bs_speed: **6/6**

## Key numbers

- Median bs_speed speedup: **1.1113x**
- Mean bs_speed speedup: **1.253x**
- Median GSM8K throughput speedup: **1.1842x**
- Mean GSM8K throughput speedup: **1.1938x**
- Median accuracy delta: **0.01**

- Best bs_speed case: bs=4, tp=1, speedup=1.9403x (389.0986 -> 754.9588 tok/s)
- Worst bs_speed case: bs=8, tp=4, speedup=1.0334x (833.1781 -> 860.981 tok/s)

## Visualizations

- [BS speed by tp](bs_speed_by_tp.svg)
- [GSM8K throughput by tp](gsm8k_throughput_by_tp.svg)
- [BS speedup heatmap](bs_speedup_heatmap.svg)
- [GSM8K speedup heatmap](gsm8k_speedup_heatmap.svg)

## Speedup by tp

| tp | bs_speedup |
| --- | --- |
| 1 | 1.4206 |
| 4 | 1.0853 |

## Speedup by bs

| bs | bs_speedup |
| --- | --- |
| 1 | 1.0984 |
| 4 | 1.5306 |
| 8 | 1.13 |

## Matched case summary

| bs | tp | bf16_bs_speed | int8_bs_speed | bs_speedup | bf16_gsm8k_throughput | int8_gsm8k_throughput | gsm8k_speedup | acc_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 550.5916 | 602.9559 | 1.0951 | 315.5687 | 381.0572 | 1.2075 | 0.01 |
| 4 | 1 | 389.0986 | 754.9588 | 1.9403 | 381.0255 | 507.5738 | 1.3321 | 0.01 |
| 8 | 1 | 600.7995 | 736.8993 | 1.2265 | 491.8814 | 696.1072 | 1.4152 | 0.01 |
| 1 | 4 | 521.0514 | 574.0573 | 1.1017 | 390.9037 | 390.4189 | 0.9988 | 0.005 |
| 4 | 4 | 637.2442 | 714.2805 | 1.1209 | 520.3786 | 545.4332 | 1.0481 | 0.035 |
| 8 | 4 | 833.1781 | 860.981 | 1.0334 | 670.9719 | 778.8979 | 1.1609 | 0.01 |

## How to read this

- `bs_speedup = int8_bs_speed / bf16_bs_speed`. Values above 1 mean INT8 is faster.
- `gsm8k_speedup = int8_gsm8k_output_throughput / bf16_gsm8k_output_throughput`.
- Latency is normalized as latency per output token because output token counts differ across runs.
- Warmup rows are excluded by default.

## Unmatched cases

- {'model_size': 'mini', 'bs': 16, 'tp': 4, 'ep': 1, 'dp': 1, 'moe_a2a_backend': 'none'}: ['bf16']
- {'model_size': 'mini', 'bs': 32, 'tp': 4, 'ep': 1, 'dp': 1, 'moe_a2a_backend': 'none'}: ['bf16']
