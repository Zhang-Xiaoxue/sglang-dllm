# LLaDA2 BF16 vs INT8 Quantization Analysis

Source CSV: `/data/home/z84301856/proj_sglang/sglang_dllm_yuan/sglang-dllm/test/registered/dllm/flash_bf16_int8_quant_analysis/llada2_flash_gsm8k_bf16_int8.csv`
Matched bf16/int8 cases: **10**
INT8 wins on bs_speed: **10/10**

## Key numbers

- Median bs_speed speedup: **1.141x**
- Mean bs_speed speedup: **1.16x**
- Median GSM8K throughput speedup: **1.2249x**
- Mean GSM8K throughput speedup: **1.2171x**
- Median accuracy delta: **-0.0075**

- Best bs_speed case: bs=4, tp=8, speedup=1.348x (320.3544 -> 431.84 tok/s)
- Worst bs_speed case: bs=32, tp=8, speedup=1.0554x (912.0602 -> 962.6142 tok/s)

## Visualizations

- [BS speed by tp](bs_speed_by_tp.svg)
- [GSM8K throughput by tp](gsm8k_throughput_by_tp.svg)
- [BS speedup heatmap](bs_speedup_heatmap.svg)
- [GSM8K speedup heatmap](gsm8k_speedup_heatmap.svg)

## Speedup by tp

| tp | bs_speedup |
| --- | --- |
| 4 | 1.1701 |
| 8 | 1.1499 |

## Speedup by bs

| bs | bs_speedup |
| --- | --- |
| 1 | 1.1827 |
| 4 | 1.2098 |
| 8 | 1.1968 |
| 16 | 1.1195 |
| 32 | 1.0913 |

## Matched case summary

| bs | tp | bf16_bs_speed | int8_bs_speed | bs_speedup | bf16_gsm8k_throughput | int8_gsm8k_throughput | gsm8k_speedup | acc_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 265.7777 | 321.7513 | 1.2106 | 174.2558 | 220.0488 | 1.2628 | 0.025 |
| 4 | 4 | 287.5767 | 308.1712 | 1.0716 | 208.764 | 266.2044 | 1.2751 | -0.02 |
| 8 | 4 | 385.5273 | 495.6043 | 1.2855 | 244.8431 | 320.903 | 1.3106 | -0.01 |
| 16 | 4 | 525.7495 | 607.624 | 1.1557 | 327.3274 | 408.7104 | 1.2486 | -0.01 |
| 32 | 4 | 633.2724 | 713.7627 | 1.1271 | 437.7235 | 486.6977 | 1.1119 | 0.03 |
| 1 | 8 | 301.3314 | 348.0059 | 1.1549 | 206.6837 | 240.2667 | 1.1625 | -0.005 |
| 4 | 8 | 320.3544 | 431.84 | 1.348 | 241.6465 | 296.8055 | 1.2283 | 0.01 |
| 8 | 8 | 448.579 | 497.0474 | 1.108 | 345.0084 | 410.5663 | 1.19 | 0.005 |
| 16 | 8 | 683.3534 | 740.3114 | 1.0834 | 425.482 | 519.7407 | 1.2215 | -0.03 |
| 32 | 8 | 912.0602 | 962.6142 | 1.0554 | 561.3326 | 651.071 | 1.1599 | -0.04 |

## How to read this

- `bs_speedup = int8_bs_speed / bf16_bs_speed`. Values above 1 mean INT8 is faster.
- `gsm8k_speedup = int8_gsm8k_output_throughput / bf16_gsm8k_output_throughput`.
- Latency is normalized as latency per output token because output token counts differ across runs.
- Warmup rows are excluded by default.
