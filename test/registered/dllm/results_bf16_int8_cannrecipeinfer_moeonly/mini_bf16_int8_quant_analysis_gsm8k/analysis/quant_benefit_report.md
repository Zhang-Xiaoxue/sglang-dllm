# LLaDA2 Mini BF16 vs INT8 Quantization Analysis

Source CSV: `/data/home/z84301856/proj_sglang/sglang_dllm_yuan/sglang-dllm/test/registered/dllm/llada2_mini_gsm8k_bf16_int8.csv`
Matched bf16/int8 cases: **14**
INT8 wins on bs_speed: **12/14**

## Key numbers

- Median bs_speed speedup: **1.0785x**
- Mean bs_speed speedup: **1.3982x**
- Median GSM8K throughput speedup: **1.0518x**
- Mean GSM8K throughput speedup: **1.0745x**
- Median accuracy delta: **-0.005**

- Best bs_speed case: bs=4, tp=1, speedup=3.9714x (144.8053 -> 575.084 tok/s)
- Worst bs_speed case: bs=16, tp=8, speedup=0.9349x (1297.4593 -> 1213.0025 tok/s)

## Visualizations

- [BS speed by tp](bs_speed_by_tp.svg)
- [GSM8K throughput by tp](gsm8k_throughput_by_tp.svg)
- [BS speedup heatmap](bs_speedup_heatmap.svg)
- [GSM8K speedup heatmap](gsm8k_speedup_heatmap.svg)

## Speedup by tp

| tp | bs_speedup |
| --- | --- |
| 1 | 2.3743 |
| 4 | 1.2241 |
| 8 | 1.0553 |

## Speedup by bs

| bs | bs_speedup |
| --- | --- |
| 1 | 1.068 |
| 4 | 2.0433 |
| 8 | 1.3676 |
| 16 | 1.2341 |
| 32 | 1.2805 |
| 64 | 1.1092 |

## Matched case summary

| bs | tp | bf16_bs_speed | int8_bs_speed | bs_speedup | bf16_gsm8k_throughput | int8_gsm8k_throughput | gsm8k_speedup | acc_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 523.7803 | 552.3527 | 1.0546 | 316.4916 | 393.5384 | 1.2434 | 0 |
| 4 | 1 | 144.8053 | 575.084 | 3.9714 | 393.5346 | 504.7559 | 1.2826 | 0.035 |
| 8 | 1 | 418.9697 | 878.5294 | 2.0969 | 482.942 | 649.1586 | 1.3442 | -0.01 |
| 1 | 4 | 549.562 | 580.792 | 1.0568 | 397.4307 | 421.1053 | 1.0596 | -0.02 |
| 4 | 4 | 671.4664 | 710.7988 | 1.0586 | 531.5149 | 632.7796 | 1.1905 | 0.03 |
| 8 | 4 | 865.0137 | 843.447 | 0.9751 | 687.4983 | 775.167 | 1.1275 | 0.02 |
| 16 | 4 | 760.9938 | 1166.8258 | 1.5333 | 969.4157 | 954.4011 | 0.9845 | -0.005 |
| 32 | 4 | 1014.9826 | 1518.9933 | 1.4966 | 1118.9237 | 1168.178 | 1.044 | 0.055 |
| 1 | 8 | 475.4554 | 519.4244 | 1.0925 | 404.7245 | 415.8753 | 1.0276 | -0.01 |
| 4 | 8 | 656.8209 | 722.4206 | 1.0999 | 526.2393 | 560.6501 | 1.0654 | -0.005 |
| 8 | 8 | 874.5321 | 901.548 | 1.0309 | 808.9443 | 811.2924 | 1.0029 | -0.01 |
| 16 | 8 | 1297.4593 | 1213.0025 | 0.9349 | 1130.3463 | 872.4352 | 0.7718 | -0.01 |
| 32 | 8 | 1605.4621 | 1708.9742 | 1.0645 | 1307.9173 | 1277.3713 | 0.9766 | -0.005 |
| 64 | 8 | 1797.3861 | 1993.7081 | 1.1092 | 1632.8538 | 1505.6071 | 0.9221 | 0 |

## How to read this

- `bs_speedup = int8_bs_speed / bf16_bs_speed`. Values above 1 mean INT8 is faster.
- `gsm8k_speedup = int8_gsm8k_output_throughput / bf16_gsm8k_output_throughput`.
- Latency is normalized as latency per output token because output token counts differ across runs.
- Warmup rows are excluded by default.
