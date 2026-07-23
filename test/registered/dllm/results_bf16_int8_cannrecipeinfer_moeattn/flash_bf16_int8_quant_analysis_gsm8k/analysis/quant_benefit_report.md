# LLaDA2 BF16 vs INT8 Quantization Analysis

Source CSV: `/data/home/z84301856/proj_sglang/sglang_dllm_yuan/sglang-dllm/test/registered/dllm/results_bf16_int8_cannrecipeinfer_moeattn/flash_bf16_int8_quant_analysis_gsm8k/llada2_flash_gsm8k.csv`
Matched bf16/int8 cases: **5**
INT8 wins on bs_speed: **5/5**

## Key numbers

- Median bs_speed speedup: **1.2051x**
- Mean bs_speed speedup: **1.2242x**
- Median GSM8K throughput speedup: **1.2514x**
- Mean GSM8K throughput speedup: **1.2855x**
- Median accuracy delta: **0**

- Best bs_speed case: bs=8, tp=4, speedup=1.3719x (318.513 -> 436.9696 tok/s)
- Worst bs_speed case: bs=4, tp=4, speedup=1.1116x (310.7957 -> 345.4824 tok/s)

## Visualizations

- [BS speed by tp](bs_speed_by_tp.svg)
- [GSM8K throughput by tp](gsm8k_throughput_by_tp.svg)
- [BS speedup heatmap](bs_speedup_heatmap.svg)
- [GSM8K speedup heatmap](gsm8k_speedup_heatmap.svg)

## Speedup by tp

| tp | bs_speedup |
| --- | --- |
| 4 | 1.2242 |

## Speedup by bs

| bs | bs_speedup |
| --- | --- |
| 1 | 1.2519 |
| 4 | 1.1116 |
| 8 | 1.3719 |
| 16 | 1.1804 |
| 32 | 1.2051 |

## Matched case summary

| bs | tp | bf16_bs_speed | int8_bs_speed | bs_speedup | bf16_gsm8k_throughput | int8_gsm8k_throughput | gsm8k_speedup | acc_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 299.3683 | 374.7925 | 1.2519 | 177.5086 | 214.2642 | 1.2071 | -0.04 |
| 4 | 4 | 310.7957 | 345.4824 | 1.1116 | 206.0266 | 257.8247 | 1.2514 | 0 |
| 8 | 4 | 318.513 | 436.9696 | 1.3719 | 233.7733 | 343.9467 | 1.4713 | -0.01 |
| 16 | 4 | 514.4731 | 607.2677 | 1.1804 | 319.4929 | 440.4893 | 1.3787 | 0.02 |
| 32 | 4 | 621.6981 | 749.1936 | 1.2051 | 436.6233 | 488.5 | 1.1188 | 0.01 |

## How to read this

- `bs_speedup = int8_bs_speed / bf16_bs_speed`. Values above 1 mean INT8 is faster.
- `gsm8k_speedup = int8_gsm8k_output_throughput / bf16_gsm8k_output_throughput`.
- Latency is normalized as latency per output token because output token counts differ across runs.
- Warmup rows are excluded by default.
