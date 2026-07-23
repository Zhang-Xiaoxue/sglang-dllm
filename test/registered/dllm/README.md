# LLaDA2 Ascend Test Guide

本目录用于串行跑 LLaDA2 mini/flash 在 Ascend 上的回归测试，并把结果写入 CSV。

## 进入目录

```bash
cd /data/home/z84301856/proj_sglang/sglang_dllm_yuan/sglang-dllm/test/registered/dllm
```

## `test.sh` 总入口

`test.sh` 主要使用位置参数选择模型、测试集和输出 CSV。

参数格式：

```bash
bash test.sh [csv_output_path] [mini|flash] [gsm8k|gpqa|piqa|gpqa,piqa|gsm8k,gpqa|all]
```

默认跑 `mini` 的 `all`，其中 `all` 表示：

- `gsm8k`：调用 `run_llada2_ascend_gsm8k_throughput_csv.sh`，会跑 GSM8K accuracy + bs throughput，覆盖 bf16/int8
- `gpqa`：调用 `run_llada2_ascend_gpqa_piqa_csv.sh`，覆盖 bf16/int8
- 如果设置了 `SGLANG_DLLM_PIQA_DATA_PATH`，`all` 会额外包含 `piqa`

```bash
./test.sh
```

跑 flash 的默认测试：

```bash
bash test.sh flash
```

只跑 flash + GSM8K/throughput，并写到指定 CSV：

```bash
bash test.sh results/flash_bf16_int8_quant_analysis_gsm8k/llada2_flash_gsm8k.csv flash gsm8k
```

只跑 flash + GPQA，并写到指定 CSV：

```bash
bash test.sh results/flash_bf16_int8_quant_analysis_gpqa/llada2_flash_gpqa.csv flash gpqa
```

只跑 flash + PIQA：

```bash
SGLANG_DLLM_PIQA_DATA_PATH=/path/to/piqa/validation.jsonl \
  bash test.sh results/flash_piqa.csv flash piqa
```

同时跑 GSM8K + GPQA：

```bash
bash test.sh results/flash_eval flash gsm8k,gpqa
```

如果一个命令同时包含 `gsm8k` 和其他 eval，`test.sh` 会自动拆成两个 CSV，避免不同表头写到同一个文件。例如上一条命令会写出：

- `results/flash_eval_gsm8k.csv`
- `results/flash_eval_gpqa.csv`

输出文件示例：

- `llada2_mini_gsm8k_bf16_int8.csv`
- `llada2_mini_gpqa.csv`
- `llada2_mini_gpqa_piqa.csv`
- `llada2_flash_gsm8k_bf16_int8.csv`

## 只跑 GSM8K + Throughput

脚本会分别跑 bf16/int8，并写入同一个 CSV。

```bash
bash run_llada2_ascend_gsm8k_throughput_csv.sh llada2_mini_gsm8k_bf16_int8.csv mini
```

跑 flash：

```bash
bash run_llada2_ascend_gsm8k_throughput_csv.sh llada2_flash_gsm8k_bf16_int8.csv flash
```

注意：flash 下 `tp` 必须大于等于 4，脚本会自动跳过不合法 case。

## 只跑 GPQA

```bash
bash run_llada2_ascend_gpqa_piqa_csv.sh llada2_mini_gpqa.csv mini gpqa
```

跑 flash：

```bash
bash run_llada2_ascend_gpqa_piqa_csv.sh llada2_flash_gpqa.csv flash gpqa
```

GPQA 默认使用 `run_eval.py` 里的 `gpqa_diamond.csv` 远端地址。如果环境无法访问外网，可以准备本地 CSV 后指定：

```bash
SGLANG_DLLM_GPQA_DATA_PATH=/path/to/gpqa_diamond.csv \
  bash run_llada2_ascend_gpqa_piqa_csv.sh llada2_mini_gpqa.csv mini gpqa
```

## 只跑 PIQA

PIQA 需要本地数据路径。支持：

- 单个 JSONL/CSV/JSON 文件，字段包含 `goal, sol1, sol2, label`
- PIQA 原始目录形式，例如包含 `validation.jsonl` 和 `validation-labels.lst`

```bash
SGLANG_DLLM_PIQA_DATA_PATH=/path/to/piqa/validation.jsonl \
  bash run_llada2_ascend_gpqa_piqa_csv.sh llada2_mini_piqa.csv mini piqa
```

跑 flash：

```bash
SGLANG_DLLM_PIQA_DATA_PATH=/path/to/piqa/validation.jsonl \
  bash run_llada2_ascend_gpqa_piqa_csv.sh llada2_flash_piqa.csv flash piqa
```

## 同时跑 GPQA + PIQA

```bash
SGLANG_DLLM_PIQA_DATA_PATH=/path/to/piqa/validation.jsonl \
  bash run_llada2_ascend_gpqa_piqa_csv.sh llada2_mini_gpqa_piqa.csv mini gpqa,piqa
```

## 常用环境变量

模型尺寸、测试名、CSV 路径优先用位置参数传入；下面这些环境变量主要用于数据路径和高级 eval 参数。

| 环境变量 | 作用 | 默认值 |
| --- | --- | --- |
| `SGLANG_DLLM_PIQA_DATA_PATH` | PIQA 本地数据路径 | 空，不跑 PIQA |
| `SGLANG_DLLM_GPQA_DATA_PATH` | GPQA 本地 CSV 路径 | 空，使用远端 `gpqa_diamond.csv` |
| `SGLANG_DLLM_GPQA_NUM_EXAMPLES` | GPQA 样本数；空表示全量 | 空 |
| `SGLANG_DLLM_PIQA_NUM_EXAMPLES` | PIQA 样本数 | `200` |
| `SGLANG_DLLM_NUM_THREADS` | eval 默认并发线程数 | `128` |
| `SGLANG_DLLM_SERVER_TIMEOUT` | 服务启动超时时间，秒 | `3600` |
| `ASCEND_RT_VISIBLE_DEVICES` | Ascend 可见设备 | `0,1,2,3,4,5,6,7` |

每个 eval 也支持单独覆盖，例如：

```bash
SGLANG_DLLM_GPQA_NUM_EXAMPLES=100 \
SGLANG_DLLM_GPQA_NUM_THREADS=64 \
SGLANG_DLLM_PIQA_NUM_EXAMPLES=500 \
SGLANG_DLLM_PIQA_NUM_THREADS=64 \
bash run_llada2_ascend_gpqa_piqa_csv.sh llada2_mini_gpqa_piqa.csv mini gpqa,piqa
```

## 修改测试矩阵

GSM8K + throughput 的 bs/tp/ep/dp 组合在：

```text
run_llada2_ascend_gsm8k_throughput_csv.sh
```

修改数组：

- `BF16_CASES`
- `INT8_CASES`

GPQA/PIQA 的 bs/tp/ep/dp 组合在：

```text
run_llada2_ascend_gpqa_piqa_csv.sh
```

修改数组：

- `BF16_CASES`
- `INT8_CASES`

数组每行格式都是：

```text
"bs tp ep dp moe_a2a_backend"
```

例如：

```bash
"4 8 1 1 none"
```

## 结果分析

GSM8K + throughput 量化收益分析脚本：

```bash
python3 analyze_llada2_gsm8k_quant_results.py \
  llada2_mini_gsm8k_bf16_int8.csv \
  --out-dir mini_bf16_int8_quant_analysis/analysis
```

GPQA/PIQA eval 量化收益分析脚本，会生成 summary CSV、report 和 SVG 图：

```bash
python3 analyze_llada2_eval_quant_results.py \
  results/mini_bf16_int8_quant_analysis_gpqa/llada2_mini_gpqa.csv \
  --out-dir results/mini_bf16_int8_quant_analysis_gpqa/analysis
```

flash 示例：

```bash
python3 analyze_llada2_gsm8k_quant_results.py \
  flash_bf16_int8_quant_analysis/llada2_flash_gsm8k_bf16_int8.csv \
  --out-dir flash_bf16_int8_quant_analysis/analysis
```
