#!/usr/bin/env bash
set -euo pipefail

# ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Run LLaDA2 Ascend regression tests and write CSV results.
# Usage:
#   ./test.sh
#   ./test.sh mini|flash
#   ./test.sh mini|flash gsm8k|gpqa|piqa|all
#   ./test.sh <csv> mini|flash gsm8k|gpqa|piqa|gpqa,piqa|gsm8k,gpqa|all

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

is_model_size() {
  [[ "$1" == "mini" || "$1" == "flash" ]]
}

usage() {
  cat >&2 <<'EOF'
Usage:
  ./test.sh
  ./test.sh mini|flash
  ./test.sh mini|flash gsm8k|gpqa|piqa|all
  ./test.sh <csv> mini|flash gsm8k|gpqa|piqa|gpqa,piqa|gsm8k,gpqa|all

Examples:
  bash test.sh flash
  bash test.sh flash gsm8k
  bash test.sh flash gpqa
  bash test.sh results/llada2_flash_gsm8k.csv flash gsm8k
  bash test.sh results/llada2_flash_gpqa.csv flash gpqa
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODEL_SIZE=mini
CSV_ARG=
TEST_NAMES=all

case "$#" in
  0)
    ;;
  1)
    if is_model_size "$1"; then
      MODEL_SIZE=$1
    else
      CSV_ARG=$1
    fi
    ;;
  2)
    if is_model_size "$1"; then
      MODEL_SIZE=$1
      TEST_NAMES=$2
    else
      CSV_ARG=$1
      MODEL_SIZE=$2
    fi
    ;;
  *)
    CSV_ARG=$1
    MODEL_SIZE=$2
    TEST_NAMES=$3
    ;;
esac

if ! is_model_size "$MODEL_SIZE"; then
  echo "MODEL_SIZE must be mini or flash, got: $MODEL_SIZE" >&2
  usage
  exit 1
fi

TEST_NAMES=${TEST_NAMES// /}
TEST_NAMES=${TEST_NAMES,,}
if [[ -z "$TEST_NAMES" ]]; then
  TEST_NAMES=all
fi

if [[ "$TEST_NAMES" == "all" ]]; then
  TEST_NAMES=gsm8k,gpqa
  if [[ -n "${SGLANG_DLLM_PIQA_DATA_PATH:-}" ]]; then
    TEST_NAMES=${TEST_NAMES},piqa
  fi
fi

contains_test() {
  local name=$1
  [[ ",$TEST_NAMES," == *",$name,"* ]]
}

DO_GSM8K=0
if contains_test gsm8k || contains_test bench || contains_test throughput; then
  DO_GSM8K=1
fi

EVAL_NAMES=
for name in ${TEST_NAMES//,/ }; do
  case "$name" in
    gsm8k|bench|throughput)
      ;;
    gpqa|piqa)
      if [[ -z "$EVAL_NAMES" ]]; then
        EVAL_NAMES=$name
      else
        EVAL_NAMES=${EVAL_NAMES},$name
      fi
      ;;
    *)
      echo "Unknown test name: $name" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$DO_GSM8K" -eq 0 && -z "$EVAL_NAMES" ]]; then
  echo "No valid test selected: $TEST_NAMES" >&2
  usage
  exit 1
fi

if contains_test piqa && [[ -z "${SGLANG_DLLM_PIQA_DATA_PATH:-}" ]]; then
  echo "PIQA was requested but SGLANG_DLLM_PIQA_DATA_PATH is not set." >&2
  echo "Set SGLANG_DLLM_PIQA_DATA_PATH=/path/to/piqa/validation.jsonl or remove piqa from test names." >&2
  exit 1
fi

BENCH_CSV=
EVAL_CSV=
if [[ -n "$CSV_ARG" ]]; then
  if [[ "$DO_GSM8K" -eq 1 && -n "$EVAL_NAMES" ]]; then
    csv_base=${CSV_ARG%.csv}
    BENCH_CSV="${csv_base}_gsm8k.csv"
    EVAL_CSV="${csv_base}_${EVAL_NAMES//,/_}.csv"
  elif [[ "$DO_GSM8K" -eq 1 ]]; then
    BENCH_CSV=$CSV_ARG
  else
    EVAL_CSV=$CSV_ARG
  fi
fi

BENCH_CSV=${BENCH_CSV:-"llada2_${MODEL_SIZE}_gsm8k_bf16_int8.csv"}
EVAL_CSV=${EVAL_CSV:-"llada2_${MODEL_SIZE}_${EVAL_NAMES//,/_}.csv"}

echo "Run dir: $SCRIPT_DIR"
echo "Model size: $MODEL_SIZE"
echo "Tests: $TEST_NAMES"

if [[ "$DO_GSM8K" -eq 1 ]]; then
  echo
  echo "===== GSM8K + throughput benchmark ====="
  echo "Bench CSV: $BENCH_CSV"
  SGLANG_DLLM_MODEL_SIZE="$MODEL_SIZE" \
    bash run_llada2_ascend_gsm8k_throughput_csv.sh "$BENCH_CSV" "$MODEL_SIZE"
fi

if [[ -n "$EVAL_NAMES" ]]; then
  echo
  echo "===== Accuracy eval: $EVAL_NAMES ====="
  echo "Eval CSV: $EVAL_CSV"
  SGLANG_DLLM_MODEL_SIZE="$MODEL_SIZE" \
  SGLANG_DLLM_EVAL_NAMES="$EVAL_NAMES" \
    bash run_llada2_ascend_gpqa_piqa_csv.sh "$EVAL_CSV" "$MODEL_SIZE" "$EVAL_NAMES"
fi

echo
echo "Done."
