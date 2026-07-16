#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

usage() {
  cat <<'EOF'
Usage:
  bash debug_sglang_LLaDA2_deepep_perf.sh [case] [mode]

Cases:
  none_ep1              No EP baseline: backend=none, TP=4, EP=1
  none_ep4              EP baseline without token A2A: backend=none, TP=4, EP=4
  deepep_graph          DeepEP with decode/full and prefill/tc_piecewise graphs
  deepep_eager          DeepEP with decode and prefill graphs disabled
  deepep_low_latency    Same as deepep_graph, but force low_latency mode
  deepep_int8           Same as deepep_graph, but use INT8 dispatch
  deepep_no_moe_split   Same as deepep_graph, but keep MoE inside one graph segment
  deepep_sbo            Experimental single-batch overlap case
  deepep_tbo            Experimental two-batch overlap case
  deepep_dp_attention   Experimental DP-attention case (LLaDA2 is not officially supported)
  matrix                Run a list of cases sequentially

Modes:
  serve      Launch the server in the foreground (default)
  run        Launch, warm up, send one fixed-length request, then stop
  profile    Same as run, with an NPU torch-profiler capture
  dry-run    Print the resolved environment and launch command only

Examples:
  bash debug_sglang_LLaDA2_deepep_perf.sh deepep_graph serve
  bash debug_sglang_LLaDA2_deepep_perf.sh deepep_graph profile
  bash debug_sglang_LLaDA2_deepep_perf.sh matrix run

Useful overrides:
  ASCEND_RT_VISIBLE_DEVICES=1,2,3,5
  SGLANG_DEEPEP_DEBUG_BATCH_SIZE=1
  SGLANG_DEEPEP_DEBUG_MAX_NEW_TOKENS=32
  SGLANG_DEEPEP_DEBUG_MAX_RUNNING_REQUESTS=4
  SGLANG_DEEPEP_DEBUG_CASES="none_ep1 none_ep4 deepep_eager deepep_graph"
  SGLANG_DEEPEP_DEBUG_EXPECTED_TEXT=460
  SGLANG_DEEPEP_DEBUG_VERBOSE_GRAPH=1
  SGLANG_DEEPEP_DEBUG_EXTRA_ARGS="--skip-server-warmup"
EOF
}

CASE_NAME=${1:-deepep_graph}
MODE=${2:-serve}

if [[ "${CASE_NAME}" == "-h" || "${CASE_NAME}" == "--help" ]]; then
  usage
  exit 0
fi

case "${MODE}" in
  serve|run|profile|dry-run) ;;
  *)
    echo "Unsupported mode: ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac

RUN_ID=${SGLANG_DEEPEP_DEBUG_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${SGLANG_DEEPEP_DEBUG_RUN_ROOT:-${REPO_ROOT}/logs/deepep_perf_debug/${RUN_ID}}
mkdir -p "${RUN_ROOT}"

if [[ "${CASE_NAME}" == "matrix" ]]; then
  if [[ "${MODE}" == "serve" ]]; then
    echo "matrix requires mode=run, profile, or dry-run" >&2
    exit 2
  fi

  read -r -a MATRIX_CASES <<< "${SGLANG_DEEPEP_DEBUG_CASES:-none_ep1 none_ep4 deepep_eager deepep_graph deepep_int8}"
  matrix_status=0
  echo "Run root: ${RUN_ROOT}"
  echo "Cases: ${MATRIX_CASES[*]}"
  for matrix_case in "${MATRIX_CASES[@]}"; do
    echo
    echo "===== ${matrix_case} (${MODE}) ====="
    if ! SGLANG_DEEPEP_DEBUG_RUN_ROOT="${RUN_ROOT}" \
      bash "${SCRIPT_PATH}" "${matrix_case}" "${MODE}"; then
      echo "Case failed: ${matrix_case}; continuing" >&2
      matrix_status=1
    fi
  done
  echo
  echo "Matrix summary: ${RUN_ROOT}/summary.csv"
  exit "${matrix_status}"
fi

MODEL_PATH=${SGLANG_DEEPEP_DEBUG_MODEL_PATH:-/data/public_models/LLaDA/LLaDA2.1-mini}
SERVED_MODEL_NAME=${SGLANG_DEEPEP_DEBUG_SERVED_MODEL_NAME:-LLaDA2.1-mini}
HOST=${SGLANG_DEEPEP_DEBUG_HOST:-0.0.0.0}
CLIENT_HOST=${SGLANG_DEEPEP_DEBUG_CLIENT_HOST:-127.0.0.1}
PORT=${SGLANG_DEEPEP_DEBUG_PORT:-8001}
TP_SIZE=${SGLANG_DEEPEP_DEBUG_TP:-4}
EP_SIZE=${TP_SIZE}
DP_SIZE=1
MOE_DP_SIZE=${SGLANG_DEEPEP_DEBUG_MOE_DP_SIZE:-1}
MOE_A2A_BACKEND=deepep
DEEPEP_MODE=${SGLANG_DEEPEP_DEBUG_DEEPEP_MODE:-auto}
DISPATCH_DTYPE=${SGLANG_DEEPEP_DEBUG_DISPATCH_DTYPE:-auto}
ENABLE_GRAPH=1
ENABLE_SBO=0
ENABLE_TBO=0
ENABLE_DP_ATTENTION=0
DISABLE_MOE_SPLIT=0
CASE_PURPOSE="DeepEP graph target"

case "${CASE_NAME}" in
  none_ep1)
    MOE_A2A_BACKEND=none
    EP_SIZE=1
    CASE_PURPOSE="No-EP baseline; isolates TP-sharded expert compute"
    ;;
  none_ep4)
    MOE_A2A_BACKEND=none
    EP_SIZE=${TP_SIZE}
    CASE_PURPOSE="EP baseline without token all-to-all"
    ;;
  deepep_graph)
    CASE_PURPOSE="DeepEP graph target"
    ;;
  deepep_eager)
    ENABLE_GRAPH=0
    CASE_PURPOSE="Isolate graph replay contribution"
    ;;
  deepep_low_latency)
    DEEPEP_MODE=low_latency
    CASE_PURPOSE="Verify AUTO versus forced low-latency dispatch"
    ;;
  deepep_int8)
    DISPATCH_DTYPE=int8
    CASE_PURPOSE="Estimate BF16 dispatch communication cost"
    ;;
  deepep_no_moe_split)
    DISABLE_MOE_SPLIT=1
    CASE_PURPOSE="Isolate piecewise MoE graph splitting"
    ;;
  deepep_sbo)
    ENABLE_SBO=1
    CASE_PURPOSE="Test whether LLaDA2 wires single-batch overlap"
    ;;
  deepep_tbo)
    ENABLE_TBO=1
    CASE_PURPOSE="Test two-batch overlap compatibility"
    ;;
  deepep_dp_attention)
    DP_SIZE=${TP_SIZE}
    ENABLE_DP_ATTENTION=1
    CASE_PURPOSE="Isolate TP-attention all-gather/reduce-scatter cost"
    ;;
  *)
    echo "Unsupported case: ${CASE_NAME}" >&2
    usage >&2
    exit 2
    ;;
esac

EP_SIZE=${SGLANG_DEEPEP_DEBUG_EP:-${EP_SIZE}}
DP_SIZE=${SGLANG_DEEPEP_DEBUG_DP:-${DP_SIZE}}
MEM_FRACTION_STATIC=${SGLANG_DEEPEP_DEBUG_MEM_FRACTION_STATIC:-0.90}
MAX_RUNNING_REQUESTS=${SGLANG_DEEPEP_DEBUG_MAX_RUNNING_REQUESTS:-4}
BATCH_SIZE=${SGLANG_DEEPEP_DEBUG_BATCH_SIZE:-1}
WARMUP_NEW_TOKENS=${SGLANG_DEEPEP_DEBUG_WARMUP_NEW_TOKENS:-32}
MAX_NEW_TOKENS=${SGLANG_DEEPEP_DEBUG_MAX_NEW_TOKENS:-32}
REQUEST_TIMEOUT=${SGLANG_DEEPEP_DEBUG_REQUEST_TIMEOUT:-900}
SERVER_TIMEOUT=${SGLANG_DEEPEP_DEBUG_SERVER_TIMEOUT:-1800}
PROFILE_STEPS=${SGLANG_DEEPEP_DEBUG_PROFILE_STEPS:-8}
PROFILE_TIMEOUT=${SGLANG_DEEPEP_DEBUG_PROFILE_TIMEOUT:-300}
PROFILE_START_DELAY=${SGLANG_DEEPEP_DEBUG_PROFILE_START_DELAY:-5}
ALGORITHM_CONFIG=${SGLANG_DEEPEP_DEBUG_ALGORITHM_CONFIG:-${REPO_ROOT}/test/registered/dllm/joint_threshold.yaml}
PROMPT=${SGLANG_DEEPEP_DEBUG_PROMPT:-Question: Eliza earns 10 dollars per hour for 40 hours and 1.2 times that rate for overtime. If she works 45 hours, what are her total earnings? Answer:}
EXPECTED_TEXT=${SGLANG_DEEPEP_DEBUG_EXPECTED_TEXT:-}

if (( EP_SIZE > TP_SIZE )); then
  echo "Invalid configuration: EP_SIZE=${EP_SIZE} > TP_SIZE=${TP_SIZE}" >&2
  exit 2
fi
if [[ "${MOE_A2A_BACKEND}" == "deepep" && "${EP_SIZE}" != "${TP_SIZE}" ]]; then
  echo "DeepEP requires EP_SIZE == TP_SIZE in this codebase" >&2
  exit 2
fi
if (( ENABLE_DP_ATTENTION == 1 )) && [[ "${DP_SIZE}" != "${TP_SIZE}" ]]; then
  echo "DP attention diagnostic requires DP_SIZE == TP_SIZE" >&2
  exit 2
fi

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-1,2,3,5}
IFS=',' read -r -a VISIBLE_DEVICES <<< "${ASCEND_RT_VISIBLE_DEVICES}"
visible_count=0
for visible_device in "${VISIBLE_DEVICES[@]}"; do
  visible_device=${visible_device//[[:space:]]/}
  [[ -n "${visible_device}" ]] && ((visible_count += 1))
done
if (( visible_count < TP_SIZE )); then
  echo "TP_SIZE=${TP_SIZE}, but only ${visible_count} NPU(s) are visible: ${ASCEND_RT_VISIBLE_DEVICES}" >&2
  exit 2
fi

CASE_DIR=${RUN_ROOT}/${CASE_NAME}
SERVER_LOG=${CASE_DIR}/server.log
DRIVER_LOG=${CASE_DIR}/driver.log
MANIFEST_FILE=${CASE_DIR}/manifest.txt
WARMUP_RESPONSE=${CASE_DIR}/warmup_response.json
WARMUP_METRICS=${CASE_DIR}/warmup_metrics.env
RESPONSE_FILE=${CASE_DIR}/response.json
METRICS_FILE=${CASE_DIR}/request_metrics.env
PROFILER_LOG=${CASE_DIR}/profiler_control.log
PROFILE_DIR=${CASE_DIR}/profile
SUMMARY_CSV=${RUN_ROOT}/summary.csv
mkdir -p "${CASE_DIR}" "${PROFILE_DIR}"

export PYTHONPATH=${REPO_ROOT}/python:${PYTHONPATH:-}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1024}
export SGLANG_TORCH_PROFILER_DIR=${PROFILE_DIR}
export SGLANG_PROFILE_WITH_STACK=${SGLANG_DEEPEP_DEBUG_PROFILE_WITH_STACK:-0}
export SGLANG_PROFILE_RECORD_SHAPES=${SGLANG_DEEPEP_DEBUG_PROFILE_RECORD_SHAPES:-1}
export SGLANG_LOG_FORWARD_ITERS=${SGLANG_DEEPEP_DEBUG_LOG_FORWARD_ITERS:-1}
export SGLANG_NPU_PIECEWISE_STATIC_INPUT_COPY=${SGLANG_NPU_PIECEWISE_STATIC_INPUT_COPY:-1}
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}

if (( ENABLE_GRAPH == 1 )); then
  export SGLANG_NPU_DLLM_DEEPEP_PREFILL_GRAPH=1
else
  export SGLANG_NPU_DLLM_DEEPEP_PREFILL_GRAPH=0
fi

if (( DISABLE_MOE_SPLIT == 1 )); then
  export SGLANG_NPU_DEEPEP_DISABLE_MOE_SPLIT=1
else
  unset SGLANG_NPU_DEEPEP_DISABLE_MOE_SPLIT || true
fi

if [[ "${SGLANG_DEEPEP_DEBUG_VERBOSE_GRAPH:-0}" == "1" ]]; then
  export SGLANG_NPU_DEEPEP_DEBUG_GRAPH_LOG=1
else
  unset SGLANG_NPU_DEEPEP_DEBUG_GRAPH_LOG || true
fi

unset SGLANG_NPU_DEEPEP_EAGER_POST_MOE_GRAPH || true
unset SGLANG_NPU_PIECEWISE_EAGER_GRAPH || true
unset SGLANG_NPU_PIECEWISE_EAGER_FROM_GRAPH || true
unset SGLANG_NPU_PIECEWISE_EAGER_LAST_GRAPH || true
unset SGLANG_NPU_PIECEWISE_SYNC_REPLAY || true

read -r -a CUDA_GRAPH_BS_DECODE <<< "${SGLANG_DEEPEP_DEBUG_CUDA_GRAPH_BS_DECODE:-1 2 4}"
read -r -a CUDA_GRAPH_BS_PREFILL <<< "${SGLANG_DEEPEP_DEBUG_CUDA_GRAPH_BS_PREFILL:-32}"

CMD=(
  python3 -m sglang.launch_server
  --model-path "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --device npu
  --attention-backend ascend
  --dtype bfloat16
  --kv-cache-dtype auto
  --trust-remote-code
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --enable-tokenizer-batch-encode
  --tp "${TP_SIZE}"
  --ep "${EP_SIZE}"
  --dp-size "${DP_SIZE}"
  --moe-dp-size "${MOE_DP_SIZE}"
  --moe-a2a-backend "${MOE_A2A_BACKEND}"
  --disable-radix-cache
  --dllm-algorithm JointThreshold
  --dllm-algorithm-config "${ALGORITHM_CONFIG}"
)

if [[ "${MOE_A2A_BACKEND}" == "deepep" ]]; then
  CMD+=(
    --deepep-mode "${DEEPEP_MODE}"
    --deepep-dispatcher-output-dtype "${DISPATCH_DTYPE}"
  )
fi

if (( ENABLE_GRAPH == 1 )); then
  CMD+=(
    --cuda-graph-backend-decode full
    --cuda-graph-bs-decode "${CUDA_GRAPH_BS_DECODE[@]}"
    --cuda-graph-backend-prefill tc_piecewise
    --cuda-graph-bs-prefill "${CUDA_GRAPH_BS_PREFILL[@]}"
    --cuda-graph-tc-compiler eager
  )
else
  CMD+=(
    --cuda-graph-backend-decode disabled
    --cuda-graph-backend-prefill disabled
  )
fi

(( ENABLE_SBO == 1 )) && CMD+=(--enable-single-batch-overlap)
(( ENABLE_TBO == 1 )) && CMD+=(--enable-two-batch-overlap)
(( ENABLE_DP_ATTENTION == 1 )) && CMD+=(--enable-dp-attention)
[[ "${SGLANG_DEEPEP_DEBUG_SKIP_SERVER_WARMUP:-1}" == "1" ]] && CMD+=(--skip-server-warmup)
[[ "${SGLANG_DEEPEP_DEBUG_ENABLE_CACHE_REPORT:-0}" == "1" ]] && CMD+=(--enable-cache-report)

if [[ -n "${SGLANG_DEEPEP_DEBUG_EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "${SGLANG_DEEPEP_DEBUG_EXTRA_ARGS}"
  CMD+=("${EXTRA_ARGS[@]}")
fi

print_command() {
  printf '%q ' "${CMD[@]}"
  printf '\n'
}

{
  echo "case=${CASE_NAME}"
  echo "purpose=${CASE_PURPOSE}"
  echo "mode=${MODE}"
  echo "model_path=${MODEL_PATH}"
  echo "visible_devices=${ASCEND_RT_VISIBLE_DEVICES}"
  echo "tp=${TP_SIZE}"
  echo "ep=${EP_SIZE}"
  echo "dp=${DP_SIZE}"
  echo "moe_dp_size=${MOE_DP_SIZE}"
  echo "backend=${MOE_A2A_BACKEND}"
  echo "deepep_mode=${DEEPEP_MODE}"
  echo "dispatch_dtype=${DISPATCH_DTYPE}"
  echo "decode_graph=$([[ ${ENABLE_GRAPH} == 1 ]] && echo full || echo disabled)"
  echo "prefill_graph=$([[ ${ENABLE_GRAPH} == 1 ]] && echo tc_piecewise || echo disabled)"
  echo "sbo=${ENABLE_SBO}"
  echo "tbo=${ENABLE_TBO}"
  echo "dp_attention=${ENABLE_DP_ATTENTION}"
  echo "disable_moe_split=${DISABLE_MOE_SPLIT}"
  echo "batch_size=${BATCH_SIZE}"
  echo "max_new_tokens=${MAX_NEW_TOKENS}"
  echo "profile_dir=${PROFILE_DIR}"
  printf 'command='
  print_command
} > "${MANIFEST_FILE}"

print_case_info() {
  echo "Case: ${CASE_NAME}"
  echo "Purpose: ${CASE_PURPOSE}"
  echo "Mode: ${MODE}"
  echo "Run directory: ${CASE_DIR}"
  echo "NPU devices: ${ASCEND_RT_VISIBLE_DEVICES}"
  echo "Parallelism: TP=${TP_SIZE} EP=${EP_SIZE} DP=${DP_SIZE} MOE_DP=${MOE_DP_SIZE}"
  echo "MoE: backend=${MOE_A2A_BACKEND} mode=${DEEPEP_MODE} dispatch_dtype=${DISPATCH_DTYPE}"
  echo "Graph: enabled=${ENABLE_GRAPH} decode_bs=${CUDA_GRAPH_BS_DECODE[*]} prefill_bs=${CUDA_GRAPH_BS_PREFILL[*]}"
  echo "Overlap: SBO=${ENABLE_SBO} TBO=${ENABLE_TBO} DP_attention=${ENABLE_DP_ATTENTION}"
  if (( ENABLE_DP_ATTENTION == 1 )); then
    echo "WARNING: DP attention is an isolation experiment; server_args does not list LLaDA2 as supported."
  fi
  echo "Launch command:"
  print_command
}

if [[ "${MODE}" == "dry-run" ]]; then
  print_case_info
  exit 0
fi

if [[ "${MODE}" == "serve" ]]; then
  exec > >(tee -a "${SERVER_LOG}") 2>&1
  print_case_info
  echo "Profiler output: ${PROFILE_DIR}"
  echo "For an automated fixed-length run, use:"
  echo "  bash ${SCRIPT_PATH} ${CASE_NAME} profile"
  exec "${CMD[@]}"
fi

exec > >(tee -a "${DRIVER_LOG}") 2>&1
print_case_info

SERVER_PID=""
PROFILER_PID=""

cleanup_server() {
  if [[ -z "${SERVER_PID}" ]]; then
    return
  fi
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    python3 -c \
      'import sys; from sglang.srt.utils import kill_process_tree; kill_process_tree(int(sys.argv[1]))' \
      "${SERVER_PID}" >/dev/null 2>&1 || kill "${SERVER_PID}" 2>/dev/null || true
  fi
  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
}

cleanup_all() {
  if [[ -n "${PROFILER_PID}" ]] && kill -0 "${PROFILER_PID}" 2>/dev/null; then
    kill "${PROFILER_PID}" 2>/dev/null || true
  fi
  cleanup_server
}
trap cleanup_all EXIT INT TERM

wait_for_server() {
  local deadline=$((SECONDS + SERVER_TIMEOUT))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "Server exited before becoming healthy" >&2
      tail -n 120 "${SERVER_LOG}" >&2 || true
      return 1
    fi
    if curl -fsS "http://${CLIENT_HOST}:${PORT}/health_generate" >/dev/null 2>&1; then
      echo "Server is ready"
      return 0
    fi
    sleep 5
  done
  echo "Server readiness timeout after ${SERVER_TIMEOUT}s" >&2
  tail -n 120 "${SERVER_LOG}" >&2 || true
  return 1
}

run_fixed_request() {
  local max_new_tokens=$1
  local response_file=$2
  local metrics_file=$3

  python3 - \
    "http://${CLIENT_HOST}:${PORT}" \
    "${BATCH_SIZE}" \
    "${max_new_tokens}" \
    "${REQUEST_TIMEOUT}" \
    "${PROMPT}" \
    "${EXPECTED_TEXT}" \
    "${response_file}" \
    "${metrics_file}" <<'PY'
import json
import sys
import time
from collections import Counter
from pathlib import Path

import requests

base_url = sys.argv[1]
batch_size = int(sys.argv[2])
max_new_tokens = int(sys.argv[3])
timeout = float(sys.argv[4])
prompt = sys.argv[5]
expected = sys.argv[6]
response_path = Path(sys.argv[7])
metrics_path = Path(sys.argv[8])

text = prompt if batch_size == 1 else [prompt] * batch_size
payload = {
    "text": text,
    "sampling_params": {
        "temperature": 0.0,
        "max_new_tokens": max_new_tokens,
        "ignore_eos": True,
        "sampling_seed": 0,
    },
    "stream": False,
}

start = time.perf_counter()
response = requests.post(f"{base_url}/generate", json=payload, timeout=timeout)
wall_time = time.perf_counter() - start
response_path.write_text(response.text, encoding="utf-8")
response.raise_for_status()
result = response.json()
items = result if isinstance(result, list) else [result]

total_tokens = sum(int(item.get("meta_info", {}).get("completion_tokens", 0)) for item in items)
server_latency = max(
    (float(item.get("meta_info", {}).get("e2e_latency", 0.0)) for item in items),
    default=0.0,
)
throughput = total_tokens / wall_time if wall_time > 0 else 0.0
combined_text = "\n".join(str(item.get("text", "")) for item in items)
expected_match = int(not expected or expected in combined_text)
non_whitespace = [char for char in combined_text if not char.isspace()]
max_char_ratio = (
    max(Counter(non_whitespace).values()) / len(non_whitespace)
    if non_whitespace
    else 0.0
)
degenerate_output = int(len(non_whitespace) >= 16 and max_char_ratio >= 0.90)

metrics_path.write_text(
    "\n".join(
        [
            f"wall_time={wall_time:.9f}",
            f"server_latency={server_latency:.9f}",
            f"total_tokens={total_tokens}",
            f"throughput={throughput:.9f}",
            f"expected_match={expected_match}",
            f"max_char_ratio={max_char_ratio:.9f}",
            f"degenerate_output={degenerate_output}",
        ]
    )
    + "\n",
    encoding="utf-8",
)

print(f"wall_time={wall_time:.3f}s")
print(f"server_latency={server_latency:.3f}s")
print(f"completion_tokens={total_tokens}")
print(f"throughput={throughput:.2f} token/s")
print(f"contains_expected({expected!r})={bool(expected_match)}")
print(f"max_char_ratio={max_char_ratio:.3f}")
print(f"degenerate_output={bool(degenerate_output)}")
print("output_preview:")
print(combined_text[:1000])
PY
}

echo "Starting server; log: ${SERVER_LOG}"
"${CMD[@]}" > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
wait_for_server

echo
echo "Warmup request: batch=${BATCH_SIZE}, max_new_tokens=${WARMUP_NEW_TOKENS}"
run_fixed_request "${WARMUP_NEW_TOKENS}" "${WARMUP_RESPONSE}" "${WARMUP_METRICS}"

if [[ "${MODE}" == "profile" ]]; then
  echo
  echo "Starting profiler: steps=${PROFILE_STEPS}, output=${PROFILE_DIR}"
  PYTHONUNBUFFERED=1 timeout "${PROFILE_TIMEOUT}" \
    python3 -m sglang.profiler \
      --url "http://${CLIENT_HOST}:${PORT}" \
      --output-dir "${PROFILE_DIR}" \
      --num-steps "${PROFILE_STEPS}" \
      --profile-prefix "${CASE_NAME}" \
      > "${PROFILER_LOG}" 2>&1 &
  PROFILER_PID=$!
  sleep "${PROFILE_START_DELAY}"
fi

echo
echo "Measured request: batch=${BATCH_SIZE}, max_new_tokens=${MAX_NEW_TOKENS}"
request_status=0
if run_fixed_request "${MAX_NEW_TOKENS}" "${RESPONSE_FILE}" "${METRICS_FILE}"; then
  request_status=0
else
  request_status=$?
  echo "Measured request failed with exit=${request_status}" >&2
fi

if [[ "${MODE}" == "profile" ]]; then
  profiler_status=0
  if wait "${PROFILER_PID}"; then
    profiler_status=0
  else
    profiler_status=$?
    echo "Profiler controller failed or timed out with exit=${profiler_status}" >&2
    curl -fsS -X POST "http://${CLIENT_HOST}:${PORT}/stop_profile" >/dev/null 2>&1 || true
  fi
  PROFILER_PID=""
  echo "Profiler controller log: ${PROFILER_LOG}"
fi

cleanup_server
trap - EXIT INT TERM

graph_true=$(awk '/npu graph: True/{count += 1} END{print count + 0}' "${SERVER_LOG}")
graph_false=$(awk '/npu graph: False/{count += 1} END{print count + 0}' "${SERVER_LOG}")
resolved_low_latency=$(awk '/resolved_mode=DeepEPMode.LOW_LATENCY/{count += 1} END{print count + 0}' "${SERVER_LOG}")
resolved_normal=$(awk '/resolved_mode=DeepEPMode.NORMAL/{count += 1} END{print count + 0}' "${SERVER_LOG}")

wall_time=""
server_latency=""
total_tokens=""
throughput=""
expected_match=""
max_char_ratio=""
degenerate_output=""
if [[ -f "${METRICS_FILE}" ]]; then
  # The generated file contains numeric assignments only.
  source "${METRICS_FILE}"
fi

if [[ ! -s "${SUMMARY_CSV}" ]]; then
  echo "case,backend,tp,ep,dp,graph,dispatch_dtype,sbo,tbo,batch,max_new_tokens,request_exit,wall_seconds,server_latency,total_tokens,tokens_per_second,expected_match,max_char_ratio,degenerate_output,graph_true,graph_false,resolved_low_latency,resolved_normal,case_dir" > "${SUMMARY_CSV}"
fi
printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
  "${CASE_NAME}" "${MOE_A2A_BACKEND}" "${TP_SIZE}" "${EP_SIZE}" "${DP_SIZE}" \
  "${ENABLE_GRAPH}" "${DISPATCH_DTYPE}" "${ENABLE_SBO}" "${ENABLE_TBO}" \
  "${BATCH_SIZE}" "${MAX_NEW_TOKENS}" "${request_status}" "${wall_time}" \
  "${server_latency}" "${total_tokens}" "${throughput}" "${expected_match}" \
  "${max_char_ratio}" "${degenerate_output}" "${graph_true}" "${graph_false}" \
  "${resolved_low_latency}" "${resolved_normal}" \
  "${CASE_DIR}" >> "${SUMMARY_CSV}"

echo
echo "Diagnostic summary"
echo "  graph_true=${graph_true} graph_false=${graph_false}"
echo "  resolved_low_latency=${resolved_low_latency} resolved_normal=${resolved_normal}"
echo "  request_exit=${request_status} tokens=${total_tokens} throughput=${throughput}"
echo "  expected_match=${expected_match} max_char_ratio=${max_char_ratio} degenerate_output=${degenerate_output}"
echo "  server_log=${SERVER_LOG}"
echo "  response=${RESPONSE_FILE}"
echo "  profile_dir=${PROFILE_DIR}"
echo "  summary_csv=${SUMMARY_CSV}"
echo
echo "Useful log filter:"
echo "  rg -n 'npu graph:|DeepEP graph debug|Prefill batch|Decode batch|ERROR|Traceback' ${SERVER_LOG}"

exit "${request_status}"
