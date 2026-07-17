import csv
import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import requests
from prometheus_client.parser import text_string_to_metric_families

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
os.environ["PYTHONPATH"] = f"{PYTHON_DIR}:{os.environ.get('PYTHONPATH', '')}"

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)
from sglang.utils import download_and_cache_file, read_jsonl

RESULT_DIR = THIS_FILE.parent
DEFAULT_CSV_PATH = RESULT_DIR / "llada2_mini_ascend_results.csv"
GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)
ORIGINAL_WORKLOAD = "original"
GSM8K_VARIABLE_WORKLOAD = "gsm8k_variable"
GSM8K_FIXED_WORKLOAD = "gsm8k_fixed"
SYNTHETIC_FIXED_WORKLOAD = "synthetic_fixed"
BENCHMARK_WORKLOADS = {
    GSM8K_VARIABLE_WORKLOAD,
    GSM8K_FIXED_WORKLOAD,
    SYNTHETIC_FIXED_WORKLOAD,
}
CSV_COLUMNS = [
    "run_name",
    "precision",
    "model_size",
    "bs",
    "tp",
    "ep",
    "dp",
    "moe_dp_size",
    "fixed_workload",
    "workload_mode",
    "benchmark_dataset",
    "benchmark_data_path",
    "input_length_mode",
    "output_length_mode",
    "input_tokens_per_request",
    "input_tokens_total",
    "input_tokens_min",
    "input_tokens_max",
    "input_tokens_mean",
    "input_tokens_p50",
    "input_tokens_p95",
    "fixed_input_length_ok",
    "output_tokens_per_request",
    "dllm_block_size",
    "bs_status",
    "bs_error",
    "bs_batch_wall_latency",
    "bs_batch_wall_speed",
    "fixed_output_length_ok",
    "graph_replay_passes",
    "graph_eager_passes",
    "graph_hit_rate",
    "graph_device_seconds",
    "graph_avg_replay_ms",
    "graph_tokens_per_replay",
    "graph_static_tokens",
    "graph_replay_throughput",
    "graph_metrics_valid",
    "graph_metrics_error",
    "graph_only_replay",
    "benchmark_valid",
    "profile_steps",
    "profile_artifact_count",
    "graph_profile_dir",
    "graph_profile_artifact",
    "decode_graph_bs",
    "prefill_graph_backend",
    "moe_a2a_backend",
    "max_running_requests",
    "model",
    "quantization",
    "bs_latency",
    "bs_tokens",
    "bs_acc_length",
    "bs_speed",
    "gsm8k_status",
    "gsm8k_error",
    "gsm8k_acc",
    "gsm8k_invalid",
    "gsm8k_latency",
    "gsm8k_output_throughput",
]


def _env(name, default=""):
    return os.environ.get(f"SGLANG_DLLM_{name}", str(default))


def _env_int(name, default):
    return int(_env(name, default))


def _env_bool(name, default=False):
    value = _env(name, "1" if default else "0").lower()
    return value in ("1", "true", "yes", "on")


def _env_list(name, default):
    return _env(name, default).replace(",", " ").split()


def _env_path(name, default):
    value = _env(name, default)
    if not value:
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((RESULT_DIR / path).resolve())


def _append_optional_arg(args, flag, value):
    value = str(value)
    if value and value.lower() not in ("none", "null"):
        args.extend([flag, value])


def _set_runtime_env(name, dllm_name, default=""):
    value = _env(dllm_name, default)
    if value:
        os.environ[name] = value
    else:
        os.environ.pop(name, None)


def _clear_debug_envs_unless_kept():
    if _env_bool("KEEP_DEBUG_ENVS", False):
        return
    for name in (
        "SGLANG_NPU_DEEPEP_DEBUG_GRAPH_LOG",
        "SGLANG_NPU_DEEPEP_EAGER_POST_MOE_GRAPH",
        "SGLANG_NPU_PIECEWISE_EAGER_GRAPH",
        "SGLANG_NPU_PIECEWISE_EAGER_FROM_GRAPH",
        "SGLANG_NPU_PIECEWISE_EAGER_LAST_GRAPH",
        "SGLANG_NPU_PIECEWISE_SYNC_REPLAY",
        "SGLANG_NPU_DEEPEP_DISABLE_MOE_SPLIT",
    ):
        os.environ.pop(name, None)


def _append_csv_row(row):
    csv_path = Path(_env("CSV", DEFAULT_CSV_PATH))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if need_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})
    return csv_path


def _metric(metrics, key):
    if not metrics:
        return ""
    value = metrics.get(key, "")
    if hasattr(value, "item"):
        value = value.item()
    return value


def _format_error(exc):
    return f"{type(exc).__name__}: {exc}".replace("\n", " ")[:1000]


def _percentile(values, fraction):
    values = sorted(values)
    if not values:
        return ""
    index = round((len(values) - 1) * fraction)
    return values[index]


def _add_input_token_stats(metrics):
    values = metrics.get("submitted_input_tokens_per_request") or metrics.get(
        "prompt_tokens_per_request", []
    )
    if not values:
        return
    metrics.update(
        {
            "input_tokens_total": sum(values),
            "input_tokens_min": min(values),
            "input_tokens_max": max(values),
            "input_tokens_mean": sum(values) / len(values),
            "input_tokens_p50": _percentile(values, 0.50),
            "input_tokens_p95": _percentile(values, 0.95),
        }
    )


def _load_gsm8k_prompts(data_path, offset=0):
    filename = data_path or download_and_cache_file(GSM8K_URL)
    lines = list(read_jsonl(filename))
    prompts = [
        f"Question: {line['question']}\nAnswer:"
        for line in lines
        if line.get("question")
    ]
    if offset >= len(prompts):
        raise ValueError(
            f"GSM8K benchmark offset {offset} exceeds {len(prompts)} prompts"
        )
    return prompts[offset:], str(filename)


def _fixed_gsm8k_input_ids(model, prompts, batch_size, input_len):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    input_ids = []
    for prompt in prompts:
        token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        if len(token_ids) >= input_len:
            input_ids.append(token_ids[:input_len])
        if len(input_ids) == batch_size:
            break
    if len(input_ids) != batch_size:
        raise ValueError(
            f"Need {batch_size} GSM8K prompts with at least {input_len} tokens, "
            f"found {len(input_ids)}"
        )
    return input_ids


def _read_replay_metrics(base_url):
    response = requests.get(f"{base_url}/metrics", timeout=30)
    response.raise_for_status()

    samples = []
    for family in text_string_to_metric_families(response.text):
        samples.extend(family.samples)

    def total(name, **labels):
        return sum(
            sample.value
            for sample in samples
            if sample.name == name
            and all(sample.labels.get(key) == value for key, value in labels.items())
        )

    graph_modes = ("graph_replay", "prefill_graph")
    graph_passes = sum(
        total("sglang:dllm_forward_execution_passes_total", mode=mode)
        for mode in graph_modes
    )
    eager_passes = total(
        "sglang:dllm_forward_execution_passes_total", mode="eager"
    )
    return {
        "forward_seconds": sum(
            total("sglang:dllm_forward_execution_seconds_total", mode=mode)
            for mode in graph_modes
        ),
        "graph_passes": graph_passes,
        "eager_passes": eager_passes,
    }


def _wait_for_replay_metrics(base_url, previous=None, timeout=10):
    deadline = time.monotonic() + timeout
    last = None
    stable_reads = 0
    while time.monotonic() < deadline:
        current = _read_replay_metrics(base_url)
        progressed = previous is None or (
            current["graph_passes"] + current["eager_passes"]
            > previous["graph_passes"] + previous["eager_passes"]
        )
        if progressed and current == last:
            stable_reads += 1
            if stable_reads >= 2:
                return current
        else:
            stable_reads = 0
        last = current
        time.sleep(0.1)

    raise RuntimeError(
        "Timed out waiting for replay metrics to settle: "
        f"previous={previous}, last={last}"
    )


def _flush_device_timer(base_url):
    response = requests.get(f"{base_url}/health_generate", timeout=30)
    response.raise_for_status()


def _completed_profile_artifacts(profile_dir):
    root = Path(profile_dir)
    if not root.is_dir():
        return []

    artifacts = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if (
            path.name in {"kernel_details.csv", "trace_view.json"}
            or path.name.endswith(".trace.json.gz")
        ):
            artifacts.append(path)
    return sorted(artifacts)


def _wait_for_profile_artifacts(profile_dir, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        artifacts = _completed_profile_artifacts(profile_dir)
        if artifacts:
            return artifacts
        time.sleep(0.5)

    raise RuntimeError(
        f"Timed out waiting for completed profiler artifacts in {profile_dir}"
    )


def _replay_metric_delta(before, after, bs, block_size):
    graph_passes = int(round(after["graph_passes"] - before["graph_passes"]))
    eager_passes = int(round(after["eager_passes"] - before["eager_passes"]))
    device_seconds = max(0.0, after["forward_seconds"] - before["forward_seconds"])
    total_passes = graph_passes + eager_passes
    graph_hit_rate = graph_passes / total_passes if total_passes else 0.0
    tokens_per_replay = bs * block_size
    static_tokens = graph_passes * tokens_per_replay
    valid = graph_passes > 0 and device_seconds > 0

    return {
        "graph_replay_passes": graph_passes,
        "graph_eager_passes": eager_passes,
        "graph_hit_rate": graph_hit_rate,
        "graph_device_seconds": device_seconds,
        "graph_avg_replay_ms": (
            device_seconds / graph_passes * 1000 if graph_passes else 0.0
        ),
        "graph_tokens_per_replay": tokens_per_replay,
        "graph_static_tokens": static_tokens,
        "graph_replay_throughput": static_tokens / device_seconds if valid else 0.0,
        "graph_metrics_valid": valid,
        "graph_only_replay": valid and eager_passes == 0,
    }



class TestLLaDA2(CustomTestCase):
    result_label = "bf16_ep"
    default_model_size = "mini"
    model_paths = {
        "mini": "/data/public_models/LLaDA/LLaDA2.1-mini",
        "flash": "/data/public_models/LLaDA/LLaDA2.1-flash",
    }
    default_tp = "4"
    default_ep = "4"
    default_dp = "1"
    default_bs = 1
    default_moe_a2a_backend = "deepep"
    quantization = ""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault(
            "ASCEND_RT_VISIBLE_DEVICES",
            _env("ASCEND_RT_VISIBLE_DEVICES", "1,2,3,5"),
        )
        # _set_runtime_env("HCCL_BUFFSIZE", "HCCL_BUFFSIZE", "1024")
        _set_runtime_env("SGLANG_DEBUG_GRAPH_CAN_RUN", "DEBUG_GRAPH_CAN_RUN", "0")
        _set_runtime_env(
            "SGLANG_NPU_DLLM_DEEPEP_PREFILL_GRAPH",
            "NPU_DLLM_DEEPEP_PREFILL_GRAPH",
            "1",
        )
        _set_runtime_env(
            "SGLANG_NPU_PIECEWISE_STATIC_INPUT_COPY",
            "NPU_PIECEWISE_STATIC_INPUT_COPY",
            "1",
        )
        _set_runtime_env(
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK",
            "DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK",
            "",
        )
        _set_runtime_env(
            "SGLANG_DEEPEP_BF16_DISPATCH",
            "DEEPEP_BF16_DISPATCH",
            "",
        )
        _clear_debug_envs_unless_kept()

        cls.model_size = _env("MODEL_SIZE", cls.default_model_size).lower()
        if cls.model_size not in cls.model_paths:
            raise ValueError(
                f"Unsupported SGLANG_DLLM_MODEL_SIZE={cls.model_size!r}; "
                "use mini or flash"
            )
        cls.model = _env("MODEL", cls.model_paths[cls.model_size])
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.bs = _env_int("BS", cls.default_bs)
        cls.tp = _env("TP", cls.default_tp)
        cls.ep = _env("EP", cls.default_ep)
        cls.dp = _env("DP", cls.default_dp)
        cls.moe_dp_size = _env("MOE_DP_SIZE", "1")
        cls.moe_a2a_backend = _env(
            "MOE_A2A_BACKEND", cls.default_moe_a2a_backend
        )
        cls.max_running_requests = _env("MAX_RUNNING_REQUESTS", cls.bs)
        workload_mode = _env("WORKLOAD_MODE", "").strip().lower()
        if not workload_mode:
            workload_mode = (
                SYNTHETIC_FIXED_WORKLOAD
                if _env_bool("FIXED_WORKLOAD", False)
                else ORIGINAL_WORKLOAD
            )
        valid_workloads = {ORIGINAL_WORKLOAD, *BENCHMARK_WORKLOADS}
        if workload_mode not in valid_workloads:
            raise ValueError(
                f"Unsupported SGLANG_DLLM_WORKLOAD_MODE={workload_mode!r}; "
                f"use one of {sorted(valid_workloads)}"
            )
        cls.workload_mode = workload_mode
        cls.benchmark_mode = workload_mode in BENCHMARK_WORKLOADS
        cls.fixed_workload = workload_mode in (
            GSM8K_FIXED_WORKLOAD,
            SYNTHETIC_FIXED_WORKLOAD,
        )
        cls.input_tokens_per_request = _env_int("BS_INPUT_LEN", 32)
        cls.output_tokens_per_request = _env_int("BS_OUTPUT_LEN", 512)
        cls.dllm_block_size = _env_int("BLOCK_SIZE", 32)
        cls.warmup_runs = _env_int("BS_WARMUP_RUNS", 1)
        cls.profile_graph = _env_bool("PROFILE_GRAPH", False)
        cls.profile_steps = _env_int("PROFILE_STEPS", 5)
        cls.profile_output_tokens = _env_int(
            "PROFILE_OUTPUT_TOKENS",
            cls.dllm_block_size * (cls.profile_steps + 1),
        )
        cls.profile_flush_timeout = _env_int("PROFILE_FLUSH_TIMEOUT", 600)
        cls.profile_root = _env_path("PROFILE_DIR", "profiles")
        cls.benchmark_data_offset = _env_int("BENCHMARK_DATA_OFFSET", 0)
        cls.gsm8k_data_path = os.environ.get("SGLANG_DLLM_GSM8K_DATA_PATH")
        if cls.benchmark_mode:
            if cls.moe_a2a_backend != "none":
                raise ValueError(
                    "Fixed-output benchmarks currently target "
                    "moe_a2a_backend=none, got "
                    f"{cls.moe_a2a_backend!r}"
                )
            if int(cls.max_running_requests) != cls.bs:
                raise ValueError(
                    "Fixed-BS benchmark requires max_running_requests == bs, got "
                    f"{cls.max_running_requests} != {cls.bs}"
                )
            os.environ["SGLANG_ENABLE_METRICS_DEVICE_TIMER"] = "1"
            os.environ["SGLANG_NPU_DLLM_DEEPEP_PREFILL_GRAPH"] = "0"
        if cls.profile_graph and not cls.fixed_workload:
            raise ValueError(
                "Kernel profiling requires a fixed-input workload; "
                f"got {cls.workload_mode!r}"
            )
        if cls.profile_graph:
            minimum_profile_tokens = cls.dllm_block_size * (cls.profile_steps + 1)
            if cls.profile_output_tokens < minimum_profile_tokens:
                raise ValueError(
                    "Kernel profiling output is too short to stop and flush: "
                    f"got {cls.profile_output_tokens}, need at least "
                    f"{minimum_profile_tokens} for {cls.profile_steps} steps"
                )
            if cls.profile_output_tokens % cls.dllm_block_size != 0:
                raise ValueError(
                    "Kernel profiling output must be a multiple of block size: "
                    f"{cls.profile_output_tokens} % {cls.dllm_block_size} != 0"
                )
        cls.run_name = _env("RUN_NAME", cls.result_label)
        cls.bs_metrics = None
        cls.gsm8k_metrics = None
        cls.bs_status = "not_run"
        cls.bs_error = ""
        cls.gsm8k_status = "not_run"
        cls.gsm8k_error = ""
        cls.benchmark_request_kwargs = None
        cls.benchmark_data_path = ""
        cls.benchmark_dataset = (
            "gsm8k"
            if cls.workload_mode
            in (GSM8K_VARIABLE_WORKLOAD, GSM8K_FIXED_WORKLOAD)
            else "synthetic"
            if cls.workload_mode == SYNTHETIC_FIXED_WORKLOAD
            else ""
        )

        other_args = [
            "--trust-remote-code",
            "--device",
            "npu",
            "--dtype",
            "bfloat16",
            "--disable-radix-cache",
            "--mem-fraction-static",
            _env("MEM_FRACTION_STATIC", "0.90"),
            "--max-running-requests",
            str(cls.max_running_requests),
            "--attention-backend",
            "ascend",
            "--tp",
            cls.tp,
            "--ep",
            cls.ep,
            "--dp-size",
            cls.dp,
            "--moe-dp-size",
            cls.moe_dp_size,
        ]
        if cls.fixed_workload:
            other_args.extend(
                [
                    "--cuda-graph-backend-prefill",
                    "disabled",
                    "--cuda-graph-backend-decode",
                    "full",
                    "--cuda-graph-bs-decode",
                    str(cls.bs),
                    "--cuda-graph-max-bs-decode",
                    str(cls.bs),
                    "--disable-cuda-graph-padding",
                    "--enable-metrics",
                    "--moe-a2a-backend",
                    cls.moe_a2a_backend,
                ]
            )
        elif cls.benchmark_mode:
            other_args.extend(
                [
                    "--disable-cuda-graph",
                    "--enable-metrics",
                    "--moe-a2a-backend",
                    cls.moe_a2a_backend,
                ]
            )
        else:
            _append_optional_arg(other_args, "--moe-a2a-backend", cls.moe_a2a_backend)
        _append_optional_arg(
            other_args,
            "--deepep-mode",
            _env("DEEPEP_MODE", "auto" if cls.moe_a2a_backend == "deepep" else ""),
        )
        if not cls.benchmark_mode and _env_bool(
            "ENABLE_PREFILL_GRAPH", cls.moe_a2a_backend == "deepep"
        ):
            _append_optional_arg(
                other_args,
                "--cuda-graph-backend-prefill",
                _env("CUDA_GRAPH_BACKEND_PREFILL", "tc_piecewise"),
            )
            prefill_bs = _env_list("CUDA_GRAPH_BS_PREFILL", "32")
            if prefill_bs:
                other_args.extend(["--cuda-graph-bs-prefill", *prefill_bs])
            _append_optional_arg(
                other_args,
                "--cuda-graph-tc-compiler",
                _env("CUDA_GRAPH_TC_COMPILER", "eager"),
            )
        if _env_bool("ENABLE_PIECEWISE_GRAPH", False):
            other_args.extend(
                [
                    "--piecewise-cuda-graph-compiler",
                    _env("PIECEWISE_CUDA_GRAPH_COMPILER", "eager"),
                ]
            )
            tokens = _env_list("PIECEWISE_CUDA_GRAPH_TOKENS", "32,64,96,128")
            other_args.extend(["--piecewise-cuda-graph-tokens", *tokens])
            other_args.append("--enforce-piecewise-cuda-graph")
        other_args.extend(
            [
                "--dllm-algorithm",
                _env("ALGORITHM", "JointThreshold"),
                "--dllm-algorithm-config",
                _env_path("ALGORITHM_CONFIG", "joint_threshold.yaml"),
            ]
        )

        print("LLaDA2 EP bf16 launch args:", " ".join(map(str, other_args)))
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=_env_int("SERVER_TIMEOUT", 3600),
            other_args=other_args,
            device="npu",
        )

    @classmethod
    def tearDownClass(cls):
        try:
            kill_process_tree(cls.process.pid)
        finally:
            cls.write_csv_result()

    @classmethod
    def write_csv_result(cls):
        csv_path = _append_csv_row(
            {
                "run_name": cls.run_name,
                "precision": cls.result_label,
                "model_size": cls.model_size,
                "bs": cls.bs,
                "tp": cls.tp,
                "ep": cls.ep,
                "dp": cls.dp,
                "moe_dp_size": cls.moe_dp_size,
                "moe_a2a_backend": cls.moe_a2a_backend,
                "max_running_requests": cls.max_running_requests,
                "model": cls.model,
                "quantization": cls.quantization,
                "fixed_workload": cls.fixed_workload,
                "workload_mode": cls.workload_mode,
                "benchmark_dataset": cls.benchmark_dataset,
                "benchmark_data_path": cls.benchmark_data_path,
                "input_length_mode": (
                    "variable"
                    if cls.workload_mode == GSM8K_VARIABLE_WORKLOAD
                    else "fixed"
                    if cls.workload_mode
                    in (GSM8K_FIXED_WORKLOAD, SYNTHETIC_FIXED_WORKLOAD)
                    else "original"
                ),
                "output_length_mode": (
                    "fixed" if cls.benchmark_mode else "original"
                ),
                "input_tokens_per_request": (
                    cls.input_tokens_per_request
                    if cls.workload_mode
                    in (GSM8K_FIXED_WORKLOAD, SYNTHETIC_FIXED_WORKLOAD)
                    else ""
                ),
                "input_tokens_total": _metric(
                    cls.bs_metrics, "input_tokens_total"
                ),
                "input_tokens_min": _metric(cls.bs_metrics, "input_tokens_min"),
                "input_tokens_max": _metric(cls.bs_metrics, "input_tokens_max"),
                "input_tokens_mean": _metric(cls.bs_metrics, "input_tokens_mean"),
                "input_tokens_p50": _metric(cls.bs_metrics, "input_tokens_p50"),
                "input_tokens_p95": _metric(cls.bs_metrics, "input_tokens_p95"),
                "fixed_input_length_ok": _metric(
                    cls.bs_metrics, "fixed_input_length_ok"
                ),
                "output_tokens_per_request": (
                    (
                        cls.profile_output_tokens
                        if cls.profile_graph
                        else cls.output_tokens_per_request
                    )
                    if cls.benchmark_mode
                    else ""
                ),
                "dllm_block_size": cls.dllm_block_size,
                "bs_status": cls.bs_status,
                "bs_error": cls.bs_error,
                "bs_batch_wall_latency": _metric(
                    cls.bs_metrics, "batch_wall_latency"
                ),
                "bs_batch_wall_speed": _metric(cls.bs_metrics, "batch_wall_speed"),
                "fixed_output_length_ok": _metric(
                    cls.bs_metrics, "fixed_output_length_ok"
                ),
                "graph_replay_passes": _metric(
                    cls.bs_metrics, "graph_replay_passes"
                ),
                "graph_eager_passes": _metric(cls.bs_metrics, "graph_eager_passes"),
                "graph_hit_rate": _metric(cls.bs_metrics, "graph_hit_rate"),
                "graph_device_seconds": _metric(
                    cls.bs_metrics, "graph_device_seconds"
                ),
                "graph_avg_replay_ms": _metric(
                    cls.bs_metrics, "graph_avg_replay_ms"
                ),
                "graph_tokens_per_replay": _metric(
                    cls.bs_metrics, "graph_tokens_per_replay"
                ),
                "graph_static_tokens": _metric(cls.bs_metrics, "graph_static_tokens"),
                "graph_replay_throughput": _metric(
                    cls.bs_metrics, "graph_replay_throughput"
                ),
                "graph_metrics_valid": _metric(cls.bs_metrics, "graph_metrics_valid"),
                "graph_metrics_error": _metric(cls.bs_metrics, "graph_metrics_error"),
                "graph_only_replay": _metric(
                    cls.bs_metrics, "graph_only_replay"
                ),
                "benchmark_valid": _metric(cls.bs_metrics, "benchmark_valid"),
                "profile_steps": cls.profile_steps if cls.profile_graph else "",
                "profile_artifact_count": _metric(
                    cls.bs_metrics, "profile_artifact_count"
                ),
                "graph_profile_dir": _metric(cls.bs_metrics, "profile_dir"),
                "graph_profile_artifact": _metric(
                    cls.bs_metrics, "profile_artifact"
                ),
                "decode_graph_bs": cls.bs if cls.fixed_workload else "",
                "prefill_graph_backend": (
                    "disabled"
                    if cls.benchmark_mode
                    else _env("CUDA_GRAPH_BACKEND_PREFILL")
                ),
                "bs_latency": _metric(cls.bs_metrics, "latency"),
                "bs_tokens": _metric(cls.bs_metrics, "tokens"),
                "bs_acc_length": _metric(cls.bs_metrics, "acc_length"),
                "bs_speed": _metric(cls.bs_metrics, "speed"),
                "gsm8k_status": cls.gsm8k_status,
                "gsm8k_error": cls.gsm8k_error,
                "gsm8k_acc": _metric(cls.gsm8k_metrics, "accuracy"),
                "gsm8k_invalid": _metric(cls.gsm8k_metrics, "invalid"),
                "gsm8k_latency": _metric(cls.gsm8k_metrics, "latency"),
                "gsm8k_output_throughput": _metric(
                    cls.gsm8k_metrics, "output_throughput"
                ),
            }
        )
        print(f"LLaDA2 mini Ascend CSV updated: {csv_path}")

    def test_gsm8k(self):
        if self.benchmark_mode:
            type(self).gsm8k_status = "skipped_fixed_output_benchmark"
            self.skipTest("Fixed-output benchmarks do not run GSM8K accuracy")

        type(self).gsm8k_status = "running"
        try:
            args = SimpleNamespace(
                num_shots=_env_int("GSM8K_NUM_SHOTS", 0),
                data_path=os.environ.get("SGLANG_DLLM_GSM8K_DATA_PATH"),
                num_questions=_env_int("GSM8K_NUM_QUESTIONS", 200),
                max_new_tokens=_env_int("GSM8K_MAX_NEW_TOKENS", 512),
                parallel=_env_int("GSM8K_PARALLEL", 128),
                host="http://127.0.0.1",
                port=int(self.base_url.split(":")[-1]),
            )
            metrics = run_eval_few_shot_gsm8k(args)
            print(f"{metrics=}")
            type(self).gsm8k_metrics = metrics

            self.assertGreater(
                metrics["accuracy"], float(_env("MIN_GSM8K_ACC", 0.68))
            )
            self.assertGreater(
                metrics["output_throughput"],
                float(_env("MIN_GSM8K_OUTPUT_THROUGHPUT", 10)),
            )
        except Exception as exc:
            type(self).gsm8k_status = "failed"
            type(self).gsm8k_error = _format_error(exc)
            raise
        else:
            type(self).gsm8k_status = "passed"

    def _make_bs_bench_args(self, profile=False):
        if not self.benchmark_mode:
            return BenchArgs(
                port=int(self.base_url.split(":")[-1]),
                batch_size=self.bs,
                different_prompts=_env_bool("DIFFERENT_PROMPTS", self.bs > 1),
                max_new_tokens=_env_int("BS_MAX_NEW_TOKENS", 2048),
            )

        output_tokens = (
            self.profile_output_tokens
            if self.profile_graph
            else self.output_tokens_per_request
        )
        # The scheduler checks its stop target before executing that batch.
        profile_api_steps = self.profile_steps + 1 if profile else self.profile_steps
        return BenchArgs(
            port=int(self.base_url.split(":")[-1]),
            batch_size=self.bs,
            different_prompts=True,
            random_input_len=(
                self.input_tokens_per_request
                if self.workload_mode == SYNTHETIC_FIXED_WORKLOAD
                else None
            ),
            seed=_env_int("BS_SEED", 12345),
            max_new_tokens=output_tokens,
            min_new_tokens=output_tokens,
            ignore_eos=True,
            stop=None,
            profile=profile,
            profile_steps=profile_api_steps,
            profile_prefix=self.run_name,
            profile_output_dir=self.profile_root,
        )

    def _get_benchmark_request_kwargs(self):
        if type(self).benchmark_request_kwargs is not None:
            return type(self).benchmark_request_kwargs

        if self.workload_mode == SYNTHETIC_FIXED_WORKLOAD:
            request_kwargs = {}
        else:
            prompts, data_path = _load_gsm8k_prompts(
                self.gsm8k_data_path,
                offset=self.benchmark_data_offset,
            )
            type(self).benchmark_data_path = data_path
            if self.workload_mode == GSM8K_VARIABLE_WORKLOAD:
                if len(prompts) < self.bs:
                    raise ValueError(
                        f"Need {self.bs} GSM8K prompts, found {len(prompts)}"
                    )
                request_kwargs = {"prompts": prompts[: self.bs]}
            elif self.workload_mode == GSM8K_FIXED_WORKLOAD:
                request_kwargs = {
                    "input_ids": _fixed_gsm8k_input_ids(
                        self.model,
                        prompts,
                        batch_size=self.bs,
                        input_len=self.input_tokens_per_request,
                    )
                }
            else:
                raise AssertionError(f"Unexpected workload mode {self.workload_mode}")

        type(self).benchmark_request_kwargs = request_kwargs
        return request_kwargs

    def _fixed_input_length_ok(self, metrics):
        submitted_lengths = metrics.get("submitted_input_tokens_per_request", [])
        return len(submitted_lengths) == self.bs and all(
            length == self.input_tokens_per_request for length in submitted_lengths
        )

    def _run_kernel_profile_case(self, request_kwargs):
        args = self._make_bs_bench_args()
        for warmup_index in range(self.warmup_runs):
            warmup_metrics = send_one_prompt(
                args,
                label=f"{self.workload_mode} short warmup {warmup_index + 1}",
                print_output=False,
                return_metrics=True,
                **request_kwargs,
            )
            self.assertTrue(
                warmup_metrics["fixed_output_length_ok"],
                "Profiler warmup did not return exactly "
                f"{self.profile_output_tokens} tokens per request",
            )

        print(
            f"Capturing {self.profile_steps} complete scheduler steps; "
            f"profiler stop target={self.profile_steps + 1}"
        )
        profile_metrics = send_one_prompt(
            self._make_bs_bench_args(profile=True),
            label=f"{self.workload_mode} kernel profile batch",
            print_output=False,
            return_metrics=True,
            **request_kwargs,
        )
        type(self).bs_metrics = profile_metrics
        _add_input_token_stats(profile_metrics)
        profile_metrics["fixed_input_length_ok"] = self._fixed_input_length_ok(
            profile_metrics
        )

        profile_dir = profile_metrics.get("profile_dir")
        self.assertTrue(profile_dir, "Profiler did not return an output directory")
        artifacts = _wait_for_profile_artifacts(
            profile_dir, timeout=self.profile_flush_timeout
        )
        profile_metrics["profile_artifact_count"] = len(artifacts)
        profile_metrics["profile_artifact"] = str(artifacts[0])
        profile_metrics["graph_metrics_error"] = ""
        profile_metrics["benchmark_valid"] = (
            profile_metrics["fixed_output_length_ok"]
            and profile_metrics["fixed_input_length_ok"]
            and bool(artifacts)
        )
        print(
            "Profiler artifacts are ready; finishing this setting: "
            f"count={len(artifacts)}, representative={artifacts[0]}"
        )
        return profile_metrics

    def test_bs_speed(self):
        type(self).bs_status = "running"
        try:
            args = self._make_bs_bench_args()
            if self.benchmark_mode:
                request_kwargs = self._get_benchmark_request_kwargs()
                requires_fixed_input = self.workload_mode in (
                    GSM8K_FIXED_WORKLOAD,
                    SYNTHETIC_FIXED_WORKLOAD,
                )
                if self.profile_graph:
                    bs_metrics = self._run_kernel_profile_case(request_kwargs)
                else:
                    for warmup_index in range(self.warmup_runs):
                        warmup_metrics = send_one_prompt(
                            args,
                            label=(
                                f"{self.workload_mode} warmup {warmup_index + 1}"
                            ),
                            print_output=False,
                            return_metrics=True,
                            **request_kwargs,
                        )
                        self.assertTrue(
                            warmup_metrics["fixed_output_length_ok"],
                            "Warmup did not return exactly "
                            f"{self.output_tokens_per_request} tokens per request",
                        )

                    _flush_device_timer(self.base_url)
                    before = _wait_for_replay_metrics(self.base_url)
                    bs_metrics = send_one_prompt(
                        args,
                        label=f"{self.workload_mode} measured batch",
                        print_output=False,
                        return_metrics=True,
                        **request_kwargs,
                    )
                    type(self).bs_metrics = bs_metrics
                    _add_input_token_stats(bs_metrics)
                    bs_metrics["fixed_input_length_ok"] = (
                        self._fixed_input_length_ok(bs_metrics)
                        if requires_fixed_input
                        else ""
                    )

                    _flush_device_timer(self.base_url)
                    after = _wait_for_replay_metrics(self.base_url, previous=before)
                    bs_metrics.update(
                        _replay_metric_delta(
                            before,
                            after,
                            bs=self.bs,
                            block_size=self.dllm_block_size,
                        )
                    )
                    bs_metrics["graph_metrics_error"] = ""
                    execution_mode_valid = (
                        bs_metrics["graph_metrics_valid"]
                        if requires_fixed_input
                        else (
                            bs_metrics["graph_replay_passes"] == 0
                            and bs_metrics["graph_eager_passes"] > 0
                        )
                    )
                    bs_metrics["benchmark_valid"] = (
                        bs_metrics["fixed_output_length_ok"]
                        and execution_mode_valid
                        and (
                            not requires_fixed_input
                            or bs_metrics["fixed_input_length_ok"]
                        )
                    )
            else:
                bs_metrics = send_one_prompt(args, return_metrics=True)
                type(self).bs_metrics = bs_metrics
                _add_input_token_stats(bs_metrics)

            speed = bs_metrics["speed"]
            print(f"{speed=:.2f}")

            if self.benchmark_mode:
                output_tokens = (
                    self.profile_output_tokens
                    if self.profile_graph
                    else self.output_tokens_per_request
                )
                expected_tokens = self.bs * output_tokens
                if self.profile_graph:
                    print(
                        f"{self.workload_mode} profile: "
                        f"steps={self.profile_steps}, output={output_tokens}, "
                        f"dir={bs_metrics['profile_dir']}"
                    )
                else:
                    print(
                        f"{self.workload_mode} graph metrics: "
                        f"passes={bs_metrics['graph_replay_passes']}, "
                        f"eager={bs_metrics['graph_eager_passes']}, "
                        f"hit_rate={bs_metrics['graph_hit_rate']:.2%}, "
                        f"avg_replay={bs_metrics['graph_avg_replay_ms']:.3f} ms, "
                        "static_throughput="
                        f"{bs_metrics['graph_replay_throughput']:.2f} token/s"
                    )
                self.assertEqual(bs_metrics["tokens"], expected_tokens)
                self.assertTrue(bs_metrics["fixed_output_length_ok"])
                if self.fixed_workload:
                    if self.profile_graph:
                        self.assertTrue(bs_metrics.get("profile_dir"))
                        self.assertGreater(
                            bs_metrics.get("profile_artifact_count", 0), 0
                        )
                    else:
                        self.assertTrue(
                            bs_metrics["graph_metrics_valid"],
                            "Expected the measured DLLM_EXTEND request to "
                            f"replay the exact-BS graph, got {bs_metrics}",
                        )
                    self.assertTrue(bs_metrics["fixed_input_length_ok"])
                else:
                    self.assertEqual(bs_metrics["graph_replay_passes"], 0)
                    self.assertGreater(bs_metrics["graph_eager_passes"], 0)

            if is_in_ci() and not self.profile_graph:
                write_github_step_summary(
                    f"### test_bs_speed (llada2-{self.model_size}) {self.run_name}\n"
                    f"{speed=:.2f} token/s\n"
                )
                self.assertGreater(speed, float(_env("MIN_BS_SPEED", 100)))
        except Exception as exc:
            type(self).bs_status = "failed"
            type(self).bs_error = _format_error(exc)
            raise
        else:
            type(self).bs_status = "passed"


if __name__ == "__main__":
    unittest.main()
