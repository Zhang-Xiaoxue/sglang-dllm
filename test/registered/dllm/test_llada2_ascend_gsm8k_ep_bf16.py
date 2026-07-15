import csv
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

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

RESULT_DIR = THIS_FILE.parent
DEFAULT_CSV_PATH = RESULT_DIR / "llada2_mini_ascend_results.csv"
CSV_COLUMNS = [
    "run_name",
    "precision",
    "model_size",
    "bs",
    "tp",
    "ep",
    "dp",
    "moe_dp_size",
    "moe_a2a_backend",
    "max_running_requests",
    "model",
    "quantization",
    "bs_latency",
    "bs_tokens",
    "bs_acc_length",
    "bs_speed",
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
        _set_runtime_env("HCCL_BUFFSIZE", "HCCL_BUFFSIZE", "1024")
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
        cls.run_name = _env("RUN_NAME", cls.result_label)
        cls.bs_metrics = None
        cls.gsm8k_metrics = None

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
        _append_optional_arg(other_args, "--moe-a2a-backend", cls.moe_a2a_backend)
        _append_optional_arg(
            other_args,
            "--deepep-mode",
            _env("DEEPEP_MODE", "auto" if cls.moe_a2a_backend == "deepep" else ""),
        )
        if _env_bool("ENABLE_PREFILL_GRAPH", cls.moe_a2a_backend == "deepep"):
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
                "bs_latency": _metric(cls.bs_metrics, "latency"),
                "bs_tokens": _metric(cls.bs_metrics, "tokens"),
                "bs_acc_length": _metric(cls.bs_metrics, "acc_length"),
                "bs_speed": _metric(cls.bs_metrics, "speed"),
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

        self.assertGreater(metrics["accuracy"], float(_env("MIN_GSM8K_ACC", 0.68)))
        self.assertGreater(
            metrics["output_throughput"],
            float(_env("MIN_GSM8K_OUTPUT_THROUGHPUT", 10)),
        )

    def test_bs_speed(self):
        args = BenchArgs(
            port=int(self.base_url.split(":")[-1]),
            batch_size=self.bs,
            different_prompts=_env_bool("DIFFERENT_PROMPTS", self.bs > 1),
            max_new_tokens=_env_int("BS_MAX_NEW_TOKENS", 2048),
        )
        bs_metrics = send_one_prompt(args, return_metrics=True)
        type(self).bs_metrics = bs_metrics
        speed = bs_metrics["speed"]

        print(f"{speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_speed (llada2-{self.model_size}) {self.run_name}\n"
                f"{speed=:.2f} token/s\n"
            )
            self.assertGreater(speed, float(_env("MIN_BS_SPEED", 100)))


if __name__ == "__main__":
    unittest.main()
