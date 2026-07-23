import csv
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

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

RESULT_DIR = Path(__file__).resolve().parent
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
    "deepep_dispatch_dtype",
    "random_seed",
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


def _env_list(name):
    return _env(name).replace(",", " ").split()


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


def _append_csv_row(row):
    csv_path = Path(_env("CSV", DEFAULT_CSV_PATH))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        if need_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})
    return csv_path


def _metric(metrics, key):
    if not metrics:
        return ""
    value = metrics.get(key, "")
    return value.item() if hasattr(value, "item") else value


class TestLLaDA2(CustomTestCase):
    result_label = "bf16_ep"
    model_paths = {
        "mini": "/data/public_models/LLaDA/LLaDA2.1-mini",
        "flash": "/data/public_models/LLaDA/LLaDA2.1-flash",
    }
    quantization = ""

    @classmethod
    def setUpClass(cls):
        cls.model_size = _env("MODEL_SIZE", "mini").lower()
        if cls.model_size not in cls.model_paths:
            raise ValueError(
                f"Unsupported SGLANG_DLLM_MODEL_SIZE={cls.model_size!r}; "
                "use mini or flash"
            )

        cls.model = _env("MODEL", cls.model_paths[cls.model_size])
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.bs = _env_int("BS", 1)
        cls.tp = _env("TP", 2)
        cls.ep = _env("EP", cls.tp)
        cls.dp = _env("DP", 1)
        cls.moe_dp_size = _env("MOE_DP_SIZE", 1)
        cls.moe_a2a_backend = _env("MOE_A2A_BACKEND", "none")
        cls.deepep_dispatch_dtype = _env("DEEPEP_DISPATCH_DTYPE", "auto")
        cls.random_seed = _env("RANDOM_SEED", 0)
        cls.max_running_requests = _env("MAX_RUNNING_REQUESTS", cls.bs)
        cls.run_name = _env("RUN_NAME", cls.result_label)
        cls.bs_metrics = None
        cls.gsm8k_metrics = None

        if cls.moe_a2a_backend == "deepep":
            os.environ.setdefault("HCCL_BUFFSIZE", _env("HCCL_BUFFSIZE", "1024"))
            capacity = _env("DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "")
            if capacity:
                os.environ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = capacity
            else:
                os.environ.setdefault(
                    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK",
                    str(int(cls.max_running_requests) * 32),
                )

        other_args = [
            "--trust-remote-code",
            "--device",
            "npu",
            "--dtype",
            "bfloat16",
            "--disable-radix-cache",
            "--mem-fraction-static",
            _env("MEM_FRACTION_STATIC", "0.80"),
            "--max-running-requests",
            str(cls.max_running_requests),
            "--random-seed",
            cls.random_seed,
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
        if _env_bool("ENABLE_DP_ATTENTION"):
            other_args.extend(["--enable-dp-attention", "--enable-dp-lm-head"])
        _append_optional_arg(other_args, "--moe-a2a-backend", cls.moe_a2a_backend)
        decode_graph_bs = _env_list("CUDA_GRAPH_BS_DECODE")
        if decode_graph_bs:
            other_args.extend(["--cuda-graph-bs-decode", *decode_graph_bs])
        if cls.moe_a2a_backend == "deepep":
            _append_optional_arg(
                other_args, "--deepep-mode", _env("DEEPEP_MODE", "")
            )
            _append_optional_arg(
                other_args,
                "--deepep-dispatcher-output-dtype",
                cls.deepep_dispatch_dtype,
            )
        other_args.extend(
            [
                "--dllm-algorithm",
                _env("ALGORITHM", "JointThreshold"),
                "--dllm-algorithm-config",
                _env_path("ALGORITHM_CONFIG", "joint_threshold.yaml"),
            ]
        )

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
                "deepep_dispatch_dtype": (
                    cls.deepep_dispatch_dtype
                    if cls.moe_a2a_backend == "deepep"
                    else ""
                ),
                "random_seed": cls.random_seed,
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
        print(f"LLaDA2 {cls.model_size} Ascend CSV updated: {csv_path}")

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
        cls = type(self)
        cls.bs_metrics = send_one_prompt(args, return_metrics=True)
        speed = cls.bs_metrics["speed"]
        print(f"{speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_speed (llada2-{self.model_size}) {self.run_name}\n"
                f"{speed=:.2f} token/s\n"
            )
            self.assertGreater(speed, float(_env("MIN_BS_SPEED", 100)))


if __name__ == "__main__":
    unittest.main()
