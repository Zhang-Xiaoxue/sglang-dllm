import csv
import os
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

RESULT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = RESULT_DIR / "llada2_ascend_eval_results.csv"
CSV_COLUMNS = [
    "run_name",
    "precision",
    "model_size",
    "eval_name",
    "status",
    "bs",
    "tp",
    "ep",
    "dp",
    "moe_dp_size",
    "moe_a2a_backend",
    "max_running_requests",
    "model",
    "quantization",
    "api",
    "num_examples",
    "num_threads",
    "max_tokens",
    "score",
    "mean_score",
    "latency",
    "output_throughput",
    "invalid",
    "chars",
    "error",
]

MODEL_PATHS = {
    "bf16": {
        "mini": "/data/home/z84301856/proj_sglang/models/LLaDA/LLaDA2.1-mini",
        "flash": "/data/home/z84301856/proj_sglang/models/LLaDA/LLaDA2.1-flash",
    },
    "int8": {
        # "mini": "/data/home/z84301856/proj_sglang/models/LLaDA/llada2.1-mini-cannrecipe-w8a8c16-moe",
        # "flash": "/data/home/z84301856/proj_sglang/models/LLaDA/llada2.1-flash-cannrecipe-w8a8c16-moe",
        "mini": "/data/home/z84301856/proj_sglang/models/LLaDA/llada2.1-mini-cannrecipe-w8a8c16-moe-attn-v1",
        "flash": "/data/home/z84301856/proj_sglang/models/LLaDA/llada2.1-flash-cannrecipe-w8a8c16-moe-attn-v1",
    },
}
QUANTIZATION = {
    "bf16": "",
    "int8": "compressed-tensors",
}


def _env(name, default=""):
    return os.environ.get(f"SGLANG_DLLM_{name}", str(default))


def _env_int(name, default):
    return int(_env(name, default))


def _env_optional_int(name, default=None):
    value = _env(name, "" if default is None else default)
    if value == "" or str(value).lower() in ("none", "null"):
        return None
    return int(value)


def _env_bool(name, default=False):
    value = _env(name, "1" if default else "0").lower()
    return value in ("1", "true", "yes", "on")


def _env_path(name, default):
    value = _env(name, default)
    if not value:
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((RESULT_DIR / path).resolve())


def _eval_env(eval_name, suffix, default=""):
    return _env(f"{eval_name.upper()}_{suffix}", _env(suffix, default))


def _eval_env_int(eval_name, suffix, default):
    return int(_eval_env(eval_name, suffix, default))


def _eval_env_optional_int(eval_name, suffix, default=None):
    value = _eval_env(eval_name, suffix, "" if default is None else default)
    if value == "" or str(value).lower() in ("none", "null"):
        return None
    return int(value)


def _append_optional_arg(args, flag, value):
    value = str(value)
    if value and value.lower() not in ("none", "null"):
        args.extend([flag, value])


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


def _split_csv_env(value):
    return [item.strip().lower() for item in value.split(",") if item.strip()]


class TestLLaDA2AscendEval(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.precision = _env("PRECISION", "bf16").lower()
        if cls.precision not in MODEL_PATHS:
            raise ValueError(
                f"Unsupported SGLANG_DLLM_PRECISION={cls.precision!r}; use bf16 or int8"
            )

        cls.model_size = _env("MODEL_SIZE", "mini").lower()
        if cls.model_size not in MODEL_PATHS[cls.precision]:
            raise ValueError(
                f"Unsupported SGLANG_DLLM_MODEL_SIZE={cls.model_size!r}; use mini or flash"
            )

        cls.model = _env("MODEL", MODEL_PATHS[cls.precision][cls.model_size])
        cls.quantization = QUANTIZATION[cls.precision]
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.bs = _env_int("BS", 1)
        cls.tp = _env("TP", "4")
        cls.ep = _env("EP", "1")
        cls.dp = _env("DP", "1")
        cls.moe_dp_size = _env("MOE_DP_SIZE", "1")
        cls.moe_a2a_backend = _env("MOE_A2A_BACKEND", "none")
        cls.max_running_requests = _env("MAX_RUNNING_REQUESTS", cls.bs)
        cls.run_name = _env(
            "RUN_NAME",
            f"llada2_{cls.model_size}_{cls.precision}_eval_bs{cls.bs}_tp{cls.tp}_ep{cls.ep}_dp{cls.dp}",
        )
        cls.eval_names = _split_csv_env(_env("EVAL_NAMES", "gpqa,piqa"))
        if not cls.eval_names:
            raise ValueError("SGLANG_DLLM_EVAL_NAMES is empty")

        if cls.model_size == "flash" and int(cls.tp) < 4:
            raise ValueError("LLaDA2 flash requires SGLANG_DLLM_TP >= 4")

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
        _append_optional_arg(other_args, "--deepep-mode", _env("DEEPEP_MODE", ""))
        other_args.extend(
            [
                "--dllm-algorithm",
                _env("ALGORITHM", "JointThreshold"),
                "--dllm-algorithm-config",
                _env_path("ALGORITHM_CONFIG", "joint_threshold.yaml"),
            ]
        )
        _append_optional_arg(other_args, "--quantization", cls.quantization)

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=_env_int("SERVER_TIMEOUT", 3600),
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        process = getattr(cls, "process", None)
        if process is not None:
            kill_process_tree(process.pid)

    @classmethod
    def _base_row(cls, eval_name, status, error=""):
        return {
            "run_name": cls.run_name,
            "precision": cls.precision,
            "model_size": cls.model_size,
            "eval_name": eval_name,
            "status": status,
            "bs": cls.bs,
            "tp": cls.tp,
            "ep": cls.ep,
            "dp": cls.dp,
            "moe_dp_size": cls.moe_dp_size,
            "moe_a2a_backend": cls.moe_a2a_backend,
            "max_running_requests": cls.max_running_requests,
            "model": cls.model,
            "quantization": cls.quantization,
            "error": error,
        }

    @classmethod
    def _write_eval_result(cls, eval_name, metrics, args, status="passed", error=""):
        row = cls._base_row(eval_name, status, error)
        row.update(
            {
                "api": args.api,
                "num_examples": args.num_examples,
                "num_threads": args.num_threads,
                "max_tokens": args.max_tokens,
                "score": _metric(metrics, "score"),
                "mean_score": _metric(metrics, "mean_score"),
                "latency": _metric(metrics, "latency"),
                "output_throughput": _metric(metrics, "output_throughput"),
                "invalid": _metric(metrics, "invalid"),
                "chars": _metric(metrics, "chars"),
            }
        )
        csv_path = _append_csv_row(row)
        print(f"LLaDA2 Ascend eval CSV updated: {csv_path}")

    def _make_eval_args(self, eval_name):
        default_max_tokens = 64 if eval_name == "piqa" else 2048
        default_num_examples = 200 if eval_name == "piqa" else None
        return SimpleNamespace(
            base_url=self.base_url,
            host="127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
            model=self.model,
            eval_name=eval_name,
            api=_eval_env(eval_name, "API", "chat"),
            repeat=_eval_env_int(eval_name, "REPEAT", 1),
            num_examples=_eval_env_optional_int(eval_name, "NUM_EXAMPLES", default_num_examples),
            num_threads=_eval_env_int(eval_name, "NUM_THREADS", 128),
            max_tokens=_eval_env_int(eval_name, "MAX_TOKENS", default_max_tokens),
            temperature=float(_eval_env(eval_name, "TEMPERATURE", 0.0)),
            top_p=float(_eval_env(eval_name, "TOP_P", 1.0)),
            top_k=_env_optional_int("TOP_K"),
            min_p=None,
            chat_template_kwargs=None,
            reasoning_effort=None,
            thinking_mode=None,
            gpqa_data_path=os.environ.get("SGLANG_DLLM_GPQA_DATA_PATH"),
            piqa_data_path=os.environ.get("SGLANG_DLLM_PIQA_DATA_PATH"),
        )

    def test_evals(self):
        failed = []
        for eval_name in self.eval_names:
            args = self._make_eval_args(eval_name)
            if eval_name == "piqa" and not args.piqa_data_path:
                message = (
                    "Set SGLANG_DLLM_PIQA_DATA_PATH to a local PIQA JSONL/CSV/JSON "
                    "file or dataset directory."
                )
                self._write_eval_result(eval_name, {}, args, status="skipped", error=message)
                print(f"Skip PIQA: {message}")
                continue

            try:
                tic = time.perf_counter()
                metrics = run_eval(args)
                elapsed = time.perf_counter() - tic
                print(f"{eval_name} metrics={metrics}, elapsed={elapsed:.3f}s")
                self._write_eval_result(eval_name, metrics, args)

                score = _metric(metrics, "score")
                min_score = float(_eval_env(eval_name, "MIN_SCORE", 0.0))
                if score != "" and min_score > 0:
                    self.assertGreaterEqual(float(score), min_score)

                if is_in_ci():
                    write_github_step_summary(
                        f"### {eval_name} (llada2-{self.model_size}) {self.run_name}\n"
                        f"score={score}\n"
                    )
            except Exception as exc:
                self._write_eval_result(eval_name, {}, args, status="failed", error=str(exc))
                if _env_bool("FAIL_ON_EVAL_ERROR", True):
                    raise
                failed.append(f"{eval_name}: {exc}")

        if failed:
            print("Ignored eval failures: " + "; ".join(failed))


if __name__ == "__main__":
    unittest.main()
