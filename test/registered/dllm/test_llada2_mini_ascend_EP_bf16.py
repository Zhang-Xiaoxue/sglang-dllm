import os
import unittest
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


class TestLLaDA2Mini(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # cls._old_disable_acl = os.environ.get("SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT")
        os.environ["SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT"] = "1"

        os.environ.setdefault("HCCL_BUFFSIZE", "1024")
        os.environ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = "256"
        os.environ["SGLANG_DEEPEP_BF16_DISPATCH"] = "1"
        os.environ["SGLANG_DEBUG_GRAPH_CAN_RUN"] = "0"

        cls.model = "/home/ma-user/work/z84301856/models/LLaDA2.1-mini"
        # cls.model = "/home/ma-user/work/z84301856/models/LLaDA2.1-flash"

        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--trust-remote-code",
            "--device", "npu",
            "--dtype", "bfloat16",
            "--disable-radix-cache",
            "--mem-fraction-static", "0.90",
            "--attention-backend", "ascend",
            "--tp", "4",
            "--ep", "4", 
            "--dp-size", "1",       
            "--moe-dp-size", "1",
            "--moe-a2a-backend", "deepep", # "ascend_fuseep", "deepep" , "none"
            "--deepep-mode", "auto",
            "--max-running-requests", "4",            
            "--cuda-graph-max-bs", "4",
            # "--disable-cuda-graph",
            "--piecewise-cuda-graph-compiler", "eager",
            "--piecewise-cuda-graph-tokens", "32", "64", "96", "128",
            "--enforce-piecewise-cuda-graph",

            # "--dllm-algorithm", "LowConfidence",  # TODO: Add dLLM configurations
            # "--dllm-algorithm-config", "/home/ma-user/work/z84301856/sglang-dllm/bashs/dllm_config_zxx.yaml",

            "--dllm-algorithm", "JointThreshold",  # TODO: Add dLLM configurations
            "--dllm-algorithm-config", "joint_threshold.yaml",

            # "--model-loader-extra-config", 
            # '{"weights_path":"/workspace/sglang/sglang_zxx/cann-recipes-infer/models/llada/quant/w8a8c16_quantALL","int8_dequant":true,"int8_scale_suffix":"_scale"}'
            # '{"weights_path":"/workspace/sglang/sglang_zxx/cann-recipes-infer/models/llada/quant/w8a8c16_layer_all_mlp_false","w8a8c16_quantALL":true,"int8_scale_suffix":"_scale"}'
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3600,  # downloading model takes time, may change back to DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH after caching the model locally
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

        # if cls._old_disable_acl is None:
        #     os.environ.pop("SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT", None)
        # else:
        #     os.environ["SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT"] = cls._old_disable_acl

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=0,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.68)
        self.assertGreater(metrics["output_throughput"], 10)

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        print(f"{speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (llada2-mini) with tp1\n"
                f"{speed=:.2f} token/s\n"
            )
            # if test speed lower than 130 tps, have to check the environ
            self.assertGreater(speed, 100)


if __name__ == "__main__":
    unittest.main()
