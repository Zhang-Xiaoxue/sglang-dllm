"""
Run one test prompt.

Usage:
python3 -m sglang.test.send_one
python3 -m sglang.test.send_one --profile --profile-steps 5
python3 -m sglang.test.send_one --profile --profile-by-stage
python3 -m sglang.test.send_one --stop "<|separator|>" "<|eos|>" --max-new-tokens 2048
"""

import argparse
import dataclasses
import json
import random
import time
from typing import Optional

import requests
import tabulate

from sglang.profiler import run_profile


@dataclasses.dataclass
class BenchArgs:
    host: str = "localhost"
    port: int = 30000
    batch_size: int = 1
    different_prompts: bool = False
    random_input_len: Optional[int] = None
    random_input_vocab_size: int = 32768
    seed: Optional[int] = None
    temperature: float = 0.0
    max_new_tokens: int = 512
    min_new_tokens: int = 0
    ignore_eos: bool = False
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    json: bool = False
    return_logprob: bool = False
    prompt: str = (
        "Human: Give me a fully functional FastAPI server. Show the python code.\n\nAssistant:"
    )
    image: bool = False
    many_images: bool = False
    stop: Optional[list] = None
    stream: bool = False
    profile: bool = False
    profile_steps: int = 5
    profile_by_stage: bool = False
    profile_prefix: Optional[str] = None
    profile_output_dir: Optional[str] = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--host", type=str, default=BenchArgs.host)
        parser.add_argument("--port", type=int, default=BenchArgs.port)
        parser.add_argument("--batch-size", type=int, default=BenchArgs.batch_size)
        parser.add_argument(
            "--different-prompts",
            action="store_true",
            default=BenchArgs.different_prompts,
        )
        parser.add_argument(
            "--random-input-len",
            type=int,
            default=BenchArgs.random_input_len,
            help="Generate a random prompt of exactly this many tokens (random token IDs). "
            "Each request in the batch gets unique random IDs, avoiding radix cache hits. "
            "Useful for profiling to ensure the full prefill is captured.",
        )
        parser.add_argument(
            "--random-input-vocab-size",
            type=int,
            default=BenchArgs.random_input_vocab_size,
            help="Vocab size for --random-input-len. Token IDs are sampled from "
            "[0, vocab_size). Default: 32768.",
        )
        parser.add_argument("--seed", type=int, default=BenchArgs.seed)
        parser.add_argument("--temperature", type=float, default=BenchArgs.temperature)
        parser.add_argument(
            "--max-new-tokens", type=int, default=BenchArgs.max_new_tokens
        )
        parser.add_argument(
            "--min-new-tokens", type=int, default=BenchArgs.min_new_tokens
        )
        parser.add_argument(
            "--ignore-eos",
            action=argparse.BooleanOptionalAction,
            default=BenchArgs.ignore_eos,
        )
        parser.add_argument(
            "--frequency-penalty", type=float, default=BenchArgs.frequency_penalty
        )
        parser.add_argument(
            "--presence-penalty", type=float, default=BenchArgs.presence_penalty
        )
        parser.add_argument("--json", action="store_true")
        parser.add_argument("--return-logprob", action="store_true")
        parser.add_argument("--prompt", type=str, default=BenchArgs.prompt)
        parser.add_argument("--stop", type=str, nargs="*", default=None)
        parser.add_argument("--image", action="store_true")
        parser.add_argument("--many-images", action="store_true")
        parser.add_argument("--stream", action="store_true")
        parser.add_argument("--profile", action="store_true")
        parser.add_argument(
            "--profile-steps", type=int, default=BenchArgs.profile_steps
        )
        parser.add_argument("--profile-by-stage", action="store_true")
        parser.add_argument(
            "--profile-prefix", type=str, default=BenchArgs.profile_prefix
        )
        parser.add_argument(
            "--profile-output-dir", type=str, default=BenchArgs.profile_output_dir
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


def send_one_prompt(
    args: BenchArgs,
    label: Optional[str] = None,
    print_output: bool = True,
    return_metrics: bool = False,
    input_ids: Optional[list[int] | list[list[int]]] = None,
    prompts: Optional[list[str]] = None,
):
    base_url = f"http://{args.host}:{args.port}"

    # Construct the input
    if input_ids is not None and prompts is not None:
        raise ValueError("input_ids and prompts are mutually exclusive")

    submitted_input_tokens_per_request = []
    if input_ids is not None:
        if args.batch_size == 1:
            if input_ids and isinstance(input_ids[0], (list, tuple)):
                if len(input_ids) != 1:
                    raise ValueError(
                        f"Expected one input_ids item, got {len(input_ids)}"
                    )
                request_input_ids = list(input_ids[0])
            else:
                request_input_ids = list(input_ids)
            submitted_input_tokens_per_request = [len(request_input_ids)]
        else:
            if len(input_ids) != args.batch_size or not all(
                isinstance(item, (list, tuple)) for item in input_ids
            ):
                raise ValueError(
                    f"Expected {args.batch_size} input_ids items for the batch"
                )
            request_input_ids = [list(item) for item in input_ids]
            submitted_input_tokens_per_request = [
                len(item) for item in request_input_ids
            ]
        prompt = None
    elif prompts is not None:
        if len(prompts) != args.batch_size:
            raise ValueError(
                f"Expected {args.batch_size} prompts, got {len(prompts)}"
            )
        request_input_ids = None
        prompt = prompts[0] if args.batch_size == 1 else list(prompts)
    elif args.random_input_len is not None:
        # Generate random input ids within the vocab size
        n = args.random_input_len
        v = args.random_input_vocab_size
        rng = random.Random(args.seed)
        if args.batch_size == 1:
            request_input_ids = rng.choices(range(v), k=n)
            submitted_input_tokens_per_request = [n]
        else:
            if args.different_prompts:
                request_input_ids = [
                    rng.choices(range(v), k=n) for _ in range(args.batch_size)
                ]
            else:
                request_input_ids = [rng.choices(range(v), k=n)] * args.batch_size
            submitted_input_tokens_per_request = [n] * args.batch_size
        prompt = None
    else:
        # Use the user inputs
        request_input_ids = None
        if args.batch_size == 1:
            prompt = args.prompt
        else:
            if args.different_prompts:
                prompt = [
                    f"Test case {i+1}: " + args.prompt for i in range(args.batch_size)
                ]
            else:
                prompt = [args.prompt] * args.batch_size

    # If need image
    if args.image:
        assert args.batch_size == 1 and not args.random_input_len
        args.prompt = (
            "Human: Describe this image in a very short sentence.\n\nAssistant:"
        )
        image_data = "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png"
    elif args.many_images:
        args.prompt = (
            "Human: I have one reference image and many images."
            "Describe their relationship in a very short sentence.\n\nAssistant:"
        )
        image_data = [
            "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png",
            "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png",
            "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png",
            "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png",
        ]
    else:
        image_data = None

    # If need json output
    if args.json:
        assert args.batch_size == 1 and not args.random_input_len
        prompt = (
            "Human: What is the capital of France and how is that city like. "
            "Give me 3 trivial information about that city. "
            "Write in a format of json.\nAssistant:"
        )
        json_schema = "$$ANY$$"
    else:
        json_schema = None

    json_data = {
        **(
            {"input_ids": request_input_ids}
            if request_input_ids is not None
            else {"text": prompt}
        ),
        "image_data": image_data,
        "sampling_params": {
            "sampling_seed": args.seed,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "ignore_eos": args.ignore_eos,
            "frequency_penalty": args.frequency_penalty,
            "presence_penalty": args.presence_penalty,
            "json_schema": json_schema,
            "stop": args.stop,
        },
        "return_logprob": args.return_logprob,
        "stream": args.stream,
    }

    # Run profiler if requested
    profile_dir = None
    if args.profile:
        print(f"Running profiler with {args.profile_steps} steps...")
        profile_dir = run_profile(
            url=base_url,
            num_steps=args.profile_steps,
            activities=["CPU", "GPU"],
            output_dir=args.profile_output_dir,
            profile_by_stage=args.profile_by_stage,
            profile_prefix=args.profile_prefix,
        )

    # Send the request
    request_start = time.perf_counter()
    response = requests.post(
        f"{base_url}/generate",
        json=json_data,
        stream=args.stream,
    )

    if args.stream:
        last_len = 0
        for chunk in response.iter_lines(decode_unicode=False):
            chunk = chunk.decode("utf-8")
            if chunk and chunk.startswith("data:"):
                if chunk == "data: [DONE]":
                    break
                ret = json.loads(chunk[5:].strip("\n"))
                chunk_str = ret["text"][last_len:]
                last_len = len(ret["text"])
                print(chunk_str, end="", flush=True)
    else:
        ret = response.json()
    batch_wall_latency = time.perf_counter() - request_start

    raw_ret = ret
    if args.batch_size > 1 and isinstance(ret, list):
        ret = ret[0]

    if response.status_code != 200:
        print(ret)
        if return_metrics:
            return {
                "latency": 0,
                "batch_wall_latency": batch_wall_latency,
                "tokens": 0,
                "acc_length": 0,
                "speed": 0,
                "batch_wall_speed": 0,
                "fixed_output_length_ok": False,
                "submitted_input_tokens_per_request": (
                    submitted_input_tokens_per_request
                ),
                "prompt_tokens_per_request": [],
                "profile_dir": profile_dir,
            }
        return 0, 0

    # Print results
    if return_metrics and isinstance(raw_ret, list):
        meta_infos = [item["meta_info"] for item in raw_ret]
        prompt_tokens_per_request = [
            meta_info["prompt_tokens"]
            for meta_info in meta_infos
            if "prompt_tokens" in meta_info
        ]
        completion_tokens_per_request = [
            meta_info["completion_tokens"] for meta_info in meta_infos
        ]
        latency = max(meta_info["e2e_latency"] for meta_info in meta_infos)
        tokens = sum(completion_tokens_per_request)
        acc_lengths = []
        for meta_info in meta_infos:
            spec_verify_ct = meta_info.get("spec_verify_ct", 0)
            if spec_verify_ct > 0:
                acc_lengths.append(meta_info["completion_tokens"] / spec_verify_ct)
            else:
                acc_lengths.append(1.0)
        acc_length = sum(acc_lengths) / len(acc_lengths)
    else:
        prompt_tokens_per_request = (
            [ret["meta_info"]["prompt_tokens"]]
            if "prompt_tokens" in ret["meta_info"]
            else []
        )
        completion_tokens_per_request = [ret["meta_info"]["completion_tokens"]]
        if "spec_verify_ct" in ret["meta_info"] and ret["meta_info"]["spec_verify_ct"] > 0:
            acc_length = (
                ret["meta_info"]["completion_tokens"]
                / ret["meta_info"]["spec_verify_ct"]
            )
        else:
            acc_length = 1.0

        latency = ret["meta_info"]["e2e_latency"]
        tokens = ret["meta_info"]["completion_tokens"]

    speed = tokens / latency if latency > 0 else 0
    batch_wall_speed = tokens / batch_wall_latency if batch_wall_latency > 0 else 0
    fixed_output_length_ok = (
        len(completion_tokens_per_request) == args.batch_size
        and all(
            completion_tokens == args.max_new_tokens
            for completion_tokens in completion_tokens_per_request
        )
    )

    if not args.stream and print_output:
        print(ret["text"])

    print()
    if label is not None:
        print(label)
    headers = [
        "Server E2E (s)",
        "Batch Wall (s)",
        "Tokens",
        "Output Range",
        "Acc Length",
        "Server E2E (token/s)",
        "Batch Wall (token/s)",
    ]
    output_range = (
        f"{min(completion_tokens_per_request)}-{max(completion_tokens_per_request)}"
    )
    rows = [
        [
            f"{latency:.3f}",
            f"{batch_wall_latency:.3f}",
            f"{tokens}",
            output_range,
            f"{acc_length:.3f}",
            f"{speed:.2f}",
            f"{batch_wall_speed:.2f}",
        ]
    ]
    msg = tabulate.tabulate(rows, headers=headers, tablefmt="pretty")
    print(msg)

    if return_metrics:
        return {
            "latency": latency,
            "batch_wall_latency": batch_wall_latency,
            "tokens": tokens,
            "submitted_input_tokens_per_request": (
                submitted_input_tokens_per_request
            ),
            "prompt_tokens_per_request": prompt_tokens_per_request,
            "completion_tokens_per_request": completion_tokens_per_request,
            "fixed_output_length_ok": fixed_output_length_ok,
            "acc_length": acc_length,
            "speed": speed,
            "batch_wall_speed": batch_wall_speed,
            "profile_dir": profile_dir,
        }

    return acc_length, speed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BenchArgs.add_cli_args(parser)
    args = BenchArgs.from_cli_args(parser.parse_args())

    send_one_prompt(args)
