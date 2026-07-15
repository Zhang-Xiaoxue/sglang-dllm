import logging
from contextlib import ExitStack
from typing import Any, Callable
from unittest.mock import patch

import torch
import torch.fx as fx

from sglang.srt.environ import envs
from sglang.srt.compilation.compilation_config import CompilationConfig
from sglang.srt.compilation.compilation_counter import compilation_counter
from sglang.srt.compilation.compile_phase import (
    get_pcg_capture_stream,
    is_in_torch_compile_warmup,
)
from sglang.srt.compilation.cuda_piecewise_backend import (
    CUDAPiecewiseBackend,
    weak_ref_tensors,
)
from sglang.srt.utils import log_info_on_rank0


logger = logging.getLogger(__name__)


def _debug_piecewise_tensor_summary(name: str, value: Any) -> str:
    if isinstance(value, (list, tuple)):
        if not value:
            return f"{name}=empty_{type(value).__name__}"
        return _debug_piecewise_tensor_summary(f"{name}[0]", value[0])
    if not isinstance(value, torch.Tensor):
        return f"{name}=type({type(value).__name__})"
    try:
        x = value.detach()
        shape = tuple(x.shape)
        if x.numel() == 0:
            return f"{name}: shape={shape} dtype={x.dtype} empty=True"
        xf = x.float()
        return (
            f"{name}: shape={shape} dtype={x.dtype} "
            f"finite={bool(torch.isfinite(xf).all().item())} "
            f"min={float(xf.min().item()):.6g} "
            f"max={float(xf.max().item()):.6g} "
            f"mean={float(xf.mean().item()):.6g} "
            f"std={float(xf.std(unbiased=False).item()):.6g}"
        )
    except Exception as exc:  # pragma: no cover - debug aid only
        return f"{name}: debug_failed={exc!r}"


class NPUPiecewiseBackend(CUDAPiecewiseBackend):
    def __init__(
        self,
        graph: fx.GraphModule,
        compile_config: CompilationConfig,
        inductor_config: dict[str, Any],
        graph_pool: Any,
        piecewise_compile_index: int,
        total_piecewise_compiles: int,
        sym_shape_indices: list[int],
        compiled_graph_for_general_shape: Callable,
        sglang_backend,
    ):
        super().__init__(
            graph,
            compile_config,
            inductor_config,
            graph_pool,
            piecewise_compile_index,
            total_piecewise_compiles,
            sym_shape_indices,
            compiled_graph_for_general_shape,
            sglang_backend,
        )
        self.placeholder_names = [
            node.name for node in graph.graph.nodes if node.op == "placeholder"
        ]
        self.force_eager_reason = self._get_force_eager_reason(graph)

    def _is_stable_model_tensor_arg(self, index: int) -> bool:
        if index >= len(self.placeholder_names):
            return False
        name = self.placeholder_names[index]
        # Model parameters, buffers, and config tensors are stable across replays.
        return name.startswith("l_self_modules_")

    def _should_static_copy_arg(self, index: int, value: Any) -> bool:
        return (
            envs.SGLANG_NPU_PIECEWISE_STATIC_INPUT_COPY.get()
            and isinstance(value, torch.Tensor)
            and value.device.type == "npu"
            and not self._is_stable_model_tensor_arg(index)
        )

    def _copy_static_arg(self, dst: torch.Tensor, src: torch.Tensor, stream=None) -> None:
        if stream is None:
            dst.copy_(src)
            return

        current_stream = torch.npu.current_stream()
        stream.wait_stream(current_stream)
        with torch.npu.stream(stream):
            dst.copy_(src)
        current_stream.wait_stream(stream)

    def _prepare_static_args(
        self, entry, args: tuple[Any, ...], stream=None
    ) -> tuple[Any, ...]:
        static_args = getattr(entry, "static_args", None)
        static_tensor_indices = getattr(entry, "static_tensor_indices", None)
        if static_args is None:
            static_args_list = list(args)
            static_tensor_indices = []
            for index, value in enumerate(args):
                if not self._should_static_copy_arg(index, value):
                    continue
                static_value = torch.empty_strided(
                    tuple(value.shape),
                    tuple(value.stride()),
                    dtype=value.dtype,
                    device=value.device,
                )
                self._copy_static_arg(static_value, value, stream)
                static_args_list[index] = static_value
                static_tensor_indices.append(index)
            static_args = tuple(static_args_list)
            entry.static_args = static_args
            entry.static_tensor_indices = static_tensor_indices
        else:
            for index in static_tensor_indices:
                self._copy_static_arg(static_args[index], args[index], stream)
        return static_args

    @staticmethod
    def _get_force_eager_reason(graph: fx.GraphModule) -> str | None:
        if not envs.SGLANG_NPU_DEEPEP_EAGER_POST_MOE_GRAPH.get():
            return None
        placeholder_names = [
            node.name for node in graph.graph.nodes if node.op == "placeholder"
        ]
        consumes_moe_output = any(
            name.startswith("final_hidden_states") for name in placeholder_names
        )
        consumes_shared_output = any(
            name.startswith("output_parallel") for name in placeholder_names
        )
        if consumes_moe_output and consumes_shared_output:
            return "deepep_post_moe_mutable_input"
        return None

    def __call__(self, *args):
        runtime_shape = args[self.sym_shape_indices[0]]
        if runtime_shape not in self.concrete_size_entries:
            # we don't need to do anything for this shape
            return self.compiled_graph_for_general_shape(*args)

        entry = self.concrete_size_entries[runtime_shape]

        if entry.runnable is None:
            entry.runnable = self.compiled_graph_for_general_shape

        if is_in_torch_compile_warmup():
            return entry.runnable(*args)

        if entry.cudagraph is None:
            if entry.num_finished_warmup < 1:  # noqa
                entry.num_finished_warmup += 1
                return entry.runnable(*args)

            stream = get_pcg_capture_stream()
            assert (
                stream is not None
            ), "PCG capture stream is not set, please check if runtime recompilation happened"
            entry.capture_stream = stream
            static_args = self._prepare_static_args(entry, args, stream)
            if self.compile_config.get_enable_debug_mode():
                input_addresses = [
                    x.data_ptr()
                    for x in static_args
                    if isinstance(x, torch.Tensor)
                ]
                entry.input_addresses = input_addresses
            npugraph = torch.npu.NPUGraph()

            with ExitStack() as stack:
                if not self.is_first_graph:
                    # during every model forward, we will capture
                    # many pieces of cudagraphs (roughly one per layer).
                    # running gc again and again across layers will
                    # make the cudagraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(patch("torch.npu.empty_cache", lambda: None))

                # mind-exploding: carefully manage the reference and memory.
                with torch.npu.graph(
                    npugraph,
                    pool=self.graph_pool,
                    stream=stream,
                    auto_dispatch_capture=True,
                ):
                    # `output` is managed by pytorch's cudagraph pool
                    output = entry.runnable(*static_args)
                    if self.is_last_graph:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph, because the output of the last graph
                        # will not be used by any other cuda graph.
                        output = weak_ref_tensors(output)

            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)
            entry.cudagraph = npugraph

            compilation_counter.num_cudagraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during cuda graph capture
            return output

        if self.compile_config.get_enable_debug_mode():
            # check if the input addresses are the same
            static_args = self._prepare_static_args(
                entry, args, getattr(entry, "capture_stream", None)
            )
            new_input_addresses = [
                x.data_ptr() for x in static_args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                "Input addresses for cudagraphs are different during replay."
                f" Expected {entry.input_addresses}, got {new_input_addresses}"
            )
        graph_index = self.piecewise_compile_index + 1
        eager_from_graph = envs.SGLANG_NPU_PIECEWISE_EAGER_FROM_GRAPH.get()
        eager_graph = envs.SGLANG_NPU_PIECEWISE_EAGER_GRAPH.get()
        force_eager_graph = (
            self.force_eager_reason is not None
            or (eager_graph > 0 and graph_index == eager_graph)
            or (eager_from_graph > 0 and graph_index >= eager_from_graph)
            or (
                self.is_last_graph
                and envs.SGLANG_NPU_PIECEWISE_EAGER_LAST_GRAPH.get()
            )
        )
        if force_eager_graph:
            count = getattr(self, "_debug_eager_graph_log_count", 0)
            if count < 1:
                log_info_on_rank0(
                    logger,
                    "NPU piecewise eager graph debug: "
                    f"graph={graph_index}/{self.total_piecewise_compiles} "
                    f"runtime_shape={runtime_shape} "
                    f"reason={self.force_eager_reason} "
                    f"eager_graph={eager_graph} "
                    f"eager_from_graph={eager_from_graph} "
                    f"eager_last_graph={envs.SGLANG_NPU_PIECEWISE_EAGER_LAST_GRAPH.get()}",
                )
                self._debug_eager_graph_log_count = count + 1
            return entry.runnable(*args)

        self._prepare_static_args(entry, args, getattr(entry, "capture_stream", None))
        if envs.SGLANG_NPU_PIECEWISE_SYNC_REPLAY.get():
            # Keep eager split ops, such as DeepEP dispatch/combine, ordered with graph segments.
            torch.npu.synchronize()
        entry.cudagraph.replay()
        if envs.SGLANG_NPU_PIECEWISE_SYNC_REPLAY.get():
            torch.npu.synchronize()
        if envs.SGLANG_NPU_DEEPEP_DEBUG_GRAPH_LOG.get():
            count = getattr(self, "_debug_replay_output_log_count", 0)
            should_log = (
                count < 1
                and (
                    self.piecewise_compile_index < 12
                    or self.piecewise_compile_index % 16 == 0
                    or self.is_last_graph
                )
            )
            if should_log:
                log_info_on_rank0(
                    logger,
                    "NPU piecewise replay output debug: "
                    f"graph={self.piecewise_compile_index + 1}/{self.total_piecewise_compiles} "
                    f"runtime_shape={runtime_shape} "
                    f"{_debug_piecewise_tensor_summary('output', entry.output)}",
                )
            self._debug_replay_output_log_count = count + 1
        return entry.output
