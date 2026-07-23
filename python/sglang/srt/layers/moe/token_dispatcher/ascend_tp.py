from __future__ import annotations

from typing import NamedTuple, Optional

import torch

from sglang.srt.hardware_backend.npu.moe.finalize_routing import (
    AllGatherFinalizeRoutingWrapper,
    NPUFinalizeRouting,
)
from sglang.srt.hardware_backend.npu.moe.init_routing import (
    NPUMoEInitRouting_v2,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInputFormat,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import (
    DispatcherOutputDtype,
    get_ascend_dispatcher_output_dtype,
)
from sglang.srt.runtime_context import get_parallel


class AscendTPDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    expanded_row_idx: torch.Tensor
    expert_tokens: torch.Tensor
    group_list_type: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.ASCEND_TP


class AscendTPCombineInput(NamedTuple):
    hidden_states: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.ASCEND_TP


class AscendTPDispatcher(BaseDispatcher):
    def __init__(self, moe_runner_config: MoeRunnerConfig):
        super().__init__()
        self.num_experts = moe_runner_config.num_experts
        self.num_local_experts = moe_runner_config.num_local_experts
        self.num_local_shared_experts = moe_runner_config.num_fused_shared_experts
        self.num_local_routed_experts = (
            self.num_local_experts - self.num_local_shared_experts
        )
        self.moe_ep_size = get_parallel().moe_ep_size
        self.moe_ep_rank = get_parallel().moe_ep_rank
        self.top_k = moe_runner_config.top_k
        self._dispatch_output: Optional[AscendTPDispatchOutput] = None
        self.local_expert_mapping: Optional[torch.Tensor] = None

        self.quant_config: Optional[dict] = None

        # Initialise routing kernels with default (no quant config yet)
        self.set_ascend_dispatcher_output_dtype()

    def set_quant_config(self, quant_config: dict) -> None:
        self.quant_config = quant_config
        self.set_ascend_dispatcher_output_dtype()

        # If the quantisation is GGUF and TP is active, wrap the finalizer
        # with an all‑gather so that the dispatcher stays completely clean.
        if (
            isinstance(self.quant_config, dict)
            and self.quant_config.get("quant_type") == "gguf"
            and get_parallel().tp_size > 1
        ):
            self.finalize = AllGatherFinalizeRoutingWrapper(self.finalize, dim=-1)

    def set_ascend_dispatcher_output_dtype(self) -> None:
        """Choose init & finalize routing kernels based on quant config."""
        self.ascend_dispatcher_output_dtype = get_ascend_dispatcher_output_dtype(self)

        if self.ascend_dispatcher_output_dtype == DispatcherOutputDtype.BF16:
            self.init = NPUMoEInitRouting_v2(quant_mode=-1)
            self.finalize = NPUFinalizeRouting(drop_pad_mode=2)
            self.group_list_type = 1
        elif self.ascend_dispatcher_output_dtype == DispatcherOutputDtype.INT8:
            self.init = NPUMoEInitRouting_v2(quant_mode=1)
            self.finalize = NPUFinalizeRouting(drop_pad_mode=2)
            self.group_list_type = 1
        else:
            raise ValueError(
                f"Unsupported ascend_dispatcher_output_dtype: {self.ascend_dispatcher_output_dtype}"
            )

    def _get_local_expert_mapping(self, device: torch.device) -> torch.Tensor:
        if (
            self.local_expert_mapping is not None
            and self.local_expert_mapping.device == device
        ):
            return self.local_expert_mapping

        mapping = torch.full(
            (self.num_experts,), -1, dtype=torch.int64, device=device
        )
        local_start = self.moe_ep_rank * self.num_local_routed_experts
        local_end = local_start + self.num_local_routed_experts
        mapping[local_start:local_end] = torch.arange(
            0, self.num_local_routed_experts, dtype=torch.int64, device=device
        )

        if self.num_local_shared_experts > 0:
            mapping[-self.num_local_shared_experts :] = torch.arange(
                self.num_local_routed_experts,
                self.num_local_routed_experts + self.num_local_shared_experts,
                dtype=torch.int64,
                device=device,
            )

        self.local_expert_mapping = mapping
        return mapping

    def _map_topk_ids_to_local(
        self, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_ids = topk_ids.to(torch.int64)

        if self.moe_ep_size > 1:
            mapping = self._get_local_expert_mapping(topk_ids.device)
            valid_global = (topk_ids >= 0) & (topk_ids < self.num_experts)
            safe_global_ids = torch.where(
                valid_global, topk_ids, torch.zeros_like(topk_ids)
            )
            local_topk_ids = mapping[safe_global_ids]
        else:
            valid_global = (topk_ids >= 0) & (topk_ids < self.num_local_experts)
            local_topk_ids = topk_ids

        valid_local = (
            valid_global
            & (local_topk_ids >= 0)
            & (local_topk_ids < self.num_local_experts)
        )
        local_topk_ids = torch.where(
            valid_local, local_topk_ids, torch.zeros_like(local_topk_ids)
        )
        topk_weights = torch.where(
            valid_local, topk_weights, torch.zeros_like(topk_weights)
        )
        return local_topk_ids.to(torch.int32), topk_weights

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> AscendTPDispatchOutput:
        topk_weights, topk_ids, _ = topk_output
        topk_weights = topk_weights.to(hidden_states.dtype)
        topk_ids, topk_weights = self._map_topk_ids_to_local(topk_ids, topk_weights)

        (
            permuted_hidden_states,
            expanded_row_idx,
            expert_tokens,
            hidden_states_scale,
        ) = self.init._init_routing(
            hidden_states,
            topk_ids,
            self.num_local_experts,
            self.top_k,
        )

        self._dispatch_output = AscendTPDispatchOutput(
            hidden_states=permuted_hidden_states,
            hidden_states_scale=hidden_states_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expanded_row_idx=expanded_row_idx,
            expert_tokens=expert_tokens,
            group_list_type=self.group_list_type,
        )
        return self._dispatch_output

    def combine(self, combine_input: AscendTPCombineInput) -> torch.Tensor:
        if self._dispatch_output is None:
            raise RuntimeError("combine() called before dispatch()")

        dispatch_out = self._dispatch_output

        # The finalizer (possibly wrapped with TP all‑gather) does all the work.
        final_hidden_states = self.finalize._finalize_routing(
            combine_input.hidden_states,
            topk_weights=dispatch_out.topk_weights,
            expanded_row_idx=dispatch_out.expanded_row_idx,
            topk_ids=dispatch_out.topk_ids,
        )

        self._dispatch_output = None
        return final_hidden_states
