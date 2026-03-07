from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.srt.batch_overlap import operations
from sglang.srt.batch_overlap.operations import Operation
from sglang.srt.layers.moe.token_dispatcher import DeepEPConfig
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_hip

_is_hip = is_hip()


@dataclass
class OperationsStrategy:
    operations: List[Operation]
    deep_gemm_num_sms: Optional[int] = None
    tbo_delta_stages: Optional[int] = None

    @classmethod
    def concat(cls, items: List["OperationsStrategy"]) -> "OperationsStrategy":
        return OperationsStrategy(
            operations=[x for item in items for x in item.operations],
            deep_gemm_num_sms=_assert_all_same(
                [item.deep_gemm_num_sms for item in items]
            ),
            tbo_delta_stages=_assert_all_same(
                [item.tbo_delta_stages for item in items]
            ),
        )

    @staticmethod
    def init_new_tbo(
        layers: torch.nn.ModuleList,
        forward_mode: ForwardMode,
    ) -> "OperationsStrategy":
        layer_name = layers[0].__class__.__name__
        if layer_name == "DeepseekV2DecoderLayer":
            return OperationsStrategy.concat(
                [
                    _compute_moe_deepseek_layer_operations_strategy_tbo(
                        layer, forward_mode
                    )
                    for layer in layers
                ]
            )
        elif layer_name == "Qwen3MoeDecoderLayer":
            return OperationsStrategy.concat(
                [
                    _compute_moe_qwen3_layer_operations_strategy_tbo(
                        layer, forward_mode
                    )
                    for layer in layers
                ]
            )
        elif layer_name == "MiMoV2DecoderLayer":
            return OperationsStrategy.concat(
                [
                    _compute_moe_mimov2_layer_operations_strategy_tbo(
                        layer, forward_mode
                    )
                    for layer in layers
                ]
            )
        elif layer_name == "GptOssDecoderLayer":
            return OperationsStrategy.concat(
                [
                    _compute_moe_gptoss_layer_operations_strategy_tbo(
                        layer, forward_mode
                    )
                    for layer in layers
                ]
            )
        else:
            raise NotImplementedError


def _assert_all_same(items: List):
    assert all(item == items[0] for item in items)
    return items[0]


# -------------------------------- Strategy for DeepSeek ---------------------------------------


# TODO can refactor to make it more fancy if we have more complex strategies
def _compute_moe_deepseek_layer_operations_strategy_tbo(
    layer: torch.nn.Module,
    forward_mode: ForwardMode,
) -> OperationsStrategy:
    assert layer.is_layer_sparse, "dense layer TBO not yet implemented"
    if forward_mode == ForwardMode.EXTEND:
        return _compute_moe_deepseek_blog_prefill(layer)
    elif (
        forward_mode == ForwardMode.DECODE or forward_mode == ForwardMode.TARGET_VERIFY
    ):
        return _compute_moe_deepseek_blog_decode(layer)
    else:
        raise NotImplementedError(f"Unsupported {forward_mode=}")


def _compute_moe_deepseek_blog_prefill(layer):
    device_properties = torch.cuda.get_device_properties(device="cuda")
    total_num_sms = device_properties.multi_processor_count
    deep_gemm_num_sms = None
    if not _is_hip:
        deep_gemm_num_sms = total_num_sms - DeepEPConfig.get_instance().num_sms

    return OperationsStrategy(
        deep_gemm_num_sms=deep_gemm_num_sms,
        tbo_delta_stages=0,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            layer.mlp.op_dispatch_a,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_shared_experts,
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
        ],
    )


def _compute_moe_deepseek_blog_decode(layer):
    return OperationsStrategy(
        deep_gemm_num_sms=None,
        tbo_delta_stages=2,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            operations.YieldOperation(),
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_a,
            layer.mlp.op_shared_experts,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_combine_b,
            operations.YieldOperation(),
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
        ],
    )


# -------------------------------- Strategy for Qwen3 ---------------------------------------


# TODO: unstable, current strategy is almost the same as DeepSeek, keep redundant code here for
# convenience to adjust strategy
def _compute_moe_qwen3_layer_operations_strategy_tbo(
    layer: torch.nn.Module,
    forward_mode: ForwardMode,
) -> OperationsStrategy:
    assert layer.is_layer_sparse, "qwen3 moe only support sparse layers"
    if forward_mode == ForwardMode.EXTEND:
        return _compute_moe_qwen3_prefill(layer)
    elif (
        forward_mode == ForwardMode.DECODE or forward_mode == ForwardMode.TARGET_VERIFY
    ):
        return _compute_moe_qwen3_decode(layer)
    else:
        raise NotImplementedError(f"Unsupported {forward_mode=}")


def _compute_moe_qwen3_prefill(layer):
    device_properties = torch.cuda.get_device_properties(device="cuda")
    total_num_sms = device_properties.multi_processor_count
    deep_gemm_num_sms = None
    if not _is_hip:
        deep_gemm_num_sms = total_num_sms - DeepEPConfig.get_instance().num_sms

    return OperationsStrategy(
        deep_gemm_num_sms=deep_gemm_num_sms,
        tbo_delta_stages=0,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            layer.mlp.op_dispatch_a,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
        ],
    )


def _compute_moe_qwen3_decode(layer):
    return OperationsStrategy(
        deep_gemm_num_sms=None,
        tbo_delta_stages=2,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            operations.YieldOperation(),
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_a,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
            operations.YieldOperation(),
        ],
    )


# -------------------------------- Strategy for MiMoV2DecoderLayer ---------------------------------------


# TODO: unstable; current strategy matches DeepSeek for the common operations (MiMoV2 has no op_shared_experts),
# so we keep this redundant code here for convenience when adjusting the strategy
def _compute_moe_mimov2_layer_operations_strategy_tbo(
    layer: torch.nn.Module,
    forward_mode: ForwardMode,
) -> OperationsStrategy:
    assert layer.is_layer_sparse, "MiMoV2DecoderLayer moe only support sparse layers"
    if forward_mode == ForwardMode.EXTEND:
        return _compute_moe_mimov2_prefill(layer)
    elif (
        forward_mode == ForwardMode.DECODE or forward_mode == ForwardMode.TARGET_VERIFY
    ):
        return _compute_moe_mimov2_decode(layer)
    else:
        raise NotImplementedError(f"Unsupported {forward_mode=}")


def _compute_moe_mimov2_prefill(layer):
    device_properties = torch.cuda.get_device_properties(device="cuda")
    total_num_sms = device_properties.multi_processor_count
    deep_gemm_num_sms = total_num_sms - DeepEPConfig.get_instance().num_sms

    return OperationsStrategy(
        deep_gemm_num_sms=deep_gemm_num_sms,
        tbo_delta_stages=0,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            layer.mlp.op_dispatch_a,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
        ],
    )


def _compute_moe_mimov2_decode(layer):
    return OperationsStrategy(
        deep_gemm_num_sms=None,
        tbo_delta_stages=2,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            operations.YieldOperation(),
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_a,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
            operations.YieldOperation(),
        ],
    )


# -------------------------------- Strategy for GPT-OSS ---------------------------------------
# GPT-OSS uses TP with AllReduce (not EP with dispatch/combine).
# Overlap is achieved by launching AllReduce on a dedicated NCCL stream
# and interleaving the other sub-batch's compute during the AllReduce.
# 5 stages per layer with delta_stages=1:
#   S0: [attn_compute + launch_attn_AR]     -- launch AR at end of compute
#   S1: [wait_attn_AR + norm + MoE_no_ar]   -- wait AR (hidden behind other batch's S0)
#   S2: [launch_MoE_AR]                     -- launch AR
#   S3: [wait_MoE_AR + postprocess]         -- wait AR (hidden behind other batch's S1)
#   Across layer boundaries: B.L(i).S3's MoE_AR overlaps with A.L(i+1).S0's attn compute.


def _compute_moe_gptoss_layer_operations_strategy_tbo(
    layer: torch.nn.Module,
    forward_mode: ForwardMode,
) -> OperationsStrategy:
    assert layer.is_layer_sparse, "GPT-OSS TBO only supports sparse layers"
    if forward_mode == ForwardMode.EXTEND:
        return _compute_moe_gptoss_prefill(layer)
    elif (
        forward_mode == ForwardMode.DECODE or forward_mode == ForwardMode.TARGET_VERIFY
    ):
        return _compute_moe_gptoss_decode(layer)
    else:
        raise NotImplementedError(f"Unsupported {forward_mode=}")


def _compute_moe_gptoss_prefill(layer):
    return OperationsStrategy(
        deep_gemm_num_sms=None,
        tbo_delta_stages=1,
        operations=[
            # Stage 0: Attention compute + launch async AllReduce
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            layer.self_attn.op_core,
            layer.op_launch_attn_ar,
            operations.YieldOperation(),
            # Stage 1: Wait AllReduce + norm + MoE (no AllReduce inside MoE)
            layer.op_wait_attn_ar_and_norm,
            layer.op_mlp_no_ar,
            operations.YieldOperation(),
            # Stage 2: Launch MoE AllReduce
            layer.op_launch_mlp_ar,
            operations.YieldOperation(),
            # Stage 3: Wait MoE AllReduce + postprocess
            layer.op_wait_mlp_ar_and_post,
        ],
    )


def _compute_moe_gptoss_decode(layer):
    # For decode, use synchronous path (no async AR needed, decode is fast)
    return OperationsStrategy(
        deep_gemm_num_sms=None,
        tbo_delta_stages=0,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.op_mlp_with_ar,
            layer.op_comm_postprocess_layer,
        ],
    )
