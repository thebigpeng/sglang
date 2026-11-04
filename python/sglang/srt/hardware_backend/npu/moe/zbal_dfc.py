"""ZBAL dispatch/FFN/combine entry and its offline expert-weight layouts."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.topk import TopKOutput

logger = logging.getLogger(__name__)


def _reject(message: str) -> None:
    logger.error("ZBAL DFC: %s", message)
    raise ValueError(message)


def use_zbal_dfc() -> bool:
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe import get_moe_a2a_backend

    return (
        envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get() > 0
        and get_moe_a2a_backend().is_ascend_fuseep()
    )


def validate_zbal_dfc(layer: FusedMoE) -> None:
    """Reject layer semantics which the DFC ABI cannot express."""
    from sglang.srt.environ import envs
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
        NPUMXFP8MoEMethod,
        NPUW4A4MXFP4MoEMethod,
        NPUW8A8Int8MoEMethod,
    )
    from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
    from sglang.srt.layers.moe import get_moe_runner_backend
    from sglang.srt.runtime_context import get_exec, get_parallel

    config = layer.moe_runner_config
    execution = get_exec()
    moe = execution.moe
    if moe.fuseep_mode != 2:
        _reject("ZBAL requires --fuseep-mode 2 (dispatch_ffn_combine)")
    if get_parallel().pp_size != 1:
        _reject(f"ZBAL DFC requires pp_size=1, got pp_size={get_parallel().pp_size}")
    overlap = execution.overlap
    if overlap.enable_two_batch_overlap or overlap.enable_single_batch_overlap:
        _reject(
            "ZBAL DFC requires overlap disabled, got "
            f"{overlap.enable_two_batch_overlap=}, {overlap.enable_single_batch_overlap=}"
        )
    runner = get_moe_runner_backend()
    if not (runner.is_auto() or runner.is_ascend()):
        _reject(
            f"layer_id={layer.layer_id}: DFC requires --moe-runner-backend auto or ascend, got {runner.value}"
        )
    # Restrict selection before any generic NZ transform can destroy the ND MX layout.
    supported = (NPUW8A8Int8MoEMethod, NPUMXFP8MoEMethod, NPUW4A4MXFP4MoEMethod)
    for prefix in ("w13", "w2"):
        kernel = getattr(getattr(layer, f"{prefix}_scheme", None), "kernel", None)
        if type(kernel) not in supported:
            _reject(
                f"layer_id={layer.layer_id}, {prefix}: DFC supports ModelSlim W8A8 INT8, MXFP8, and W4A4 MXFP4 only"
            )
    if type(layer.w13_scheme.kernel) is not type(layer.w2_scheme.kernel):
        _reject(
            f"layer_id={layer.layer_id}: DFC requires matching w13/w2 quantization schemes"
        )
    if (
        type(layer.w13_scheme.kernel) is not NPUW8A8Int8MoEMethod
        and not is_npu_arch35()
    ):
        _reject(f"layer_id={layer.layer_id}: MX DFC requires Ascend A5 architecture 35")
    if config.activation != "silu" or not config.is_gated:
        _reject(
            f"layer_id={layer.layer_id}: DFC requires gated silu, got {config.activation=}, {config.is_gated=}"
        )
    for name in ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit", "swiglu_limit"):
        if getattr(config, name) is not None:
            _reject(
                f"layer_id={layer.layer_id}: DFC does not support {name}={getattr(config, name)}"
            )
    if config.apply_router_weight_on_input or config.no_combine or layer.with_bias:
        _reject(
            f"layer_id={layer.layer_id}: unsupported "
            f"{config.apply_router_weight_on_input=}, {config.no_combine=}, {layer.with_bias=}"
        )
    if layer.moe_tp_size != 1 or layer.num_fused_shared_experts:
        _reject(
            f"layer_id={layer.layer_id}: unsupported {layer.moe_tp_size=}, {layer.num_fused_shared_experts=}"
        )
    if not 1 <= layer.moe_ep_size <= 16 or layer.num_experts % layer.moe_ep_size:
        _reject(
            f"layer_id={layer.layer_id}: unsupported {layer.moe_ep_size=}, {layer.num_experts=}"
        )
    if (
        not 1 <= layer.num_local_experts <= 128
        or layer.num_local_experts != layer.num_experts // layer.moe_ep_size
        or moe.ep_num_redundant_experts
    ):
        _reject(
            f"layer_id={layer.layer_id}: DFC requires 1 to 128 local experts without redundant slots, "
            f"got {layer.num_local_experts=}, {layer.num_experts=}, {moe.ep_num_redundant_experts=}"
        )
    if (
        moe.enable_eplb
        or moe.ep_dispatch_algorithm is not None
        or moe.ep_join_mode is not None
    ):
        _reject(
            f"layer_id={layer.layer_id}: DFC requires static contiguous expert placement, "
            f"got {moe.enable_eplb=}, {moe.ep_dispatch_algorithm=}, {moe.ep_join_mode=}"
        )
    if os.environ.get("ZBAL_DFC_BALANCE_TOKEN", "0") not in ("", "0"):
        _reject(
            f"layer_id={layer.layer_id}: SGLang DFC requires ZBAL_DFC_BALANCE_TOKEN=0"
        )
    if (
        envs.SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT.get()
        and layer.w13_weight.dtype == torch.int8
    ):
        _reject(
            f"layer_id={layer.layer_id}: INT8 DFC requires NZ weights; ACL format conversion is disabled"
        )


def _pack_mx_scale(scale: torch.Tensor, logical_k: int) -> torch.Tensor:
    """Keep E8M0 bytes in [E,N,ceil(K/256)*8], including the dead K-tile tail."""
    if (
        scale.dtype != torch.uint8
        or scale.ndim != 3
        or scale.shape[-1] != logical_k // 32
    ):
        _reject(
            f"expected raw E8M0 [E,N,K/32], got dtype={scale.dtype}, shape={tuple(scale.shape)}, K={logical_k}"
        )
    padded_k32 = ((logical_k + 255) // 256) * 8
    return F.pad(scale, (0, padded_k32 - scale.shape[-1])).flatten(1).contiguous()


def process_zbal_dfc_weights(layer: FusedMoE, prefix: str, kind: str) -> bool:
    """Process each ModelSlim weight group before the generic NPU NZ transform."""
    if not use_zbal_dfc():
        return False
    validate_zbal_dfc(layer)
    processed = getattr(layer, "_zbal_dfc_weight_prefixes", set())
    if prefix in processed:
        return True
    weight = getattr(layer, f"{prefix}_weight").data
    scale = getattr(layer, f"{prefix}_weight_scale").data
    e = layer.num_local_experts
    h, d = layer.hidden_size, layer.intermediate_size_per_partition
    n, k = (2 * d, h) if prefix == "w13" else (h, d)
    expected_shape = (e, n, k // 2 if kind == "mxfp4" else k)
    if tuple(weight.shape) != expected_shape:
        _reject(
            f"layer_id={layer.layer_id}, {prefix}: expected {expected_shape}, got {tuple(weight.shape)}"
        )
    for suffix in ("weight_offset", "weight_bias", "scale_bias"):
        value = getattr(layer, f"{prefix}_{suffix}", None)
        if value is not None and torch.count_nonzero(value).item():
            _reject(
                f"layer_id={layer.layer_id}, {prefix}: DFC does not support nonzero {suffix}"
            )
    if kind == "int8":
        if weight.dtype != torch.int8 or scale.numel() != e * n or k % 16 or n % 32:
            _reject(
                f"layer_id={layer.layer_id}, {prefix}: INT8 requires [E,N,K], per-channel scales, K%16=0, N%32=0"
            )
        from sglang.srt.hardware_backend.npu.utils import NPUACLFormat

        weight = torch.ops.npu.npu_format_cast(
            weight.transpose(1, 2).contiguous(), int(NPUACLFormat.ACL_FORMAT_FRACTAL_NZ)
        )
        scale = (
            scale.float().contiguous().view(torch.int32).to(torch.int64).reshape(e, n)
        )
    else:
        if h % 64 or d % 64 or (2 * d) % 128:
            _reject(
                f"layer_id={layer.layer_id}: MX DFC requires hidden and intermediate dimensions divisible by 64"
            )
        if kind == "mxfp4":
            fp4 = getattr(torch, "float4_e2m1fn_x2", None)
            if fp4 is None or weight.dtype != torch.uint8:
                _reject(
                    f"layer_id={layer.layer_id}, {prefix}: MXFP4 requires native torch float4 dtype "
                    "and packed uint8 weights"
                )
            weight = weight.contiguous().view(fp4)
        elif kind != "mxfp8" or weight.dtype != torch.float8_e4m3fn:
            _reject(
                f"layer_id={layer.layer_id}, {prefix}: DFC requires offline MXFP8 E4M3 or W4A4 MXFP4 weights"
            )
        if tuple(scale.shape) != (e, n, k // 32):
            _reject(
                f"layer_id={layer.layer_id}, {prefix}: expected MX scale {(e, n, k // 32)}, "
                f"got {tuple(scale.shape)}"
            )
        scale = _pack_mx_scale(scale, k)
        weight = weight.contiguous()
    # ModelSlim loads [gate, up]; the DFC SwiGLU consumes the same split order.
    setattr(layer, f"{prefix}_weight", torch.nn.Parameter(weight, requires_grad=False))
    setattr(
        layer, f"{prefix}_weight_scale", torch.nn.Parameter(scale, requires_grad=False)
    )
    layer._zbal_dfc_weight_prefixes = processed | {prefix}
    return True


def init_zbal_dfc_buffer() -> None:
    from sglang.srt.runtime_context import get_parallel, get_resources
    from zbal.zbal.deepep_adaptor import Buffer as NativeBuffer
    from zbal.zbal_buffer import Buffer

    buffers = get_resources().buffers
    if "zbal_dfc" in buffers:
        return
    if not callable(getattr(NativeBuffer, "dispatch_ffn_combine", None)):
        _reject(
            "installed ZBAL native Buffer has no dispatch_ffn_combine; rebuild with DFC enabled"
        )
    group = get_parallel().moe_ep_group
    world_size = get_parallel().world_group.world_size
    if group.world_size != world_size:
        _reject(
            f"ZBAL DFC requires EP to span WORLD, got ep_size={group.world_size}, {world_size=}"
        )
    # The wrapper clones the group with a WORLD-wide new_group call.
    buffers["zbal_dfc"] = Buffer(group.device_group)


def _get_zbal_dfc_buffer():
    from sglang.srt.runtime_context import get_resources

    buffer = get_resources().buffers.get("zbal_dfc")
    if buffer is None:
        _reject("ZBAL DFC Buffer was not initialized before model execution")
    return buffer


@contextmanager
def zbal_dfc_batch(mode: str, token_bound: int):
    from sglang.srt.runtime_context import get_forward

    if not use_zbal_dfc():
        yield
        return
    # Idle ranks submit one dummy row. All EP ranks use the same stride.
    with get_forward().scoped(zbal_dfc_batch=(mode, max(1, token_bound))):
        yield


def forward_zbal_dfc(
    layer: FusedMoE, hidden_states: torch.Tensor, topk_output: TopKOutput
) -> torch.Tensor:
    from sglang.srt.runtime_context import get_forward

    if getattr(layer, "_zbal_dfc_weight_prefixes", set()) != {"w13", "w2"}:
        _reject(
            f"layer_id={layer.layer_id}: ZBAL DFC weights were not prepared; "
            "use ModelSlim W8A8 INT8, MXFP8, or W4A4 MXFP4"
        )
    if hidden_states.ndim != 2 or hidden_states.dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        _reject(
            f"layer_id={layer.layer_id}: expected BF16/FP16 [tokens,hidden], "
            f"got {hidden_states.dtype}, {tuple(hidden_states.shape)}"
        )
    tokens, hidden = hidden_states.shape
    topk_ids, topk_weights = topk_output.topk_ids, topk_output.topk_weights
    expected = (tokens, layer.top_k)
    if (
        hidden != layer.hidden_size
        or topk_ids.shape != expected
        or topk_weights.shape != expected
    ):
        _reject(
            f"layer_id={layer.layer_id}: incompatible input/routing shapes, "
            f"x={tuple(hidden_states.shape)}, {topk_ids.shape=}, {topk_weights.shape=}"
        )
    batch = get_forward().zbal_dfc_batch
    if batch is None:
        _reject(
            f"layer_id={layer.layer_id}: DFC batch metadata was not set by the model runner"
        )
    mode, capacity = batch
    if tokens > capacity:
        _reject(
            f"layer_id={layer.layer_id}: local {tokens=} exceeds the synchronized DFC bound {capacity=}"
        )
    if tokens == 0:
        # Idle DP ranks must still join the cross-rank handshake and run experts.
        hidden_states = hidden_states.new_zeros((1, hidden))
        topk_ids = torch.zeros(
            (1, layer.top_k), dtype=torch.int32, device=hidden_states.device
        )
        topk_weights = torch.zeros(
            (1, layer.top_k), dtype=torch.float32, device=hidden_states.device
        )
    result = _get_zbal_dfc_buffer().dispatch_ffn_combine(
        x=hidden_states.contiguous(),
        topk_idx=topk_ids.to(torch.int32).contiguous(),
        topk_weights=topk_weights.float().contiguous(),
        weight1=layer.w13_weight,
        scale1=layer.w13_weight_scale,
        weight2=layer.w2_weight,
        scale2=layer.w2_weight_scale,
        # Common bound across ranks: remote combine offsets depend on this value.
        # Each source can route every token/top-k slot to this destination rank.
        max_output_size=capacity * layer.moe_ep_size * layer.top_k,
        num_experts=layer.num_experts,
        quant_mode=1,
        mode=mode,
    )
    return result[0][:tokens]
