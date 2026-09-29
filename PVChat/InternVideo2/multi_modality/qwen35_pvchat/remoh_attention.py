"""Qwen3.5使用的恒等初始化ReMoH注意力。

模块完整保留Qwen3.5预训练注意力投影和原有16个query head，只把它们划分为
12个shared head与4个routed head。ReMoH只调节文本token接收到的视频信息，
不会改写视频token之间或纯文本之间原有的注意力结果。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)


DEFAULT_REMOH_LAYER_INDICES = (7, 11, 15, 19)
DEFAULT_ROUTED_HEAD_INDICES = (3, 7, 11, 15)


@dataclass
class ReMoHRouterOutput:
    """保存每个token的路由值，供ReMoH前向和辅助loss共同使用。"""

    alpha: torch.Tensor
    routed_logits: torch.Tensor
    routed_gates: torch.Tensor
    shared_head_weights: torch.Tensor
    routed_head_weights: torch.Tensor
    head_weights: torch.Tensor


class Qwen35ReMoHRouter(nn.Module):
    """带严格恒等初始化的双分支ReMoH路由器。

    两个分支softmax从``(0.5, 0.5)``开始，再统一乘2；routed ReLU分数从1
    开始。因此第一次参数更新前，每个预训练head的实际乘数都严格等于1。
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        routed_head_indices: Sequence[int] = DEFAULT_ROUTED_HEAD_INDICES,
    ) -> None:
        super().__init__()
        routed_head_indices = tuple(int(index) for index in routed_head_indices)
        if not routed_head_indices:
            raise ValueError("At least one routed head is required.")
        if len(set(routed_head_indices)) != len(routed_head_indices):
            raise ValueError("routed_head_indices must not contain duplicates.")
        if min(routed_head_indices) < 0 or max(routed_head_indices) >= num_attention_heads:
            raise ValueError(
                f"Routed head indices {routed_head_indices} are invalid for "
                f"{num_attention_heads} attention heads."
            )

        self.hidden_size = int(hidden_size)
        self.num_attention_heads = int(num_attention_heads)
        self.routed_head_indices = routed_head_indices
        self.num_routed_heads = len(routed_head_indices)
        self.num_shared_heads = self.num_attention_heads - self.num_routed_heads

        self.router = nn.Linear(self.hidden_size, self.num_routed_heads, bias=True)
        self.alpha_proj = nn.Linear(self.hidden_size, 2, bias=False)

        routed_selector = torch.zeros(self.num_routed_heads, self.num_attention_heads)
        for routed_position, head_index in enumerate(self.routed_head_indices):
            routed_selector[routed_position, head_index] = 1.0
        shared_selector = 1.0 - routed_selector.sum(dim=0)
        self.register_buffer("routed_selector", routed_selector, persistent=False)
        self.register_buffer("shared_selector", shared_selector, persistent=False)

        self.reset_identity_parameters()

    def reset_identity_parameters(self) -> None:
        """只重置新增路由参数，绝不改动预训练注意力参数。"""

        with torch.no_grad():
            self.alpha_proj.weight.zero_()
            self.router.weight.zero_()
            self.router.bias.fill_(1.0)

    def forward(self, hidden_states: torch.Tensor) -> ReMoHRouterOutput:
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, got {hidden_states.shape[-1]}."
            )

        alpha_logits = self.alpha_proj(hidden_states)
        alpha = F.softmax(alpha_logits.float(), dim=-1).to(hidden_states.dtype)

        routed_logits = self.router(hidden_states)
        routed_gates = F.relu(routed_logits)

        # 用2抵消初始分支概率0.5，在保留softmax竞争关系的同时保持原head输出。
        shared_weight = 2.0 * alpha[..., 0:1]
        routed_weights = 2.0 * alpha[..., 1:2] * routed_gates

        shared_selector = self.shared_selector.to(dtype=hidden_states.dtype)
        routed_selector = self.routed_selector.to(dtype=hidden_states.dtype)
        head_weights = (
            shared_weight * shared_selector
            + torch.matmul(routed_weights, routed_selector)
        )
        shared_head_weights = shared_weight.expand(
            *shared_weight.shape[:-1], self.num_shared_heads
        )

        return ReMoHRouterOutput(
            alpha=alpha,
            routed_logits=routed_logits,
            routed_gates=routed_gates,
            shared_head_weights=shared_head_weights,
            routed_head_weights=routed_weights,
            head_weights=head_weights,
        )


def apply_head_weights(context: torch.Tensor, head_weights: torch.Tensor) -> torch.Tensor:
    """把``[batch, query, heads, dim]``上下文乘以逐head权重。"""

    if context.ndim != 4:
        raise ValueError(f"Expected a 4D attention context, got shape {tuple(context.shape)}.")
    if head_weights.shape != context.shape[:-1]:
        raise ValueError(
            f"Head-weight shape {tuple(head_weights.shape)} does not match "
            f"context shape {tuple(context.shape)}."
        )
    return context * head_weights.unsqueeze(-1)


def _align_token_mask(mask: torch.Tensor, target_length: int, pad_value: bool) -> torch.Tensor:
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    if mask.ndim != 2:
        raise ValueError(f"Token mask must have shape [batch, sequence], got {tuple(mask.shape)}.")
    mask = mask.to(dtype=torch.bool)
    current_length = mask.shape[1]
    if current_length < target_length:
        mask = F.pad(mask, (0, target_length - current_length), value=pad_value)
    elif current_length > target_length:
        mask = mask[:, :target_length]
    return mask


class Qwen35ReMoHAttention(nn.Module):
    """可直接替换Qwen3.5 full-attention层的包装器。

    第二次注意力计算复用同一组Q/K，只把V中的非视频位置清零，从而在不显式
    保存完整注意力矩阵的情况下分离视频贡献。Qwen3.5-9B注意力dropout为0，
    因此该分解是严格等价的，也能继续使用节省显存的注意力后端。
    """

    def __init__(
        self,
        original_attention: nn.Module,
        routed_head_indices: Sequence[int] = DEFAULT_ROUTED_HEAD_INDICES,
    ) -> None:
        super().__init__()
        self.config = original_attention.config
        self.layer_idx = original_attention.layer_idx
        self.head_dim = original_attention.head_dim
        self.num_key_value_groups = original_attention.num_key_value_groups
        self.scaling = original_attention.scaling
        self.attention_dropout = original_attention.attention_dropout
        self.is_causal = original_attention.is_causal

        # 直接复用原投影模块，替换容器后预训练参数的值不会发生变化。
        self.q_proj = original_attention.q_proj
        self.k_proj = original_attention.k_proj
        self.v_proj = original_attention.v_proj
        self.o_proj = original_attention.o_proj
        self.q_norm = original_attention.q_norm
        self.k_norm = original_attention.k_norm

        self.num_attention_heads = int(self.config.num_attention_heads)
        self.num_key_value_heads = int(self.config.num_key_value_heads)
        self.router = Qwen35ReMoHRouter(
            hidden_size=int(self.config.hidden_size),
            num_attention_heads=self.num_attention_heads,
            routed_head_indices=routed_head_indices,
        )
        # 基座通常以bfloat16加载；新建模块默认是float32。这里显式继承
        # 原注意力投影的设备和dtype，避免真实模型前向时发生类型冲突。
        projection_weight = self.q_proj.weight
        self.router.to(device=projection_weight.device, dtype=projection_weight.dtype)
        self.last_router_output: ReMoHRouterOutput | None = None
        self.generation_video_token_mask: torch.Tensor | None = None
        self.generation_text_token_mask: torch.Tensor | None = None

    def reset_identity_parameters(self) -> None:
        self.router.reset_identity_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        pvchat_video_token_mask: torch.Tensor | None = None,
        pvchat_text_token_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, native_gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        native_gate = native_gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
            )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            eager_attention_forward,
        )
        dropout = 0.0 if not self.training else self.attention_dropout
        attention_output, attention_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=dropout,
            scaling=self.scaling,
            **kwargs,
        )

        # generate()会拒绝模型签名之外的自定义参数，因此生成阶段通过模块上的
        # 临时mask传入；普通训练前向仍优先使用函数参数中的mask。
        if pvchat_video_token_mask is None:
            pvchat_video_token_mask = self.generation_video_token_mask
        if pvchat_text_token_mask is None:
            pvchat_text_token_mask = self.generation_text_token_mask

        self.last_router_output = None
        if pvchat_video_token_mask is not None:
            key_length = value_states.shape[-2]
            query_length = query_states.shape[-2]
            video_mask = _align_token_mask(
                pvchat_video_token_mask.to(value_states.device),
                key_length,
                pad_value=False,
            )
            video_values = value_states * video_mask[:, None, :, None].to(value_states.dtype)
            video_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                video_values,
                attention_mask,
                dropout=dropout,
                scaling=self.scaling,
                **kwargs,
            )

            router_output = self.router(hidden_states)
            weighted_video_output = apply_head_weights(
                video_output,
                router_output.head_weights,
            )

            if pvchat_text_token_mask is None:
                full_text_mask = ~video_mask
            else:
                full_text_mask = _align_token_mask(
                    pvchat_text_token_mask.to(value_states.device),
                    key_length,
                    pad_value=True,
                )
            query_start = max(0, key_length - query_length)
            text_query_mask = full_text_mask[:, query_start:key_length]
            if text_query_mask.shape[1] != query_length:
                text_query_mask = _align_token_mask(
                    text_query_mask,
                    query_length,
                    pad_value=True,
                )
            text_query_mask = text_query_mask[:, :, None, None].to(attention_output.dtype)

            attention_output = attention_output + text_query_mask * (
                weighted_video_output - video_output
            )
            self.last_router_output = router_output

        attention_output = attention_output.reshape(*input_shape, -1).contiguous()
        attention_output = attention_output * torch.sigmoid(native_gate)
        attention_output = self.o_proj(attention_output)
        return attention_output, attention_weights


def _resolve_language_layers(model) -> Iterable[nn.Module]:
    candidates = (
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("layers",),
    )
    for path in candidates:
        current = model
        for attribute in path:
            if not hasattr(current, attribute):
                break
            current = getattr(current, attribute)
        else:
            return current
    raise ValueError("Could not find Qwen3.5 language-model layers on the supplied model.")


def patch_qwen35_remoh_layers(
    model,
    layer_indices: Sequence[int] = DEFAULT_REMOH_LAYER_INDICES,
    routed_head_indices: Sequence[int] = DEFAULT_ROUTED_HEAD_INDICES,
) -> tuple[int, ...]:
    """基础权重加载完成后，只替换指定的Qwen3.5 full-attention容器。"""

    layers = _resolve_language_layers(model)
    patched = []
    for layer_index in tuple(int(index) for index in layer_indices):
        if layer_index < 0 or layer_index >= len(layers):
            raise ValueError(f"ReMoH layer index {layer_index} is outside 0..{len(layers) - 1}.")
        layer = layers[layer_index]
        if getattr(layer, "block_type", None) != "full_attention":
            raise ValueError(f"Layer {layer_index} is not a Qwen3.5 full-attention layer.")
        if isinstance(layer.self_attn, Qwen35ReMoHAttention):
            patched.append(layer_index)
            continue
        layer.self_attn = Qwen35ReMoHAttention(
            layer.self_attn,
            routed_head_indices=routed_head_indices,
        )
        patched.append(layer_index)
    return tuple(patched)


def reset_all_remoh_identity_parameters(model) -> int:
    """把所有已替换ReMoH层恢复到恒等初始化，并返回层数。"""

    count = 0
    for module in model.modules():
        if isinstance(module, Qwen35ReMoHAttention):
            module.reset_identity_parameters()
            count += 1
    return count


@contextmanager
def remoh_generation_masks(model, video_token_mask, text_token_mask):
    """在一次generate调用期间临时向所有ReMoH层提供token mask。

    上下文退出时恢复原值，即使生成抛出异常也不会把上一个视频的mask留给
    下一个样本。缓存解码时序列会逐token增长，层内的_align_token_mask会
    自动把新增token视为文本。
    """

    patched_modules = []
    for module in model.modules():
        if not isinstance(module, Qwen35ReMoHAttention):
            continue
        previous = (
            module.generation_video_token_mask,
            module.generation_text_token_mask,
        )
        module.generation_video_token_mask = video_token_mask
        module.generation_text_token_mask = text_token_mask
        patched_modules.append((module, previous))
    try:
        yield
    finally:
        for module, previous in patched_modules:
            (
                module.generation_video_token_mask,
                module.generation_text_token_mask,
            ) = previous
