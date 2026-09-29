"""Qwen3.5基础模型、个性化token、ReMoH和LoRA的统一加载入口。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .checkpoint import TRAINABLE_STATE_NAME, load_checkpoint_metadata, load_trainable_state
from .personalization import install_personalized_token_modules
from .remoh_attention import (
    DEFAULT_REMOH_LAYER_INDICES,
    DEFAULT_ROUTED_HEAD_INDICES,
    patch_qwen35_remoh_layers,
)


@dataclass
class Qwen35ModelBundle:
    model: nn.Module
    processor: object
    personalized_tokens: list[str]
    personalized_token_ids: list[int]
    metadata: dict


class LoRALinear(nn.Module):
    """不依赖PEFT的标准LoRA线性层。

    原始线性层完整保留并冻结；训练时只更新两个很小的低秩矩阵A和B。
    B初始化为0，所以安装LoRA后的第一次前向与原模型逐元素完全相同，
    不会在训练开始前破坏Qwen3.5已有能力。
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear只能包装nn.Linear。")
        if int(rank) <= 0:
            raise ValueError("LoRA rank必须大于0。")

        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))

        # A使用常规随机初始化，B从0开始。这样A和B的乘积初始为0。
        weight = self.base_layer.weight
        self.lora_A = nn.Parameter(
            torch.empty(
                self.rank,
                self.base_layer.in_features,
                device=weight.device,
                dtype=weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                self.base_layer.out_features,
                self.rank,
                device=weight.device,
                dtype=weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        """兼容少数会读取linear.weight的Transformers代码。"""

        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    @property
    def in_features(self):
        return self.base_layer.in_features

    @property
    def out_features(self):
        return self.base_layer.out_features

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(hidden_states)
        low_rank = F.linear(self.dropout(hidden_states), self.lora_A)
        low_rank = F.linear(low_rank, self.lora_B)
        return base_output + low_rank * self.scaling


def parse_int_tuple(value, default=()):
    if value is None:
        return tuple(default)
    if isinstance(value, (tuple, list)):
        return tuple(int(item) for item in value)
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def find_language_lora_targets(model) -> list[str]:
    """沿用YoLLaVA：语言模型中的Linear都加LoRA，但排除视觉塔和lm_head。"""

    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "language_model" not in name:
            continue
        if any(
            value in name
            for value in ("router", "alpha_proj", "personal_rows", "lm_head", "base_layer")
        ):
            continue
        targets.append(name)
    if not targets:
        raise RuntimeError("没有找到Qwen3.5语言模型LoRA目标层。")
    return sorted(targets)


def _resolve_parent_module(model: nn.Module, module_name: str):
    """根据named_modules返回的点分路径找到父模块和子模块名。"""

    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_language_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> list[str]:
    """把Qwen语言模型中的目标Linear原地替换为LoRALinear。

    先收集全部路径，再执行替换，避免一边遍历模块树一边修改它。
    返回值用于日志和测试，能够明确看到LoRA实际装到了哪些层。
    """

    targets = find_language_lora_targets(model)
    for target in targets:
        parent, child_name = _resolve_parent_module(model, target)
        base_layer = getattr(parent, child_name)
        setattr(
            parent,
            child_name,
            LoRALinear(base_layer, rank=rank, alpha=alpha, dropout=dropout),
        )
    return targets


def set_lora_dropout(model: nn.Module, probability: float) -> int:
    """统一修改自定义LoRA dropout，并返回受影响层数。"""

    probability = float(probability)
    if not 0.0 <= probability < 1.0:
        raise ValueError("LoRA dropout必须位于[0, 1)。")
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.dropout.p = probability
            count += 1
    return count


def _enable_pvchat_parameters(model):
    """只打开PVChat新增的小参数，确保基础Qwen权重始终冻结。"""

    for name, parameter in model.named_parameters():
        if (
            "personal_rows" in name
            or ".router.router." in name
            or ".router.alpha_proj." in name
            or "lora_A" in name
            or "lora_B" in name
        ):
            parameter.requires_grad_(True)


def load_qwen35_pvchat_model(
    model_path: str,
    person_token: str,
    device: torch.device,
    checkpoint_path: str | None = None,
    num_detail_tokens: int = 16,
    remoh_layers=DEFAULT_REMOH_LAYER_INDICES,
    routed_heads=DEFAULT_ROUTED_HEAD_INDICES,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    attn_implementation: str = "sdpa",
    gradient_checkpointing: bool = True,
) -> Qwen35ModelBundle:
    """按不会破坏预训练权重的顺序构建模型。"""

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    metadata_from_checkpoint = None
    if checkpoint_path:
        metadata_from_checkpoint = load_checkpoint_metadata(checkpoint_path)
        person_token = metadata_from_checkpoint.get("person_token", person_token)
        num_detail_tokens = int(metadata_from_checkpoint.get("num_detail_tokens", num_detail_tokens))
        remoh_layers = metadata_from_checkpoint.get("remoh_layers", remoh_layers)
        routed_heads = metadata_from_checkpoint.get("routed_heads", routed_heads)
        lora_r = int(metadata_from_checkpoint.get("lora_r", lora_r))
        lora_alpha = int(metadata_from_checkpoint.get("lora_alpha", lora_alpha))
        lora_dropout = float(metadata_from_checkpoint.get("lora_dropout", lora_dropout))

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # 同一视频对应几十条QA，缓存解码结果避免每条都重新解码整个视频。
    from .data import install_video_decode_cache

    install_video_decode_cache(processor)

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    # 先冻结完整基座，再安装个性化token、ReMoH和LoRA。后面新创建的参数
    # 默认可训练，因此不会误把9B基座权重放进优化器。
    model.requires_grad_(False)

    personal = install_personalized_token_modules(
        model,
        processor.tokenizer,
        person_token,
        num_detail_tokens=num_detail_tokens,
    )
    remoh_layers = parse_int_tuple(remoh_layers, DEFAULT_REMOH_LAYER_INDICES)
    routed_heads = parse_int_tuple(routed_heads, DEFAULT_ROUTED_HEAD_INDICES)
    patch_qwen35_remoh_layers(model, remoh_layers, routed_heads)

    if lora_r > 0:
        inject_language_lora(
            model,
            rank=int(lora_r),
            alpha=int(lora_alpha),
            dropout=float(lora_dropout),
        )
    _enable_pvchat_parameters(model)

    if checkpoint_path:
        load_trainable_state(model, Path(checkpoint_path) / TRAINABLE_STATE_NAME)

    model.config.use_cache = False
    if getattr(model.config, "text_config", None) is not None:
        model.config.text_config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.to(device)

    metadata = {
        "base_model_path": str(Path(model_path).resolve()),
        "person_token": personal.tokens[0],
        "num_detail_tokens": num_detail_tokens,
        "personalized_tokens": personal.tokens,
        "remoh_layers": list(remoh_layers),
        "routed_heads": list(routed_heads),
        "lora_r": int(lora_r),
        "lora_alpha": int(lora_alpha),
        "lora_dropout": float(lora_dropout),
        "attn_implementation": attn_implementation,
    }
    return Qwen35ModelBundle(
        model=model,
        processor=processor,
        personalized_tokens=personal.tokens,
        personalized_token_ids=personal.token_ids,
        metadata=metadata,
    )


def build_optimizer(model, token_lr=1e-4, remoh_lr=1e-6, lora_lr=1e-6, weight_decay=0.0):
    personal, remoh, lora = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "personal_rows" in name:
            personal.append(parameter)
        elif ".router.router." in name or ".router.alpha_proj." in name:
            remoh.append(parameter)
        elif "lora_" in name:
            lora.append(parameter)
        else:
            raise RuntimeError(f"出现未分类的可训练参数: {name}")
    groups = []
    if personal:
        groups.append({"params": personal, "lr": float(token_lr), "weight_decay": 0.0})
    if remoh:
        groups.append({"params": remoh, "lr": float(remoh_lr), "weight_decay": 0.0})
    if lora:
        groups.append({"params": lora, "lr": float(lora_lr), "weight_decay": float(weight_decay)})
    if not groups:
        raise RuntimeError("模型中没有可训练参数。")
    return torch.optim.AdamW(groups)


def trainable_parameter_summary(model):
    groups = {"personal": 0, "remoh": 0, "lora": 0, "other": 0}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        count = parameter.numel()
        if "personal_rows" in name:
            groups["personal"] += count
        elif ".router.router." in name or ".router.alpha_proj." in name:
            groups["remoh"] += count
        elif "lora_" in name:
            groups["lora"] += count
        else:
            groups["other"] += count
    groups["total"] = sum(groups.values())
    return groups
