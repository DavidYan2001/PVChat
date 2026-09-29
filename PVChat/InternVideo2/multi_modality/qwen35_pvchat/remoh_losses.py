"""PVChat论文中的SPR和HAE辅助损失。

SPR推动routed heads变稀疏；HAE在过度稀疏时把休眠head重新拉回来。
硬激活比例本身不可导，因此这里使用straight-through形式：前向仍按
ReLU是否大于0统计，反向使用sigmoid近似梯度。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .remoh_attention import Qwen35ReMoHAttention


@dataclass
class ReMoHLossOutput:
    total_loss: torch.Tensor
    spr_loss: torch.Tensor
    hae_loss: torch.Tensor
    active_ratio: torch.Tensor
    spr_weight: torch.Tensor


class AdaptiveReMoHLoss(nn.Module):
    def __init__(
        self,
        target_active_ratio: float = 0.5,
        initial_spr_weight: float = 1e-8,
        adaptation_rate: float = 0.1,
        hae_weight: float = 0.5,
        temperature: float = 10.0,
    ) -> None:
        super().__init__()
        if not 0.0 < target_active_ratio <= 1.0:
            raise ValueError("target_active_ratio必须位于(0, 1]。")
        self.target_active_ratio = float(target_active_ratio)
        self.adaptation_rate = float(adaptation_rate)
        self.hae_weight = float(hae_weight)
        self.temperature = float(temperature)
        self.register_buffer("spr_weight", torch.tensor(float(initial_spr_weight)))

    def from_gates(self, gate_pairs) -> ReMoHLossOutput:
        gate_pairs = list(gate_pairs)
        if not gate_pairs:
            zero = self.spr_weight * 0.0
            return ReMoHLossOutput(zero, zero, zero, zero, self.spr_weight.detach().clone())

        logits = torch.cat([pair[0].reshape(-1) for pair in gate_pairs])
        gates = torch.cat([pair[1].reshape(-1) for pair in gate_pairs])

        soft_active = torch.sigmoid(self.temperature * logits)
        hard_active = (logits > 0).to(logits.dtype)
        straight_through_active = hard_active.detach() - soft_active.detach() + soft_active
        active_ratio = straight_through_active.mean()
        current_sparsity = 1.0 - active_ratio
        target_sparsity = 1.0 - self.target_active_ratio

        # 当前过于稠密时增大SPR权重，过于稀疏时减小。
        with torch.no_grad():
            delta = target_sparsity - current_sparsity.detach()
            self.spr_weight.mul_(torch.exp(self.adaptation_rate * delta))
            self.spr_weight.clamp_(1e-15, 10.0)

        spr_loss = self.spr_weight.to(gates.dtype) * gates.mean()
        too_sparse = current_sparsity > target_sparsity
        hae_raw = torch.exp(2.0 * (current_sparsity - target_sparsity)) - 1.0
        hae_loss = torch.where(too_sparse, hae_raw, torch.zeros_like(hae_raw))
        total = spr_loss + self.hae_weight * hae_loss
        return ReMoHLossOutput(total, spr_loss, hae_loss, active_ratio, self.spr_weight.detach().clone())

    def from_model(self, model) -> ReMoHLossOutput:
        pairs = []
        for module in model.modules():
            if isinstance(module, Qwen35ReMoHAttention) and module.last_router_output is not None:
                output = module.last_router_output
                pairs.append((output.routed_logits, output.routed_gates))
        return self.from_gates(pairs)
