"""简化且可检查的token级GRPO数学函数。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def normalize_advantages(rewards, eps: float = 1e-6) -> list[float]:
    rewards = [float(value) for value in rewards]
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
    std = math.sqrt(variance)
    if std < eps:
        return [0.0 for _ in rewards]
    return [(value - mean) / (std + eps) for value in rewards]


def select_causal_positions(labels: torch.Tensor):
    """找出真正需要预测答案token的隐藏状态位置。

    因果语言模型用位置i预测位置i+1。PVChat的prompt标签都是-100，
    所以没有必要为几千个视频token计算完整词表logits。batch内不同样本的
    答案长度可以不同，这里取它们有效位置的并集，其他样本仍用-100屏蔽。
    """

    if labels.ndim != 2:
        raise ValueError("labels必须是[batch, sequence]二维tensor。")
    shifted_targets = labels[:, 1:]
    positions = shifted_targets.ne(-100).any(dim=0).nonzero(as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError("当前batch没有可训练的答案token。")
    targets = shifted_targets.index_select(1, positions)
    return positions, targets


def selected_causal_cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    """对已经筛选出的答案位置计算标准交叉熵。"""

    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1).to(logits.device),
        ignore_index=-100,
    )


def selected_token_logprobs(logits: torch.Tensor, targets: torch.Tensor):
    """返回筛选后答案token的log probability与有效mask。"""

    targets = targets.to(logits.device)
    mask = targets.ne(-100)
    safe_targets = targets.masked_fill(~mask, 0)
    logprobs = F.log_softmax(logits.float(), dim=-1).gather(
        -1,
        safe_targets.unsqueeze(-1),
    ).squeeze(-1)
    return logprobs.masked_fill(~mask, 0.0), mask


def completion_token_logprobs(logits, input_ids, labels):
    """返回每个答案token的log probability和有效mask。"""

    shifted_logits = logits[:, :-1, :].float()
    shifted_targets = input_ids[:, 1:].long()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe_targets = shifted_targets.masked_fill(~mask, 0)
    logprobs = F.log_softmax(shifted_logits, dim=-1).gather(
        -1, safe_targets.unsqueeze(-1)
    ).squeeze(-1)
    return logprobs.masked_fill(~mask, 0.0), mask


def build_completion_labels(sequences, prompt_length: int, stop_token_ids) -> torch.Tensor:
    """屏蔽prompt，并在每行第一个停止token之后屏蔽padding。"""

    labels = sequences.clone()
    labels[:, :prompt_length] = -100
    stop_token_ids = {int(token_id) for token_id in stop_token_ids if token_id is not None}
    for row_index in range(labels.shape[0]):
        for column in range(prompt_length, labels.shape[1]):
            if int(sequences[row_index, column]) in stop_token_ids:
                labels[row_index, column + 1 :] = -100
                break
    return labels


def token_grpo_loss(
    new_logprobs,
    old_logprobs,
    advantages,
    mask,
    clip_range: float = 0.2,
    drift_beta: float = 0.02,
):
    """PPO裁剪目标；drift项限制当前策略偏离本轮采样策略。"""

    advantages = advantages.to(new_logprobs.dtype).unsqueeze(-1)
    ratio = torch.exp(new_logprobs - old_logprobs)
    clipped_ratio = ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
    policy_terms = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    valid = mask.to(new_logprobs.dtype)
    policy_loss = -(policy_terms * valid).sum() / valid.sum().clamp_min(1.0)
    drift_loss = (((new_logprobs - old_logprobs) ** 2) * valid).sum() / valid.sum().clamp_min(1.0)
    total = policy_loss + float(drift_beta) * drift_loss
    return total, {"policy_loss": policy_loss.detach(), "drift_loss": drift_loss.detach()}


def reference_kl_loss(new_logprobs, ref_logprobs, mask):
    """k3 KL估计 (Schulman): exp(ref-new) - (ref-new) - 1, 对有效token取均值。

    锚向冻结参照模型的绝对约束; 与drift(锚向滚动行为快照)互补。
    diff截断到[-20, 20]防exp溢出。
    """
    valid = mask.to(new_logprobs.dtype)
    diff = (ref_logprobs - new_logprobs).clamp(-20.0, 20.0)
    k3 = torch.exp(diff) - diff - 1.0
    return (k3 * valid).sum() / valid.sum().clamp_min(1.0)
