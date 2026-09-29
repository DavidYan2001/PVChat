"""Pure policy-objective math for Qwen3.5 Stage 3 experiments."""

from __future__ import annotations

import math

import torch


def _validated_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1 inclusive")
    return alpha


def normalize_group_advantages(
    rewards,
    alpha: float = 1.0,
    eps: float = 1e-6,
) -> list[float]:
    """Normalize rewards with configurable standard-deviation scaling."""

    alpha = _validated_alpha(alpha)
    rewards = [float(value) for value in rewards]
    if not rewards:
        return []

    mean = sum(rewards) / len(rewards)
    variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
    std = math.sqrt(variance)
    # Keep the current token-GRPO baseline behavior for numerically flat groups.
    if std < float(eps):
        return [0.0 for _ in rewards]

    denominator = (std + float(eps)) ** alpha
    return [(value - mean) / denominator for value in rewards]


def masked_sequence_logprobs(
    token_logprobs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Average valid completion-token log probabilities for each row."""

    if token_logprobs.ndim != 2 or mask.shape != token_logprobs.shape:
        raise ValueError("token_logprobs and mask must have the same 2D shape")

    valid = mask.to(device=token_logprobs.device, dtype=torch.bool)
    token_counts = valid.sum(dim=1)
    if torch.any(token_counts == 0).item():
        raise ValueError("each row must contain at least one valid completion token")

    masked_logprobs = token_logprobs.masked_fill(~valid, 0.0)
    return masked_logprobs.sum(dim=1) / token_counts.to(token_logprobs.dtype)


def supervised_anchor_loss(
    token_logprobs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return mean negative log likelihood over gold completion tokens only."""

    if token_logprobs.ndim != 2 or mask.shape != token_logprobs.shape:
        raise ValueError("token_logprobs and mask must have the same 2D shape")
    valid = mask.to(device=token_logprobs.device, dtype=torch.bool)
    count = valid.sum()
    if count.item() == 0:
        raise ValueError("supervised anchor requires at least one valid completion token")
    return -(token_logprobs * valid.to(token_logprobs.dtype)).sum() / count


def token_policy_loss(
    new_token_logprobs,
    old_token_logprobs,
    advantages,
    mask,
    clip_range: float = 0.2,
    drift_beta: float = 0.02,
):
    """Apply the existing token-level clipped policy objective."""

    advantages = advantages.to(new_token_logprobs.dtype).unsqueeze(-1)
    ratio = torch.exp(new_token_logprobs - old_token_logprobs)
    clipped_ratio = ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
    policy_terms = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    valid = mask.to(new_token_logprobs.dtype)
    policy_loss = -(policy_terms * valid).sum() / valid.sum().clamp_min(1.0)
    drift_loss = (
        ((new_token_logprobs - old_token_logprobs) ** 2) * valid
    ).sum() / valid.sum().clamp_min(1.0)
    total = policy_loss + float(drift_beta) * drift_loss
    return total, {
        "policy_loss": policy_loss.detach(),
        "drift_loss": drift_loss.detach(),
    }


def sequence_policy_loss(
    new_token_logprobs,
    old_token_logprobs,
    advantages,
    mask,
    clip_range: float = 0.2,
    drift_beta: float = 0.02,
):
    """Apply one clipped importance ratio per completion sequence."""

    new_sequence_logprobs = masked_sequence_logprobs(new_token_logprobs, mask)
    old_sequence_logprobs = masked_sequence_logprobs(old_token_logprobs, mask)
    advantages = advantages.to(new_sequence_logprobs.dtype)

    ratio = torch.exp(new_sequence_logprobs - old_sequence_logprobs)
    clipped_ratio = ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
    policy_terms = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    policy_loss = -policy_terms.mean()
    drift_loss = ((new_sequence_logprobs - old_sequence_logprobs) ** 2).mean()
    total = policy_loss + float(drift_beta) * drift_loss
    return total, {
        "policy_loss": policy_loss.detach(),
        "drift_loss": drift_loss.detach(),
    }


def decoupled_component_advantages(
    component_rows,
    component_weights,
    alpha: float,
    eps: float = 1e-6,
) -> list[float]:
    """Normalize reward components separately, then combine their advantages."""

    alpha = _validated_alpha(alpha)
    rows = list(component_rows)
    combined = [0.0 for _ in rows]

    for component_name in sorted(component_weights):
        rewards = [float(row.get(component_name, 0.0)) for row in rows]
        advantages = normalize_group_advantages(rewards, alpha=alpha, eps=eps)
        weight = float(component_weights[component_name])
        for index, advantage in enumerate(advantages):
            combined[index] += weight * advantage

    return combined


def connected_zero_loss(tensor: torch.Tensor) -> torch.Tensor:
    """Return an exact scalar zero that remains connected to ``tensor``."""

    return tensor.sum() * 0.0
