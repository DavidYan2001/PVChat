"""Identity-gated overlay for the existing Dynamic-GSPO policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping, Sequence

from .adaptive_policy import SUPPORTED_ROLLOUT_LENGTHS, decide_dynamic_rollout
from .identity_presence import ABSENT, PRESENT, UNKNOWN, detect_target_identity_presence
from .policy_objectives import normalize_group_advantages
from .rewards import RewardResult, score_answer


IDENTITY_CORRECT = 1.0
IDENTITY_UNKNOWN = 0.0
IDENTITY_WRONG = -1.0
_IDENTITY_VALUES = {IDENTITY_CORRECT, IDENTITY_UNKNOWN, IDENTITY_WRONG}


def _finite_float(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _identity_values(values: Sequence[float]) -> list[float]:
    result = [_finite_float(value, "identity state") for value in values]
    if any(value not in _IDENTITY_VALUES for value in result):
        raise ValueError("identity states must be -1, 0, or 1")
    return result


def identity_gate_value(presence: str, is_positive: bool) -> float:
    if presence == UNKNOWN:
        return IDENTITY_UNKNOWN
    if presence not in {PRESENT, ABSENT}:
        raise ValueError(f"unknown presence label: {presence!r}")
    expected = PRESENT if bool(is_positive) else ABSENT
    return IDENTITY_CORRECT if presence == expected else IDENTITY_WRONG


def score_identity_gated_answer(
    question: str,
    gold_answer: str,
    candidate_answer: str,
    person_token: str,
    is_positive: bool,
    qa_type: str,
) -> RewardResult:
    """Keep the legacy scalar reward and append conservative gate evidence."""

    legacy = score_answer(
        question,
        gold_answer,
        candidate_answer,
        person_token,
        is_positive,
        qa_type,
    )
    presence = detect_target_identity_presence(
        candidate_answer,
        person_token,
        is_identity_question=qa_type == "identity",
    )
    components = dict(legacy.components)
    components["identity_gate"] = identity_gate_value(presence, is_positive)
    return RewardResult(legacy.reward, components)


@dataclass(frozen=True)
class IdentityGatedRolloutDecision:
    target_count: int
    should_update: bool
    reason: str
    observed_count: int
    identity_triggered: bool
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    @property
    def needs_more(self) -> bool:
        return self.target_count > self.observed_count


def decide_identity_gated_rollout(
    rewards: Sequence[float],
    identity_states: Sequence[float],
) -> IdentityGatedRolloutDecision:
    rewards = [_finite_float(value, "reward") for value in rewards]
    states = _identity_values(identity_states)
    count = len(rewards)
    if count not in SUPPORTED_ROLLOUT_LENGTHS:
        raise ValueError(f"rewards must have length 2, 4, or 8; got {count}")
    if len(states) != count:
        raise ValueError("identity states must align with rewards")

    correct = sum(value == IDENTITY_CORRECT for value in states)
    wrong = sum(value == IDENTITY_WRONG for value in states)
    unknown = count - correct - wrong
    diagnostics = {
        "identity_correct": float(correct),
        "identity_wrong": float(wrong),
        "identity_unknown": float(unknown),
    }

    if wrong == 0:
        legacy = decide_dynamic_rollout(rewards)
        return IdentityGatedRolloutDecision(
            legacy.target_count,
            legacy.should_update,
            legacy.reason,
            count,
            False,
            diagnostics,
        )
    if correct > 0:
        return IdentityGatedRolloutDecision(
            count,
            True,
            "identity_mixed",
            count,
            True,
            diagnostics,
        )
    if count < 8:
        target = 4 if count == 2 else 8
        return IdentityGatedRolloutDecision(
            target,
            False,
            "identity_expand_no_correct",
            count,
            True,
            diagnostics,
        )
    return IdentityGatedRolloutDecision(
        8,
        False,
        "identity_no_correct",
        count,
        True,
        diagnostics,
    )


def identity_gated_advantages(
    rewards: Sequence[float],
    identity_states: Sequence[float],
    identity_margin: float = 1.0,
    soft_clip: float = 0.25,
    eps: float = 1e-6,
) -> list[float]:
    rewards = [_finite_float(value, "reward") for value in rewards]
    states = _identity_values(identity_states)
    if not rewards:
        return []
    if len(states) != len(rewards):
        raise ValueError("identity states must align with rewards")
    identity_margin = _finite_float(identity_margin, "identity_margin")
    soft_clip = _finite_float(soft_clip, "soft_clip")
    if soft_clip < 0:
        raise ValueError("soft_clip must be non-negative")
    if identity_margin <= 2.0 * soft_clip:
        raise ValueError("identity_margin must be greater than 2 * soft_clip")

    legacy = normalize_group_advantages(rewards, alpha=1.0, eps=eps)
    if IDENTITY_WRONG not in states:
        return legacy

    clipped = [max(-soft_clip, min(soft_clip, value)) for value in legacy]
    clipped_mean = sum(clipped) / len(clipped)
    centered_soft = [value - clipped_mean for value in clipped]
    tier_mean = sum(states) / len(states)
    return [
        identity_margin * (state - tier_mean) + soft
        for state, soft in zip(states, centered_soft)
    ]
