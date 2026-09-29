"""ICD-GSPO 使用的身份约束 reward、advantage 与 rollout 决策。

这个模块不依赖模型或训练器，所有函数都可以单独测试。身份正确性是
不可被其他分数补偿的可行性约束；语义、人物特异性和描述覆盖度仅在
身份可行候选中提供软优化信号。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import isfinite
import re
from typing import Mapping, Sequence

from .personalization import normalize_person_token
from .rewards import (
    RewardResult,
    _pa_person_presence,
    _semantic_score,
    clean_text,
    words,
)


SOFT_COMPONENTS = ("semantic", "specificity", "coverage")

_CONTENT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "being",
    "by", "can", "cannot", "could", "does", "doing", "for", "from", "has",
    "have", "he", "her", "hers", "him", "his", "i", "in", "is", "it", "its",
    "not", "of", "on", "or", "she", "that", "the", "their", "them", "they",
    "this", "to", "video", "visible", "was", "wearing", "were", "what", "where",
    "who", "with",
}


def _finite(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def active_soft_weights(qa_type: str, is_positive: bool) -> dict[str, float]:
    """返回当前题目真正启用的软 reward，并保证权重和为一。"""

    if not bool(is_positive):
        # 负样本只奖励与“人物缺席”GT一致，避免详细幻觉获得额外分数。
        return {"semantic": 1.0}
    if str(qa_type) == "identity":
        return {"semantic": 0.5, "specificity": 0.5}
    return {"semantic": 0.5, "specificity": 0.2, "coverage": 0.3}


def _content_words(text: str, person_token: str) -> set[str]:
    bare = normalize_person_token(person_token).strip("<>").lower()
    return {
        token
        for token in words(text)
        if token != bare and token not in _CONTENT_STOPWORDS
    }


def _coverage_score(gold_answer: str, candidate_answer: str, person_token: str) -> float:
    gold = _content_words(gold_answer, person_token)
    if not gold:
        return 1.0 if clean_text(candidate_answer) else 0.0
    candidate = _content_words(candidate_answer, person_token)
    return len(gold & candidate) / len(gold)


def _specificity_score(answer: str, person_token: str) -> float:
    """奖励明确点名，同时避免短名字误命中普通单词。"""

    text = clean_text(answer).lower()
    token = normalize_person_token(person_token).lower()
    bare = token.strip("<>")
    if token in text:
        return 1.0
    if bare and re.search(rf"(?<!\w){re.escape(bare)}(?!\w)", text):
        return 0.8
    if re.search(r"\b(?:the person|the man|the woman|he|she)\b", text):
        return 0.3
    return 0.0


def score_icd_answer(
    question: str,
    gold_answer: str,
    candidate_answer: str,
    person_token: str,
    is_positive: bool,
    qa_type: str,
) -> RewardResult:
    """计算统一四维 reward；标量 reward 仅用于日志，不参与 advantage。"""

    del question  # qa_type 已由调用方统一分类，避免这里重复启发式判断。
    candidate_answer = clean_text(candidate_answer)
    predicted_presence = _pa_person_presence(candidate_answer, person_token)
    if predicted_presence is None:
        identity = 0.0
    else:
        identity = 1.0 if predicted_presence is bool(is_positive) else -1.0

    semantic = _semantic_score(gold_answer, candidate_answer, person_token)
    specificity = _specificity_score(candidate_answer, person_token) if is_positive else 0.0
    coverage = _coverage_score(gold_answer, candidate_answer, person_token) if is_positive else 0.0
    components = {
        "identity": identity,
        "semantic": semantic,
        "specificity": specificity,
        "coverage": coverage,
    }
    weights = active_soft_weights(qa_type, is_positive)
    soft_score = sum(weights[name] * components[name] for name in weights)
    # 这个标量也保持身份优先，方便日志排序时不重新引入补偿问题。
    reward = -1.0 if identity < 0 else 0.0 if identity == 0 else 0.5 + 0.5 * soft_score
    return RewardResult(float(reward), components)


def _validate_margin(identity_margin: float, soft_clip: float) -> tuple[float, float]:
    identity_margin = _finite(identity_margin, "identity_margin")
    soft_clip = _finite(soft_clip, "soft_clip")
    if identity_margin < 0 or soft_clip < 0:
        raise ValueError("identity_margin and soft_clip must be non-negative")
    if identity_margin < soft_clip:
        raise ValueError("identity_margin must be at least soft_clip")
    return identity_margin, soft_clip


def identity_constrained_advantages(
    component_rows: Sequence[Mapping[str, float]],
    qa_type: str,
    is_positive: bool,
    identity_threshold: float = 0.5,
    identity_margin: float = 1.0,
    soft_clip: float = 0.5,
) -> list[float]:
    """计算带可行性排序保证的组内 advantage。

    当 ``identity_margin >= soft_clip`` 时，任意可行候选的 advantage 都
    不小于任意不可行候选。只有一个可行候选时不估计软相对优势。
    """

    rows = [dict(row) for row in component_rows]
    if not rows:
        return []
    identity_threshold = _finite(identity_threshold, "identity_threshold")
    identity_margin, soft_clip = _validate_margin(identity_margin, soft_clip)
    identities = [_finite(row.get("identity", 0.0), "identity") for row in rows]
    feasible = [value >= identity_threshold for value in identities]
    feasible_indices = [index for index, value in enumerate(feasible) if value]
    feasible_rate = len(feasible_indices) / len(rows)

    if not feasible_indices:
        return [0.0 for _ in rows]

    soft_scores = [0.0 for _ in rows]
    if len(feasible_indices) > 1:
        weights = active_soft_weights(qa_type, is_positive)
        means = {
            name: sum(_finite(rows[index].get(name, 0.0), name) for index in feasible_indices)
            / len(feasible_indices)
            for name in weights
        }
        for index in feasible_indices:
            centered = sum(
                weight * (_finite(rows[index].get(name, 0.0), name) - means[name])
                for name, weight in weights.items()
            )
            soft_scores[index] = max(-soft_clip, min(soft_clip, centered))

    return [
        identity_margin * ((1.0 if is_feasible else 0.0) - feasible_rate)
        + ((1.0 if is_feasible else 0.0) * soft_scores[index])
        for index, is_feasible in enumerate(feasible)
    ]


def _sign(value: float, tolerance: float = 1e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _rank_disagreement(
    rows: Sequence[Mapping[str, float]],
    indices: Sequence[int],
    component_names: Sequence[str],
) -> float:
    """计算 reward 维度之间平均的 Kendall 式候选排序冲突。"""

    if len(indices) < 2 or len(component_names) < 2:
        return 0.0
    disagreements = 0
    comparisons = 0
    for first_name, second_name in combinations(component_names, 2):
        for left, right in combinations(indices, 2):
            first = _sign(
                _finite(rows[left].get(first_name, 0.0), first_name)
                - _finite(rows[right].get(first_name, 0.0), first_name)
            )
            second = _sign(
                _finite(rows[left].get(second_name, 0.0), second_name)
                - _finite(rows[right].get(second_name, 0.0), second_name)
            )
            if first == 0 or second == 0:
                continue
            comparisons += 1
            disagreements += int(first != second)
    return disagreements / comparisons if comparisons else 0.0


def vector_reward_diagnostics(
    component_rows: Sequence[Mapping[str, float]],
    qa_type: str,
    is_positive: bool,
    identity_threshold: float = 0.5,
) -> dict[str, float]:
    rows = [dict(row) for row in component_rows]
    if not rows:
        raise ValueError("component_rows must not be empty")
    threshold = _finite(identity_threshold, "identity_threshold")
    feasible_indices = [
        index
        for index, row in enumerate(rows)
        if _finite(row.get("identity", 0.0), "identity") >= threshold
    ]
    feasibility_rate = len(feasible_indices) / len(rows)
    identity_uncertainty = 1.0 - abs(2.0 * feasibility_rate - 1.0)
    component_names = tuple(active_soft_weights(qa_type, is_positive))

    if len(feasible_indices) < 2:
        soft_dispersion = 0.0
    else:
        ranges = []
        for name in component_names:
            values = [_finite(rows[index].get(name, 0.0), name) for index in feasible_indices]
            ranges.append(max(values) - min(values))
        soft_dispersion = sum(ranges) / len(ranges) if ranges else 0.0
    rank_disagreement = _rank_disagreement(rows, feasible_indices, component_names)
    uncertainty = 0.5 * identity_uncertainty + 0.25 * soft_dispersion + 0.25 * rank_disagreement
    return {
        "feasibility_rate": feasibility_rate,
        "identity_uncertainty": identity_uncertainty,
        "soft_dispersion": soft_dispersion,
        "rank_disagreement": rank_disagreement,
        "vector_uncertainty": uncertainty,
    }


@dataclass(frozen=True)
class ICDRolloutDecision:
    target_count: int
    should_update: bool
    reason: str
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    @property
    def needs_more(self) -> bool:
        return self.reason.startswith("expand")


def _has_advantage_signal(values: Sequence[float], tolerance: float = 1e-12) -> bool:
    return any(abs(float(value)) > tolerance for value in values)


def decide_fixed_icd_rollout(
    component_rows: Sequence[Mapping[str, float]],
    qa_type: str,
    is_positive: bool,
    identity_threshold: float = 0.5,
    identity_margin: float = 1.0,
    soft_clip: float = 0.5,
) -> ICDRolloutDecision:
    if len(component_rows) != 8:
        raise ValueError("ICD-GSPO V0 requires exactly eight candidates")
    diagnostics = vector_reward_diagnostics(
        component_rows,
        qa_type,
        is_positive,
        identity_threshold=identity_threshold,
    )
    if diagnostics["feasibility_rate"] == 0.0:
        return ICDRolloutDecision(8, False, "all_identity_infeasible", diagnostics)
    advantages = identity_constrained_advantages(
        component_rows,
        qa_type,
        is_positive,
        identity_threshold=identity_threshold,
        identity_margin=identity_margin,
        soft_clip=soft_clip,
    )
    if not _has_advantage_signal(advantages):
        return ICDRolloutDecision(8, False, "zero_advantage", diagnostics)
    return ICDRolloutDecision(8, True, "fixed8_identity_margin", diagnostics)


def decide_vector_icd_rollout(
    component_rows: Sequence[Mapping[str, float]],
    qa_type: str,
    is_positive: bool,
    identity_threshold: float = 0.5,
    identity_margin: float = 1.0,
    soft_clip: float = 0.5,
    conflict_threshold: float = 0.25,
) -> ICDRolloutDecision:
    count = len(component_rows)
    if count not in (4, 8):
        raise ValueError("ICD-GSPO V1 requires four or eight candidates")
    conflict_threshold = _finite(conflict_threshold, "conflict_threshold")
    if conflict_threshold < 0:
        raise ValueError("conflict_threshold must be non-negative")
    diagnostics = vector_reward_diagnostics(
        component_rows,
        qa_type,
        is_positive,
        identity_threshold=identity_threshold,
    )

    if count == 4:
        if diagnostics["feasibility_rate"] == 0.0:
            return ICDRolloutDecision(8, False, "expand_all_identity_infeasible", diagnostics)
        if diagnostics["vector_uncertainty"] > conflict_threshold:
            return ICDRolloutDecision(8, False, "expand_vector_uncertainty", diagnostics)

    advantages = identity_constrained_advantages(
        component_rows,
        qa_type,
        is_positive,
        identity_threshold=identity_threshold,
        identity_margin=identity_margin,
        soft_clip=soft_clip,
    )
    if diagnostics["feasibility_rate"] == 0.0:
        return ICDRolloutDecision(count, False, "all_identity_infeasible", diagnostics)
    if not _has_advantage_signal(advantages):
        return ICDRolloutDecision(count, False, "zero_advantage", diagnostics)
    return ICDRolloutDecision(count, True, "vector_informative", diagnostics)
