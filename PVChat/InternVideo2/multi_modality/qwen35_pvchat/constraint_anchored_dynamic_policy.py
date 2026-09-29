"""Constraint-anchored reward and rollout policy for Qwen3.5 Stage 3.

The policy keeps Dynamic-GSPO's 2/4/8 sampling idea, but makes three signals
explicit: output validity, target-identity consistency, and answer content.
No API model is used here; every score is deterministic and local.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Mapping, Sequence

from .identity_gated_dynamic_policy import (
    IDENTITY_CORRECT,
    IDENTITY_UNKNOWN,
    IDENTITY_WRONG,
    identity_gate_value,
)
from .identity_presence import detect_target_identity_presence
from .personalization import normalize_person_token
from .policy_objectives import normalize_group_advantages
from .rewards import RewardResult, clean_text


VALIDITY_VALID = 1.0
VALIDITY_INVALID = -1.0
SUPPORTED_ROLLOUT_LENGTHS = (2, 4, 8)

DEFAULT_SEMANTIC_THRESHOLD = 0.60
DEFAULT_INFORMATIVE_MARGIN = 0.10
EASY_CONTENT_THRESHOLD = 0.85
EASY_CONTENT_SPREAD = 0.05


_PHRASE_ALIASES = (
    ("long sleeved", "long_sleeve"),
    ("long sleeve", "long_sleeve"),
    ("short sleeved", "short_sleeve"),
    ("short sleeve", "short_sleeve"),
    ("in front of", "in_front"),
    ("next to", "next_to"),
    ("beside", "next_to"),
    ("at the beginning", "beginning"),
    ("at the start", "beginning"),
    ("from the start", "beginning"),
    ("at the end", "end"),
    ("from the end", "end"),
    ("all the way through", "throughout"),
)

_TOKEN_ALIASES = {
    "seated": "sitting",
    "sits": "sitting",
    "sit": "sitting",
    "stands": "standing",
    "stand": "standing",
    "walks": "walking",
    "walk": "walking",
    "talks": "talking",
    "speaks": "talking",
    "speaking": "talking",
    "conversing": "talking",
    "smiles": "smiling",
    "smile": "smiling",
    "shown": "present",
    "visible": "present",
    "appears": "present",
    "missing": "absent",
    "angrily": "angry",
    "happiness": "happy",
    "sadness": "sad",
    "seriously": "serious",
}

_COMMON_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "he", "her", "him", "his", "in", "is", "it", "of", "on", "or",
    "she", "that", "the", "their", "them", "they", "this", "to", "video",
    "wearing", "with",
}

_FACT_VOCABULARY = {
    "action": {
        "sitting", "standing", "walking", "talking", "smiling", "looking",
        "holding", "eating", "drinking", "reading", "writing", "working",
        "gesturing", "moving", "watching", "listening", "entering", "leaving",
    },
    "location": {
        "left", "right", "center", "in_front", "behind", "next_to", "alone",
        "sitting", "standing", "walking", "couch", "sofa", "chair", "table",
        "kitchen", "bedroom", "office", "room", "hallway", "indoors", "outdoors",
        "beginning", "middle", "end", "throughout",
    },
    "clothing": {
        "black", "blue", "brown", "gray", "green", "grey", "orange", "pink",
        "purple", "red", "white", "yellow", "shirt", "tshirt", "jacket", "coat",
        "suit", "sweater", "hoodie", "dress", "top", "pants", "jeans", "shorts",
        "hat", "glasses", "tie", "long_sleeve", "short_sleeve",
    },
    "emotion": {
        "angry", "annoyed", "calm", "confused", "contemplative", "excited",
        "focused", "happy", "neutral", "relaxed", "sad", "serious", "surprised",
        "tense", "worried",
    },
}

_CONTRADICTION_PAIRS = (
    ("left", "right"),
    ("sitting", "standing"),
    ("beginning", "end"),
    ("indoors", "outdoors"),
    ("present", "absent"),
    ("happy", "sad"),
    ("calm", "angry"),
)


def _finite_float(value, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _canonical_text(text: str, person_token: str) -> str:
    text = clean_text(text).lower().replace("-", " ")
    person = normalize_person_token(person_token).lower()
    text = text.replace(person, " ").replace(person.strip("<>"), " ")
    for phrase, replacement in _PHRASE_ALIASES:
        text = text.replace(phrase, replacement)
    return " ".join(text.split())


def _canonical_tokens(text: str, person_token: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9_']+", _canonical_text(text, person_token))
    return [_TOKEN_ALIASES.get(token, token) for token in tokens]


def _counter_f1(reference: Sequence[str], candidate: Sequence[str]) -> float:
    reference_counts = Counter(reference)
    candidate_counts = Counter(candidate)
    overlap = sum((reference_counts & candidate_counts).values())
    if not reference_counts or not candidate_counts or overlap == 0:
        return 0.0
    precision = overlap / sum(candidate_counts.values())
    recall = overlap / sum(reference_counts.values())
    return 2.0 * precision * recall / (precision + recall)


def _semantic_score(gold: str, candidate: str, person_token: str) -> float:
    gold_tokens = _canonical_tokens(gold, person_token)
    candidate_tokens = _canonical_tokens(candidate, person_token)
    token_f1 = _counter_f1(gold_tokens, candidate_tokens)
    sequence = SequenceMatcher(None, " ".join(gold_tokens), " ".join(candidate_tokens)).ratio()
    return max(0.0, min(1.0, 0.70 * token_f1 + 0.30 * sequence))


def _category_vocabulary(qa_type: str) -> set[str]:
    if qa_type in _FACT_VOCABULARY:
        return set(_FACT_VOCABULARY[qa_type])
    result = set()
    for values in _FACT_VOCABULARY.values():
        result.update(values)
    return result


def _fact_score(gold: str, candidate: str, person_token: str, qa_type: str) -> float:
    gold_tokens = set(_canonical_tokens(gold, person_token))
    candidate_tokens = set(_canonical_tokens(candidate, person_token))
    vocabulary = _category_vocabulary(qa_type)
    gold_facts = gold_tokens & vocabulary
    candidate_facts = candidate_tokens & vocabulary

    if gold_facts:
        coverage = len(gold_facts & candidate_facts) / len(gold_facts)
    else:
        content_gold = {token for token in gold_tokens if token not in _COMMON_STOPWORDS}
        coverage = (
            len(content_gold & candidate_tokens) / len(content_gold)
            if content_gold
            else 0.0
        )

    contradictions = 0
    relevant_pairs = 0
    for first, second in _CONTRADICTION_PAIRS:
        if first in gold_tokens and second not in gold_tokens:
            relevant_pairs += 1
            contradictions += int(second in candidate_tokens and first not in candidate_tokens)
        elif second in gold_tokens and first not in gold_tokens:
            relevant_pairs += 1
            contradictions += int(first in candidate_tokens and second not in candidate_tokens)
    contradiction_rate = contradictions / relevant_pairs if relevant_pairs else 0.0
    return max(0.0, min(1.0, coverage - 0.5 * contradiction_rate))


def _length_score(gold: str, candidate: str, person_token: str) -> float:
    gold_length = len(_canonical_tokens(gold, person_token))
    candidate_length = len(_canonical_tokens(candidate, person_token))
    if candidate_length == 0:
        return 0.0
    return min(gold_length + 1, candidate_length + 1) / max(gold_length + 1, candidate_length + 1)


def _has_degeneration(answer: str, gold_answer: str, person_token: str) -> bool:
    lowered = clean_text(answer).lower()
    if not lowered:
        return True
    if re.search(r"<?sks_token\d+>?", lowered):
        return True

    tokens = _canonical_tokens(answer, person_token)
    gold_tokens = _canonical_tokens(gold_answer, person_token)
    if len(tokens) >= 8 and max(Counter(tokens).values(), default=0) / len(tokens) >= 0.5:
        return True
    if len(tokens) >= 10:
        bigrams = list(zip(tokens, tokens[1:]))
        if max(Counter(bigrams).values(), default=0) >= 4:
            return True
    return len(tokens) > max(64, 3 * max(len(gold_tokens), 1))


def score_constraint_anchored_answer(
    question: str,
    gold_answer: str,
    candidate_answer: str,
    person_token: str,
    is_positive: bool,
    qa_type: str,
) -> RewardResult:
    """Score one answer with always-on semantic and factual components."""

    candidate_answer = clean_text(candidate_answer)
    semantic = _semantic_score(gold_answer, candidate_answer, person_token)
    facts = _fact_score(gold_answer, candidate_answer, person_token, qa_type)
    length = _length_score(gold_answer, candidate_answer, person_token)
    degeneration = float(_has_degeneration(candidate_answer, gold_answer, person_token))
    validity = VALIDITY_INVALID if degeneration else VALIDITY_VALID
    presence = detect_target_identity_presence(
        candidate_answer,
        person_token,
        is_identity_question=qa_type == "identity",
    )
    identity = identity_gate_value(presence, bool(is_positive))
    identity_soft = {
        IDENTITY_CORRECT: 1.0,
        IDENTITY_UNKNOWN: 0.5,
        IDENTITY_WRONG: 0.0,
    }[identity]
    format_ok = 1.0 if candidate_answer else 0.0
    content = 0.60 * semantic + 0.40 * facts

    if qa_type == "identity":
        reward = 0.75 * identity_soft + 0.15 * content + 0.05 * length + 0.05 * format_ok
    else:
        reward = (
            0.50 * semantic
            + 0.30 * facts
            + 0.10 * identity_soft
            + 0.05 * length
            + 0.05 * format_ok
        )
    reward -= degeneration
    components = {
        "semantic": float(semantic),
        "fact_coverage": float(facts),
        "content_score": float(content),
        "identity_gate": float(identity),
        "validity": float(validity),
        "degeneration": float(degeneration),
        "length": float(length),
        "format": float(format_ok),
    }
    return RewardResult(float(max(-1.5, min(1.0, reward))), components)


def constraint_anchored_advantages(
    component_rows: Sequence[Mapping[str, float]],
    eps: float = 1e-6,
) -> list[float]:
    """Build GSPO advantages from validity, identity, then content.

    The fixed tier gaps make the ordering lexicographic. Content remains fully
    ordered inside each tier instead of being clipped to a small interval.
    """

    utilities = []
    for row in component_rows:
        validity = _finite_float(row.get("validity", VALIDITY_INVALID), "validity")
        identity = _finite_float(row.get("identity_gate", IDENTITY_UNKNOWN), "identity_gate")
        content = _finite_float(row.get("content_score", 0.0), "content_score")
        utilities.append(4.0 * validity + 2.0 * identity + content)
    return normalize_group_advantages(utilities, alpha=1.0, eps=eps)


@dataclass(frozen=True)
class ConstraintAnchoredRolloutDecision:
    target_count: int
    should_update: bool
    reason: str
    observed_count: int
    use_sft_fallback: bool = False
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    @property
    def needs_more(self) -> bool:
        return self.target_count > self.observed_count


def decide_constraint_anchored_rollout(
    component_rows: Sequence[Mapping[str, float]],
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    informative_margin: float = DEFAULT_INFORMATIVE_MARGIN,
) -> ConstraintAnchoredRolloutDecision:
    """Choose 2/4/8 candidates and use GT fallback for uniformly poor groups."""

    rows = list(component_rows)
    count = len(rows)
    if count not in SUPPORTED_ROLLOUT_LENGTHS:
        raise ValueError(f"component_rows must have length 2, 4, or 8; got {count}")
    semantic_threshold = _finite_float(semantic_threshold, "semantic_threshold")
    informative_margin = _finite_float(informative_margin, "informative_margin")
    if not 0.0 <= semantic_threshold <= 1.0:
        raise ValueError("semantic_threshold must be between zero and one")
    if informative_margin < 0.0:
        raise ValueError("informative_margin must be non-negative")

    contents = [_finite_float(row.get("content_score", 0.0), "content_score") for row in rows]
    validities = [_finite_float(row.get("validity", VALIDITY_INVALID), "validity") for row in rows]
    identities = [_finite_float(row.get("identity_gate", IDENTITY_UNKNOWN), "identity_gate") for row in rows]
    feasible = [
        validity == VALIDITY_VALID and identity == IDENTITY_CORRECT
        for validity, identity in zip(validities, identities)
    ]
    feasible_contents = [value for value, is_feasible in zip(contents, feasible) if is_feasible]
    best_content = max(feasible_contents, default=max(contents))
    content_spread = max(contents) - min(contents)
    constraint_mixed = len(set(zip(validities, identities))) > 1
    diagnostics = {
        "feasible_count": float(sum(feasible)),
        "invalid_count": float(sum(value == VALIDITY_INVALID for value in validities)),
        "identity_wrong_count": float(sum(value == IDENTITY_WRONG for value in identities)),
        "best_content": float(best_content),
        "content_spread": float(content_spread),
    }

    if count == 2:
        if all(feasible) and min(contents) >= EASY_CONTENT_THRESHOLD and content_spread < EASY_CONTENT_SPREAD:
            return ConstraintAnchoredRolloutDecision(2, False, "easy_saturated", 2, diagnostics=diagnostics)
        if constraint_mixed:
            return ConstraintAnchoredRolloutDecision(2, True, "constraint_informative", 2, diagnostics=diagnostics)
        if len(feasible_contents) >= 2 and content_spread >= informative_margin:
            return ConstraintAnchoredRolloutDecision(2, True, "content_informative", 2, diagnostics=diagnostics)
        return ConstraintAnchoredRolloutDecision(4, False, "expand", 2, diagnostics=diagnostics)

    if count == 4:
        if len(feasible_contents) >= 2 and (constraint_mixed or content_spread >= informative_margin):
            reason = "constraint_informative" if constraint_mixed else "content_informative"
            return ConstraintAnchoredRolloutDecision(4, True, reason, 4, diagnostics=diagnostics)
        if all(feasible) and min(contents) >= EASY_CONTENT_THRESHOLD and content_spread < EASY_CONTENT_SPREAD:
            return ConstraintAnchoredRolloutDecision(4, False, "easy_saturated", 4, diagnostics=diagnostics)
        return ConstraintAnchoredRolloutDecision(8, False, "expand_hard", 4, diagnostics=diagnostics)

    if len(feasible_contents) < 2 or best_content < semantic_threshold:
        return ConstraintAnchoredRolloutDecision(
            8,
            True,
            "sft_fallback",
            8,
            use_sft_fallback=True,
            diagnostics=diagnostics,
        )
    if constraint_mixed:
        return ConstraintAnchoredRolloutDecision(8, True, "constraint_informative", 8, diagnostics=diagnostics)
    if content_spread >= informative_margin:
        return ConstraintAnchoredRolloutDecision(8, True, "content_informative", 8, diagnostics=diagnostics)
    return ConstraintAnchoredRolloutDecision(8, False, "easy_saturated", 8, diagnostics=diagnostics)


__all__ = [
    "IDENTITY_CORRECT",
    "IDENTITY_UNKNOWN",
    "IDENTITY_WRONG",
    "VALIDITY_INVALID",
    "VALIDITY_VALID",
    "ConstraintAnchoredRolloutDecision",
    "constraint_anchored_advantages",
    "decide_constraint_anchored_rollout",
    "score_constraint_anchored_answer",
]
