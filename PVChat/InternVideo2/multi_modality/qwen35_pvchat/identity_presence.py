"""Shared target-presence parsing for training and metric evaluation."""

from __future__ import annotations

import re


PRESENT = "present"
ABSENT = "absent"
UNKNOWN = "unknown"


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _lower_text(text: str) -> str:
    return _normalize_text(text).lower()


_LEGACY_ABSENT_PATTERNS = (
    r"\bno\b",
    r"does\s+not\s+appear",
    r"do\s+not\s+appear",
    r"does\s+not\s+show",
    r"does\s+not\s+contain",
    r"does\s+not\s+include",
    r"(?:does|do)\s+not\s+exist",
    r"doesn't\s+appear",
    r"doesn't\s+exist",
    r"don't\s+appear",
    r"is\s+not\s+in",
    r"not\s+here",
    r"not\s+in\s+(this\s+)?(video|clip|recording|frame)",
    r"not\s+visible",
    r"not\s+present",
    r"not\s+shown",
    r"cannot\s+(find|see|identify|recognize|detect)",
    r"can't\s+(find|see|identify|recognize|detect)",
    r"do\s+not\s+(find|see|identify|recognize|detect)",
    r"don't\s+(find|see|identify|recognize|detect)",
    r"is\s+absent",
    r"absent\s+from",
    r"no\s+sign\s+of",
    r"no\s+trace\s+of",
)

_PRESENT_PATTERNS = (
    r"\byes\b",
    r"is\s+in\s+(this\s+)?(video|clip|recording|frame)",
    r"appears?\s+in",
    r"is\s+present",
    r"present\s+in",
    r"is\s+visible",
    r"visible\s+in",
    r"can\s+(find|see|identify|recognize)",
    r"in\s+this\s+(video|clip|recording)",
    r"shown\s+in",
)

# Training excludes a bare "no" because it can describe clothing or objects.
_TARGET_ABSENT_PATTERNS = _LEGACY_ABSENT_PATTERNS[1:]


def _target_is_mentioned(text: str, person_token: str) -> bool:
    person = _lower_text(person_token)
    bare = person.strip("<>")
    if person and person in text:
        return True
    return bool(bare and re.search(rf"(?<!\w){re.escape(bare)}(?!\w)", text))


def detect_identity_presence(answer: str, person_token: str) -> str:
    """Preserve the historical metric evaluator's presence classification."""

    text = _lower_text(answer)
    if not text:
        return UNKNOWN
    if any(re.search(pattern, text) for pattern in _LEGACY_ABSENT_PATTERNS):
        return ABSENT

    person = _lower_text(person_token)
    has_person = bool(person and person in text)
    if has_person and not any(pattern in text for pattern in ("not", "cannot", "can't", "absent")):
        return PRESENT
    if any(re.search(pattern, text) for pattern in _PRESENT_PATTERNS):
        return PRESENT
    return UNKNOWN


def detect_target_identity_presence(
    answer: str,
    person_token: str,
    is_identity_question: bool,
) -> str:
    """Return only target-scoped evidence suitable for activating a gate."""

    text = _lower_text(answer)
    if not text:
        return UNKNOWN
    has_target = _target_is_mentioned(text, person_token)
    question_scopes_answer = bool(is_identity_question)

    if (has_target or question_scopes_answer) and any(
        re.search(pattern, text) for pattern in _TARGET_ABSENT_PATTERNS
    ):
        return ABSENT
    if (has_target or question_scopes_answer) and any(
        re.search(pattern, text) for pattern in _PRESENT_PATTERNS[1:]
    ):
        return PRESENT
    if has_target:
        return PRESENT
    if question_scopes_answer:
        if re.search(r"\bno\b", text):
            return ABSENT
        if re.search(r"\byes\b", text):
            return PRESENT
    return UNKNOWN
