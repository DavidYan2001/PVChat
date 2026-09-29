"""Stage 3使用的可解释本地Reward。

身份题直接使用数据中的is_positive作为GT，不再要求答案必须包含字面Yes/No，
从而修复旧版本把“The footage shows <Sheldon>.”误判为低分的问题。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .identity_presence import ABSENT, PRESENT, detect_identity_presence
from .personalization import normalize_person_token


@dataclass(frozen=True)
class RewardResult:
    reward: float
    components: dict[str, float]


def clean_text(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").split()).strip()


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", clean_text(text).lower())


def classify_qa_type(question: str, is_special: bool = False) -> str:
    if is_special:
        return "identity"
    text = question.lower()
    if any(value in text for value in ("wear", "clothing", "outfit", "attire")):
        return "clothing"
    if any(value in text for value in ("where", "location", "position", "scene")):
        return "location"
    if any(value in text for value in ("emotion", "mood", "expression", "feel")):
        return "emotion"
    if any(value in text for value in ("doing", "action", "activity", "behavior")):
        return "action"
    return "open"


def detect_person_presence(answer: str, person_token: str):
    """训练与评测共用同一套存在性分类器（identity_presence）。

    旧实现的否定白名单只有9个固定短语，"X is not in this video"/
    "There's no X present"等自然否定会因提到token被误判为在场，
    训练reward因此与评测accuracy系统性不一致。
    """

    label = detect_identity_presence(
        clean_text(answer), normalize_person_token(person_token)
    )
    if label == PRESENT:
        return True
    if label == ABSENT:
        return False
    return None


def _pa_person_presence(answer: str, person_token: str):
    """PA-only presence parsing; legacy presence semantics remain untouched."""
    text = clean_text(answer).lower()
    token = normalize_person_token(person_token).lower()
    bare = token.strip("<>")
    absent_patterns = (
        "not present", "not visible", "not shown", "does not appear", "doesn't appear",
        "cannot see", "can't see", "is absent", "no sign of", "nobody", "no person",
        "not in the footage", "not in the video", "not here", "without the person",
    )
    if any(pattern in text for pattern in absent_patterns):
        return False
    if token in text or re.search(rf"(?<!\w){re.escape(bare)}(?!\w)", text):
        return True
    if re.search(r"\bno\b", text):
        return False
    if re.search(r"\byes\b", text):
        return True
    return None


def _is_degenerate(answer: str) -> bool:
    """标准RLHF卫生检查: 特殊token刷屏 / 连续复读 / 词表塌缩。

    组内相对advantage在全垃圾组里依然会奖励"相对赢家", 需要绝对的退化惩罚
    把退化文本的reward拉开量级(CA的打分器自带同类惩罚, legacy此前没有)。
    """

    text = clean_text(answer)
    if not text:
        return False
    if text.count("<sks_token") >= 2:
        return True
    tokens = words(text)
    if len(tokens) >= 6:
        run = 1
        for previous, current in zip(tokens, tokens[1:]):
            run = run + 1 if current == previous else 1
            if run >= 4:
                return True
        if len(set(tokens)) / len(tokens) < 0.3:
            return True
    return False


def _name_score(answer: str, person_token: str) -> float:
    text = clean_text(answer).lower()
    token = normalize_person_token(person_token).lower()
    bare = token.strip("<>")
    if token in text:
        return 1.0
    if bare in text:
        return 0.8
    if any(value in text for value in ("the person", "the man", "the woman", " he ", " she ")):
        return 0.3
    return 0.0


def _conciseness(answer: str, identity: bool) -> float:
    count = len(words(answer))
    limit = 24 if identity else 60
    return 1.0 if 0 < count <= limit else 0.5 if count <= limit * 2 else 0.0


def _semantic_score(gold: str, candidate: str, person_token: str) -> float:
    token = normalize_person_token(person_token).lower()
    gold_text = clean_text(gold).lower().replace(token, "")
    candidate_text = clean_text(candidate).lower().replace(token, "")
    gold_words = set(words(gold_text))
    candidate_words = set(words(candidate_text))
    overlap = len(gold_words & candidate_words) / max(len(gold_words | candidate_words), 1)
    sequence = SequenceMatcher(None, gold_text, candidate_text).ratio()
    return max(0.0, min(1.0, 0.6 * overlap + 0.4 * sequence))


def score_answer(
    question: str,
    gold_answer: str,
    candidate_answer: str,
    person_token: str,
    is_positive: bool,
    qa_type: str | None = None,
) -> RewardResult:
    qa_type = qa_type or classify_qa_type(question)
    candidate_answer = clean_text(candidate_answer)
    name = _name_score(candidate_answer, person_token)
    concise = _conciseness(candidate_answer, qa_type == "identity")
    format_ok = 1.0 if candidate_answer else 0.0

    if qa_type == "identity":
        predicted = detect_person_presence(candidate_answer, person_token)
        correctness = 1.0 if predicted is bool(is_positive) else -1.0
        reward = 0.80 * correctness + 0.10 * name + 0.05 * concise + 0.05 * format_ok
        components = {
            "identity_correctness": correctness,
            "identity_name": name,
            "conciseness": concise,
            "format": format_ok,
        }
    else:
        semantic = _semantic_score(gold_answer, candidate_answer, person_token)
        reward = 0.70 * semantic + 0.15 * name + 0.10 * concise + 0.05 * format_ok
        components = {
            "semantic": semantic,
            "identity_name": name,
            "conciseness": concise,
            "format": format_ok,
        }
    return RewardResult(float(max(-1.5, min(1.0, reward))), components)


def pa_component_weights(qa_type: str, is_positive: bool) -> dict[str, float]:
    if qa_type == "identity":
        return {"identity_correctness": 0.80, "identity_name": 0.10, "conciseness": 0.05, "format": 0.05}
    if is_positive:
        return {"semantic": 0.65, "presence_consistency": 0.15, "identity_name": 0.10, "conciseness": 0.05, "format": 0.05}
    return {"absence_consistency": 0.60, "semantic": 0.25, "identity_name": 0.05, "conciseness": 0.05, "format": 0.05}


def score_pa_answer(
    question: str,
    gold_answer: str,
    candidate_answer: str,
    person_token: str,
    is_positive: bool,
    qa_type: str | None = None,
) -> RewardResult:
    qa_type = qa_type or classify_qa_type(question)
    if qa_type == "identity":
        return score_answer(question, gold_answer, candidate_answer, person_token, is_positive, qa_type)

    candidate_answer = clean_text(candidate_answer)
    semantic = _semantic_score(gold_answer, candidate_answer, person_token)
    name = _name_score(candidate_answer, person_token)
    concise = _conciseness(candidate_answer, False)
    format_ok = 1.0 if candidate_answer else 0.0
    presence = _pa_person_presence(candidate_answer, person_token)
    consistency = 0.0 if presence is None else (1.0 if presence is is_positive else -1.0)
    if is_positive:
        components = {"semantic": semantic, "presence_consistency": consistency, "identity_name": name, "conciseness": concise, "format": format_ok}
    else:
        components = {"absence_consistency": consistency, "semantic": semantic, "identity_name": name, "conciseness": concise, "format": format_ok}
    weights = pa_component_weights(qa_type, is_positive)
    reward = sum(weights[key] * value for key, value in components.items())
    return RewardResult(float(max(-1.5, min(1.0, reward))), components)
