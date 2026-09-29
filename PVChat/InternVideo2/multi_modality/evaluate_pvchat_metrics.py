import argparse
import concurrent.futures
import gc
import hashlib
import json
import math
import os
import re
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

from generate_grpo_rollouts import is_quota_error, parse_model_fallbacks
from qwen35_pvchat.identity_presence import (
    ABSENT,
    PRESENT,
    UNKNOWN,
    detect_identity_presence,
)


DEFAULT_BLEU_ORDER = 1


def normalize_person_token(sks_name):
    if not sks_name:
        return ""
    sks_name = str(sks_name).strip()
    if sks_name.startswith("<") and sks_name.endswith(">"):
        return sks_name
    return f"<{sks_name}>"


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def lower_text(text):
    return normalize_text(text).lower()


def tokenize(text):
    return re.findall(r"[a-z0-9]+|<[^>\s]+>", lower_text(text))


def extract_person_from_data(data):
    text_parts = []
    if isinstance(data, dict):
        text_parts.append(str(data.get("model_name", "")))
        entries = data.get("results") or data.get("data") or []
    elif isinstance(data, list):
        entries = data
    else:
        entries = []

    for entry in entries[:5]:
        for qa in entry.get("qa_pairs", [])[:10]:
            text_parts.append(str(qa.get("question", "")))
            text_parts.append(str(qa.get("answer", "")))
    joined = "\n".join(text_parts)
    match = re.search(r"<([^>\s]+)>", joined)
    if match:
        return f"<{match.group(1)}>"
    return ""


def classify_qa_category(question, is_special=False):
    q = lower_text(question)
    if is_special:
        return "identity"
    if any(word in q for word in ("wear", "wearing", "clothing", "clothes", "outfit", "attire", "dressed")):
        return "clothing"
    if any(word in q for word in ("where", "location", "position", "place", "setting", "scene", "environment", "room")):
        return "location"
    if any(word in q for word in ("emotion", "mood", "expression", "feel", "feeling", "display")):
        return "emotion"
    if any(word in q for word in ("doing", "action", "activity", "behavior", "engaged", "performing", "happening")):
        return "action"
    # 仅在没有更具体语义类别时使用身份题启发式，避免把
    # "What emotion does <sks> appear ... in this video?"误归为身份题。
    if re.search(r"\b(find|identify|recognize|present|appear|visible)\b", q) and re.search(
        r"\b(video|clip|recording|frame)\b", q
    ):
        return "identity"
    return "other"


def iter_video_entries(data):
    if isinstance(data, dict):
        if isinstance(data.get("results"), list):
            return data["results"]
        if isinstance(data.get("data"), list):
            return data["data"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported result JSON format: expected dict with results/data or a list.")


def flatten_result_json(data, person_token=None):
    person_token = person_token or extract_person_from_data(data)
    records = []
    for video_index, entry in enumerate(iter_video_entries(data)):
        qa_pairs = entry.get("qa_pairs") or entry.get("QA") or []
        for qa_index, qa in enumerate(qa_pairs):
            question = normalize_text(qa.get("question", ""))
            gold = normalize_text(qa.get("answer", qa.get("gold_answer", "")))
            generated = normalize_text(qa.get("generated_answer", qa.get("prediction", "")))
            is_special = bool(qa.get("is_special", False))
            raw_is_positive = qa.get("is_positive", entry.get("is_positive"))
            is_positive = raw_is_positive if isinstance(raw_is_positive, bool) else None
            records.append(
                {
                    "record_id": f"{video_index}:{qa_index}",
                    "video_index": video_index,
                    "qa_index": qa_index,
                    "video_path": entry.get("video_path", ""),
                    "question": question,
                    "gold_answer": gold,
                    "generated_answer": generated,
                    "is_special": is_special,
                    "is_positive": is_positive,
                    "category": classify_qa_category(question, is_special=is_special),
                    "person_token": person_token,
                }
            )
    return records


def compute_accuracy(records, person_token):
    stats = {
        "total": 0,
        "correct": 0,
        "accuracy": None,
        "gold_present_total": 0,
        "gold_present_correct": 0,
        "gold_absent_total": 0,
        "gold_absent_correct": 0,
        "pred_unknown": 0,
        "gold_unknown": 0,
    }
    for record in records:
        if record["category"] != "identity":
            continue
        if record.get("is_positive") is True:
            gold = PRESENT
        elif record.get("is_positive") is False:
            gold = ABSENT
        else:
            # 兼容尚未保存is_positive字段的旧测试结果。
            gold = detect_identity_presence(record["gold_answer"], person_token)
        pred = detect_identity_presence(record["generated_answer"], person_token)
        if gold == UNKNOWN:
            stats["gold_unknown"] += 1
            continue
        stats["total"] += 1
        if pred == UNKNOWN:
            stats["pred_unknown"] += 1
        correct = pred == gold
        if correct:
            stats["correct"] += 1
        if gold == PRESENT:
            stats["gold_present_total"] += 1
            stats["gold_present_correct"] += int(correct)
        elif gold == ABSENT:
            stats["gold_absent_total"] += 1
            stats["gold_absent_correct"] += int(correct)
    if stats["total"]:
        stats["accuracy"] = stats["correct"] / stats["total"]
    stats["gold_present_accuracy"] = (
        stats["gold_present_correct"] / stats["gold_present_total"]
        if stats["gold_present_total"]
        else None
    )
    stats["gold_absent_accuracy"] = (
        stats["gold_absent_correct"] / stats["gold_absent_total"]
        if stats["gold_absent_total"]
        else None
    )
    return stats


def ngrams(tokens, order):
    return Counter(tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1))


def internal_corpus_bleu(
    predictions,
    references,
    max_order=DEFAULT_BLEU_ORDER,
    smooth=True,
):
    matches_by_order = [0] * max_order
    possible_by_order = [0] * max_order
    ref_length = 0
    pred_length = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize(pred)
        ref_tokens = tokenize(ref)
        pred_length += len(pred_tokens)
        ref_length += len(ref_tokens)
        for order in range(1, max_order + 1):
            pred_ngrams = ngrams(pred_tokens, order)
            ref_ngrams = ngrams(ref_tokens, order)
            overlap = pred_ngrams & ref_ngrams
            matches_by_order[order - 1] += sum(overlap.values())
            possible_by_order[order - 1] += max(len(pred_tokens) - order + 1, 0)

    precisions = []
    for matches, possible in zip(matches_by_order, possible_by_order):
        if possible == 0:
            continue
        if matches == 0 and smooth:
            precisions.append(1.0 / (2.0 * possible))
        else:
            precisions.append(matches / possible)

    if not precisions or pred_length == 0:
        return {
            "score": 0.0,
            "name": f"BLEU-{max_order}",
            "max_order": max_order,
            "backend": "internal",
            "precisions": [0.0] * max_order,
            "bp": 0.0,
            "sys_len": pred_length,
            "ref_len": ref_length,
        }

    log_precision_sum = sum(math.log(max(p, 1e-16)) for p in precisions) / len(precisions)
    bp = 1.0 if pred_length > ref_length else math.exp(1.0 - ref_length / max(pred_length, 1))
    score = bp * math.exp(log_precision_sum)
    precision_values = [
        (matches_by_order[i] / possible_by_order[i] if possible_by_order[i] else 0.0) * 100.0
        for i in range(max_order)
    ]
    return {
        "score": score,
        "name": f"BLEU-{max_order}",
        "max_order": max_order,
        "backend": "internal",
        "precisions": precision_values,
        "bp": bp,
        "sys_len": pred_length,
        "ref_len": ref_length,
    }


def compute_corpus_bleu(predictions, references, max_order=DEFAULT_BLEU_ORDER):
    predictions = [normalize_text(item) for item in predictions]
    references = [normalize_text(item) for item in references]
    if not predictions:
        return {
            "score": None,
            "name": f"BLEU-{max_order}",
            "max_order": max_order,
            "backend": "none",
            "count": 0,
        }

    try:
        import sacrebleu

        metric = sacrebleu.metrics.BLEU(
            max_ngram_order=max_order,
            effective_order=True,
        )
        result = metric.corpus_score(predictions, [references])
        return {
            "score": float(result.score) / 100.0,
            "name": f"BLEU-{max_order}",
            "max_order": max_order,
            "backend": "sacrebleu",
            "count": len(predictions),
            "signature": str(metric.get_signature()),
        }
    except Exception as exc:
        result = internal_corpus_bleu(
            predictions,
            references,
            max_order=max_order,
        )
        result["count"] = len(predictions)
        result["sacrebleu_error"] = f"{type(exc).__name__}: {exc}"
        return result


def lexical_f1(gold_answer, generated_answer, person_token):
    stop = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "in",
        "on",
        "at",
        "of",
        "to",
        "and",
        "or",
        "this",
        "that",
        "video",
        "clip",
        "recording",
    }
    person_words = set(tokenize(person_token))
    gold = [t for t in tokenize(gold_answer) if t not in stop and t not in person_words]
    pred = [t for t in tokenize(generated_answer) if t not in stop and t not in person_words]
    if not gold or not pred:
        return 0.0
    gold_counts = Counter(gold)
    pred_counts = Counter(pred)
    overlap = sum((gold_counts & pred_counts).values())
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def local_entity_specificity(generated_answer, person_token):
    text = lower_text(generated_answer)
    person = lower_text(person_token)
    if not text:
        return {"score": 0.0, "reason": "empty answer", "source": "local"}

    other_entity = re.search(r"<([^>\s]+)>", text)
    if other_entity and person and other_entity.group(0).lower() != person:
        return {"score": 0.0, "reason": "mentions a different bracketed entity", "source": "local"}

    generic_terms = [
        "the person",
        "this person",
        "the man",
        "the woman",
        "someone",
        "somebody",
        "the individual",
        "a person",
        "a man",
        "a woman",
    ]
    generic_hits = [term for term in generic_terms if term in text]
    if person and person in text:
        score = 5.0 if not generic_hits else 4.0
        reason = "explicitly mentions target entity"
    elif generic_hits:
        score = 2.0
        reason = "uses generic person wording"
    else:
        score = 3.0
        reason = "specificity is implied by question context but target entity is not named"
    return {"score": score, "reason": reason, "source": "local"}


def local_descriptive_completeness(record, person_token):
    gold = record["gold_answer"]
    pred = record["generated_answer"]
    if not pred:
        return {"score": 0.0, "reason": "empty answer", "source": "local"}
    overlap = lexical_f1(gold, pred, person_token)
    pred_len = len(tokenize(pred))
    gold_len = max(len(tokenize(gold)), 1)
    detail_ratio = min(1.0, pred_len / gold_len)
    score = 0.75 * overlap + 0.25 * detail_ratio

    gold_presence = detect_identity_presence(gold, person_token)
    pred_presence = detect_identity_presence(pred, person_token)
    if gold_presence != UNKNOWN and pred_presence != UNKNOWN and gold_presence != pred_presence:
        score -= 0.4
    score = max(0.0, min(1.0, score)) * 5.0
    return {
        "score": score,
        "reason": "local lexical overlap and detail-length heuristic",
        "source": "local",
    }


def build_es_prompt(record, person_token):
    return f"""You are evaluating Entity Specificity for a personalized video QA system.

Target entity: {person_token}

Given a question, a ground-truth answer, and a model answer, score whether the model answer is personalized to the target entity rather than generic.

Score from 0 to 5:
5: Explicitly and correctly refers to {person_token}, and the response is clearly about this target entity.
4: Refers to {person_token} or an unambiguous target pronoun with minor generic wording.
3: The answer is likely about the target because of the question context, but does not explicitly mention {person_token}.
2: Mostly generic wording such as "the person", "the man", or "someone", with weak personalization.
1: Very generic, vague, or reusable for any person.
0: Mentions the wrong entity, contradicts whether {person_token} appears, or hallucinates target presence/absence against the ground truth.

Return only JSON:
{{"score": <0-5>, "reason": "<short reason>"}}

Question:
{record["question"]}

Ground truth answer:
{record["gold_answer"]}

Model answer:
{record["generated_answer"]}"""


def build_dc_prompt(record):
    return f"""You are evaluating Descriptive Completeness for a personalized video QA system.

Given a question, a ground-truth answer, and a model answer, score how complete and detailed the model answer is compared with the ground truth. Do not require exact wording. Focus on whether the model answer captures the key facts and sufficient details requested by the question.

Score from 0 to 5:
5: Covers all important details in the ground truth with clear, concrete wording and no contradiction.
4: Covers most key details, with only minor omissions.
3: Partially correct, includes some key information but misses important details.
2: Very incomplete or overly generic, with little concrete detail.
1: Barely answers the question or is mostly off-topic.
0: Contradicts the ground truth, answers the wrong question, or gives an invalid answer.

Return only JSON:
{{"score": <0-5>, "reason": "<short reason>"}}

Question:
{record["question"]}

Ground truth answer:
{record["gold_answer"]}

Model answer:
{record["generated_answer"]}"""


def build_packed_judge_prompt(records, metric_name, person_token):
    """Build one compact judge prompt that scores several records independently."""
    if metric_name == "ES":
        title = "Entity Specificity"
        criterion = f"""Target entity: {person_token}

Score whether each model answer is personalized to the target entity rather than generic.
5: Explicitly and correctly refers to {person_token}; clearly about this entity.
4: Refers to {person_token} or an unambiguous target pronoun, with minor generic wording.
3: Likely about the target from context, but does not explicitly mention {person_token}.
2: Mostly generic wording such as \"the person\", \"the man\", or \"someone\".
1: Very generic, vague, or reusable for any person.
0: Wrong entity, contradiction, or hallucinated presence/absence versus ground truth."""
    elif metric_name == "DC":
        title = "Descriptive Completeness"
        criterion = """Score how completely each model answer covers its ground-truth answer.
5: Covers all important details clearly, with no contradiction.
4: Covers most key details, with only minor omissions.
3: Partially correct, but misses important details.
2: Very incomplete or overly generic.
1: Barely answers the question or is mostly off-topic.
0: Contradicts the ground truth, answers the wrong question, or is invalid."""
    else:
        raise ValueError(f"Unsupported packed judge metric: {metric_name}")

    items = [
        {
            "id": index,
            "question": record["question"],
            "ground_truth_answer": record["gold_answer"],
            "model_answer": record["generated_answer"],
        }
        for index, record in enumerate(records)
    ]
    return f"""You are evaluating {title} for a personalized video QA system.

{criterion}

Evaluate every item independently. Do not let one item's content affect another item's score.
Return only one JSON object containing exactly {len(items)} scores in input order, with no prose and no reasons:
{{"scores": [<score 0-5>, ...]}}

Items:
{json.dumps(items, ensure_ascii=False, separators=(",", ":"))}"""


def parse_json_object(text):
    text = normalize_text(text)
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def clamp_score_0_5(value):
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return max(0.0, min(5.0, score))


def parse_packed_scores(text, expected_count):
    """Parse a compact {"scores": [...]} response without silently dropping items."""
    parsed = parse_json_object(text)
    values = parsed.get("scores") if isinstance(parsed, dict) else None
    if not isinstance(values, list):
        raise ValueError("packed judge response is missing a scores list")
    if len(values) != expected_count:
        raise ValueError(
            f"packed judge returned {len(values)} scores; expected {expected_count}"
        )
    scores = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("score")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid packed judge score: {value!r}") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"non-finite packed judge score: {value!r}")
        scores.append(clamp_score_0_5(numeric))
    return scores


class DashScopeJudge:
    def __init__(
            self,
            model,
            api_key=None,
            base_url=None,
            timeout=90,
            max_retries=3,
            fallback_models=None,
            client=None,
    ):
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("The openai package is required for --judge_backend dashscope.") from exc
            client = OpenAI(
                api_key=api_key or os.environ.get("DASHSCOPE_API_KEY"),
                base_url=base_url or os.environ.get(
                    "DASHSCOPE_BASE_URL",
                    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                ),
                timeout=timeout,
            )

        self.models = parse_model_fallbacks(model, fallback_models)
        if not self.models:
            raise ValueError("At least one DashScope judge model is required.")
        self.client = client
        self.max_retries = max(int(max_retries), 1)
        self._model_index = 0
        self._model_lock = threading.Lock()

    @property
    def current_model(self):
        with self._model_lock:
            return self.models[self._model_index]

    @property
    def cache_model_key(self):
        return ",".join(self.models)

    def _switch_after_quota(self, failed_model):
        with self._model_lock:
            current = self.models[self._model_index]
            if current != failed_model:
                return True
            if self._model_index + 1 >= len(self.models):
                return False
            self._model_index += 1
            next_model = self.models[self._model_index]
        print(
            f"[Judge] Quota exhausted for {failed_model}; switching to {next_model}.",
            flush=True,
        )
        return True

    @staticmethod
    def parse_scored_response(content):
        parsed = parse_json_object(content)
        score_0_5 = clamp_score_0_5(parsed.get("score"))
        return {
            "score": score_0_5,
            "reason": normalize_text(parsed.get("reason", "")),
            "source": "dashscope",
            "raw": content,
        }

    def score_prompt(self, prompt):
        last_error = None
        attempt = 0
        while attempt < self.max_retries:
            model = self.current_model
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a strict evaluator. Return only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )
                content = response.choices[0].message.content
                scored = self.parse_scored_response(content)
                scored["model"] = model
                return scored
            except Exception as exc:
                last_error = exc
                if is_quota_error(exc) and self._switch_after_quota(model):
                    attempt = 0
                    continue
                attempt += 1
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 10))
        return {
            "score": None,
            "reason": f"judge failed: {type(last_error).__name__}: {last_error}",
            "source": "dashscope",
            "raw": "",
            "model": self.current_model,
        }


class LocalQwen35Judge:
    """Batch ES/DC scoring with one local Qwen3.5-35B GPU model."""

    def __init__(
            self,
            model_path,
            device="cuda:0",
            batch_size=64,
            max_new_tokens=512,
            tokenizer=None,
            model=None,
            torch_module=None,
    ):
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.models = [self.model_path]
        self.device = str(device)
        self.batch_size = max(int(batch_size), 1)
        self.max_new_tokens = max(int(max_new_tokens), 1)

        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        if self.device.startswith("cuda") and not self.torch.cuda.is_available():
            raise RuntimeError("--local_judge_device requests CUDA, but CUDA is unavailable.")

        if tokenizer is None or model is None:
            try:
                from transformers import AutoModelForImageTextToText, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "transformers with Qwen3.5 support is required for --judge_backend local_qwen35."
                ) from exc
            model_dir = Path(self.model_path)
            if not model_dir.is_dir():
                raise FileNotFoundError(f"Local Qwen3.5 judge directory does not exist: {model_dir}")
            gc.collect()
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            dtype = self.torch.bfloat16 if self.device.startswith("cuda") else self.torch.float32
            model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                dtype=dtype,
                attn_implementation="sdpa",
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            model.to(self.device)

        self.tokenizer = tokenizer
        self.model = model.eval()
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        for config in (
                getattr(self.model, "config", None),
                getattr(getattr(self.model, "config", None), "text_config", None),
        ):
            if config is not None:
                config.use_cache = True

    @property
    def current_model(self):
        return self.model_path

    @property
    def cache_model_key(self):
        return self.model_path

    @staticmethod
    def parse_scored_response(content, model_name):
        try:
            parsed = parse_json_object(content)
            if "score" not in parsed:
                raise ValueError("missing score")
            return {
                "score": clamp_score_0_5(parsed["score"]),
                "reason": normalize_text(parsed.get("reason", "")),
                "source": "local_qwen35",
                "raw": content,
                "model": model_name,
            }
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            # max_new_tokens预算下reason字符串常被截断成非法JSON；score字段
            # 排在reason之前、截断时通常已完整输出，从残缺文本中直接恢复，
            # 避免少量截断把整份评测判为不完整（require_complete_judge退出2）。
            salvage = re.search(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', content or "")
            if salvage is not None:
                return {
                    "score": clamp_score_0_5(float(salvage.group(1))),
                    "reason": "reason truncated; score recovered from partial JSON",
                    "source": "local_qwen35",
                    "raw": content,
                    "model": model_name,
                }
            return {
                "score": None,
                "reason": f"local judge parse failed: {type(exc).__name__}: {exc}",
                "source": "local_qwen35",
                "raw": content,
                "model": model_name,
            }

    def _render_prompt(self, prompt):
        messages = [
            {"role": "system", "content": "You are a strict evaluator. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _generate_prompt_batch(self, prompts):
        rendered = [self._render_prompt(prompt) for prompt in prompts]
        inputs = self.tokenizer(rendered, return_tensors="pt", padding=True)
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        prompt_width = inputs["input_ids"].shape[1]
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
        texts = self.tokenizer.batch_decode(
            generated[:, prompt_width:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return texts

    def _score_prompt_batch(self, prompts):
        texts = self._generate_prompt_batch(prompts)
        return [self.parse_scored_response(text, self.model_path) for text in texts]

    def _score_record_group_batch(self, record_groups, metric_name, person_token):
        prompts = [
            build_packed_judge_prompt(records, metric_name, person_token)
            for records in record_groups
        ]
        texts = self._generate_prompt_batch(prompts)
        results = []
        for records, text in zip(record_groups, texts):
            try:
                scores = parse_packed_scores(text, len(records))
            except (json.JSONDecodeError, TypeError, ValueError):
                results.append(None)
                continue
            results.append(
                [
                    {
                        "score": score,
                        "reason": "packed local judge score; per-item rationale disabled",
                        "source": "local_qwen35_packed",
                        "raw": text,
                        "model": self.model_path,
                    }
                    for score in scores
                ]
            )
        return results

    def score_prompts(self, prompts):
        prompts = list(prompts)
        if not prompts:
            return []
        try:
            return self._score_prompt_batch(prompts)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or len(prompts) == 1:
                raise
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
            midpoint = len(prompts) // 2
            print(
                f"[Judge:local_qwen35] OOM at batch={len(prompts)}; retrying as "
                f"{midpoint}+{len(prompts) - midpoint}.",
                flush=True,
            )
            return self.score_prompts(prompts[:midpoint]) + self.score_prompts(prompts[midpoint:])

    def score_record_groups(self, record_groups, metric_name, person_token):
        """Score packed record groups, splitting on OOM or malformed packed output."""
        record_groups = [list(group) for group in record_groups]
        if not record_groups:
            return []
        try:
            results = self._score_record_group_batch(record_groups, metric_name, person_token)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or len(record_groups) == 1:
                raise
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
            midpoint = len(record_groups) // 2
            print(
                f"[Judge:local_qwen35] OOM at packed_prompt_batch={len(record_groups)}; "
                f"retrying as {midpoint}+{len(record_groups) - midpoint}.",
                flush=True,
            )
            return self.score_record_groups(
                record_groups[:midpoint], metric_name, person_token
            ) + self.score_record_groups(record_groups[midpoint:], metric_name, person_token)

        recovered = []
        for records, scored in zip(record_groups, results):
            if scored is not None:
                recovered.append(scored)
                continue
            if len(records) == 1:
                prompt = (
                    build_es_prompt(records[0], person_token)
                    if metric_name == "ES"
                    else build_dc_prompt(records[0])
                )
                fallback = self.score_prompts([prompt])[0]
                fallback["reason"] = "packed parse failed; " + fallback.get("reason", "")
                fallback["source"] = "local_qwen35_packed_fallback"
                recovered.append([fallback])
                continue
            midpoint = len(records) // 2
            print(
                f"[Judge:local_qwen35] malformed packed response for {len(records)} items; "
                f"retrying as {midpoint}+{len(records) - midpoint}.",
                flush=True,
            )
            left = self.score_record_groups([records[:midpoint]], metric_name, person_token)[0]
            right = self.score_record_groups([records[midpoint:]], metric_name, person_token)[0]
            recovered.append(left + right)
        return recovered


def cache_key(metric_name, model_name, person_token, record):
    payload = {
        "metric": metric_name,
        "model": model_name,
        "person": person_token,
        "question": record["question"],
        "gold": record["gold_answer"],
        "generated": record["generated_answer"],
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_judge_cache(cache_path):
    cache = {}
    path = Path(cache_path)
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = item.get("key")
            score = item.get("score_0_5", item.get("score"))
            if key and score is not None:
                cache[key] = item
    return cache


def judge_records(records, person_token, args, metric_name, judge=None):
    cache_path = Path(args.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_judge_cache(cache_path)
    if judge is None:
        judge = DashScopeJudge(
            model=args.judge_model,
            fallback_models=args.judge_fallback_models,
            api_key=args.dashscope_api_key,
            base_url=args.dashscope_base_url,
            timeout=args.judge_timeout,
            max_retries=args.judge_max_retries,
        )
    scores = [None] * len(records)
    lock = threading.Lock()

    def cached_score(cached):
        cached_value = cached.get("score_0_5", cached.get("score"))
        return {
            "score": cached_value,
            "reason": cached.get("reason", ""),
            "source": cached.get("source", "cache"),
            "model": cached.get("model", judge.current_model),
        }

    def make_item(index, record):
        key = cache_key(metric_name, judge.cache_model_key, person_token, record)
        prompt = build_es_prompt(record, person_token) if metric_name == "ES" else build_dc_prompt(record)
        return index, record, key, prompt

    def write_cache_items(items):
        if not items:
            return
        with lock:
            with cache_path.open("a", encoding="utf-8") as f:
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def cache_item(record, key, scored):
        return {
            "key": key,
            "metric": metric_name,
            "requested_model": judge.current_model,
            "model_chain": judge.models,
            "record_id": record["record_id"],
            **scored,
        }

    indexed_records = list(enumerate(records))
    if args.max_judge_items is not None:
        indexed_records = indexed_records[: args.max_judge_items]
    pending = []
    for index, record in indexed_records:
        key = cache_key(metric_name, judge.cache_model_key, person_token, record)
        if key in cache:
            scores[index] = cached_score(cache[key])
        else:
            pending.append(make_item(index, record))
    cached_count = len(indexed_records) - len(pending)
    route = "local" if hasattr(judge, "score_prompts") else "api"
    # Keep the old all-metrics override for reproducible A/B experiments.  The
    # production default is deliberately metric-specific: ES is not stable when
    # packed, while DC was stable in both the Sheldon and Do full-set checks.
    all_metrics_override = getattr(args, "local_judge_items_per_prompt", None)
    if all_metrics_override is not None:
        items_per_prompt = max(int(all_metrics_override), 1)
    elif metric_name == "ES":
        items_per_prompt = max(
            int(getattr(args, "local_judge_es_items_per_prompt", 1)), 1
        )
    else:
        items_per_prompt = max(
            int(getattr(args, "local_judge_dc_items_per_prompt", 10)), 1
        )
    print(
        f"[Judge:{metric_name}] total={len(indexed_records)}, cached={cached_count}, "
        f"need_{route}={len(pending)}, "
        + (
            f"prompt_batch_size={judge.batch_size}, items_per_prompt={items_per_prompt}"
            if route == "local"
            else f"workers={args.api_num_workers}"
        ),
        flush=True,
    )

    if route == "local":
        completed = cached_count
        if items_per_prompt == 1:
            for start in range(0, len(pending), judge.batch_size):
                batch = pending[start: start + judge.batch_size]
                scored_batch = judge.score_prompts([item[3] for item in batch])
                if len(scored_batch) != len(batch):
                    raise RuntimeError("Local judge returned a different number of scores than prompts.")
                cache_items = []
                for (index, record, key, _), scored in zip(batch, scored_batch):
                    scores[index] = scored
                    cache_items.append(cache_item(record, key, scored))
                write_cache_items(cache_items)
                completed += len(batch)
                if completed == len(indexed_records) or completed % 320 == 0:
                    print(f"[Judge:{metric_name}] completed {completed}/{len(indexed_records)}", flush=True)
        else:
            packed = [
                pending[start: start + items_per_prompt]
                for start in range(0, len(pending), items_per_prompt)
            ]
            for start in range(0, len(packed), judge.batch_size):
                prompt_batch = packed[start: start + judge.batch_size]
                scored_groups = judge.score_record_groups(
                    [[item[1] for item in group] for group in prompt_batch],
                    metric_name,
                    person_token,
                )
                if len(scored_groups) != len(prompt_batch):
                    raise RuntimeError("Local judge returned a different number of packed groups.")
                cache_items = []
                batch_records = 0
                for group, scored_group in zip(prompt_batch, scored_groups):
                    if len(scored_group) != len(group):
                        raise RuntimeError("Local judge returned the wrong packed score count.")
                    for (index, record, key, _), scored in zip(group, scored_group):
                        scores[index] = scored
                        cache_items.append(cache_item(record, key, scored))
                    batch_records += len(group)
                write_cache_items(cache_items)
                completed += batch_records
                print(
                    f"[Judge:{metric_name}] completed {completed}/{len(indexed_records)}",
                    flush=True,
                )
        return scores

    def work(item):
        index, record, key, prompt = item
        scored = judge.score_prompt(prompt)
        write_cache_items([cache_item(record, key, scored)])
        return index, scored

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.api_num_workers) as executor:
        futures = [executor.submit(work, item) for item in pending]
        completed = cached_count
        for future in concurrent.futures.as_completed(futures):
            index, scored = future.result()
            scores[index] = scored
            completed += 1
            if completed == len(indexed_records) or completed % 20 == 0:
                print(f"[Judge:{metric_name}] completed {completed}/{len(indexed_records)}", flush=True)
    return scores


def summarize_judge_usage(records):
    usage = {
        "entity_specificity": defaultdict(int),
        "descriptive_completeness": defaultdict(int),
    }
    for record in records:
        if record.get("entity_specificity") is not None:
            model = record.get("entity_specificity_model") or "unknown"
            usage["entity_specificity"][model] += 1
        if record.get("descriptive_completeness") is not None:
            model = record.get("descriptive_completeness_model") or "unknown"
            usage["descriptive_completeness"][model] += 1
    return {metric: dict(counts) for metric, counts in usage.items()}


def judge_scores_complete(records):
    return all(
        record.get("entity_specificity") is not None
        and record.get("descriptive_completeness") is not None
        for record in records
    )


def compute_bertscore(records, args):
    predictions = [record["generated_answer"] for record in records]
    references = [record["gold_answer"] for record in records]
    if not predictions:
        return [], {"score": None, "count": 0}
    try:
        from bert_score import score as bert_score
    except ImportError as exc:
        raise RuntimeError("bert_score is required for --compute_bertscore.") from exc

    kwargs = {
        "cands": predictions,
        "refs": references,
        "lang": args.bertscore_lang,
        "batch_size": args.bertscore_batch_size,
        "verbose": True,
    }
    if args.bertscore_model:
        kwargs["model_type"] = args.bertscore_model
    if args.bertscore_device:
        kwargs["device"] = args.bertscore_device
    if args.bertscore_rescale:
        kwargs["rescale_with_baseline"] = True
    _, _, f1 = bert_score(**kwargs)
    values = [float(item) for item in f1.tolist()]
    summary = {
        "score": sum(values) / len(values),
        "count": len(values),
        "model": args.bertscore_model or "default",
        "lang": args.bertscore_lang,
    }
    return values, summary


def safe_mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def summarize_group(records):
    predictions = [record["generated_answer"] for record in records]
    references = [record["gold_answer"] for record in records]
    return {
        "count": len(records),
        "bleu": compute_corpus_bleu(predictions, references),
        "bertscore_f1": safe_mean([record.get("bertscore_f1") for record in records]),
        "entity_specificity": safe_mean([record.get("entity_specificity") for record in records]),
        "descriptive_completeness": safe_mean([record.get("descriptive_completeness") for record in records]),
    }


def evaluate_records(records, person_token, args):
    predictions = [record["generated_answer"] for record in records]
    references = [record["gold_answer"] for record in records]

    accuracy = compute_accuracy(records, person_token)
    bleu = compute_corpus_bleu(predictions, references)

    bert_summary = {"score": None, "count": 0}
    if args.compute_bertscore:
        bert_values, bert_summary = compute_bertscore(records, args)
        for record, value in zip(records, bert_values):
            record["bertscore_f1"] = value

    judge = None
    if args.judge_backend == "dashscope":
        judge = DashScopeJudge(
            model=args.judge_model,
            fallback_models=args.judge_fallback_models,
            api_key=args.dashscope_api_key,
            base_url=args.dashscope_base_url,
            timeout=args.judge_timeout,
            max_retries=args.judge_max_retries,
        )
        es_scores = judge_records(records, person_token, args, "ES", judge=judge)
        dc_scores = judge_records(records, person_token, args, "DC", judge=judge)
    elif args.judge_backend == "local_qwen35":
        judge = LocalQwen35Judge(
            model_path=args.local_judge_model_path,
            device=args.local_judge_device,
            batch_size=args.local_judge_batch_size,
            max_new_tokens=args.local_judge_max_new_tokens,
        )
        es_scores = judge_records(records, person_token, args, "ES", judge=judge)
        dc_scores = judge_records(records, person_token, args, "DC", judge=judge)
    else:
        es_scores = [local_entity_specificity(record["generated_answer"], person_token) for record in records]
        dc_scores = [local_descriptive_completeness(record, person_token) for record in records]

    for record, es, dc in zip(records, es_scores, dc_scores):
        es = es or {"score": None, "reason": "not judged", "source": args.judge_backend}
        dc = dc or {"score": None, "reason": "not judged", "source": args.judge_backend}
        record["entity_specificity"] = es.get("score")
        record["entity_specificity_reason"] = es.get("reason", "")
        record["entity_specificity_source"] = es.get("source", args.judge_backend)
        record["entity_specificity_model"] = es.get("model", es.get("source", args.judge_backend))
        record["descriptive_completeness"] = dc.get("score")
        record["descriptive_completeness_reason"] = dc.get("reason", "")
        record["descriptive_completeness_source"] = dc.get("source", args.judge_backend)
        record["descriptive_completeness_model"] = dc.get("model", dc.get("source", args.judge_backend))
        record["identity_gold_label"] = detect_identity_presence(record["gold_answer"], person_token)
        record["identity_pred_label"] = detect_identity_presence(record["generated_answer"], person_token)
        record["sentence_bleu"] = compute_corpus_bleu([record["generated_answer"]], [record["gold_answer"]])["score"]

    by_category = {}
    for category in sorted({record["category"] for record in records}):
        group = [record for record in records if record["category"] == category]
        by_category[category] = summarize_group(group)
        if category == "identity":
            by_category[category]["accuracy"] = compute_accuracy(group, person_token)

    summary = {
        "input_json": str(args.input_json),
        "person_token": person_token,
        "num_records": len(records),
        "num_videos": len({record["video_path"] for record in records}),
        "judge_backend": args.judge_backend,
        "judge": {
            "requested_model": judge.current_model if judge else None,
            "model_chain": judge.models if judge else [],
            "active_model": judge.current_model if judge else None,
            "complete": judge_scores_complete(records),
            "model_usage": summarize_judge_usage(records),
        },
        "metrics": {
            "accuracy": accuracy,
            "bleu": bleu,
            "bertscore_f1": bert_summary,
            "entity_specificity": safe_mean([record.get("entity_specificity") for record in records]),
            "descriptive_completeness": safe_mean(
                [record.get("descriptive_completeness") for record in records]
            ),
        },
    }
    return summary, by_category, records


def write_outputs(output_dir, summary, by_category, records):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "metrics_summary.json"
    category_path = output_dir / "metrics_by_category.json"
    details_path = output_dir / "metrics_details.jsonl"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with category_path.open("w", encoding="utf-8") as f:
        json.dump(by_category, f, ensure_ascii=False, indent=2)
    with details_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return summary_path, category_path, details_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PVChat QA results.")
    parser.add_argument("--input_json", required=True, help="Path to test_results_*.json.")
    parser.add_argument("--output_dir", required=True, help="Directory for metric outputs.")
    parser.add_argument("--sks_name", default="", help="Person name, e.g. Sheldon or <Sheldon>.")

    parser.add_argument("--compute_bertscore", action="store_true")
    parser.add_argument("--bertscore_model", default="roberta-large")
    parser.add_argument("--bertscore_lang", default="en")
    parser.add_argument("--bertscore_device", default="")
    parser.add_argument("--bertscore_batch_size", type=int, default=16)
    parser.add_argument("--bertscore_rescale", action="store_true")

    default_local_judge = (
        Path(__file__).resolve().parents[3] / "models" / "Qwen3.5-35B-A3B"
    )
    parser.add_argument(
        "--judge_backend",
        choices=["none", "dashscope", "local_qwen35"],
        default="local_qwen35",
    )
    parser.add_argument("--judge_model", default=os.environ.get("DASHSCOPE_TEXT_MODEL", "qwen3.7-max"))
    parser.add_argument(
        "--judge_fallback_models",
        default=os.environ.get("DASHSCOPE_JUDGE_FALLBACK_MODELS", ""),
        help="Comma-separated models used after a DashScope quota error.",
    )
    parser.add_argument(
        "--dashscope_base_url",
        default=os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    )
    parser.add_argument("--dashscope_api_key", default=os.environ.get("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--api_num_workers", type=int, default=10)
    parser.add_argument("--judge_timeout", type=int, default=90)
    parser.add_argument("--judge_max_retries", type=int, default=3)
    parser.add_argument(
        "--local_judge_model_path",
        default=os.environ.get("PVCHAT_LOCAL_JUDGE_MODEL_PATH", str(default_local_judge)),
        help="Local Qwen3.5-35B-A3B directory used for ES/DC scoring.",
    )
    parser.add_argument("--local_judge_device", default="cuda:0")
    parser.add_argument("--local_judge_batch_size", type=int, default=64)
    parser.add_argument(
        "--local_judge_items_per_prompt",
        type=int,
        default=None,
        help=(
            "Compatibility/A-B override: use the same number of records per prompt "
            "for both ES and DC. By default the metric-specific settings below apply."
        ),
    )
    parser.add_argument(
        "--local_judge_es_items_per_prompt",
        type=int,
        default=1,
        help="Number of ES records per local-judge prompt; keep at 1 for score fidelity.",
    )
    parser.add_argument(
        "--local_judge_dc_items_per_prompt",
        type=int,
        default=10,
        help="Number of DC records per local-judge prompt; validated production default is 10.",
    )
    parser.add_argument("--local_judge_max_new_tokens", type=int, default=512)
    parser.add_argument("--cache_path", default="")
    parser.add_argument("--max_judge_items", type=int, default=None)
    parser.add_argument(
        "--require_complete_judge",
        action="store_true",
        help="Exit with status 2 when any ES/DC score is missing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.input_json = Path(args.input_json)
    args.output_dir = Path(args.output_dir)
    if not args.cache_path:
        args.cache_path = str(args.output_dir / "judge_cache.jsonl")

    with args.input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    person_token = normalize_person_token(args.sks_name) if args.sks_name else extract_person_from_data(data)
    if not person_token:
        raise ValueError("Cannot infer person token. Please pass --sks_name Sheldon.")

    records = flatten_result_json(data, person_token)
    records = [record for record in records if record["gold_answer"] and record["generated_answer"]]
    summary, by_category, details = evaluate_records(records, person_token, args)
    summary_path, category_path, details_path = write_outputs(args.output_dir, summary, by_category, details)

    print(f"[Saved] {summary_path}")
    print(f"[Saved] {category_path}")
    print(f"[Saved] {details_path}")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    if (
            args.judge_backend in {"dashscope", "local_qwen35"}
            and args.require_complete_judge
            and not summary["judge"]["complete"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
