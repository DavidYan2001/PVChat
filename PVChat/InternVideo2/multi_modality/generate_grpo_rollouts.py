import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from difflib import SequenceMatcher


YES_WORDS = (
    "yes",
    "present",
    "appears",
    "appear",
    "visible",
    "seen",
    "in the video",
    "can be seen",
)
NO_WORDS = (
    "no",
    "not",
    "absent",
    "does not appear",
    "do not appear",
    "doesn't appear",
    "cannot be seen",
    "can't be seen",
    "not visible",
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "does", "for", "from",
    "has", "have", "he", "her", "him", "his", "in", "is", "it", "its", "of", "on",
    "or", "she", "that", "the", "their", "there", "they", "this", "to", "video",
    "visible", "wearing", "with", "while", "who", "woman", "man", "person",
}
FORMAT_PREFIXES = ("question:", "answer:", "analysis:", "assistant:", "user:")
DEFAULT_KEYWORD_FALLBACK_MODELS = (
    "qwen3.7-max-2026-06-08",
    "qwen3.7-max-2026-05-17",
    "qwen3.6-max-preview",
)


def clean_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def parse_model_fallbacks(primary_model, fallback_models):
    models = [clean_text(primary_model)]
    if isinstance(fallback_models, str):
        extra = re.split(r"[,\s]+", fallback_models)
    else:
        extra = list(fallback_models or [])
    models.extend(clean_text(model) for model in extra)

    deduped = []
    seen = set()
    for model in models:
        if model and model not in seen:
            deduped.append(model)
            seen.add(model)
    return deduped


def is_quota_error(exc):
    text = str(exc).lower()
    status_code = getattr(exc, "status_code", None)
    return (
        "allocationquota.freetieronly" in text
        or ("allocationquota" in text and "freetieronly" in text)
        or (status_code == 403 and "quota" in text)
    )


def read_existing_group_ids(path):
    path = Path(path)
    if not path.exists():
        return set()
    group_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            group_id = item.get("group_id")
            if group_id:
                group_ids.add(str(group_id))
    return group_ids


def filter_state_dict_for_model(state_dict, model_keys):
    model_keys = set(model_keys)
    filtered = {}
    dropped = []
    for key, value in state_dict.items():
        if key in model_keys:
            filtered[key] = value
        else:
            dropped.append(key)
    return filtered, sorted(dropped)


def load_model_state_compat(model, state_path, torch_module):
    state_dict = torch_module.load(state_path, map_location="cpu")
    filtered_state, dropped = filter_state_dict_for_model(state_dict, model.state_dict().keys())
    if dropped:
        print(
            "[Checkpoint] Ignored checkpoint keys not present in current model: "
            + ", ".join(dropped[:10])
            + (" ..." if len(dropped) > 10 else ""),
            flush=True,
        )
    model.load_state_dict(filtered_state, strict=True)


def ensure_append_newline(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb") as handle:
        handle.seek(-1, os.SEEK_END)
        last = handle.read(1)
    if last != b"\n":
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n")


def parse_number(text):
    match = re.search(r"[-+]?\d*\.?\d+", str(text))
    return float(match.group(0)) if match else None


def select_available_gpus_from_smi(smi_output, threshold):
    available = []
    for line in str(smi_output or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        gpu_id = parse_number(parts[0])
        used = parse_number(parts[1])
        total = parse_number(parts[2])
        if gpu_id is None or used is None or not total:
            continue
        if used / total <= threshold:
            available.append(int(gpu_id))
    return available


def query_available_gpus(threshold):
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd, text=True)
    return select_available_gpus_from_smi(output, threshold)


def limit_gpu_ids(gpu_ids, maximum):
    gpu_ids = list(gpu_ids)
    return gpu_ids[:maximum] if maximum and maximum > 0 else gpu_ids


def shard_rows(rows, shard_id, num_shards):
    if num_shards <= 1:
        return rows
    return [row for idx, row in enumerate(rows) if idx % num_shards == shard_id]


def shard_output_path(output_jsonl, shard_id):
    path = Path(output_jsonl)
    return path.with_name(f"{path.stem}.shard{shard_id}{path.suffix}")


def backfill_shard_output_path(output_jsonl, shard_id):
    path = Path(output_jsonl)
    return path.with_name(f"{path.stem}.backfill_shard{shard_id}{path.suffix}")


def merge_shard_outputs(output_jsonl, shard_paths):
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_handle:
        for shard_path in shard_paths:
            shard_path = Path(shard_path)
            if not shard_path.exists():
                continue
            with shard_path.open("r", encoding="utf-8") as shard_handle:
                for line in shard_handle:
                    if line.strip():
                        output_handle.write(line if line.endswith("\n") else line + "\n")
    return output_path


def resolve_rollout_video_path(video_path, rollout_path):
    if os.path.isabs(video_path):
        return video_path
    rollout_dir = Path(rollout_path).resolve().parent
    for base_dir in (rollout_dir, rollout_dir.parent, rollout_dir.parent.parent):
        candidate = base_dir / video_path
        if candidate.exists():
            return str(candidate)
    return video_path


def parse_gpu_id_list(gpu_ids):
    if isinstance(gpu_ids, str):
        items = re.split(r"[,\s]+", gpu_ids.strip())
    else:
        items = list(gpu_ids or [])
    return [int(item) for item in items if str(item).strip()]


def count_jsonl_records(path):
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def contiguous_ranges(total, num_ranges):
    ranges = []
    for shard_id in range(num_ranges):
        start = total * shard_id // num_ranges
        end = total * (shard_id + 1) // num_ranges
        ranges.append((start, end))
    return ranges


def build_auto_gpu_child_command(args, shard_id, num_shards):
    child_output = shard_output_path(args.output_jsonl, shard_id)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--train_json", args.train_json,
        "--output_jsonl", str(child_output),
        "--model_path", args.model_path,
        "--sks_name", args.sks_name,
        "--num_samples", str(args.num_samples),
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
        "--top_k", str(args.top_k),
        "--max_new_tokens", str(args.max_new_tokens),
        "--seed", str(args.seed + shard_id),
        "--keyword_backend", args.keyword_backend,
        "--keyword_model", args.keyword_model,
        "--keyword_fallback_models", args.keyword_fallback_models,
        "--keyword_batch_size", str(args.keyword_batch_size),
        "--keyword_sleep", str(args.keyword_sleep),
        "--keyword_cache", args.keyword_cache,
        "--judge_device", args.judge_device,
        "--num_shards", str(num_shards),
        "--shard_id", str(shard_id),
    ]
    if args.checkpoint_path:
        command.extend(["--checkpoint_path", args.checkpoint_path])
    if args.max_groups:
        command.extend(["--max_groups", str(args.max_groups)])
    if args.judge_model_path:
        command.extend(["--judge_model_path", args.judge_model_path])
    return command


def build_backfill_child_command(args, shard_id, source_path, output_path, start, end):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--backfill_old_logprob",
        "--backfill_source_jsonl", str(source_path),
        "--train_json", args.train_json,
        "--output_jsonl", str(output_path),
        "--model_path", args.model_path,
        "--sks_name", args.sks_name,
        "--record_start", str(start),
        "--record_end", str(end),
    ]
    if args.checkpoint_path:
        command.extend(["--checkpoint_path", args.checkpoint_path])
    return command


def normalize_person_token(person_token):
    person_token = clean_text(person_token)
    if not person_token:
        return person_token
    if person_token.startswith("<") and person_token.endswith(">"):
        return person_token
    return f"<{person_token}>"


def strip_person_tokens(text, person_token):
    person_token = normalize_person_token(person_token)
    bare_name = person_token.strip("<>")
    text = re.sub(re.escape(person_token), " ", text, flags=re.IGNORECASE)
    text = re.sub(re.escape(bare_name), " ", text, flags=re.IGNORECASE)
    return text


def tokenize_words(text):
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text.lower())


def parse_yes_no(answer):
    text = clean_text(answer).lower()
    no_hit = any(word in text for word in NO_WORDS)
    if no_hit:
        return "no"
    yes_hit = any(word in text for word in YES_WORDS)
    if yes_hit and not no_hit:
        return "yes"
    return None


def classify_qa_type(question, gold_answer, is_special=False):
    question_l = clean_text(question).lower()
    answer_l = clean_text(gold_answer).lower().strip(" .")
    if is_special or answer_l in {"yes", "no"}:
        return "identity"
    if "wear" in question_l or "clothing" in question_l or "dress" in question_l:
        return "clothing"
    if "where" in question_l or "location" in question_l or "background" in question_l:
        return "location"
    if "emotion" in question_l or "feel" in question_l or "mood" in question_l:
        return "emotion"
    if "doing" in question_l or "action" in question_l or "activity" in question_l:
        return "action"
    return "open"


def extract_rule_keywords(gold_answer, person_token, max_keywords=8):
    text = strip_person_tokens(clean_text(gold_answer), person_token)
    words = [
        word for word in tokenize_words(text)
        if len(word) > 2 and word not in STOPWORDS
    ]
    phrases = []
    for idx, word in enumerate(words):
        if word not in phrases:
            phrases.append(word)
        if idx + 1 < len(words):
            phrase = f"{word} {words[idx + 1]}"
            if phrase not in phrases:
                phrases.append(phrase)
        if len(phrases) >= max_keywords:
            break
    return phrases


def keyword_overlap(candidate_answer, keywords):
    if not keywords:
        return 0.0
    text = clean_text(candidate_answer).lower()
    matches = sum(1 for keyword in keywords if keyword.lower() in text)
    return matches / max(len(keywords), 1)


def lexical_semantic_score(gold_answer, candidate_answer, person_token):
    gold = strip_person_tokens(clean_text(gold_answer).lower(), person_token)
    candidate = strip_person_tokens(clean_text(candidate_answer).lower(), person_token)
    gold_words = {w for w in tokenize_words(gold) if w not in STOPWORDS}
    cand_words = {w for w in tokenize_words(candidate) if w not in STOPWORDS}
    if not gold_words or not cand_words:
        return 0.0
    jaccard = len(gold_words & cand_words) / len(gold_words | cand_words)
    ratio = SequenceMatcher(None, gold, candidate).ratio()
    return max(0.0, min(1.0, 0.6 * jaccard + 0.4 * ratio))


def identity_name_score(candidate_answer, person_token, is_positive=True):
    text = clean_text(candidate_answer).lower()
    person_token = normalize_person_token(person_token)
    token = person_token.lower()
    bare = person_token.strip("<>").lower()
    if is_positive:
        if token in text:
            return 1.0
        if bare and bare in text:
            return 0.7
        if any(ref in text for ref in ("he ", "she ", "the person", "the man", "the woman")):
            return 0.3
        return 0.0
    if token in text or (bare and bare in text):
        return -1.0
    return 1.0


def conciseness_score(candidate_answer, qa_type):
    word_count = len(tokenize_words(candidate_answer))
    if qa_type == "identity":
        if word_count <= 20:
            return 1.0
        if word_count <= 40:
            return 0.5
        return 0.0
    if word_count <= 30:
        return 1.0
    if word_count <= 50:
        return 0.7
    if word_count <= 80:
        return 0.3
    return 0.0


def format_score(candidate_answer):
    text = clean_text(candidate_answer).lower()
    if not text:
        return 0.0
    if any(prefix in text for prefix in FORMAT_PREFIXES):
        return 0.5
    if len(set(text.split())) <= 2 and len(text.split()) > 8:
        return 0.0
    return 1.0


def hallucination_penalty(candidate_answer, person_token, qa_type, is_positive=True):
    text = clean_text(candidate_answer).lower()
    person_token = normalize_person_token(person_token)
    token = person_token.lower()
    bare = person_token.strip("<>").lower()
    mentions_person = token in text or (bare and bare in text)
    if not is_positive and mentions_person:
        return 1.0
    other_name = re.search(r"<[^>]+>", text)
    if is_positive and other_name and other_name.group(0) != token:
        return 0.5
    return 0.0


def score_candidate_answer(
    question,
    gold_answer,
    candidate_answer,
    person_token,
    qa_type=None,
    is_positive=True,
    keywords=None,
    judge_result=None,
):
    qa_type = qa_type or classify_qa_type(question, gold_answer)
    candidate_answer = clean_text(candidate_answer)
    person_token = normalize_person_token(person_token)
    if keywords is None:
        keywords = extract_rule_keywords(gold_answer, person_token)

    name_score = identity_name_score(candidate_answer, person_token, is_positive=is_positive)
    concise = conciseness_score(candidate_answer, qa_type)
    fmt = format_score(candidate_answer)
    hallucination = hallucination_penalty(candidate_answer, person_token, qa_type, is_positive=is_positive)

    if qa_type == "identity":
        gold_yn = parse_yes_no(gold_answer)
        answer_yn = parse_yes_no(candidate_answer)
        yes_no = 1.0 if gold_yn and gold_yn == answer_yn else -1.0 if gold_yn and answer_yn else -0.3
        reward = 0.70 * yes_no + 0.15 * name_score + 0.10 * concise + 0.05 * fmt - hallucination
        components = {
            "yes_no_score": yes_no,
            "identity_name_score": name_score,
            "conciseness_score": concise,
            "format_score": fmt,
            "hallucination_penalty": hallucination,
        }
    else:
        keyword = keyword_overlap(candidate_answer, keywords)
        semantic = lexical_semantic_score(gold_answer, candidate_answer, person_token)
        contradiction = 0.0
        if judge_result:
            semantic = float(judge_result.get("score", semantic))
            contradiction = 1.0 if judge_result.get("contradiction") else 0.0
        elif semantic < 0.18 and keyword == 0.0:
            contradiction = 0.5
        reward = (
            0.50 * semantic
            + 0.20 * keyword
            + 0.15 * name_score
            + 0.10 * concise
            + 0.05 * fmt
            - 0.50 * contradiction
            - hallucination
        )
        components = {
            "semantic_score": semantic,
            "keyword_score": keyword,
            "identity_name_score": name_score,
            "conciseness_score": concise,
            "format_score": fmt,
            "contradiction_penalty": contradiction,
            "hallucination_penalty": hallucination,
        }

    return {
        "reward": float(max(-2.0, min(1.5, reward))),
        "components": components,
    }


class DashScopeKeywordExtractor:
    def __init__(self, model_name, batch_size=10, fallback_models=None):
        from openai import OpenAI

        api_key = (
            os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("QWEN_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set.")
        self.client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
            timeout=120,
            max_retries=0,
        )
        self.model_names = parse_model_fallbacks(model_name, fallback_models)
        self.model_index = 0
        self.batch_size = batch_size

    def extract_batch(self, answers, person_token):
        prompt = (
            "Extract 2 to 6 short English key phrases from each answer for reward matching. "
            "Remove person names/tokens and generic verbs. Return only a JSON array of arrays.\n\n"
            f"Person token: {person_token}\n"
            f"Answers:\n{json.dumps(answers, ensure_ascii=False)}"
        )
        last_error = None
        while self.model_index < len(self.model_names):
            model_name = self.model_names[self.model_index]
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=512,
                )
                break
            except Exception as exc:
                last_error = exc
                if is_quota_error(exc) and self.model_index + 1 < len(self.model_names):
                    next_model = self.model_names[self.model_index + 1]
                    print(f"[Keyword] {model_name} quota error, switch to {next_model}", flush=True)
                    self.model_index += 1
                    continue
                raise
        else:
            raise last_error or RuntimeError("No DashScope keyword model is available.")
        text = response.choices[0].message.content or "[]"
        match = re.search(r"\[.*\]", text, flags=re.S)
        parsed = json.loads(match.group(0) if match else text)
        if not isinstance(parsed, list):
            raise ValueError("Keyword extractor did not return a list.")
        result = []
        for item in parsed[:len(answers)]:
            if isinstance(item, list):
                result.append([clean_text(x).lower() for x in item if clean_text(x)])
            else:
                result.append([])
        while len(result) < len(answers):
            result.append([])
        return result


class LocalTextJudge:
    def __init__(self, model_path, device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_available() else torch.float32,
            device_map="auto" if device.startswith("cuda") else None,
        )
        self.device = next(self.model.parameters()).device
        self.model.eval()

    def score(self, question, gold_answer, candidate_answer, qa_type):
        prompt = (
            "You are a strict text-only QA judge. Compare candidate_answer with gold_answer. "
            "Return only JSON with keys: score (0 to 1), contradiction (true/false).\n"
            f"qa_type: {qa_type}\n"
            f"question: {question}\n"
            f"gold_answer: {gold_answer}\n"
            f"candidate_answer: {candidate_answer}\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                temperature=0.0,
            )
        text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        score = float(parsed.get("score", 0.0))
        return {
            "score": max(0.0, min(1.0, score)),
            "contradiction": bool(parsed.get("contradiction", False)),
        }


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def load_keyword_cache(path):
    if not path or not os.path.exists(path):
        return {}
    return load_json(path)


def collect_unique_gold_answers(data):
    answers = []
    seen = set()
    for video in data.get("videos", []):
        for qa in video.get("qa_pairs", []):
            answer = clean_text(qa.get("answer", ""))
            if answer and answer not in seen:
                seen.add(answer)
                answers.append(answer)
    return answers


def build_keyword_cache(data, person_token, args):
    cache = load_keyword_cache(args.keyword_cache)
    missing = [answer for answer in collect_unique_gold_answers(data) if answer not in cache]
    total = len(collect_unique_gold_answers(data))
    print(
        f"[Keyword] total_unique={total}, cached={total - len(missing)}, "
        f"missing={len(missing)}, backend={args.keyword_backend}",
        flush=True,
    )
    if missing and args.keyword_backend == "api":
        extractor = DashScopeKeywordExtractor(
            args.keyword_model,
            batch_size=args.keyword_batch_size,
            fallback_models=args.keyword_fallback_models,
        )
        print(f"[Keyword] model fallback order: {', '.join(extractor.model_names)}", flush=True)
        for start in range(0, len(missing), args.keyword_batch_size):
            batch = missing[start:start + args.keyword_batch_size]
            try:
                keywords_list = extractor.extract_batch(batch, person_token)
            except Exception as exc:
                print(f"[Keyword] API extraction failed, fallback to rules: {exc}")
                keywords_list = [extract_rule_keywords(answer, person_token) for answer in batch]
            for answer, keywords in zip(batch, keywords_list):
                cache[answer] = keywords or extract_rule_keywords(answer, person_token)
            write_json(args.keyword_cache, cache)
            print(
                f"[Keyword] cached {min(start + len(batch), len(missing))}/{len(missing)} missing answers",
                flush=True,
            )
            time.sleep(args.keyword_sleep)
    for answer in missing:
        cache.setdefault(answer, extract_rule_keywords(answer, person_token))
    if args.keyword_cache and (missing or not os.path.exists(args.keyword_cache)):
        write_json(args.keyword_cache, cache)
    return cache


def prepare_keyword_cache(args):
    data = load_json(args.train_json)
    person_token = normalize_person_token(args.sks_name)
    cache = build_keyword_cache(data, person_token, args)
    print(
        f"[Keyword] Cache ready with {len(cache)} entries: "
        f"{getattr(args, 'keyword_cache', None)}",
        flush=True,
    )
    return cache


def is_positive_video(video, person_token):
    expected = normalize_person_token(person_token)
    return video.get("sks_present") == expected or video.get("is_positive") is True


def flatten_qa(data, person_token):
    rows = []
    for video_idx, video in enumerate(data.get("videos", [])):
        positive = is_positive_video(video, person_token)
        for qa_idx, qa in enumerate(video.get("qa_pairs", [])):
            rows.append({
                "flat_idx": len(rows),
                "video_idx": video_idx,
                "qa_idx": qa_idx,
                "video": video,
                "qa": qa,
                "is_positive": positive,
            })
    return rows


def normalize_group_advantages(candidates):
    rewards = [float(candidate["reward"]) for candidate in candidates]
    mean = sum(rewards) / max(len(rewards), 1)
    variance = sum((reward - mean) ** 2 for reward in rewards) / max(len(rewards), 1)
    std = math.sqrt(variance)
    for candidate in candidates:
        candidate["advantage"] = 0.0 if std < 1e-6 else (float(candidate["reward"]) - mean) / (std + 1e-6)
    return candidates


def load_pvchat_model(args):
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from finetune_internvideo_REOMH_one_person2_stage import ensure_complete_config

    config_source = args.checkpoint_path if args.checkpoint_path else args.model_path
    config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
    config.sks_name = args.sks_name
    config.model_config["sks_name"] = args.sks_name
    config = ensure_complete_config(config, os.path.join(args.model_path, "config.json"))

    tokenizer = AutoTokenizer.from_pretrained(config_source, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda()

    sks_tokens = [args.sks_name]
    prefix_tokens = [f"<token{i}>" for i in range(config.model_config["bridge"]["num_prefix_token"])]
    tokenizer.add_tokens(sks_tokens + prefix_tokens)
    model.tokenizer = tokenizer
    model.lm.resize_token_embeddings(len(tokenizer))
    if hasattr(model.lm, "lm_head"):
        old_lm_head = model.lm.lm_head
        if old_lm_head.out_features != len(tokenizer):
            model.lm.lm_head = nn.Linear(old_lm_head.in_features, len(tokenizer), bias=False).to(model.lm.device)
            model.lm.lm_head.weight.data[:old_lm_head.out_features] = old_lm_head.weight.data
    if args.checkpoint_path:
        from internvideo_compact_checkpoint import load_checkpoint_state

        load_checkpoint_state(model, args.checkpoint_path, torch_module=torch)
    model.eval()
    return model, tokenizer, config


def masked_answer_logprobs(logits, input_ids, labels):
    import torch.nn.functional as F

    shifted_logits = logits[:, :-1, :].float()
    shifted_input_ids = input_ids[:, 1:].long()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe_targets = shifted_input_ids.masked_fill(~mask, 0)
    token_logprobs = F.log_softmax(shifted_logits, dim=-1).gather(
        -1, safe_targets.unsqueeze(-1)
    ).squeeze(-1)
    token_logprobs = token_logprobs.masked_fill(~mask, 0.0)
    token_count = mask.sum(dim=1).clamp_min(1)
    return token_logprobs.sum(dim=1) / token_count


def build_grpo_answer_inputs(tokenizer, config, video_tensor, question, answer, max_length=512):
    import torch
    from finetune_internvideo_REOMH_one_person2_stage import IMG_TOKEN, VID_TOKEN, build_input_ids

    conversation = "[INST] "
    if video_tensor.shape[1] == 1:
        ilen = video_tensor.shape[0]
        conversation += ("<Image>" + IMG_TOKEN + "</Image>") * ilen
    else:
        ilen = video_tensor.shape[1]
        conversation += ("<Video>" + VID_TOKEN + "</Video>") * ilen
    conversation += "[/INST] "
    sks_and_tokens = (
        f"[{config.sks_name}]"
        + "is"
        + "".join([f"<token{i}>" for i in range(config.model_config["bridge"]["num_prefix_token"])])
        + " "
    )
    if config.model_config["bridge"]["num_prefix_token"] == 0:
        sks_and_tokens = f"[{config.sks_name}]"
    conversation += "[INST]" + sks_and_tokens + question + "[/INST]"
    conversation += clean_text(answer) + "</s>"

    tokenized = build_input_ids(
        tokenizer,
        conversation,
        max_length=max_length,
        add_special_tokens=True,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        video_placeholder="[<VID_PLH>]",
    )
    labels = tokenized["input_ids"].clone()
    inst_tokens = tokenizer.encode("[/INST]")
    inst_end_indices = torch.where(labels == inst_tokens[-1])[0]
    if len(inst_end_indices) >= 2:
        labels[:inst_end_indices[1] + 1] = -100
    labels[tokenized["index"]] = -100
    return tokenized, labels.to(torch.int64)


def compute_answer_old_logprobs(model, tokenizer, config, video_tensor, question, answers):
    import torch
    from torch.nn.utils import rnn as rnn_utils

    answers = [clean_text(answer) for answer in answers]
    if not answers:
        return []

    tokenized_items = []
    label_items = []
    for answer in answers:
        tokenized, labels = build_grpo_answer_inputs(tokenizer, config, video_tensor, question, answer)
        tokenized_items.append(tokenized)
        label_items.append(labels)

    input_ids = rnn_utils.pad_sequence(
        [item["input_ids"] for item in tokenized_items],
        batch_first=True,
        padding_value=0,
    ).cuda()
    attention_mask = rnn_utils.pad_sequence(
        [item["attention_mask"] for item in tokenized_items],
        batch_first=True,
        padding_value=0,
    ).cuda()
    video_idx = rnn_utils.pad_sequence(
        [item["index"] for item in tokenized_items],
        batch_first=True,
        padding_value=0,
    ).cuda()
    labels = rnn_utils.pad_sequence(label_items, batch_first=True, padding_value=-100).to(torch.int64).cuda()

    video = video_tensor.squeeze(0) if video_tensor.shape[0] == 1 else video_tensor
    video_batch = video.unsqueeze(0).expand(len(answers), *video.shape).contiguous().cuda()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            video=video_batch,
            labels=None,
            video_idx=video_idx,
        )
        logprobs = masked_answer_logprobs(outputs.logits, input_ids, labels)
    return [float(value) for value in logprobs.detach().cpu()]


def sample_model_answers(model, tokenizer, config, dataset, index, args):
    """Sample num_samples answers for one QA group.

    The group shares one prompt, so the samples are generated as one batch of
    identical rows (no padding needed): each row draws independently from the
    same softmax, i.e. the sampling distribution is unchanged versus the old
    one-by-one loop, only the RNG stream differs. --sample_batch_size 1 restores
    the serial behaviour exactly.
    """
    import torch

    sample = dataset[index]
    video = sample["video"]
    video = video.squeeze(0) if video.shape[0] == 1 else video
    num_samples = int(args.num_samples)
    chunk = int(getattr(args, "sample_batch_size", 0) or 0) or num_samples
    answers = []
    with torch.no_grad():
        remaining = num_samples
        while remaining > 0:
            b = min(chunk, remaining)
            outputs = model.generate_caption(
                input_ids=sample["input_ids"].unsqueeze(0).expand(b, -1).contiguous().cuda(),
                attention_mask=sample["attention_mask"].unsqueeze(0).expand(b, -1).contiguous().cuda(),
                video=video.unsqueeze(0).expand(b, *video.shape).contiguous().cuda(),
                video_idx=sample["video_idx"].unsqueeze(0).expand(b, -1).contiguous().cuda(),
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            for row in outputs:
                answers.append(clean_text(tokenizer.decode(row, skip_special_tokens=True)))
            remaining -= b
    old_logprobs = compute_answer_old_logprobs(
        model,
        tokenizer,
        config,
        sample["video"],
        sample["question"],
        answers,
    )
    return list(zip(answers, old_logprobs))



def install_rollout_caches(model, stage2_module):
    """Exact, rollout-only caches (no change to sampling semantics):

    1) load_video memo (2 entries): rows are grouped by video, so the ~25 QA
       groups of one video decode it once instead of 25 times.
    2) vision-encoder memo: the 4 identical rows of one group (and the next
       groups of the same video) reuse the encoder output instead of re-encoding
       the same frames 8x per group (4x in generate, 4x in old_logprob scoring).
    """
    import torch

    orig_load = stage2_module.load_video
    load_cache, load_order = {}, []

    def cached_load_video(video_path, *a, **k):
        key = (video_path, a, tuple(sorted(k.items())))
        hit = load_cache.get(key)
        if hit is not None:
            return hit
        out = orig_load(video_path, *a, **k)
        load_cache[key] = out
        load_order.append(key)
        while len(load_order) > 2:
            load_cache.pop(load_order.pop(0), None)
        return out

    stage2_module.load_video = cached_load_video

    orig_encode = model.encode_vision
    memo = {"input": None, "output": None, "instruction": None}

    def _clone(value):
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, (tuple, list)):
            return type(value)(_clone(v) for v in value)
        return value

    def cached_encode_vision(image, instruction=None):
        cached = memo["input"]
        if (
            cached is not None
            and instruction is None and memo["instruction"] is None
            and cached.shape == image.shape and cached.dtype == image.dtype
            and cached.device == image.device and torch.equal(cached, image)
        ):
            return _clone(memo["output"])
        out = orig_encode(image, instruction)
        memo.update(input=image.detach().clone(), output=_clone(out), instruction=instruction)
        return out

    model.encode_vision = cached_encode_vision


def generate_rollouts(args):
    from tqdm import tqdm
    from finetune_internvideo_REOMH_one_person2_stage import PersonalizedVideoDataset

    random.seed(args.seed)
    person_token = normalize_person_token(args.sks_name)
    data = load_json(args.train_json)
    keyword_cache = build_keyword_cache(data, person_token, args)
    rows = flatten_qa(data, person_token)
    if args.max_groups:
        rows = rows[:args.max_groups]
    if args.num_shards > 1:
        rows = shard_rows(rows, args.shard_id, args.num_shards)
        print(
            f"[GRPO] Shard {args.shard_id}/{args.num_shards}: {len(rows)} groups",
            flush=True,
        )

    judge = LocalTextJudge(args.judge_model_path, device=args.judge_device) if args.judge_model_path else None
    model, tokenizer, config = load_pvchat_model(args)
    if not getattr(args, "disable_rollout_caches", False):
        import finetune_internvideo_REOMH_one_person2_stage as stage2_module
        install_rollout_caches(model, stage2_module)
        print("[GRPO] rollout caches enabled (video decode memo + vision-encoder memo; exact)", flush=True)
    dataset = PersonalizedVideoDataset(args.train_json, tokenizer=tokenizer, device=model.device, config=config, split="test")

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_group_ids = read_existing_group_ids(output_path)
    if existing_group_ids:
        ensure_append_newline(output_path)
        print(f"[GRPO] Resume rollout generation, existing groups: {len(existing_group_ids)}", flush=True)
    mode = "a" if output_path.exists() else "w"
    with open(output_path, mode, encoding="utf-8") as handle:
        for row in tqdm(rows, desc="Generating GRPO rollouts"):
            group_id = f"{row['video_idx']}:{row['qa_idx']}"
            if group_id in existing_group_ids:
                continue
            qa = row["qa"]
            video = row["video"]
            question = qa.get("question", "")
            gold_answer = qa.get("answer", "")
            qa_type = classify_qa_type(question, gold_answer, qa.get("is_special", False))
            keywords = keyword_cache.get(clean_text(gold_answer)) or extract_rule_keywords(gold_answer, person_token)
            sampled_answers = sample_model_answers(model, tokenizer, config, dataset, row["flat_idx"], args)
            candidates = []
            for sample_idx, (answer, old_logprob) in enumerate(sampled_answers):
                judge_result = None
                if judge and qa_type != "identity":
                    judge_result = judge.score(question, gold_answer, answer, qa_type)
                score = score_candidate_answer(
                    question=question,
                    gold_answer=gold_answer,
                    candidate_answer=answer,
                    person_token=person_token,
                    qa_type=qa_type,
                    is_positive=row["is_positive"],
                    keywords=keywords,
                    judge_result=judge_result,
                )
                candidates.append({
                    "sample_idx": sample_idx,
                    "answer": answer,
                    "old_logprob": old_logprob,
                    "reward": score["reward"],
                    "components": score["components"],
                })
            normalize_group_advantages(candidates)
            record = {
                "group_id": group_id,
                "video_idx": row["video_idx"],
                "qa_idx": row["qa_idx"],
                "video_path": video.get("video_path"),
                "sks_present": video.get("sks_present"),
                "is_positive": row["is_positive"],
                "question": question,
                "gold_answer": gold_answer,
                "qa_type": qa_type,
                "keywords": keywords,
                "candidates": candidates,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    print(f"[GRPO] Rollouts saved to {output_path}")


def backfill_old_logprobs(args):
    from tqdm import tqdm
    from finetune_internvideo_REOMH_one_person2_stage import load_video

    source_path = Path(args.backfill_source_jsonl or args.output_jsonl)
    output_path = Path(args.output_jsonl)
    if not source_path.exists():
        raise FileNotFoundError(f"Rollout file not found: {source_path}")

    model, tokenizer, config = load_pvchat_model(args)
    in_place = source_path.resolve() == output_path.resolve()
    write_path = output_path.with_name(output_path.name + ".tmp_old_logprob") if in_place else output_path
    write_path.parent.mkdir(parents=True, exist_ok=True)
    updated_candidates = 0
    total_candidates = 0
    total_records = 0
    written_records = 0
    record_start = max(args.record_start, 0)
    record_end = args.record_end if args.record_end and args.record_end > record_start else None

    with source_path.open("r", encoding="utf-8") as input_handle, write_path.open("w", encoding="utf-8") as output_handle:
        for record_idx, line in enumerate(tqdm(input_handle, desc="Backfilling old_logprob")):
            line = line.strip()
            if not line:
                continue
            if record_idx < record_start:
                continue
            if record_end is not None and record_idx >= record_end:
                break
            record = json.loads(line)
            total_records += 1
            candidates = record.get("candidates", [])
            missing = [
                idx for idx, candidate in enumerate(candidates)
                if candidate.get("old_logprob") is None and clean_text(candidate.get("answer", ""))
            ]
            total_candidates += len(candidates)
            if missing:
                video_path = resolve_rollout_video_path(record.get("video_path", ""), source_path)
                video_tensor = load_video(
                    video_path,
                    num_segments=8,
                    return_msg=False,
                    resolution=224,
                    hd_num=6,
                )
                answers = [candidates[idx].get("answer", "") for idx in missing]
                old_logprobs = compute_answer_old_logprobs(
                    model,
                    tokenizer,
                    config,
                    video_tensor,
                    record.get("question", ""),
                    answers,
                )
                for idx, old_logprob in zip(missing, old_logprobs):
                    candidates[idx]["old_logprob"] = old_logprob
                    updated_candidates += 1
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_handle.flush()
            written_records += 1

    if in_place:
        os.replace(write_path, output_path)
    print(
        f"[GRPO] Backfilled old_logprob for {updated_candidates}/{total_candidates} "
        f"candidates in {total_records} records: {output_path}",
        flush=True,
    )
    return written_records


def run_auto_gpu_rollouts(args):
    gpus = limit_gpu_ids(
        query_available_gpus(args.gpu_memory_threshold),
        args.max_auto_gpus,
    )
    if not gpus:
        raise RuntimeError(
            f"No GPU has memory usage <= {args.gpu_memory_threshold:.0%}. "
            "Lower --gpu_memory_threshold or free a GPU."
        )

    print(f"[GPU] Using available GPUs: {gpus}", flush=True)
    data = load_json(args.train_json)
    build_keyword_cache(data, normalize_person_token(args.sks_name), args)

    shard_paths = [shard_output_path(args.output_jsonl, shard_id) for shard_id in range(len(gpus))]
    processes = []
    for shard_id, gpu_id in enumerate(gpus):
        command = build_auto_gpu_child_command(args, shard_id, len(gpus))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(
            f"[GPU {gpu_id}] Start shard {shard_id}/{len(gpus)} -> {shard_paths[shard_id]}",
            flush=True,
        )
        processes.append((gpu_id, subprocess.Popen(command, env=env)))

    failed = []
    for gpu_id, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failed.append((gpu_id, return_code))
    if failed:
        raise RuntimeError(f"GRPO rollout shard processes failed: {failed}")

    merged_path = merge_shard_outputs(args.output_jsonl, shard_paths)
    print(f"[GRPO] Merged shard outputs to {merged_path}", flush=True)


def run_backfill_on_gpus(args):
    gpu_ids = parse_gpu_id_list(args.backfill_gpu_ids)
    if not gpu_ids:
        raise ValueError("--backfill_gpu_ids must contain at least one GPU id")

    source_path = Path(args.output_jsonl)
    total_records = count_jsonl_records(source_path)
    ranges = contiguous_ranges(total_records, len(gpu_ids))
    shard_paths = [backfill_shard_output_path(args.output_jsonl, shard_id) for shard_id in range(len(gpu_ids))]

    print(
        f"[GRPO] Backfill old_logprob with GPUs {gpu_ids}, records={total_records}",
        flush=True,
    )
    processes = []
    for shard_id, (gpu_id, (start, end)) in enumerate(zip(gpu_ids, ranges)):
        command = build_backfill_child_command(args, shard_id, source_path, shard_paths[shard_id], start, end)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(
            f"[GPU {gpu_id}] Backfill records [{start}:{end}) -> {shard_paths[shard_id]}",
            flush=True,
        )
        processes.append((gpu_id, subprocess.Popen(command, env=env)))

    failed = []
    for gpu_id, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failed.append((gpu_id, return_code))
    if failed:
        raise RuntimeError(f"old_logprob backfill shard processes failed: {failed}")

    tmp_merged = Path(args.output_jsonl).with_name(Path(args.output_jsonl).name + ".tmp_backfill_merged")
    merge_shard_outputs(tmp_merged, shard_paths)
    os.replace(tmp_merged, source_path)
    print(f"[GRPO] Backfilled old_logprob merged to {source_path}", flush=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--sks_name", required=True)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument(
        "--disable_rollout_caches",
        action="store_true",
        help="关闭rollout阶段的精确缓存(同一视频只解码/视觉编码一次); 仅影响速度不影响采样分布。",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=0,
        help="每次generate并行采样的候选数; 0=一组的num_samples一次生成(同一prompt批量, 分布不变), 1=旧的逐个串行。",
    )
    parser.add_argument("--max_groups", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keyword_backend", choices=["rule", "api"], default="rule")
    parser.add_argument("--keyword_model", default=os.getenv("DASHSCOPE_TEXT_MODEL", "qwen3.7-max-preview"))
    parser.add_argument(
        "--keyword_fallback_models",
        default=",".join(DEFAULT_KEYWORD_FALLBACK_MODELS),
        help="Comma-separated DashScope keyword models used after the primary model hits quota.",
    )
    parser.add_argument("--keyword_batch_size", type=int, default=10)
    parser.add_argument("--keyword_sleep", type=float, default=0.2)
    parser.add_argument("--keyword_cache", default=None)
    parser.add_argument("--judge_model_path", default=None)
    parser.add_argument("--judge_device", default="cuda")
    parser.add_argument(
        "--backfill_old_logprob",
        action="store_true",
        help="Fill missing candidate old_logprob values in an existing rollout JSONL without regenerating answers.",
    )
    parser.add_argument("--backfill_gpu_ids", default="")
    parser.add_argument(
        "--prepare_keywords_only",
        action="store_true",
        help="Build the shared keyword cache without loading the video model.",
    )
    parser.add_argument("--backfill_source_jsonl", default=None)
    parser.add_argument("--record_start", type=int, default=0)
    parser.add_argument("--record_end", type=int, default=0)
    parser.add_argument("--auto_gpus", action="store_true")
    parser.add_argument("--gpu_memory_threshold", type=float, default=0.20)
    parser.add_argument(
        "--max_auto_gpus",
        type=int,
        default=0,
        help="Maximum GPUs used by --auto_gpus; 0 uses every available GPU.",
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    args = parser.parse_args()
    if args.max_groups <= 0:
        args.max_groups = None
    if args.num_shards < 1:
        parser.error("--num_shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        parser.error("--shard_id must be in [0, num_shards)")
    if args.keyword_cache is None:
        args.keyword_cache = str(Path(args.output_jsonl).with_suffix(".keywords.json"))
    return args


if __name__ == "__main__":
    args = get_args()
    if args.prepare_keywords_only:
        prepare_keyword_cache(args)
    elif args.backfill_old_logprob and args.backfill_gpu_ids:
        run_backfill_on_gpus(args)
    elif args.backfill_old_logprob:
        backfill_old_logprobs(args)
    elif args.auto_gpus:
        run_auto_gpu_rollouts(args)
    else:
        generate_rollouts(args)
