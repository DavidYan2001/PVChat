"""Qwen3.5 PVChat多卡生成测试结果，并调用现有五指标评测脚本。"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm

from .data import collate_qwen35_features, encode_record, load_qa_records
from .distributed import barrier, move_batch_to_device, unwrap_model
from .remoh_attention import remoh_generation_masks


def strip_thinking(text: str) -> str:
    """评测只保留最终回答，不把Qwen内部thinking文本计入指标。"""

    text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL | re.IGNORECASE)
    # 如果达到max_new_tokens时思考段尚未闭合，直接丢弃未完成的私有推理。
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    return " ".join(text.split()).strip()


def build_result_qa_pair(record, generated_answer):
    """保留评测所需的QA文本和数据集显式正负标签。"""

    return {
        "question": record.question,
        "answer": record.answer,
        "generated_answer": generated_answer,
        "is_special": record.is_special,
        "is_positive": record.is_positive,
    }


@contextmanager
def generation_runtime(model):
    """临时切到高效生成状态，并在退出时完整恢复训练设置。"""

    base_model = unwrap_model(model)
    was_training = base_model.training
    cache_configs = [base_model.config]
    text_config = getattr(base_model.config, "text_config", None)
    if text_config is not None and all(text_config is not item for item in cache_configs):
        cache_configs.append(text_config)
    old_cache_values = [getattr(config, "use_cache", None) for config in cache_configs]
    had_gradient_checkpointing = bool(
        getattr(base_model, "is_gradient_checkpointing", False)
    )

    base_model.eval()
    if had_gradient_checkpointing:
        base_model.gradient_checkpointing_disable()
    for config in cache_configs:
        config.use_cache = True
    try:
        yield base_model
    finally:
        for config, previous in zip(cache_configs, old_cache_values):
            config.use_cache = previous
        if had_gradient_checkpointing:
            base_model.gradient_checkpointing_enable()
        base_model.train(was_training)


def _model_inputs(batch, include_remoh_masks=True):
    values = {key: value for key, value in batch.items() if key not in ("records", "labels")}
    if include_remoh_masks:
        mm_types = values["mm_token_type_ids"]
        values["pvchat_video_token_mask"] = mm_types.eq(2)
        values["pvchat_text_token_mask"] = mm_types.eq(0)
    return values


@torch.no_grad()
def _generate_batch_answers(
    model,
    processor,
    records,
    personalized_tokens,
    stage,
    device,
    max_new_tokens=96,
    video_overrides=None,
):
    if not records:
        return []

    features = [
        encode_record(
            processor,
            record,
            personalized_tokens,
            stage=stage,
            answer=None,
            video_overrides=video_overrides,
        )
        for record in records
    ]
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id
    batch = collate_qwen35_features(
        features,
        pad_token_id,
        padding_side="left",
    )
    batch = move_batch_to_device(batch, device)
    base_model = unwrap_model(model)
    mm_types = batch["mm_token_type_ids"]
    attention_mask = batch["attention_mask"].bool()
    with remoh_generation_masks(
        base_model,
        mm_types.eq(2) & attention_mask,
        mm_types.eq(0) & attention_mask,
    ):
        generated = base_model.generate(
            **_model_inputs(batch, include_remoh_masks=False),
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    prompt_width = batch["input_ids"].shape[1]
    answer_ids = generated[:, prompt_width:]
    texts = processor.batch_decode(
        answer_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return [
        strip_thinking(text.replace("<|im_end|>", "").replace("<|endoftext|>", ""))
        for text in texts
    ]


@torch.no_grad()
def _generate_one_answer(
    model,
    processor,
    record,
    personalized_tokens,
    stage,
    device,
    max_new_tokens=96,
    video_overrides=None,
):
    return _generate_batch_answers(
        model,
        processor,
        [record],
        personalized_tokens,
        stage,
        device,
        max_new_tokens=max_new_tokens,
        video_overrides=video_overrides,
    )[0]


@torch.no_grad()
def generate_batch_answers(
    model,
    processor,
    records,
    personalized_tokens,
    stage,
    device,
    max_new_tokens=96,
    video_overrides=None,
):
    """在一次generate调用中独立回答多条视频QA。"""

    with generation_runtime(model):
        return _generate_batch_answers(
            model,
            processor,
            records,
            personalized_tokens,
            stage,
            device,
            max_new_tokens=max_new_tokens,
            video_overrides=video_overrides,
        )


@torch.no_grad()
def generate_one_answer(
    model,
    processor,
    record,
    personalized_tokens,
    stage,
    device,
    max_new_tokens=96,
    video_overrides=None,
):
    """独立生成一个回答；批量评测会在外层复用同一生成上下文。"""

    with generation_runtime(model):
        return _generate_one_answer(
            model,
            processor,
            record,
            personalized_tokens,
            stage,
            device,
            max_new_tokens=max_new_tokens,
            video_overrides=video_overrides,
        )


def run_distributed_evaluation(
    model,
    processor,
    test_json,
    personalized_tokens,
    output_json,
    context,
    stage=2,
    max_new_tokens=96,
    batch_size=4,
    video_overrides=None,
):
    """按视频分配到不同GPU，rank0合并为旧评测脚本兼容的JSON。"""

    records = load_qa_records(test_json)
    local_records = [record for record in records if record.video_index % context.world_size == context.rank]
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("评测batch_size必须大于0。")

    local_outputs = []
    progress = tqdm(
        total=len(local_records),
        desc=f"[GPU {context.local_rank}] Qwen3.5 test",
        disable=False,
    )
    with generation_runtime(model):
        for start in range(0, len(local_records), batch_size):
            record_batch = local_records[start : start + batch_size]
            answers = _generate_batch_answers(
                model,
                processor,
                record_batch,
                personalized_tokens,
                stage,
                context.device,
                max_new_tokens=max_new_tokens,
                video_overrides=video_overrides,
            )
            local_outputs.extend(
                (record.flat_index, answer)
                for record, answer in zip(record_batch, answers)
            )
            progress.update(len(record_batch))
    progress.close()

    if context.distributed:
        gathered = [None for _ in range(context.world_size)]
        dist.all_gather_object(gathered, local_outputs)
        all_outputs = [item for shard in gathered for item in shard]
    else:
        all_outputs = local_outputs

    if context.is_main:
        generated_by_index = dict(all_outputs)
        grouped = defaultdict(list)
        video_metadata = {}
        for record in records:
            video_metadata[record.video_index] = (record.video_path, record.video_name)
            grouped[record.video_index].append(
                build_result_qa_pair(
                    record,
                    generated_by_index.get(record.flat_index, ""),
                )
            )
        payload = {
            "model_name": personalized_tokens[0].strip("<>"),
            "results": [
                {
                    "video_path": video_metadata[index][0],
                    "video_name": video_metadata[index][1],
                    "qa_pairs": grouped[index],
                }
                for index in sorted(grouped)
            ],
        }
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    barrier(context)
    return Path(output_json)


def run_metrics(
    result_json,
    output_dir,
    person_token,
    judge_backend="local_qwen35",
    judge_model="qwen3.7-max",
    judge_fallback_models="qwen3.7-max-preview,qwen3.7-max-2026-06-08,qwen3.7-max-2026-05-17,qwen3.6-max-preview",
    api_num_workers=20,
    bertscore_device=None,
    local_judge_model_path=None,
    local_judge_device="cuda:0",
    local_judge_batch_size=64,
    local_judge_es_items_per_prompt=1,
    local_judge_dc_items_per_prompt=10,
    local_judge_max_new_tokens=64,
):
    script = Path(__file__).resolve().parent.parent / "evaluate_pvchat_metrics.py"
    if local_judge_model_path is None:
        local_judge_model_path = (
            Path(__file__).resolve().parents[4] / "models" / "Qwen3.5-35B-A3B"
        )
    command = [
        sys.executable,
        str(script),
        "--input_json",
        str(result_json),
        "--output_dir",
        str(output_dir),
        "--sks_name",
        person_token.strip("<>"),
        "--compute_bertscore",
        "--bertscore_model",
        "roberta-large",
        "--judge_backend",
        judge_backend,
        "--require_complete_judge",
    ]
    if judge_backend == "dashscope":
        command.extend(
            [
                "--judge_model",
                judge_model,
                "--judge_fallback_models",
                judge_fallback_models,
                "--api_num_workers",
                str(api_num_workers),
            ]
        )
    elif judge_backend == "local_qwen35":
        command.extend(
            [
                "--local_judge_model_path",
                str(local_judge_model_path),
                "--local_judge_device",
                str(local_judge_device),
                "--local_judge_batch_size",
                str(local_judge_batch_size),
                "--local_judge_es_items_per_prompt",
                str(local_judge_es_items_per_prompt),
                "--local_judge_dc_items_per_prompt",
                str(local_judge_dc_items_per_prompt),
                "--local_judge_max_new_tokens",
                str(local_judge_max_new_tokens),
            ]
        )
    if bertscore_device:
        command.extend(["--bertscore_device", bertscore_device])
    environment = os.environ.copy()
    if judge_backend == "local_qwen35":
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    subprocess.run(command, check=True, env=environment)
    return Path(output_dir) / "metrics_summary.json"
