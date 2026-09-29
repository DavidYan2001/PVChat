"""PVChat JSON到Qwen3.5视频对话输入的转换。"""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
import subprocess
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .personalization import build_identity_prefix


# Stage 1只读取4帧。现有short video已经由同一首帧复制而成。
STAGE1_VIDEO_KWARGS = {"num_frames": 4, "fps": None}

# Stage 2遵循Qwen3.5官方默认：每秒2帧，至少4帧，最多768帧。
STAGE2_VIDEO_KWARGS = {
    "num_frames": None,
    "fps": 2.0,
    "min_frames": 4,
    "max_frames": 768,
}

# Qwen视频处理器用32x32像素块表达动态视觉预算。模型自带配置允许到
# 24576块，单条训练样本会产生数千视觉token。训练默认限制为768块，
# 与用户确认的上限一致；最小4块保留低分辨率短视频。
VIDEO_PIXEL_BLOCK = 32 * 32
DEFAULT_VIDEO_MIN_TOKENS = 4
DEFAULT_VIDEO_MAX_TOKENS = 768


@dataclass(frozen=True)
class QARecord:
    flat_index: int
    video_index: int
    qa_index: int
    video_path: str
    video_name: str
    question: str
    answer: str
    is_special: bool
    is_positive: bool
    sks_present: str


def _resolve_video_path(raw_path: str, json_path: Path) -> str:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return str(path)
    candidates = [json_path.parent / path, json_path.parent.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str((json_path.parent / path).resolve())


def load_qa_records(json_path: str | Path) -> list[QARecord]:
    """把每个视频中的QA拍平成独立训练样本。"""

    json_path = Path(json_path).expanduser().resolve()
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    videos = payload.get("videos") if isinstance(payload, dict) else None
    if not isinstance(videos, list):
        raise ValueError(f"{json_path}必须包含videos列表。")

    records = []
    for video_index, video in enumerate(videos):
        for qa_index, qa in enumerate(video.get("qa_pairs", [])):
            records.append(
                QARecord(
                    flat_index=len(records),
                    video_index=video_index,
                    qa_index=qa_index,
                    video_path=_resolve_video_path(video.get("video_path", ""), json_path),
                    video_name=video.get("video_name", Path(video.get("video_path", "")).name),
                    question=str(qa.get("question", "")),
                    answer=str(qa.get("answer", qa.get("gold_answer", ""))),
                    is_special=bool(qa.get("is_special", False)),
                    is_positive=bool(video.get("is_positive", False)),
                    sks_present=str(video.get("sks_present", "")),
                )
            )
    return records


def build_video_messages(
    record: QARecord,
    personalized_tokens: Sequence[str],
    answer: str | None = None,
):
    """构建Qwen原生的video+text聊天消息。"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "url": record.video_path},
                {
                    "type": "text",
                    "text": build_identity_prefix(record.question, personalized_tokens),
                },
            ],
        }
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def _find_last_subsequence(values: Sequence[int], pattern: Sequence[int]) -> int:
    if not pattern:
        raise ValueError("assistant marker不能为空。")
    for start in range(len(values) - len(pattern), -1, -1):
        if list(values[start : start + len(pattern)]) == list(pattern):
            return start
    return -1


def mask_prompt_labels(
    input_ids: torch.Tensor,
    assistant_marker_ids: Sequence[int],
    pad_token_id: int | None,
) -> torch.Tensor:
    """只让assistant答案产生CE loss，视频、问题和角色标记全部设为-100。"""

    values = input_ids.tolist()
    marker_start = _find_last_subsequence(values, assistant_marker_ids)
    if marker_start < 0:
        raise ValueError("输入中找不到assistant角色标记，无法构建labels。")
    answer_start = marker_start + len(assistant_marker_ids)
    labels = input_ids.clone()
    labels[:answer_start] = -100
    if pad_token_id is not None:
        labels[labels == int(pad_token_id)] = -100
    return labels


def assistant_marker_ids(tokenizer) -> list[int]:
    return tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)


def video_overrides_from_token_budget(
    min_tokens: int = DEFAULT_VIDEO_MIN_TOKENS,
    max_tokens: int = DEFAULT_VIDEO_MAX_TOKENS,
) -> dict:
    """把易读的视觉token预算换成处理器需要的像素面积参数。"""

    min_tokens = int(min_tokens)
    max_tokens = int(max_tokens)
    if min_tokens <= 0 or max_tokens <= 0:
        raise ValueError("视频视觉token预算必须大于0。")
    if min_tokens > max_tokens:
        raise ValueError("video_min_tokens不能大于video_max_tokens。")
    return {
        "size": {
            "shortest_edge": min_tokens * VIDEO_PIXEL_BLOCK,
            "longest_edge": max_tokens * VIDEO_PIXEL_BLOCK,
        }
    }


DEFAULT_VIDEO_BUDGET_WINDOW_SECONDS = 10.0


def video_duration_seconds(video_path: str) -> float:
    """读取视频实际时长（秒）。优先用torchcodec的容器元数据，缺失时退回ffprobe。"""

    try:
        from torchcodec.decoders import VideoDecoder

        meta = VideoDecoder(video_path).metadata
        for name in ("duration_seconds", "duration_seconds_from_header"):
            value = getattr(meta, name, None)
            if value is not None and float(value) > 0:
                return float(value)
    except Exception:  # noqa: BLE001 - 退回ffprobe
        pass
    ffprobe = os.environ.get("PVCHAT_FFPROBE") or shutil.which("ffprobe")
    if ffprobe:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video_path],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if out:
            return float(out)
    raise RuntimeError(f"无法读取视频时长: {video_path}")


class DynamicVideoBudget:
    """逐视频按实际时长缩放的视觉预算：tokens_per_window * ceil(秒数 / 10)。

    与新版Stage 2的``--video_budget_per_10_seconds``规则一致：10秒768、30秒
    2304、60秒4608、180秒13824，超过继续线性增加（``max_tokens``可选封顶）。
    短的扩充视频仍按自身时长处理，不再被最长视频的固定预算抬高。时长按
    路径缓存，同一进程内只探测一次。
    """

    def __init__(
        self,
        tokens_per_window: int,
        min_tokens: int = DEFAULT_VIDEO_MIN_TOKENS,
        window_seconds: float = DEFAULT_VIDEO_BUDGET_WINDOW_SECONDS,
        max_tokens: int | None = None,
    ):
        self.tokens_per_window = int(tokens_per_window)
        self.min_tokens = int(min_tokens)
        self.window_seconds = float(window_seconds)
        self.max_tokens = int(max_tokens) if max_tokens else None
        if self.tokens_per_window <= 0 or self.min_tokens <= 0 or self.window_seconds <= 0:
            raise ValueError("动态视觉预算参数必须大于0。")
        if self.max_tokens is not None and self.max_tokens < self.tokens_per_window:
            raise ValueError("动态视觉预算封顶不能小于单个时长窗口的预算。")
        self._durations: dict[str, float] = {}
        self._budgets: dict[str, int] = {}

    def duration_seconds(self, video_path: str) -> float:
        if video_path not in self._durations:
            self._durations[video_path] = video_duration_seconds(video_path)
        return self._durations[video_path]

    def budget_tokens(self, video_path: str) -> int:
        if video_path not in self._budgets:
            seconds = self.duration_seconds(video_path)
            # 1e-6容差：整30.0秒算3个窗口而不是被浮点误差推成4个。
            windows = max(1, math.ceil(seconds / self.window_seconds - 1e-6))
            tokens = self.tokens_per_window * windows
            if self.max_tokens is not None:
                tokens = min(tokens, self.max_tokens)
            self._budgets[video_path] = max(self.min_tokens, tokens)
        return self._budgets[video_path]

    def overrides_for(self, video_path: str) -> dict:
        return video_overrides_from_token_budget(self.min_tokens, self.budget_tokens(video_path))

    def describe(self) -> dict:
        return {
            "policy": "per_video_duration",
            "tokens_per_window": self.tokens_per_window,
            "window_seconds": self.window_seconds,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
        }


def resolve_video_overrides(video_overrides: Any, video_path: str) -> dict | None:
    """固定预算直接返回dict；DynamicVideoBudget按该视频的时长求预算。"""

    if video_overrides is None or isinstance(video_overrides, Mapping):
        return video_overrides
    return video_overrides.overrides_for(video_path)


def processor_video_kwargs(stage: int, overrides: dict | None = None) -> dict:
    values = dict(STAGE1_VIDEO_KWARGS if stage == 1 else STAGE2_VIDEO_KWARGS)
    values.update(video_overrides_from_token_budget())
    if overrides:
        values.update({key: value for key, value in overrides.items() if value is not None})
    values["do_sample_frames"] = True
    return values


DEFAULT_VIDEO_CACHE_ENTRIES = 256


def _clone_cached_video(video):
    if isinstance(video, torch.Tensor):
        return video.clone()
    import numpy as np

    if isinstance(video, np.ndarray):
        return video.copy()
    return copy.deepcopy(video)


class VideoDecodeCache:
    """fetch_videos结果的按路径LRU缓存句柄。"""

    def __init__(self, max_entries: int):
        self.max_entries = max(1, int(max_entries))
        self.entries: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def clear(self) -> None:
        self.entries.clear()

    def __len__(self) -> int:
        return len(self.entries)


# 缓存状态一律放在processor对象之外：processor.save_pretrained会把实例
# __dict__序列化成JSON，任何挂在实例上的缓存属性都会让checkpoint保存
# 崩溃（曾在训练收尾时真实发生）。WeakKeyDictionary保证processor释放
# 后缓存随之回收。
_VIDEO_CACHES: "weakref.WeakKeyDictionary" = None  # type: ignore[assignment]
_PATCHED_CLASSES: set = set()


def _video_cache_registry():
    global _VIDEO_CACHES
    if _VIDEO_CACHES is None:
        import weakref

        _VIDEO_CACHES = weakref.WeakKeyDictionary()
    return _VIDEO_CACHES


def install_video_decode_cache(
    processor,
    max_entries: int = DEFAULT_VIDEO_CACHE_ENTRIES,
) -> VideoDecodeCache | None:
    """给processor的视频解码装一层按路径的LRU缓存。

    数据集里同一个视频对应几十条QA，逐条重新解码是Stage 2/3 CPU侧的
    重复开销。抽帧结果只由视频文件和采样参数决定，而采样参数在单个
    训练进程内是常量（stage在进程启动时固定），因此按路径缓存是安全
    的；若未来在同一进程内混用不同采样参数，需先调用返回句柄的clear()。

    实现上patch视频处理器的类方法而非实例属性，缓存状态存放在模块级
    WeakKeyDictionary里——processor实例的__dict__保持原样，因此
    save_pretrained的JSON序列化不受任何影响。命中与未命中都返回防御
    性拷贝，避免下游原地修改污染缓存条目。设置环境变量
    PVCHAT_VIDEO_CACHE=0可整体禁用。重复安装是幂等的。
    """

    if os.environ.get("PVCHAT_VIDEO_CACHE", "1") == "0":
        return None
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is None:
        return None

    registry = _video_cache_registry()
    existing = registry.get(video_processor)
    if existing is not None:
        return existing
    cache = VideoDecodeCache(max_entries)
    registry[video_processor] = cache

    cls = type(video_processor)
    if cls not in _PATCHED_CLASSES:
        original_fetch = cls.fetch_videos

        def cached_fetch(self, video_url_or_urls, sample_indices_fn=None):
            state = _video_cache_registry().get(self)
            if state is None or not isinstance(video_url_or_urls, str):
                # 未启用缓存的实例、以及列表输入（由原实现递归处理，
                # 每个元素会再次进入本包装）都直接走原路径。
                return original_fetch(self, video_url_or_urls, sample_indices_fn=sample_indices_fn)
            key = video_url_or_urls
            if key in state.entries:
                state.entries.move_to_end(key)
                state.hits += 1
                video, metadata = state.entries[key]
            else:
                video, metadata = original_fetch(self, key, sample_indices_fn=sample_indices_fn)
                state.misses += 1
                state.entries[key] = (video, metadata)
                while len(state.entries) > state.max_entries:
                    state.entries.popitem(last=False)
            return _clone_cached_video(video), copy.copy(metadata)

        cls.fetch_videos = cached_fetch
        _PATCHED_CLASSES.add(cls)
    return cache


def encode_record(
    processor,
    record: QARecord,
    personalized_tokens: Sequence[str],
    stage: int,
    answer: str | None,
    video_overrides: Any = None,
):
    """读取一次视频并生成Qwen3.5需要的所有tensor。"""

    is_training = answer is not None
    messages = build_video_messages(record, personalized_tokens, answer=answer)
    encoded = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=not is_training,
        # 数据只提供最终答案，没有思维链；所有Stage统一使用non-thinking模板。
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs=processor_video_kwargs(stage, resolve_video_overrides(video_overrides, record.video_path)),
    )
    feature = {key: value.squeeze(0) if value.ndim > 0 and value.shape[0] == 1 else value for key, value in encoded.items()}
    if is_training:
        feature["labels"] = mask_prompt_labels(
            feature["input_ids"],
            assistant_marker_ids(processor.tokenizer),
            processor.tokenizer.pad_token_id,
        )
    feature["record"] = record
    return feature


class PVChatSFTDataset(Dataset):
    def __init__(self, json_path, processor, personalized_tokens, stage, video_overrides=None):
        self.records = load_qa_records(json_path)
        self.processor = processor
        self.personalized_tokens = list(personalized_tokens)
        self.stage = int(stage)
        self.video_overrides = video_overrides

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return encode_record(
            self.processor,
            record,
            self.personalized_tokens,
            self.stage,
            answer=record.answer,
            video_overrides=self.video_overrides,
        )


def collate_qwen35_features(
    features: list[dict],
    pad_token_id: int,
    padding_side: str = "right",
) -> dict:
    """合并多条视频样本；训练右填充，批量生成使用左填充。"""

    if padding_side not in ("left", "right"):
        raise ValueError("padding_side必须是left或right。")

    max_length = max(item["input_ids"].shape[0] for item in features)

    def pad_1d(value, fill):
        if value.shape[0] == max_length:
            return value
        padding = max_length - value.shape[0]
        pad_width = (padding, 0) if padding_side == "left" else (0, padding)
        return torch.nn.functional.pad(value, pad_width, value=fill)

    # 生成时左侧PAD不能被ReMoH误认为普通文本token。
    mm_pad_value = -1 if padding_side == "left" else 0

    batch = {
        "input_ids": torch.stack([pad_1d(item["input_ids"], pad_token_id) for item in features]),
        "attention_mask": torch.stack([pad_1d(item["attention_mask"], 0) for item in features]),
        "mm_token_type_ids": torch.stack(
            [pad_1d(item["mm_token_type_ids"], mm_pad_value) for item in features]
        ),
        "pixel_values_videos": torch.cat([item["pixel_values_videos"] for item in features], dim=0),
        "video_grid_thw": torch.cat([item["video_grid_thw"].reshape(-1, 3) for item in features], dim=0),
        "records": [item["record"] for item in features],
    }
    if "labels" in features[0]:
        batch["labels"] = torch.stack([pad_1d(item["labels"], -100) for item in features])
    return batch
