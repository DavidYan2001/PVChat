"""训练前检查PVChat JSON是否同时包含正样本和负样本。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingClassSummary:
    label: str
    path: Path
    positive_videos: int
    negative_videos: int
    positive_qa_pairs: int
    negative_qa_pairs: int

    @property
    def total_videos(self):
        return self.positive_videos + self.negative_videos

    @property
    def total_qa_pairs(self):
        return self.positive_qa_pairs + self.negative_qa_pairs


def validate_training_json_classes(json_path, label="training"):
    path = Path(json_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    videos = payload.get("videos") if isinstance(payload, dict) else None
    if not isinstance(videos, list):
        raise ValueError(f"{label} JSON必须包含videos列表: {path}")

    counts = {
        "positive_videos": 0,
        "negative_videos": 0,
        "positive_qa_pairs": 0,
        "negative_qa_pairs": 0,
    }
    for video in videos:
        prefix = "positive" if video.get("is_positive") is True else "negative"
        counts[f"{prefix}_videos"] += 1
        qa_pairs = video.get("qa_pairs", [])
        if not isinstance(qa_pairs, list):
            raise ValueError(f"{label}中的qa_pairs必须是列表: {path}")
        counts[f"{prefix}_qa_pairs"] += len(qa_pairs)

    summary = TrainingClassSummary(label=label, path=path, **counts)
    if summary.positive_videos == 0 or summary.positive_qa_pairs == 0:
        raise ValueError(f"{label}缺少positive训练样本: {path}")
    if summary.negative_videos == 0 or summary.negative_qa_pairs == 0:
        raise ValueError(f"{label}缺少negative训练样本: {path}")
    return summary


def describe_training_summary(summary):
    return (
        f"[Dataset] {summary.label}: videos={summary.total_videos} "
        f"(positive={summary.positive_videos}, negative={summary.negative_videos}), "
        f"QA={summary.total_qa_pairs} "
        f"(positive={summary.positive_qa_pairs}, negative={summary.negative_qa_pairs})"
    )
