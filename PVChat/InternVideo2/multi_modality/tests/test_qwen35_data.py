import json
import tempfile
import unittest
from pathlib import Path

import torch

from qwen35_pvchat.data import (
    QARecord,
    STAGE1_VIDEO_KWARGS,
    STAGE2_VIDEO_KWARGS,
    collate_qwen35_features,
    encode_record,
    load_qa_records,
    mask_prompt_labels,
    processor_video_kwargs,
    video_overrides_from_token_budget,
)


class _TemplateTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del text, add_special_tokens
        return [20, 21]


class _TemplateProcessor:
    def __init__(self):
        self.tokenizer = _TemplateTokenizer()
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {
            "input_ids": torch.tensor([[10, 20, 21, 30]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "mm_token_type_ids": torch.zeros(1, 4, dtype=torch.long),
            "pixel_values_videos": torch.zeros(1, 3),
            "video_grid_thw": torch.tensor([[1, 1, 1]]),
        }


class Qwen35DataTest(unittest.TestCase):
    def test_generation_collate_left_pads_variable_length_prompts(self):
        features = [
            {
                "input_ids": torch.tensor([10, 11]),
                "attention_mask": torch.tensor([1, 1]),
                "mm_token_type_ids": torch.tensor([2, 0]),
                "pixel_values_videos": torch.zeros(1, 3),
                "video_grid_thw": torch.tensor([[1, 1, 1]]),
                "record": "short",
            },
            {
                "input_ids": torch.tensor([20, 21, 22]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "mm_token_type_ids": torch.tensor([2, 2, 0]),
                "pixel_values_videos": torch.zeros(1, 3),
                "video_grid_thw": torch.tensor([[1, 1, 1]]),
                "record": "long",
            },
        ]

        batch = collate_qwen35_features(features, pad_token_id=0, padding_side="left")

        self.assertEqual(batch["input_ids"].tolist(), [[0, 10, 11], [20, 21, 22]])
        self.assertEqual(batch["attention_mask"].tolist(), [[0, 1, 1], [1, 1, 1]])
        self.assertEqual(batch["mm_token_type_ids"].tolist(), [[-1, 2, 0], [2, 2, 0]])

    def test_training_and_generation_use_the_same_non_thinking_template(self):
        processor = _TemplateProcessor()
        record = QARecord(
            flat_index=0,
            video_index=0,
            qa_index=0,
            video_path="/tmp/a.mp4",
            video_name="a.mp4",
            question="Is <Sheldon> here?",
            answer="<Sheldon> is not present.",
            is_special=True,
            is_positive=False,
            sks_present="",
        )

        encode_record(processor, record, ["<Sheldon>"], stage=1, answer=record.answer)
        encode_record(processor, record, ["<Sheldon>"], stage=1, answer=None)

        self.assertEqual(len(processor.calls), 2)
        for _, kwargs in processor.calls:
            self.assertIs(kwargs["enable_thinking"], False)

    def test_stage_video_sampling_defaults(self):
        self.assertEqual(STAGE1_VIDEO_KWARGS, {"num_frames": 4, "fps": None})
        self.assertEqual(STAGE2_VIDEO_KWARGS["fps"], 2.0)
        self.assertIsNone(STAGE2_VIDEO_KWARGS["num_frames"])
        self.assertEqual(STAGE2_VIDEO_KWARGS["max_frames"], 768)

        stage1 = processor_video_kwargs(1)
        stage2 = processor_video_kwargs(2)
        expected_size = {"shortest_edge": 4 * 32 * 32, "longest_edge": 768 * 32 * 32}
        self.assertEqual(stage1["size"], expected_size)
        self.assertEqual(stage2["size"], expected_size)

    def test_visual_token_budget_can_be_overridden(self):
        overrides = video_overrides_from_token_budget(min_tokens=8, max_tokens=512)

        self.assertEqual(
            overrides,
            {"size": {"shortest_edge": 8 * 32 * 32, "longest_edge": 512 * 32 * 32}},
        )

    def test_loads_existing_pvchat_video_qa_schema(self):
        payload = {
            "videos": [
                {
                    "video_path": "/tmp/a.mp4",
                    "is_positive": True,
                    "sks_present": "<Sheldon>",
                    "qa_pairs": [
                        {"question": "Is Sheldon here?", "answer": "Yes.", "is_special": True},
                        {"question": "What happens?", "answer": "He talks.", "is_special": False},
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            records = load_qa_records(path)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[1].video_path, "/tmp/a.mp4")
        self.assertFalse(records[1].is_special)

    def test_masks_everything_before_assistant_answer(self):
        input_ids = torch.tensor([10, 11, 20, 21, 22, 30, 31])

        labels = mask_prompt_labels(
            input_ids,
            assistant_marker_ids=[20, 21, 22],
            pad_token_id=0,
        )

        self.assertEqual(labels.tolist(), [-100, -100, -100, -100, -100, 30, 31])


if __name__ == "__main__":
    unittest.main()
