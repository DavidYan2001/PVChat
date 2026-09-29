import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from qwen35_pvchat.data import QARecord
from qwen35_pvchat.evaluation import (
    build_result_qa_pair,
    generate_batch_answers,
    generation_runtime,
    run_metrics,
    strip_thinking,
)


class _GenerationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            use_cache=False,
            text_config=SimpleNamespace(use_cache=False),
        )
        self._gradient_checkpointing = True

    @property
    def is_gradient_checkpointing(self):
        return self._gradient_checkpointing

    def gradient_checkpointing_disable(self):
        self._gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        self._gradient_checkpointing = True


class _BatchGenerationModel(_GenerationModel):
    def __init__(self):
        super().__init__()
        self.generate_calls = 0
        self.last_input_ids = None

    def generate(self, **kwargs):
        self.generate_calls += 1
        self.last_input_ids = kwargs["input_ids"].clone()
        batch_size = kwargs["input_ids"].shape[0]
        answer_ids = torch.arange(101, 101 + batch_size).reshape(batch_size, 1)
        return torch.cat([kwargs["input_ids"], answer_ids], dim=1)


class _BatchTokenizer:
    pad_token_id = 0


class _BatchProcessor:
    tokenizer = _BatchTokenizer()

    def batch_decode(self, answer_ids, **kwargs):
        del kwargs
        return [f"answer-{int(row[0])}" for row in answer_ids]


class Qwen35EvaluationTest(unittest.TestCase):
    def test_batch_generation_calls_model_once_for_four_records(self):
        records = [
            QARecord(
                flat_index=index,
                video_index=0,
                qa_index=index,
                video_path="video.mp4",
                video_name="video.mp4",
                question=f"question-{index}",
                answer=f"gold-{index}",
                is_special=False,
                is_positive=True,
                sks_present="<Sheldon>",
            )
            for index in range(4)
        ]
        model = _BatchGenerationModel()
        processor = _BatchProcessor()

        def fake_encode_record(processor, record, personalized_tokens, **kwargs):
            del processor, personalized_tokens, kwargs
            length = record.qa_index + 2
            return {
                "input_ids": torch.arange(10, 10 + length),
                "attention_mask": torch.ones(length, dtype=torch.long),
                "mm_token_type_ids": torch.zeros(length, dtype=torch.long),
                "pixel_values_videos": torch.zeros(1, 3),
                "video_grid_thw": torch.tensor([[1, 1, 1]]),
                "record": record,
            }

        with (
            patch("qwen35_pvchat.evaluation.encode_record", side_effect=fake_encode_record),
            patch("qwen35_pvchat.evaluation.remoh_generation_masks", return_value=nullcontext()),
        ):
            answers = generate_batch_answers(
                model,
                processor,
                records,
                ["<Sheldon>"],
                stage=2,
                device="cpu",
            )

        self.assertEqual(answers, ["answer-101", "answer-102", "answer-103", "answer-104"])
        self.assertEqual(model.generate_calls, 1)
        self.assertEqual(model.last_input_ids.shape[0], 4)
        self.assertEqual(model.last_input_ids[0, :3].tolist(), [0, 0, 0])

    def test_result_qa_pair_keeps_explicit_positive_label(self):
        record = QARecord(
            flat_index=0,
            video_index=0,
            qa_index=0,
            video_path="negative.mp4",
            video_name="negative.mp4",
            question="Is <Sheldon> present?",
            answer="No, <Sheldon> is absent.",
            is_special=True,
            is_positive=False,
            sks_present="",
        )

        result = build_result_qa_pair(record, "<Sheldon> is not present.")

        self.assertIs(result["is_positive"], False)

    def test_generation_runtime_temporarily_enables_cache(self):
        model = _GenerationModel().train()

        with generation_runtime(model) as base_model:
            self.assertIs(base_model, model)
            self.assertFalse(model.training)
            self.assertFalse(model.is_gradient_checkpointing)
            self.assertTrue(model.config.use_cache)
            self.assertTrue(model.config.text_config.use_cache)

        self.assertTrue(model.training)
        self.assertTrue(model.is_gradient_checkpointing)
        self.assertFalse(model.config.use_cache)
        self.assertFalse(model.config.text_config.use_cache)

    def test_strip_thinking_keeps_only_final_answer(self):
        text = "<think>private reasoning</think>\n\n<Sheldon> is speaking."

        self.assertEqual(strip_thinking(text), "<Sheldon> is speaking.")

    def test_strip_thinking_accepts_plain_answer(self):
        self.assertEqual(strip_thinking("  Yes, <Sheldon> is here.  "), "Yes, <Sheldon> is here.")

    def test_strip_thinking_drops_unfinished_private_reasoning(self):
        self.assertEqual(strip_thinking("<think>unfinished private reasoning"), "")

    def test_automatic_metrics_require_complete_api_judging(self):
        with patch("qwen35_pvchat.evaluation.subprocess.run") as run:
            run_metrics(
                Path("results.json"),
                Path("metrics"),
                "<Sheldon>",
                bertscore_device="cuda:0",
            )

        command = run.call_args.args[0]
        self.assertIn("--require_complete_judge", command)
        self.assertEqual(command[command.index("--judge_backend") + 1], "local_qwen35")
        self.assertNotIn("--judge_model", command)
        self.assertNotIn("--judge_fallback_models", command)
        self.assertIn("Qwen3.5-35B-A3B", command[command.index("--local_judge_model_path") + 1])
        self.assertEqual(command[command.index("--local_judge_device") + 1], "cuda:0")
        self.assertEqual(command[command.index("--local_judge_batch_size") + 1], "64")
        self.assertEqual(
            command[command.index("--local_judge_es_items_per_prompt") + 1], "1"
        )
        self.assertEqual(
            command[command.index("--local_judge_dc_items_per_prompt") + 1], "10"
        )
        environment = run.call_args.kwargs["env"]
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")


if __name__ == "__main__":
    unittest.main()
