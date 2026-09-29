import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from evaluate_pvchat_metrics import (
    DashScopeJudge,
    LocalQwen35Judge,
    build_packed_judge_prompt,
    classify_qa_category,
    compute_accuracy,
    compute_corpus_bleu,
    detect_identity_presence,
    flatten_result_json,
    judge_records,
    judge_scores_complete,
    local_entity_specificity,
    load_judge_cache,
    parse_packed_scores,
    summarize_judge_usage,
)


class EvaluatePvchatMetricsTest(unittest.TestCase):
    def test_detect_identity_presence_handles_present_and_absent_answers(self):
        person = "<Sheldon>"

        self.assertEqual(detect_identity_presence("<Sheldon> is in this video.", person), "present")
        self.assertEqual(detect_identity_presence("Yes, <Sheldon> appears in the clip.", person), "present")
        self.assertEqual(detect_identity_presence("<Sheldon> does not appear in the video.", person), "absent")
        self.assertEqual(detect_identity_presence("No, I cannot find <Sheldon> here.", person), "absent")
        self.assertEqual(detect_identity_presence("This video does not show <Sheldon>.", person), "absent")
        self.assertEqual(detect_identity_presence("The footage does not include <Sheldon>.", person), "absent")
        self.assertEqual(detect_identity_presence("I cannot detect <Sheldon>.", person), "absent")
        self.assertEqual(detect_identity_presence("I can confirm <Sheldon> is not here.", person), "absent")
        self.assertEqual(detect_identity_presence("<Sheldon> does not exist in this video.", person), "absent")
        self.assertEqual(detect_identity_presence("The lighting is dark.", person), "unknown")

    def test_emotion_question_with_appear_is_not_classified_as_identity(self):
        category = classify_qa_category(
            "What visible emotion does <Sheldon> appear to be expressing in this video?",
            is_special=False,
        )

        self.assertEqual(category, "emotion")

    def test_behavior_and_position_match_training_reward_categories(self):
        self.assertEqual(
            classify_qa_category(
                "Can you describe <Sheldon>'s behavior in this sequence?",
                is_special=False,
            ),
            "action",
        )
        self.assertEqual(
            classify_qa_category(
                "How does <Sheldon>'s position change throughout the video?",
                is_special=False,
            ),
            "location",
        )

    def test_flatten_and_accuracy_only_use_identity_questions(self):
        data = {
            "model_name": "demo",
            "results": [
                {
                    "video_path": "a.mp4",
                    "qa_pairs": [
                        {
                            "question": "Can you find <Sheldon> in this video?",
                            "answer": "<Sheldon> is in this video.",
                            "generated_answer": "Yes, <Sheldon> appears.",
                            "is_special": True,
                        },
                        {
                            "question": "What is <Sheldon> wearing?",
                            "answer": "<Sheldon> is wearing a blue shirt.",
                            "generated_answer": "<Sheldon> is wearing a blue shirt.",
                            "is_special": False,
                        },
                    ],
                },
                {
                    "video_path": "b.mp4",
                    "qa_pairs": [
                        {
                            "question": "Can you find <Sheldon> in this video?",
                            "answer": "<Sheldon> does not appear.",
                            "generated_answer": "Yes, <Sheldon> is visible.",
                            "is_special": True,
                        }
                    ],
                },
            ],
        }

        records = flatten_result_json(data, "<Sheldon>")
        accuracy = compute_accuracy(records, "<Sheldon>")

        self.assertEqual(len(records), 3)
        self.assertEqual(accuracy["total"], 2)
        self.assertEqual(accuracy["correct"], 1)
        self.assertAlmostEqual(accuracy["accuracy"], 0.5)

    def test_accuracy_prefers_explicit_video_label_over_gold_answer_wording(self):
        data = {
            "model_name": "demo",
            "results": [
                {
                    "video_path": "negative.mp4",
                    "qa_pairs": [
                        {
                            "question": "Is <Sheldon> present?",
                            "answer": "The identity cannot be determined from this wording.",
                            "generated_answer": "No, <Sheldon> is not present.",
                            "is_special": True,
                            "is_positive": False,
                        }
                    ],
                }
            ],
        }

        records = flatten_result_json(data, "<Sheldon>")
        accuracy = compute_accuracy(records, "<Sheldon>")

        self.assertIs(records[0]["is_positive"], False)
        self.assertEqual(accuracy["gold_unknown"], 0)
        self.assertEqual(accuracy["correct"], 1)

    def test_bleu_scores_identical_text_higher_than_unrelated_text(self):
        reference = ["<Sheldon> is wearing a blue shirt and dark jacket."]
        identical = ["<Sheldon> is wearing a blue shirt and dark jacket."]
        unrelated = ["The room contains a wooden table."]

        identical_result = compute_corpus_bleu(identical, reference)
        unrelated_result = compute_corpus_bleu(unrelated, reference)
        identical_bleu = identical_result["score"]
        unrelated_bleu = unrelated_result["score"]

        self.assertGreater(identical_bleu, 0.99)
        self.assertLess(unrelated_bleu, identical_bleu)
        self.assertNotIn("score_0_100", identical_result)

    def test_bleu_defaults_to_unigram_precision(self):
        reference = ["red blue green yellow"]
        reordered = ["yellow green blue red"]

        result = compute_corpus_bleu(reordered, reference)

        self.assertAlmostEqual(result["score"], 1.0)
        self.assertEqual(result["max_order"], 1)
        self.assertEqual(result["name"], "BLEU-1")

    def test_empty_answer_still_reports_bleu1_metadata(self):
        result = compute_corpus_bleu([""], [""])

        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["max_order"], 1)
        self.assertEqual(result["name"], "BLEU-1")

    def test_local_entity_specificity_prefers_target_name_over_generic_answer(self):
        target_score = local_entity_specificity(
            "<Sheldon> is sitting at a table.",
            "<Sheldon>",
        )
        generic_score = local_entity_specificity(
            "The person is sitting at a table.",
            "<Sheldon>",
        )

        self.assertGreater(target_score["score"], generic_score["score"])
        self.assertEqual(target_score["score"], 5.0)

    def test_dashscope_scores_keep_zero_to_five_scale(self):
        parsed = DashScopeJudge.parse_scored_response('{"score": 4, "reason": "mostly complete"}')

        self.assertEqual(parsed["score"], 4.0)
        self.assertEqual(parsed["reason"], "mostly complete")

    def test_local_qwen35_judge_scores_prompts_in_one_batch(self):
        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 9
            pad_token = "<pad>"
            padding_side = "right"

            def apply_chat_template(self, messages, **kwargs):
                self.template_kwargs = kwargs
                return messages[-1]["content"]

            def __call__(self, texts, **kwargs):
                del kwargs
                return {"input_ids": torch.ones((len(texts), 2), dtype=torch.long)}

            def batch_decode(self, answer_ids, **kwargs):
                del kwargs
                return [
                    json.dumps({"score": int(row[0]), "reason": "local"})
                    for row in answer_ids
                ]

        class FakeModel:
            def __init__(self):
                self.calls = 0
                self.config = SimpleNamespace(text_config=SimpleNamespace())

            def eval(self):
                return self

            def generate(self, input_ids, **kwargs):
                del kwargs
                self.calls += 1
                scores = torch.arange(1, input_ids.shape[0] + 1).reshape(-1, 1)
                return torch.cat([input_ids, scores], dim=1)

        tokenizer = FakeTokenizer()
        model = FakeModel()
        judge = LocalQwen35Judge(
            "/tmp/fake-qwen35",
            device="cpu",
            batch_size=4,
            tokenizer=tokenizer,
            model=model,
            torch_module=torch,
        )

        scores = judge.score_prompts(["first", "second", "third"])

        self.assertEqual(model.calls, 1)
        self.assertEqual([item["score"] for item in scores], [1.0, 2.0, 3.0])
        self.assertTrue(all(item["source"] == "local_qwen35" for item in scores))
        self.assertFalse(tokenizer.template_kwargs["enable_thinking"])

    def test_packed_judge_prompt_and_parser_keep_all_items_in_order(self):
        records = [
            {
                "question": f"question-{index}",
                "gold_answer": f"gold-{index}",
                "generated_answer": f"answer-{index}",
            }
            for index in range(3)
        ]

        prompt = build_packed_judge_prompt(records, "ES", "<Sheldon>")
        scores = parse_packed_scores('{"scores": [5, 3.5, 0]}', 3)

        self.assertIn('"id":0', prompt)
        self.assertIn('"id":2', prompt)
        self.assertEqual(scores, [5.0, 3.5, 0.0])
        with self.assertRaises(ValueError):
            parse_packed_scores('{"scores": [5, 3]}', 3)

    def test_local_judge_records_can_pack_multiple_items_per_prompt(self):
        class FakePackedJudge:
            models = ["local-35b"]
            current_model = "local-35b"
            cache_model_key = "local-35b"
            batch_size = 2

            def __init__(self):
                self.group_calls = []

            def score_prompts(self, prompts):
                raise AssertionError(f"unexpected single-item scoring: {prompts}")

            def score_record_groups(self, record_groups, metric_name, person_token):
                self.group_calls.append([len(group) for group in record_groups])
                self.metric_name = metric_name
                self.person_token = person_token
                return [
                    [
                        {
                            "score": 4.0,
                            "reason": "packed",
                            "source": "local_qwen35_packed",
                            "model": self.current_model,
                        }
                        for _ in group
                    ]
                    for group in record_groups
                ]

        records = [
            {
                "record_id": str(index),
                "question": f"question-{index}",
                "gold_answer": f"gold-{index}",
                "generated_answer": f"answer-{index}",
            }
            for index in range(5)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                cache_path=str(Path(temp_dir) / "judge.jsonl"),
                max_judge_items=None,
                api_num_workers=1,
                local_judge_es_items_per_prompt=1,
                local_judge_dc_items_per_prompt=2,
            )
            judge = FakePackedJudge()
            scores = judge_records(records, "<Sheldon>", args, "DC", judge=judge)

        self.assertEqual(judge.group_calls, [[2, 2], [1]])
        self.assertEqual(judge.metric_name, "DC")
        self.assertEqual(judge.person_token, "<Sheldon>")
        self.assertEqual([item["score"] for item in scores], [4.0] * 5)

    def test_local_judge_records_batches_and_reuses_cache(self):
        class FakeBatchJudge:
            models = ["local-35b"]
            current_model = "local-35b"
            cache_model_key = "local-35b"
            batch_size = 2

            def __init__(self):
                self.batch_calls = []

            def score_prompts(self, prompts):
                self.batch_calls.append(len(prompts))
                return [
                    {
                        "score": 4.0,
                        "reason": "local",
                        "source": "local_qwen35",
                        "model": self.current_model,
                    }
                    for _ in prompts
                ]

        records = [
            {
                "record_id": str(index),
                "question": f"question-{index}",
                "gold_answer": f"gold-{index}",
                "generated_answer": f"answer-{index}",
            }
            for index in range(5)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                cache_path=str(Path(temp_dir) / "judge.jsonl"),
                max_judge_items=None,
                api_num_workers=1,
            )
            first_judge = FakeBatchJudge()
            first = judge_records(records, "<Sheldon>", args, "ES", judge=first_judge)
            second_judge = FakeBatchJudge()
            second = judge_records(records, "<Sheldon>", args, "ES", judge=second_judge)

        self.assertEqual(first_judge.batch_calls, [2, 2, 1])
        self.assertEqual(second_judge.batch_calls, [])
        self.assertEqual([item["score"] for item in first], [4.0] * 5)
        self.assertEqual([item["score"] for item in second], [4.0] * 5)

    def test_dashscope_quota_error_switches_model_and_retries_same_prompt(self):
        class FakeCompletions:
            def __init__(self):
                self.models = []

            def create(self, model, **kwargs):
                self.models.append(model)
                if model == "qwen3.7-max":
                    raise RuntimeError("403 AllocationQuota.FreeTierOnly: quota exceeded")
                message = SimpleNamespace(content='{"score": 5, "reason": "complete"}')
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        completions = FakeCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        judge = DashScopeJudge(
            model="qwen3.7-max",
            fallback_models="qwen3.7-max-preview,qwen3.7-max-2026-06-08",
            client=client,
            max_retries=1,
        )

        scored = judge.score_prompt("score this")

        self.assertEqual(completions.models, ["qwen3.7-max", "qwen3.7-max-preview"])
        self.assertEqual(scored["score"], 5.0)
        self.assertEqual(scored["model"], "qwen3.7-max-preview")
        self.assertEqual(judge.current_model, "qwen3.7-max-preview")

    def test_judge_usage_records_actual_models_and_completeness(self):
        records = [
            {
                "entity_specificity": 5.0,
                "entity_specificity_model": "qwen3.7-max",
                "descriptive_completeness": 4.0,
                "descriptive_completeness_model": "qwen3.7-max-preview",
            },
            {
                "entity_specificity": 4.0,
                "entity_specificity_model": "qwen3.7-max-preview",
                "descriptive_completeness": None,
                "descriptive_completeness_model": "qwen3.7-max-preview",
            },
        ]

        usage = summarize_judge_usage(records)

        self.assertEqual(usage["entity_specificity"]["qwen3.7-max"], 1)
        self.assertEqual(usage["entity_specificity"]["qwen3.7-max-preview"], 1)
        self.assertEqual(usage["descriptive_completeness"]["qwen3.7-max-preview"], 1)
        self.assertFalse(judge_scores_complete(records))

        records[1]["descriptive_completeness"] = 3.0
        self.assertTrue(judge_scores_complete(records))

    def test_judge_cache_ignores_failed_scores_so_they_can_be_retried(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "judge_cache.jsonl"
            items = [
                {"key": "failed", "score": None, "reason": "rate limited"},
                {"key": "valid", "score": 4.0, "reason": "complete"},
            ]
            cache_path.write_text(
                "".join(json.dumps(item) + "\n" for item in items),
                encoding="utf-8",
            )

            cache = load_judge_cache(cache_path)

        self.assertNotIn("failed", cache)
        self.assertEqual(cache["valid"]["score"], 4.0)


if __name__ == "__main__":
    unittest.main()


class LocalJudgeParseSalvageTest(unittest.TestCase):
    def test_valid_json_is_parsed_normally(self):
        scored = LocalQwen35Judge.parse_scored_response(
            '{"score": 4, "reason": "captures the key facts"}', "judge-model"
        )
        self.assertEqual(scored["score"], 4.0)
        self.assertEqual(scored["reason"], "captures the key facts")

    def test_truncated_reason_recovers_score(self):
        # 回归：max_new_tokens预算耗尽时reason字符串被截断成非法JSON，
        # 之前整条被判失败并连锁触发require_complete_judge退出2。
        scored = LocalQwen35Judge.parse_scored_response(
            '{"score": 3, "reason": "The answer correctly ident', "judge-model"
        )
        self.assertEqual(scored["score"], 3.0)
        self.assertIn("recovered", scored["reason"])

    def test_salvaged_score_is_clamped_to_range(self):
        scored = LocalQwen35Judge.parse_scored_response(
            '{"score": 9, "reason": "truncat', "judge-model"
        )
        self.assertEqual(scored["score"], 5.0)

    def test_text_without_score_still_fails(self):
        scored = LocalQwen35Judge.parse_scored_response("no json here", "judge-model")
        self.assertIsNone(scored["score"])
        self.assertIn("parse failed", scored["reason"])
