import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import generate_grpo_rollouts as grpo_rollouts

from generate_grpo_rollouts import (
    classify_qa_type,
    extract_rule_keywords,
    filter_state_dict_for_model,
    is_quota_error,
    limit_gpu_ids,
    parse_model_fallbacks,
    parse_yes_no,
    prepare_keyword_cache,
    read_existing_group_ids,
    score_candidate_answer,
    select_available_gpus_from_smi,
    shard_output_path,
    shard_rows,
)


class RewardRuleTest(unittest.TestCase):
    def test_parse_yes_no_handles_clear_answers(self):
        self.assertEqual(parse_yes_no("Yes, <Sheldon> is in the video."), "yes")
        self.assertEqual(parse_yes_no("No, <Sheldon> does not appear."), "no")
        self.assertIsNone(parse_yes_no("The answer is unclear from the clip."))

    def test_identity_positive_rewards_correct_yes(self):
        score = score_candidate_answer(
            question="Is <Sheldon> in the video?",
            gold_answer="Yes.",
            candidate_answer="Yes, <Sheldon> is in the video.",
            person_token="<Sheldon>",
            qa_type="identity",
            is_positive=True,
        )

        self.assertGreater(score["reward"], 0.8)
        self.assertEqual(score["components"]["yes_no_score"], 1.0)

    def test_identity_negative_penalizes_hallucinated_person(self):
        score = score_candidate_answer(
            question="Is <Sheldon> in the video?",
            gold_answer="No.",
            candidate_answer="Yes, <Sheldon> appears in the video.",
            person_token="<Sheldon>",
            qa_type="identity",
            is_positive=False,
        )

        self.assertLess(score["reward"], -0.5)
        self.assertEqual(score["components"]["hallucination_penalty"], 1.0)

    def test_open_text_candidate_matching_keywords_scores_higher(self):
        gold = "<Sheldon> is wearing a blue shirt and a dark jacket."
        keywords = extract_rule_keywords(gold, "<Sheldon>")

        good = score_candidate_answer(
            question="What is <Sheldon> wearing?",
            gold_answer=gold,
            candidate_answer="<Sheldon> is wearing a blue shirt with a dark jacket.",
            person_token="<Sheldon>",
            qa_type="clothing",
            keywords=keywords,
        )
        bad = score_candidate_answer(
            question="What is <Sheldon> wearing?",
            gold_answer=gold,
            candidate_answer="The person is wearing a red hoodie.",
            person_token="<Sheldon>",
            qa_type="clothing",
            keywords=keywords,
        )

        self.assertGreater(good["reward"], bad["reward"])
        self.assertGreater(good["components"]["keyword_score"], bad["components"]["keyword_score"])

    def test_classify_identity_from_is_special_or_yes_no(self):
        self.assertEqual(classify_qa_type("Where is <Sheldon>?", "In a room.", True), "identity")
        self.assertEqual(classify_qa_type("Is <Sheldon> present?", "No.", False), "identity")
        self.assertEqual(classify_qa_type("What is <Sheldon> wearing?", "A jacket.", False), "clothing")

    def test_parse_model_fallbacks_keeps_primary_and_removes_duplicates(self):
        models = parse_model_fallbacks(
            "qwen3.7-max",
            "qwen3.7-max-2026-05-17,qwen3.7-max,qwen3.6-max-preview",
        )

        self.assertEqual(
            models,
            ["qwen3.7-max", "qwen3.7-max-2026-05-17", "qwen3.6-max-preview"],
        )

    def test_quota_error_detection_matches_dashscope_403(self):
        error = RuntimeError("403 AllocationQuota.FreeTierOnly: quota exceeded")

        self.assertTrue(is_quota_error(error))

    def test_read_existing_group_ids_ignores_invalid_lines(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            path.write_text('{"group_id": "0:0"}\nnot json\n{"group_id": "1:2"}\n', encoding="utf-8")

            self.assertEqual(read_existing_group_ids(path), {"0:0", "1:2"})

    def test_filter_state_dict_drops_unexpected_checkpoint_keys(self):
        state = {
            "a.weight": 1,
            "lm.lm_head.weight": 2,
            "lm.base_model.model.lm_head.weight": 3,
        }

        filtered, dropped = filter_state_dict_for_model(
            state,
            {"a.weight", "lm.base_model.model.lm_head.weight"},
        )

        self.assertEqual(set(filtered), {"a.weight", "lm.base_model.model.lm_head.weight"})
        self.assertEqual(dropped, ["lm.lm_head.weight"])

    def test_select_available_gpus_uses_memory_ratio_threshold(self):
        smi_output = "0, 1000, 24000\n1, 8000, 24000\n2, 0, 24000\n"

        self.assertEqual(select_available_gpus_from_smi(smi_output, 0.20), [0, 2])

    def test_limit_gpu_ids_honors_maximum(self):
        self.assertEqual(limit_gpu_ids([0, 1, 2, 3], 2), [0, 1])
        self.assertEqual(limit_gpu_ids([0, 1], 0), [0, 1])

    def test_prepare_keyword_cache_does_not_load_video_model(self):
        args = SimpleNamespace(train_json="train.json", sks_name="<Sheldon>")
        data = {"videos": []}

        with (
            patch.object(grpo_rollouts, "load_json", return_value=data),
            patch.object(grpo_rollouts, "build_keyword_cache", return_value={}) as build_cache,
            patch.object(grpo_rollouts, "load_pvchat_model") as load_model,
        ):
            prepare_keyword_cache(args)

        build_cache.assert_called_once_with(data, "<Sheldon>", args)
        load_model.assert_not_called()

    def test_complete_keyword_cache_is_not_rewritten_by_rollout_shards(self):
        data = {"videos": [{"qa_pairs": [{"answer": "Known answer"}]}]}
        args = SimpleNamespace(
            keyword_cache="keywords.json",
            keyword_backend="rule",
        )

        with (
            patch.object(grpo_rollouts, "load_keyword_cache", return_value={"Known answer": []}),
            patch.object(grpo_rollouts, "write_json") as write_json,
            patch.object(grpo_rollouts.os.path, "exists", return_value=True),
        ):
            cache = grpo_rollouts.build_keyword_cache(data, "<Sheldon>", args)

        self.assertIn("Known answer", cache)
        write_json.assert_not_called()

    def test_shard_rows_keeps_every_row_once(self):
        rows = list(range(10))

        shards = [shard_rows(rows, shard_id, 3) for shard_id in range(3)]

        self.assertEqual(shards[0], [0, 3, 6, 9])
        self.assertEqual(shards[1], [1, 4, 7])
        self.assertEqual(shards[2], [2, 5, 8])
        self.assertEqual(sorted(item for shard in shards for item in shard), rows)

    def test_shard_output_path_inserts_shard_suffix(self):
        path = shard_output_path("/tmp/train_rollouts_epoch0.jsonl", 2)

        self.assertEqual(str(path), "/tmp/train_rollouts_epoch0.shard2.jsonl")

    def test_backfill_resolves_video_relative_to_source_rollout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source.jsonl"
            output_path = Path(tmpdir) / "updated.jsonl"
            source_path.write_text(
                json.dumps({
                    "video_path": "videos/example.mp4",
                    "question": "What is <Sheldon> doing?",
                    "candidates": [{"answer": "<Sheldon> is talking.", "old_logprob": None}],
                }) + "\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                backfill_source_jsonl=str(source_path),
                output_jsonl=str(output_path),
                record_start=0,
                record_end=0,
            )
            fake_training_module = ModuleType("finetune_internvideo_REOMH_one_person2_stage")
            fake_training_module.load_video = lambda *args, **kwargs: "video_tensor"

            with (
                patch.dict(sys.modules, {
                    "finetune_internvideo_REOMH_one_person2_stage": fake_training_module,
                }),
                patch.object(grpo_rollouts, "load_pvchat_model", return_value=("model", "tokenizer", "config")),
                patch.object(
                    grpo_rollouts,
                    "resolve_rollout_video_path",
                    return_value="/resolved/example.mp4",
                ) as resolve_path,
                patch.object(grpo_rollouts, "compute_answer_old_logprobs", return_value=[-1.25]),
            ):
                written_records = grpo_rollouts.backfill_old_logprobs(args)

            self.assertEqual(written_records, 1)
            resolve_path.assert_called_once_with("videos/example.mp4", source_path)
            updated = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["candidates"][0]["old_logprob"], -1.25)


if __name__ == "__main__":
    unittest.main()
