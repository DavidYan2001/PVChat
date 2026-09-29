import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from qwen35_pvchat.stage3_experiment import (
    _generate_candidate_chunks_multi,
    execute_rollout_schedule,
    sample_experiment_group,
    sample_experiment_groups_batched,
)


def _ca_args():
    return SimpleNamespace(
        identity_threshold=0.5,
        identity_margin=1.0,
        icd_soft_clip=0.5,
        icd_conflict_threshold=0.25,
        ca_semantic_threshold=0.60,
        ca_informative_margin=0.10,
        ca_sft_weight=0.05,
        ca_fallback_sft_weight=0.20,
        advantage_eps=1e-6,
        max_new_tokens=8,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
    )


class ExecuteRolloutScheduleWindowingTest(unittest.TestCase):
    def test_buffered_windows_go_through_collect_slots(self):
        windows = []
        updates = []
        execute_rollout_schedule(
            "ca_dynamic_gspo_buffered",
            list(range(1, 11)),
            rollout_buffer_size=4,
            collect_slot=lambda slot: self.fail("per-slot collect must not run"),
            update_slot=updates.append,
            collect_slots=lambda window: windows.append(list(window)) or [("p", s) for s in window],
        )
        self.assertEqual(windows, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10]])
        self.assertEqual(updates, [("p", s) for s in range(1, 11)])

    def test_buffered_without_collect_slots_collects_per_slot(self):
        collected = []
        updates = []
        execute_rollout_schedule(
            "ca_dynamic_gspo_buffered",
            [1, 2, 3],
            rollout_buffer_size=2,
            collect_slot=lambda slot: collected.append(slot) or ("p", slot),
            update_slot=updates.append,
        )
        self.assertEqual(collected, [1, 2, 3])
        self.assertEqual(updates, [("p", 1), ("p", 2), ("p", 3)])

    def test_collect_slots_length_mismatch_is_an_error(self):
        with self.assertRaises(RuntimeError):
            execute_rollout_schedule(
                "ca_dynamic_gspo_buffered",
                [1, 2],
                rollout_buffer_size=2,
                collect_slot=lambda slot: ("p", slot),
                update_slot=lambda item: None,
                collect_slots=lambda window: [("p", window[0])],
            )

    def test_non_buffered_ignores_collect_slots(self):
        updates = []
        execute_rollout_schedule(
            "gspo",
            [1, 2],
            rollout_buffer_size=1,
            collect_slot=lambda slot: ("p", slot),
            update_slot=updates.append,
            collect_slots=lambda window: self.fail("non-buffered must not batch-collect"),
        )
        self.assertEqual(updates, [("p", 1), ("p", 2)])


class _ScriptedChunks:
    """Deterministic per-record chunk source shared by both generation fakes."""

    def __init__(self):
        self.progress = {}

    def chunk(self, record, size):
        start = self.progress.get(record.question, 0)
        self.progress[record.question] = start + size
        if record.question.startswith("GOOD"):
            answers = [record.answer] * size
        else:
            answers = [f"junk {start + offset}" for offset in range(size)]
        sequences = [torch.tensor([1000 + start + offset]) for offset in range(size)]
        return sequences, answers


def _fake_score_batch(base_model, processor, feature, sequences, context):
    count = len(sequences)
    return {"count": count}, torch.zeros(count, 2), torch.ones(count, 2, dtype=torch.bool)


class BatchedSequentialEquivalenceTest(unittest.TestCase):
    def test_batched_path_matches_sequential_path(self):
        args = _ca_args()
        processor = SimpleNamespace(
            tokenizer=SimpleNamespace(
                pad_token_id=0,
                encode=lambda text, add_special_tokens=False: [7, 8],
            )
        )
        records = [
            SimpleNamespace(
                question="GOOD what is <Sheldon> doing?",
                answer="<Sheldon> is running in the park.",
                is_special=False,
                is_positive=True,
            ),
            SimpleNamespace(
                question="BAD what is <Sheldon> wearing?",
                answer="<Sheldon> wears a red hooded jacket.",
                is_special=False,
                is_positive=True,
            ),
        ]
        features = [{"input_ids": torch.tensor([1, 2, 3]), "record": record} for record in records]

        sequential_chunks = _ScriptedChunks()

        def fake_chunk(base_model, processor_arg, feature, chunk_size, args_arg, context_arg):
            return sequential_chunks.chunk(feature["record"], chunk_size)

        with patch(
            "qwen35_pvchat.stage3_experiment.generation_runtime",
            return_value=nullcontext(object()),
        ), patch(
            "qwen35_pvchat.stage3_experiment._generate_candidate_chunk",
            side_effect=fake_chunk,
        ), patch(
            "qwen35_pvchat.stage3_experiment._build_score_batch_and_old_logprobs",
            side_effect=_fake_score_batch,
        ):
            sequential = [
                sample_experiment_group(
                    model=object(),
                    processor=processor,
                    feature=feature,
                    record=record,
                    personalized_token="<Sheldon>",
                    spec="ca_dynamic_gspo_buffered",
                    pa_state=None,
                    args=args,
                    context=object(),
                )
                for feature, record in zip(features, records)
            ]

        batched_chunks = _ScriptedChunks()

        def fake_multi(base_model, processor_arg, requests, args_arg, context_arg):
            return {
                owner: batched_chunks.chunk(feature["record"], chunk_size)
                for owner, feature, chunk_size in requests
            }

        with patch(
            "qwen35_pvchat.stage3_experiment.generation_runtime",
            return_value=nullcontext(object()),
        ), patch(
            "qwen35_pvchat.stage3_experiment._generate_candidate_chunks_multi",
            side_effect=fake_multi,
        ), patch(
            "qwen35_pvchat.stage3_experiment._build_score_batch_and_old_logprobs",
            side_effect=_fake_score_batch,
        ):
            batched = sample_experiment_groups_batched(
                model=object(),
                processor=processor,
                items=list(zip(features, records)),
                spec="ca_dynamic_gspo_buffered",
                personalized_token="<Sheldon>",
                pa_state=None,
                args=args,
                context=object(),
            )

        self.assertEqual(len(sequential), len(batched))
        expanded_groups = 0
        for expected, actual in zip(sequential, batched):
            self.assertEqual(expected.answers, actual.answers)
            self.assertEqual(
                [item.reward for item in expected.scored],
                [item.reward for item in actual.scored],
            )
            self.assertEqual(expected.advantages, actual.advantages)
            self.assertEqual(expected.decision.reason, actual.decision.reason)
            self.assertEqual(expected.decision.should_update, actual.decision.should_update)
            self.assertEqual(expected.candidate_count, actual.candidate_count)
            self.assertEqual(expected.anchor_weight, actual.anchor_weight)
            self.assertEqual(expected.decision_trace, actual.decision_trace)
            self.assertEqual(expected.qa_type, actual.qa_type)
            self.assertEqual(expected.identity_gated, actual.identity_gated)
            self.assertEqual(expected.identity_expanded, actual.identity_expanded)
            if actual.candidate_count > 2:
                expanded_groups += 1
        # The junk-answer record must have expanded past the initial CA chunk,
        # so the equivalence above covers a multi-round wavefront, not just the
        # first chunk.
        self.assertGreaterEqual(expanded_groups, 1)


class GenerateCandidateChunksMultiTest(unittest.TestCase):
    def test_left_padded_rows_are_rebuilt_per_owner(self):
        captured = {}

        class FakeModel:
            def generate(self, **kwargs):
                captured.update(kwargs)
                input_ids = kwargs["input_ids"]
                completions = torch.tensor(
                    [
                        [101, 9, 0],
                        [102, 103, 9],
                        [104, 9, 0],
                    ]
                )
                return torch.cat([input_ids, completions], dim=1)

        def batch_decode(ids, **kwargs):
            return ["ans:" + ",".join(str(int(v)) for v in row) for row in ids]

        processor = SimpleNamespace(
            tokenizer=SimpleNamespace(
                pad_token_id=0,
                eos_token_id=9,
                convert_tokens_to_ids=lambda token: 9,
            ),
            batch_decode=batch_decode,
        )

        def feature(input_ids):
            length = len(input_ids)
            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.ones(length, dtype=torch.long),
                "mm_token_type_ids": torch.zeros(length, dtype=torch.long),
                "pixel_values_videos": torch.zeros(2, 4),
                "video_grid_thw": torch.tensor([[1, 2, 2]]),
                "record": SimpleNamespace(),
            }

        short = feature([1, 2, 3])
        long = feature([4, 5, 6, 7, 8])
        requests = [(0, short, 2), (1, long, 1)]
        args = _ca_args()
        context = SimpleNamespace(device=torch.device("cpu"))

        @contextmanager
        def no_masks(model, video_mask, text_mask):
            yield

        with patch("qwen35_pvchat.stage3_experiment.remoh_generation_masks", no_masks):
            results = _generate_candidate_chunks_multi(FakeModel(), processor, requests, args, context)

        # Rows: short, short, long — the short prompts must be left-padded.
        self.assertEqual(tuple(captured["input_ids"].shape), (3, 5))
        self.assertEqual(captured["input_ids"][0].tolist(), [0, 0, 1, 2, 3])
        self.assertEqual(captured["attention_mask"][0].tolist(), [0, 0, 1, 1, 1])
        self.assertEqual(captured["mm_token_type_ids"][0].tolist(), [-1, -1, 0, 0, 0])
        self.assertEqual(captured["input_ids"][2].tolist(), [4, 5, 6, 7, 8])

        short_sequences, short_answers = results[0]
        long_sequences, long_answers = results[1]
        # Reconstructed sequences carry the true prompt (no padding) and stop
        # at the first stop token; trailing pads are dropped.
        self.assertEqual(short_sequences[0].tolist(), [1, 2, 3, 101, 9])
        self.assertEqual(short_sequences[1].tolist(), [1, 2, 3, 102, 103, 9])
        self.assertEqual(long_sequences[0].tolist(), [4, 5, 6, 7, 8, 104, 9])
        self.assertEqual(short_answers, ["ans:101,9", "ans:102,103,9"])
        self.assertEqual(long_answers, ["ans:104,9"])


class GreedyAnchorCandidateTest(unittest.TestCase):
    """--greedy_anchor_candidate: 每组第一个候选来自greedy解码, 两条路径一致。"""

    def _run_both_paths(self, args):
        processor = SimpleNamespace(
            tokenizer=SimpleNamespace(
                pad_token_id=0,
                encode=lambda text, add_special_tokens=False: [7, 8],
            )
        )
        records = [
            SimpleNamespace(
                question="GOOD what is <Sheldon> doing?",
                answer="<Sheldon> is running in the park.",
                is_special=False,
                is_positive=True,
            ),
            SimpleNamespace(
                question="BAD what is <Sheldon> wearing?",
                answer="<Sheldon> wears a red hooded jacket.",
                is_special=False,
                is_positive=True,
            ),
        ]
        features = [{"input_ids": torch.tensor([1, 2, 3]), "record": record} for record in records]

        class _AnchorChunks(_ScriptedChunks):
            def chunk(self, record, size, greedy=False):
                sequences, answers = super().chunk(record, size)
                if greedy:
                    answers = [f"greedy:{answer}" for answer in answers]
                return sequences, answers

        sequential_chunks = _AnchorChunks()

        def fake_chunk(base_model, processor_arg, feature, chunk_size, args_arg, context_arg, greedy=False):
            return sequential_chunks.chunk(feature["record"], chunk_size, greedy=greedy)

        with patch(
            "qwen35_pvchat.stage3_experiment.generation_runtime",
            return_value=nullcontext(object()),
        ), patch(
            "qwen35_pvchat.stage3_experiment._generate_candidate_chunk",
            side_effect=fake_chunk,
        ), patch(
            "qwen35_pvchat.stage3_experiment._build_score_batch_and_old_logprobs",
            side_effect=_fake_score_batch,
        ):
            sequential = [
                sample_experiment_group(
                    model=object(),
                    processor=processor,
                    feature=feature,
                    record=record,
                    personalized_token="<Sheldon>",
                    spec="ca_dynamic_gspo_buffered",
                    pa_state=None,
                    args=args,
                    context=object(),
                )
                for feature, record in zip(features, records)
            ]

        batched_chunks = _AnchorChunks()

        def fake_multi(base_model, processor_arg, requests, args_arg, context_arg, greedy=False):
            return {
                owner: batched_chunks.chunk(feature["record"], chunk_size, greedy=greedy)
                for owner, feature, chunk_size in requests
            }

        with patch(
            "qwen35_pvchat.stage3_experiment.generation_runtime",
            return_value=nullcontext(object()),
        ), patch(
            "qwen35_pvchat.stage3_experiment._generate_candidate_chunks_multi",
            side_effect=fake_multi,
        ), patch(
            "qwen35_pvchat.stage3_experiment._build_score_batch_and_old_logprobs",
            side_effect=_fake_score_batch,
        ):
            batched = sample_experiment_groups_batched(
                model=object(),
                processor=processor,
                items=list(zip(features, records)),
                spec="ca_dynamic_gspo_buffered",
                personalized_token="<Sheldon>",
                pa_state=None,
                args=args,
                context=object(),
            )
        return sequential, batched

    def test_anchor_first_candidate_and_path_equivalence(self):
        args = _ca_args()
        args.greedy_anchor_candidate = True
        sequential, batched = self._run_both_paths(args)
        for expected, actual in zip(sequential, batched):
            self.assertEqual(expected.answers, actual.answers)
            self.assertEqual(expected.advantages, actual.advantages)
            # 每组恰好一个greedy锚, 且是第一个候选
            greedy_flags = [answer.startswith("greedy:") for answer in actual.answers]
            self.assertTrue(greedy_flags[0])
            self.assertEqual(sum(greedy_flags), 1)

    def test_flag_off_produces_no_greedy_candidates(self):
        args = _ca_args()
        args.greedy_anchor_candidate = False
        sequential, batched = self._run_both_paths(args)
        for group in list(sequential) + list(batched):
            self.assertFalse(any(answer.startswith("greedy:") for answer in group.answers))


class NegativeGroupWeightTest(unittest.TestCase):
    def _advantages_with_weight(self, weight, gold_answer, is_positive):
        args = _ca_args()
        args.negative_group_weight = weight
        processor = SimpleNamespace(
            tokenizer=SimpleNamespace(
                pad_token_id=0,
                encode=lambda text, add_special_tokens=False: [7, 8],
            )
        )
        record = SimpleNamespace(
            question="BAD what is <Sheldon> wearing?",
            answer=gold_answer,
            is_special=False,
            is_positive=is_positive,
        )
        feature = {"input_ids": torch.tensor([1, 2, 3]), "record": record}
        chunks = _ScriptedChunks()

        def fake_chunk(base_model, processor_arg, feature_arg, chunk_size, args_arg, context_arg, greedy=False):
            return chunks.chunk(feature_arg["record"], chunk_size)

        with patch(
            "qwen35_pvchat.stage3_experiment.generation_runtime",
            return_value=nullcontext(object()),
        ), patch(
            "qwen35_pvchat.stage3_experiment._generate_candidate_chunk",
            side_effect=fake_chunk,
        ), patch(
            "qwen35_pvchat.stage3_experiment._build_score_batch_and_old_logprobs",
            side_effect=_fake_score_batch,
        ):
            group = sample_experiment_group(
                model=object(),
                processor=processor,
                feature=feature,
                record=record,
                personalized_token="<Sheldon>",
                spec="ca_dynamic_gspo_buffered",
                pa_state=None,
                args=args,
                context=object(),
            )
        return group.advantages

    def test_negative_gold_group_advantages_are_scaled(self):
        gold = "<Sheldon> is not present in this video."
        base = self._advantages_with_weight(1.0, gold, is_positive=False)
        scaled = self._advantages_with_weight(2.5, gold, is_positive=False)
        self.assertEqual(len(base), len(scaled))
        for a, b in zip(base, scaled):
            self.assertAlmostEqual(b, a * 2.5, places=6)

    def test_positive_gold_group_is_untouched(self):
        gold = "<Sheldon> wears a red hooded jacket."
        base = self._advantages_with_weight(1.0, gold, is_positive=True)
        scaled = self._advantages_with_weight(2.5, gold, is_positive=True)
        self.assertEqual(base, scaled)


if __name__ == "__main__":
    unittest.main()
