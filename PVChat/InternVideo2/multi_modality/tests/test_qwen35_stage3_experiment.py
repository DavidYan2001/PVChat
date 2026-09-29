import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import qwen35_pvchat.stage3_experiment as stage3_experiment
from qwen35_pvchat.cli import build_stage3_parser
from qwen35_pvchat.stage3_experiment import (
    _validate_experiment_args,
    aggregate_epoch_stats,
    active_loss_scale,
    anchor_weight_for_decision,
    build_gold_completion_sequence,
    build_checkpoint_metadata,
    build_epoch_slots,
    build_experiment_manifest,
    candidate_chunk_size,
    collect_input_fingerprints,
    compute_group_advantages,
    fixed_experiment_args,
    get_algorithm_spec,
    load_runtime_state,
    load_adaptive_state_artifact,
    merge_rollout_shards,
    metrics_completion_payload,
    parent_checkpoint_for_epoch,
    phase_marker_path,
    plan_epoch_phases,
    rollout_decision,
    sample_experiment_group,
    save_runtime_state,
    save_adaptive_state_artifact,
    validate_or_write_manifest,
    write_phase_marker,
)
from qwen35_pvchat.adaptive_policy import PersonalizedAdaptiveState
from qwen35_pvchat.remoh_losses import AdaptiveReMoHLoss


class AlgorithmRegistryTest(unittest.TestCase):
    def test_registry_preserves_existing_algorithms_and_adds_buffered_ca_dynamic(self):
        expected = {
            "token_grpo": ("token", 1.0, "fixed4", "legacy"),
            "gspo": ("sequence", 1.0, "fixed4", "legacy"),
            "dr_gspo": ("sequence", 0.0, "fixed4", "legacy"),
            "dynamic_gspo": ("sequence", 1.0, "dynamic", "legacy"),
            "pa_gspo": ("sequence", None, "dynamic", "pa"),
            "icd_gspo_v0": ("sequence", None, "fixed8", "icd"),
            "icd_gspo_v1": ("sequence", None, "vector_dynamic", "icd"),
            "ig_dynamic_gspo": ("sequence", 1.0, "identity_dynamic", "identity_gated"),
            "ca_dynamic_gspo": ("sequence", 1.0, "constraint_dynamic", "constraint_anchored"),
            "ca_dynamic_gspo_buffered": (
                "sequence",
                1.0,
                "constraint_dynamic",
                "constraint_anchored",
            ),
        }

        for name, values in expected.items():
            with self.subTest(name=name):
                spec = get_algorithm_spec(name)
                self.assertEqual(
                    (spec.loss_level, spec.advantage_alpha, spec.rollout_mode, spec.reward_mode),
                    values,
                )

        pa = get_algorithm_spec("pa_gspo")
        self.assertEqual(pa.component_alphas["identity"], 0.0)
        self.assertEqual(pa.component_alphas["action"], 0.5)
        with self.assertRaises(ValueError):
            get_algorithm_spec("unknown")
        self.assertTrue(get_algorithm_spec("ca_dynamic_gspo_buffered").is_buffered)
        self.assertTrue(get_algorithm_spec("ca_dynamic_gspo").is_buffered)

    def test_every_preexisting_algorithm_spec_is_byte_for_byte_unchanged(self):
        expected = {
            "token_grpo": ("token_grpo", "token", 1.0, "fixed4", "legacy", {}),
            "gspo": ("gspo", "sequence", 1.0, "fixed4", "legacy", {}),
            "dr_gspo": ("dr_gspo", "sequence", 0.0, "fixed4", "legacy", {}),
            "dynamic_gspo": ("dynamic_gspo", "sequence", 1.0, "dynamic", "legacy", {}),
            "pa_gspo": (
                "pa_gspo",
                "sequence",
                None,
                "dynamic",
                "pa",
                {
                    "identity": 0.0,
                    "action": 0.5,
                    "clothing": 0.5,
                    "location": 0.5,
                    "emotion": 0.5,
                    "open": 0.5,
                },
            ),
            "icd_gspo_v0": ("icd_gspo_v0", "sequence", None, "fixed8", "icd", {}),
            "icd_gspo_v1": (
                "icd_gspo_v1",
                "sequence",
                None,
                "vector_dynamic",
                "icd",
                {},
            ),
        }

        for name, snapshot in expected.items():
            spec = get_algorithm_spec(name)
            actual = (
                spec.name,
                spec.loss_level,
                spec.advantage_alpha,
                spec.rollout_mode,
                spec.reward_mode,
                dict(spec.component_alphas),
            )
            with self.subTest(name=name):
                self.assertEqual(actual, snapshot)


class ExactOnceSchedulingTest(unittest.TestCase):
    def test_3535_records_on_three_ranks_have_two_tail_dummies(self):
        slots = build_epoch_slots(3535, world_size=3, epoch_seed=42)

        self.assertEqual([len(rank_slots) for rank_slots in slots], [1179, 1179, 1179])
        self.assertEqual([sum(value is None for value in rank_slots) for rank_slots in slots], [0, 1, 1])
        real = [value for rank_slots in slots for value in rank_slots if value is not None]
        self.assertEqual(len(real), 3535)
        self.assertEqual(sorted(real), list(range(3535)))

    def test_smoke_limit_truncates_slots_not_the_shuffled_dataset(self):
        full = build_epoch_slots(10, world_size=3, epoch_seed=9)
        smoke = build_epoch_slots(10, world_size=3, epoch_seed=9, max_steps_per_epoch=2)

        self.assertEqual(smoke, [rank_slots[:2] for rank_slots in full])

    def test_rejects_a_dataset_smaller_than_the_ddp_world(self):
        with self.assertRaisesRegex(ValueError, "at least one real record per rank"):
            build_epoch_slots(2, world_size=3, epoch_seed=42)

    def test_active_loss_scaling_compensates_for_ddp_gradient_averaging(self):
        self.assertEqual(active_loss_scale(3, 3, True), 1.0)
        self.assertEqual(active_loss_scale(3, 2, True), 1.5)
        self.assertEqual(active_loss_scale(3, 1, True), 3.0)
        self.assertEqual(active_loss_scale(3, 2, False), 0.0)
        self.assertEqual(active_loss_scale(3, 0, False), 0.0)
        with self.assertRaises(ValueError):
            active_loss_scale(3, 0, True)

    def test_buffered_schedule_collects_a_window_before_updating(self):
        execute = getattr(stage3_experiment, "execute_rollout_schedule", None)
        self.assertIsNotNone(execute, "buffered rollout scheduler is missing")
        events = []

        execute(
            "ca_dynamic_gspo_buffered",
            [0, 1, 2, 3, 4],
            rollout_buffer_size=4,
            collect_slot=lambda slot: events.append(("collect", slot)) or slot,
            update_slot=lambda slot: events.append(("update", slot)),
        )

        self.assertEqual(
            events,
            [
                ("collect", 0),
                ("collect", 1),
                ("collect", 2),
                ("collect", 3),
                ("update", 0),
                ("update", 1),
                ("update", 2),
                ("update", 3),
                ("collect", 4),
                ("update", 4),
            ],
        )

    def test_formal_ca_schedule_collects_a_window_before_updating(self):
        events = []

        stage3_experiment.execute_rollout_schedule(
            "ca_dynamic_gspo",
            [0, 1, 2, 3],
            rollout_buffer_size=4,
            collect_slot=lambda slot: events.append(("collect", slot)) or slot,
            update_slot=lambda slot: events.append(("update", slot)),
        )

        self.assertEqual(
            events,
            [
                ("collect", 0),
                ("collect", 1),
                ("collect", 2),
                ("collect", 3),
                ("update", 0),
                ("update", 1),
                ("update", 2),
                ("update", 3),
            ],
        )

    def test_buffered_schedule_rejects_online_sized_buffer(self):
        execute = getattr(stage3_experiment, "execute_rollout_schedule", None)
        self.assertIsNotNone(execute, "buffered rollout scheduler is missing")
        with self.assertRaisesRegex(ValueError, "greater than one"):
            execute(
                "ca_dynamic_gspo_buffered",
                [0],
                rollout_buffer_size=1,
                collect_slot=lambda slot: slot,
                update_slot=lambda slot: None,
            )


class CandidatePlanningTest(unittest.TestCase):
    def test_fixed_variants_generate_four_candidates_once(self):
        self.assertEqual(candidate_chunk_size(get_algorithm_spec("token_grpo"), []), 4)
        self.assertEqual(candidate_chunk_size(get_algorithm_spec("gspo"), []), 4)
        self.assertEqual(candidate_chunk_size(get_algorithm_spec("token_grpo"), [0.1] * 4), 0)

    def test_dynamic_variants_expand_in_two_then_two_then_four_chunks(self):
        spec = get_algorithm_spec("dynamic_gspo")

        self.assertEqual(candidate_chunk_size(spec, []), 2)
        self.assertEqual(candidate_chunk_size(spec, [0.50, 0.55]), 2)
        self.assertEqual(candidate_chunk_size(spec, [0.40, 0.42, 0.43, 0.44]), 4)
        self.assertEqual(candidate_chunk_size(spec, [0.0] * 8), 0)
        self.assertEqual(candidate_chunk_size(spec, [0.1, 0.5]), 0)

    def test_fixed_groups_always_update_while_dynamic_groups_can_skip(self):
        fixed = rollout_decision(get_algorithm_spec("gspo"), [0.5] * 4)
        dynamic = rollout_decision(get_algorithm_spec("dynamic_gspo"), [0.9, 0.9])

        self.assertTrue(fixed.should_update)
        self.assertEqual(fixed.reason, "fixed4")
        self.assertFalse(dynamic.should_update)
        self.assertEqual(dynamic.reason, "easy_saturated")

    def test_icd_v0_uses_eight_and_v1_starts_with_four(self):
        self.assertEqual(candidate_chunk_size(get_algorithm_spec("icd_gspo_v0"), []), 8)
        self.assertEqual(candidate_chunk_size(get_algorithm_spec("icd_gspo_v0"), [0.0] * 8), 0)
        self.assertEqual(candidate_chunk_size(get_algorithm_spec("icd_gspo_v1"), []), 4)

    def test_ig_dynamic_uses_identity_overlay_only_for_explicit_conflicts(self):
        spec = get_algorithm_spec("ig_dynamic_gspo")
        unknown = [{"identity_gate": 0.0}, {"identity_gate": 0.0}]
        wrong_unknown = [{"identity_gate": -1.0}, {"identity_gate": 0.0}]
        mixed = [{"identity_gate": -1.0}, {"identity_gate": 1.0}]

        self.assertEqual(candidate_chunk_size(spec, []), 2)
        self.assertEqual(candidate_chunk_size(spec, [0.5, 0.5], unknown), 2)
        self.assertEqual(candidate_chunk_size(spec, [0.5, 0.5], wrong_unknown), 2)
        self.assertEqual(candidate_chunk_size(spec, [0.1, 0.9], mixed), 0)

        decision = rollout_decision(spec, [0.1, 0.9], component_rows=mixed)
        self.assertTrue(decision.should_update)
        self.assertEqual(decision.reason, "identity_mixed")

    def test_ca_dynamic_expands_low_quality_groups_and_falls_back_at_eight(self):
        spec = get_algorithm_spec("ca_dynamic_gspo")

        def rows(count):
            return [
                {"validity": 1.0, "identity_gate": 0.0, "content_score": 0.4}
                for _ in range(count)
            ]

        self.assertEqual(candidate_chunk_size(spec, []), 2)
        self.assertEqual(candidate_chunk_size(spec, [0.4] * 2, rows(2)), 2)
        self.assertEqual(candidate_chunk_size(spec, [0.4] * 4, rows(4)), 4)
        self.assertEqual(candidate_chunk_size(spec, [0.4] * 8, rows(8)), 0)

        decision = rollout_decision(spec, [0.4] * 8, component_rows=rows(8))
        self.assertTrue(decision.should_update)
        self.assertTrue(decision.use_sft_fallback)
        self.assertEqual(decision.reason, "sft_fallback")

    def test_ca_gold_sequence_and_anchor_weight_are_isolated_helpers(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                self.seen = (text, add_special_tokens)
                return [31, 32]

        tokenizer = Tokenizer()
        prompt = torch.tensor([10, 11, 12])
        sequence = build_gold_completion_sequence(prompt, tokenizer, "gold answer")
        normal = SimpleNamespace(should_update=True, use_sft_fallback=False)
        fallback = SimpleNamespace(should_update=True, use_sft_fallback=True)
        skipped = SimpleNamespace(should_update=False, use_sft_fallback=False)

        self.assertTrue(torch.equal(sequence, torch.tensor([10, 11, 12, 31, 32])))
        self.assertEqual(tokenizer.seen, ("gold answer<|im_end|>\n", False))
        self.assertEqual(anchor_weight_for_decision(normal, 0.05, 0.20), 0.05)
        self.assertEqual(anchor_weight_for_decision(fallback, 0.05, 0.20), 0.20)
        self.assertEqual(anchor_weight_for_decision(skipped, 0.05, 0.20), 0.0)


class TrainingMathAndStateTest(unittest.TestCase):
    def test_pa_advantages_are_component_decoupled_then_scaled_by_preupdate_multiplier(self):
        rows = [
            {"semantic": 1.0, "presence_consistency": 1.0},
            {"semantic": 0.0, "presence_consistency": -1.0},
        ]
        rewards = [0.8, -0.2]

        actual = compute_group_advantages(
            get_algorithm_spec("pa_gspo"),
            rewards,
            rows,
            qa_type="action",
            is_positive=True,
            category_multiplier=1.25,
            eps=1e-6,
        )

        unscaled = compute_group_advantages(
            get_algorithm_spec("pa_gspo"),
            rewards,
            rows,
            qa_type="action",
            is_positive=True,
            category_multiplier=1.0,
            eps=1e-6,
        )
        self.assertEqual(len(actual), 2)
        for value, baseline in zip(actual, unscaled):
            self.assertAlmostEqual(value, baseline * 1.25, places=12)

    def test_runtime_state_round_trip_restores_remoh_pa_and_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime_state.pt"
            remoh = AdaptiveReMoHLoss(initial_spr_weight=0.25)
            pa_state = PersonalizedAdaptiveState()
            pa_state.observe("<Sheldon>", "action", -1.0)
            save_runtime_state(path, remoh, pa_state)

            restored_remoh = AdaptiveReMoHLoss(initial_spr_weight=1e-8)
            restored = load_runtime_state(path, restored_remoh, use_pa=True)

            self.assertAlmostEqual(restored_remoh.spr_weight.item(), 0.25)
            self.assertEqual(restored["pa_state"].state_dict(), pa_state.state_dict())
            self.assertIn("python_rng_state", restored)
            self.assertIn("torch_rng_state", restored)

    def test_icd_group_advantage_uses_margin_gate(self):
        rows = [
            {"identity": -1.0, "semantic": 1.0, "specificity": 1.0, "coverage": 1.0},
            {"identity": 1.0, "semantic": 0.0, "specificity": 0.0, "coverage": 0.0},
        ]
        actual = compute_group_advantages(
            get_algorithm_spec("icd_gspo_v0"),
            rewards=[-1.0, 0.5],
            component_rows=rows,
            qa_type="action",
            is_positive=True,
            identity_margin=1.0,
            soft_clip=0.5,
        )
        self.assertGreater(actual[1], actual[0])

    def test_ig_group_advantage_preserves_legacy_without_conflict_and_gates_mixed(self):
        spec = get_algorithm_spec("ig_dynamic_gspo")
        rewards = [0.9, 0.1]
        fallback = compute_group_advantages(
            spec,
            rewards,
            [{"identity_gate": 1.0}, {"identity_gate": 0.0}],
            qa_type="identity",
            is_positive=True,
        )
        legacy = compute_group_advantages(
            get_algorithm_spec("dynamic_gspo"),
            rewards,
            [{}, {}],
            qa_type="identity",
            is_positive=True,
        )
        gated = compute_group_advantages(
            spec,
            rewards,
            [{"identity_gate": -1.0}, {"identity_gate": 1.0}],
            qa_type="identity",
            is_positive=True,
            ig_identity_margin=1.0,
            ig_soft_clip=0.25,
        )

        self.assertEqual(fallback, legacy)
        self.assertGreater(gated[1], gated[0])

    def test_adaptive_state_is_saved_as_a_separate_epoch_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "adaptive_state" / "epoch_2.pt"
            state = PersonalizedAdaptiveState()
            state.observe("<Sheldon>", "emotion", -0.5)

            save_adaptive_state_artifact(path, "pa_gspo", 2, state)
            restored = load_adaptive_state_artifact(path)

            self.assertEqual(restored["algorithm"], "pa_gspo")
            self.assertEqual(restored["epoch"], 2)
            self.assertEqual(restored["pa_state"].state_dict(), state.state_dict())


class RolloutAndStatsTest(unittest.TestCase):
    def test_sampled_icd_group_keeps_its_algorithm_for_epoch_stats(self):
        answers = ["<Sheldon> is running."] * 4 + ["<Sheldon> is not present."] * 4
        args = SimpleNamespace(
            identity_threshold=0.5,
            identity_margin=1.0,
            icd_soft_clip=0.5,
            icd_conflict_threshold=0.25,
            advantage_eps=1e-6,
        )
        record = SimpleNamespace(
            question="What is <Sheldon> doing?",
            answer="<Sheldon> is running.",
            is_special=False,
            is_positive=True,
        )
        fake_score = ({}, torch.zeros(8, 1), torch.ones(8, 1))

        with patch(
            "qwen35_pvchat.stage3_experiment.generation_runtime",
            return_value=nullcontext(object()),
        ), patch(
            "qwen35_pvchat.stage3_experiment._generate_candidate_chunk",
            return_value=([torch.tensor([1])] * 8, answers),
        ), patch(
            "qwen35_pvchat.stage3_experiment._build_score_batch_and_old_logprobs",
            return_value=fake_score,
        ):
            sampled = sample_experiment_group(
                model=object(),
                processor=object(),
                feature={},
                record=record,
                personalized_token="<Sheldon>",
                spec="icd_gspo_v0",
                pa_state=None,
                args=args,
                context=object(),
            )

        self.assertEqual(sampled.algorithm, "icd_gspo_v0")

    def test_ig_sampling_retains_identity_expansion_trace(self):
        args = SimpleNamespace(
            identity_threshold=0.5,
            identity_margin=1.0,
            icd_soft_clip=0.5,
            icd_conflict_threshold=0.25,
            ig_identity_margin=1.0,
            ig_soft_clip=0.25,
            advantage_eps=1e-6,
        )
        record = SimpleNamespace(
            question="Is <Sheldon> present in this video?",
            answer="Yes, <Sheldon> is present.",
            is_special=True,
            is_positive=True,
        )
        generated = [
            ([torch.tensor([1]), torch.tensor([2])], ["No.", "No."]),
            (
                [torch.tensor([3]), torch.tensor([4])],
                ["Yes, <Sheldon> is present.", "No."],
            ),
        ]
        fake_score = ({}, torch.zeros(4, 1), torch.ones(4, 1))

        with patch(
            "qwen35_pvchat.stage3_experiment.generation_runtime",
            return_value=nullcontext(object()),
        ), patch(
            "qwen35_pvchat.stage3_experiment._generate_candidate_chunk",
            side_effect=generated,
        ), patch(
            "qwen35_pvchat.stage3_experiment._build_score_batch_and_old_logprobs",
            return_value=fake_score,
        ):
            sampled = sample_experiment_group(
                model=object(),
                processor=object(),
                feature={},
                record=record,
                personalized_token="<Sheldon>",
                spec="ig_dynamic_gspo",
                pa_state=None,
                args=args,
                context=object(),
            )

        self.assertEqual(sampled.algorithm, "ig_dynamic_gspo")
        self.assertEqual(
            sampled.decision_trace,
            ("identity_expand_no_correct", "identity_mixed"),
        )
        self.assertTrue(sampled.identity_gated)
        self.assertTrue(sampled.identity_expanded)
        self.assertEqual(len(sampled.answers), 4)

    def test_parent_is_common_stage2_then_only_same_output_epoch(self):
        self.assertEqual(
            parent_checkpoint_for_epoch("/tmp/out", "/tmp/stage2", 1),
            Path("/tmp/stage2"),
        )
        self.assertEqual(
            parent_checkpoint_for_epoch("/tmp/out", "/tmp/stage2", 3),
            Path("/tmp/out/checkpoints/epoch_2"),
        )

    def test_rollout_merge_rejects_duplicates_and_requires_expected_real_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            shard_dir = output_dir / "rollouts" / "epoch_1"
            shard_dir.mkdir(parents=True)
            (shard_dir / "rank_0.jsonl").write_text(
                json.dumps({"group_id": "0:0", "flat_index": 0}) + "\n",
                encoding="utf-8",
            )
            (shard_dir / "rank_1.jsonl").write_text(
                json.dumps({"group_id": "1:0", "flat_index": 1}) + "\n",
                encoding="utf-8",
            )

            merged = merge_rollout_shards(
                output_dir,
                1,
                world_size=2,
                expected_count=2,
                expected_indices={0, 1},
            )
            self.assertEqual(len(merged.read_text(encoding="utf-8").splitlines()), 2)

            (shard_dir / "rank_1.jsonl").write_text(
                json.dumps({"group_id": "0:0", "flat_index": 1}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate group_id"):
                merge_rollout_shards(
                    output_dir,
                    1,
                    world_size=2,
                    expected_count=2,
                    expected_indices={0, 1},
                )

    def test_rollout_merge_rejects_wrong_flat_index_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            shard_dir = output_dir / "rollouts" / "epoch_1"
            shard_dir.mkdir(parents=True)
            (shard_dir / "rank_0.jsonl").write_text(
                json.dumps({"group_id": "0:0", "flat_index": 0}) + "\n",
                encoding="utf-8",
            )
            (shard_dir / "rank_1.jsonl").write_text(
                json.dumps({"group_id": "2:0", "flat_index": 2}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "flat_index coverage mismatch"):
                merge_rollout_shards(
                    output_dir,
                    1,
                    world_size=2,
                    expected_count=2,
                    expected_indices={0, 1},
                )

    def test_epoch_stats_include_exact_algorithm_fingerprint_and_efficiency_fields(self):
        stats = aggregate_epoch_stats(
            [
                {"real_groups": 2, "updates": 1, "skipped": 1, "zero_dispersion": 1,
                 "candidate_total": 6, "reward_sum": 2.5, "reward_count": 6},
                {"real_groups": 1, "updates": 1, "skipped": 0, "zero_dispersion": 0,
                 "candidate_total": 4, "reward_sum": 1.5, "reward_count": 4},
            ],
            get_algorithm_spec("dynamic_gspo"),
            epoch=2,
            input_fingerprint="f" * 64,
            elapsed_seconds=12.5,
        )

        self.assertEqual(stats["algorithm"], "dynamic_gspo")
        self.assertEqual(stats["input_fingerprint"], "f" * 64)
        self.assertEqual(stats["real_groups"], 3)
        self.assertEqual(stats["updates"], 2)
        self.assertEqual(stats["skipped"], 1)
        self.assertEqual(stats["candidate_total"], 10)
        self.assertAlmostEqual(stats["candidate_mean"], 10 / 3)
        self.assertAlmostEqual(stats["reward_mean"], 0.4)
        self.assertEqual(stats["elapsed_seconds"], 12.5)

    def test_icd_epoch_stats_report_identity_feasibility_and_v1_expansion_rate(self):
        stats = aggregate_epoch_stats(
            [
                {
                    "real_groups": 2,
                    "updates": 2,
                    "candidate_total": 12,
                    "reward_count": 12,
                    "identity_feasible_candidates": 9,
                    "identity_candidate_count": 12,
                    "expanded_groups": 1,
                }
            ],
            get_algorithm_spec("icd_gspo_v1"),
            epoch=1,
            input_fingerprint="a" * 64,
            elapsed_seconds=5.0,
        )

        self.assertAlmostEqual(stats["identity_feasibility_rate"], 0.75)
        self.assertAlmostEqual(stats["expansion_rate"], 0.5)

    def test_ig_epoch_stats_are_reported_only_for_ig_algorithm(self):
        shard = {
            "real_groups": 2,
            "updates": 1,
            "candidate_total": 6,
            "reward_count": 6,
            "ig_identity_correct_candidates": 2,
            "ig_identity_wrong_candidates": 1,
            "ig_identity_unknown_candidates": 3,
            "ig_gated_groups": 1,
            "ig_identity_expanded_groups": 1,
        }
        ig = aggregate_epoch_stats(
            [shard],
            get_algorithm_spec("ig_dynamic_gspo"),
            epoch=1,
            input_fingerprint="i" * 64,
            elapsed_seconds=1.0,
        )
        legacy = aggregate_epoch_stats(
            [shard],
            get_algorithm_spec("dynamic_gspo"),
            epoch=1,
            input_fingerprint="d" * 64,
            elapsed_seconds=1.0,
        )

        self.assertEqual(ig["identity_gate_counts"], {"correct": 2, "wrong": 1, "unknown": 3})
        self.assertAlmostEqual(ig["identity_gate_rate"], 0.5)
        self.assertAlmostEqual(ig["identity_expansion_rate"], 0.5)
        self.assertNotIn("identity_gate_counts", legacy)

    def test_ca_epoch_stats_report_fallback_and_degenerate_candidates(self):
        shard = {
            "real_groups": 3,
            "updates": 2,
            "candidate_total": 14,
            "reward_count": 14,
            "ca_fallback_groups": 1,
            "ca_anchored_groups": 2,
            "ca_degenerate_candidates": 2,
            "ca_identity_wrong_candidates": 1,
        }

        ca = aggregate_epoch_stats(
            [shard],
            get_algorithm_spec("ca_dynamic_gspo"),
            epoch=1,
            input_fingerprint="c" * 64,
            elapsed_seconds=1.0,
        )
        legacy = aggregate_epoch_stats(
            [shard],
            get_algorithm_spec("dynamic_gspo"),
            epoch=1,
            input_fingerprint="d" * 64,
            elapsed_seconds=1.0,
        )

        self.assertEqual(ca["fallback_groups"], 1)
        self.assertEqual(ca["anchored_groups"], 2)
        self.assertEqual(ca["degenerate_candidates"], 2)
        self.assertEqual(ca["identity_wrong_candidates"], 1)
        self.assertNotIn("fallback_groups", legacy)


class ManifestAndResumeTest(unittest.TestCase):
    def _make_inputs(self, root):
        stage2 = root / "stage2"
        stage2.mkdir()
        (stage2 / "pvchat_trainable.pt").write_bytes(b"weights-v1")
        (stage2 / "pvchat_config.json").write_text('{"stage": 2}', encoding="utf-8")
        train_json = root / "train.json"
        test_json = root / "test.json"
        train_json.write_text('{"videos": []}', encoding="utf-8")
        test_json.write_text('{"videos": [1]}', encoding="utf-8")
        return stage2, train_json, test_json

    def test_manifest_fingerprints_all_four_inputs_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage2, train_json, test_json = self._make_inputs(root)
            fingerprints = collect_input_fingerprints(stage2, train_json, test_json)
            args = Namespace(seed=42, num_epochs=3, temperature=0.7, output_dir=str(root / "out"))
            expected = build_experiment_manifest("gspo", args, stage2, fingerprints)
            expected["fixed_args"]["eval_batch_size"] = 32
            manifest_path = root / "out" / "experiment_manifest.json"

            self.assertEqual(
                set(fingerprints),
                {"stage2_trainable", "stage2_config", "train_json", "test_json"},
            )
            self.assertTrue(all(len(value["sha256"]) == 64 for value in fingerprints.values()))
            self.assertEqual(validate_or_write_manifest(manifest_path, expected), expected)
            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8")), expected)

            changed_eval_batch = json.loads(json.dumps(expected))
            changed_eval_batch["fixed_args"]["eval_batch_size"] = 48
            self.assertEqual(
                validate_or_write_manifest(manifest_path, changed_eval_batch),
                expected,
            )

            changed = json.loads(json.dumps(expected))
            changed["fixed_args"]["seed"] = 43
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                validate_or_write_manifest(manifest_path, changed)

    def test_test_only_manifest_refresh_keeps_training_and_invalidates_eval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage2, train_json, test_json = self._make_inputs(root)
            args = Namespace(seed=42, num_epochs=1, output_dir=str(root / "out"))
            original = build_experiment_manifest(
                "gspo",
                args,
                stage2,
                collect_input_fingerprints(stage2, train_json, test_json),
            )
            manifest_path = root / "out" / "experiment_manifest.json"
            validate_or_write_manifest(manifest_path, original)
            write_phase_marker(root / "out", 1, "training", {"checkpoint": "epoch_1"})
            write_phase_marker(root / "out", 1, "evaluation", {"result": "old.json"})
            write_phase_marker(root / "out", 1, "metrics", {"summary": "old.json"})

            test_json.write_text('{"videos": [1, 2]}', encoding="utf-8")
            refreshed = build_experiment_manifest(
                "gspo",
                args,
                stage2,
                collect_input_fingerprints(stage2, train_json, test_json),
            )

            self.assertEqual(validate_or_write_manifest(manifest_path, refreshed), refreshed)
            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8")), refreshed)
            self.assertTrue(phase_marker_path(root / "out", 1, "training").is_file())
            self.assertFalse(phase_marker_path(root / "out", 1, "evaluation").exists())
            self.assertFalse(phase_marker_path(root / "out", 1, "metrics").exists())

    def test_test_refresh_still_rejects_training_input_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage2, train_json, test_json = self._make_inputs(root)
            args = Namespace(seed=42, num_epochs=1, output_dir=str(root / "out"))
            manifest_path = root / "out" / "experiment_manifest.json"
            original = build_experiment_manifest(
                "gspo",
                args,
                stage2,
                collect_input_fingerprints(stage2, train_json, test_json),
            )
            validate_or_write_manifest(manifest_path, original)

            train_json.write_text('{"videos": ["changed"]}', encoding="utf-8")
            changed = build_experiment_manifest(
                "gspo",
                args,
                stage2,
                collect_input_fingerprints(stage2, train_json, test_json),
            )
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                validate_or_write_manifest(manifest_path, changed)

    def test_manifest_allows_runtime_bundle_relocation_with_identical_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage2, train_json, test_json = self._make_inputs(root)
            args = Namespace(
                seed=42,
                num_epochs=1,
                checkpoint_path=str(stage2),
                train_json=str(train_json),
                test_json=str(test_json),
                output_dir=str(root / "out"),
            )
            original = build_experiment_manifest(
                "gspo",
                args,
                stage2,
                collect_input_fingerprints(stage2, train_json, test_json),
            )
            manifest_path = root / "out" / "experiment_manifest.json"
            validate_or_write_manifest(manifest_path, original)

            relocated_stage2 = root / "relocated-stage2"
            relocated_stage2.mkdir()
            for name in ("pvchat_trainable.pt", "pvchat_config.json"):
                (relocated_stage2 / name).write_bytes((stage2 / name).read_bytes())
            relocated_train = root / "relocated-train.json"
            relocated_test = root / "relocated-test.json"
            relocated_train.write_bytes(train_json.read_bytes())
            relocated_test.write_bytes(test_json.read_bytes())
            relocated_args = Namespace(
                seed=42,
                num_epochs=1,
                checkpoint_path=str(relocated_stage2),
                train_json=str(relocated_train),
                test_json=str(relocated_test),
                output_dir=str(root / "out"),
            )
            relocated = build_experiment_manifest(
                "gspo",
                relocated_args,
                relocated_stage2,
                collect_input_fingerprints(
                    relocated_stage2,
                    relocated_train,
                    relocated_test,
                ),
            )

            self.assertEqual(validate_or_write_manifest(manifest_path, relocated), relocated)
            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8")), relocated)

    def test_phase_markers_are_independent_and_resume_only_missing_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            initial = plan_epoch_phases(output_dir, epoch=2)
            self.assertTrue(initial.run_training)
            self.assertTrue(initial.run_evaluation)
            self.assertTrue(initial.run_metrics)

            write_phase_marker(output_dir, 2, "training", {"checkpoint": "epoch_2"})
            after_training = plan_epoch_phases(output_dir, epoch=2)
            self.assertFalse(after_training.run_training)
            self.assertTrue(after_training.run_evaluation)
            self.assertTrue(after_training.run_metrics)

            write_phase_marker(output_dir, 2, "evaluation", {"result": "test_results.json"})
            after_evaluation = plan_epoch_phases(output_dir, epoch=2)
            self.assertFalse(after_evaluation.run_training)
            self.assertFalse(after_evaluation.run_evaluation)
            self.assertTrue(after_evaluation.run_metrics)

            write_phase_marker(output_dir, 2, "metrics", {"summary": "metrics_summary.json"})
            complete = plan_epoch_phases(output_dir, epoch=2)
            self.assertFalse(complete.run_training)
            self.assertFalse(complete.run_evaluation)
            self.assertFalse(complete.run_metrics)

    def test_manifest_ignores_phase_controls_so_metrics_can_resume_later(self):
        args = Namespace(
            model_path="/tmp/model",
            train_json="/tmp/train.json",
            skip_evaluation=True,
            skip_metrics=True,
            judge_backend="none",
            judge_model="judge-a",
            judge_fallback_models="judge-b",
            api_num_workers=1,
            no_resume=False,
            output_dir="/tmp/out",
            algorithm="gspo",
            ig_identity_margin=1.0,
            ig_soft_clip=0.25,
            ca_semantic_threshold=0.60,
            ca_informative_margin=0.10,
            ca_sft_weight=0.05,
            ca_fallback_sft_weight=0.20,
        )

        fixed = fixed_experiment_args(args)

        self.assertEqual(
            fixed,
            {"model_path": "/tmp/model", "train_json": "/tmp/train.json"},
        )

    def test_ig_manifest_parameters_are_isolated_from_existing_algorithms(self):
        args = Namespace(
            model_path="/tmp/model",
            train_json="/tmp/train.json",
            output_dir="/tmp/out",
            algorithm="ig_dynamic_gspo",
            ig_identity_margin=1.0,
            ig_soft_clip=0.25,
        )
        fingerprints = {"train_json": {"path": "/tmp/train.json", "sha256": "a" * 64}}

        legacy = build_experiment_manifest("dynamic_gspo", args, "/tmp/stage2", fingerprints)
        ig = build_experiment_manifest("ig_dynamic_gspo", args, "/tmp/stage2", fingerprints)

        self.assertNotIn("ig_identity_margin", legacy["fixed_args"])
        self.assertNotIn("ig_soft_clip", legacy["fixed_args"])
        self.assertEqual(ig["fixed_args"]["ig_identity_margin"], 1.0)
        self.assertEqual(ig["fixed_args"]["ig_soft_clip"], 0.25)

    def test_ca_manifest_parameters_are_isolated_from_existing_algorithms(self):
        args = Namespace(
            model_path="/tmp/model",
            train_json="/tmp/train.json",
            output_dir="/tmp/out",
            algorithm="ca_dynamic_gspo",
            ca_semantic_threshold=0.60,
            ca_informative_margin=0.10,
            ca_sft_weight=0.05,
            ca_fallback_sft_weight=0.20,
        )
        fingerprints = {"train_json": {"path": "/tmp/train.json", "sha256": "a" * 64}}

        legacy = build_experiment_manifest("dynamic_gspo", args, "/tmp/stage2", fingerprints)
        ca = build_experiment_manifest("ca_dynamic_gspo", args, "/tmp/stage2", fingerprints)

        self.assertNotIn("ca_semantic_threshold", legacy["fixed_args"])
        self.assertNotIn("ca_sft_weight", legacy["fixed_args"])
        self.assertEqual(ca["fixed_args"]["ca_semantic_threshold"], 0.60)
        self.assertEqual(ca["fixed_args"]["ca_informative_margin"], 0.10)
        self.assertEqual(ca["fixed_args"]["ca_sft_weight"], 0.05)
        self.assertEqual(ca["fixed_args"]["ca_fallback_sft_weight"], 0.20)

    def test_downstream_markers_are_stale_when_an_upstream_phase_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            write_phase_marker(output_dir, 1, "evaluation", {"result": "old.json"})
            write_phase_marker(output_dir, 1, "metrics", {"summary": "old.json"})

            missing_training = plan_epoch_phases(output_dir, epoch=1)

            self.assertTrue(missing_training.run_training)
            self.assertTrue(missing_training.run_evaluation)
            self.assertTrue(missing_training.run_metrics)

            write_phase_marker(output_dir, 1, "training", {"checkpoint": "epoch_1"})
            phase_marker = output_dir / "status" / "epoch_1.evaluation_complete.json"
            phase_marker.unlink()
            missing_evaluation = plan_epoch_phases(output_dir, epoch=1)
            self.assertFalse(missing_evaluation.run_training)
            self.assertTrue(missing_evaluation.run_evaluation)
            self.assertTrue(missing_evaluation.run_metrics)


class OutputMetadataAndCliTest(unittest.TestCase):
    def test_metrics_metadata_flattens_actual_judge_models_across_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary_path = Path(temporary) / "metrics_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "judge": {
                            "model_usage": {
                                "entity_specificity": {"judge-a": 3},
                                "descriptive_completeness": {"judge-a": 2},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            single = metrics_completion_payload(summary_path, get_algorithm_spec("gspo"))
            self.assertEqual(single["judge_models_used"], ["judge-a"])
            self.assertFalse(single["mixed_judges"])

            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            payload["judge"]["model_usage"]["descriptive_completeness"] = {"judge-b": 2}
            summary_path.write_text(json.dumps(payload), encoding="utf-8")
            mixed = metrics_completion_payload(summary_path, get_algorithm_spec("gspo"))
            self.assertEqual(mixed["judge_models_used"], ["judge-a", "judge-b"])
            self.assertTrue(mixed["mixed_judges"])

    def test_checkpoint_metadata_identifies_algorithm_parent_and_hyperparameters(self):
        spec = get_algorithm_spec("dr_gspo")
        fingerprints = {"train_json": {"path": "/tmp/train.json", "sha256": "a" * 64}}
        args = Namespace(
            seed=42,
            clip_range=0.2,
            drift_beta=0.02,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_new_tokens=96,
            num_samples=4,
            max_steps_per_epoch=0,
            eval_max_new_tokens=128,
            judge_model="qwen3.7-max",
        )

        metadata = build_checkpoint_metadata(
            base_metadata={"person_token": "<Sheldon>"},
            spec=spec,
            epoch=2,
            stage2_checkpoint="/tmp/stage2",
            parent_checkpoint="/tmp/out/checkpoints/epoch_1",
            args=args,
            fingerprints=fingerprints,
        )

        self.assertEqual(metadata["stage"], 3)
        self.assertEqual(metadata["algorithm"], "dr_gspo")
        self.assertEqual(metadata["epoch"], 2)
        self.assertEqual(metadata["common_stage2_checkpoint"], "/tmp/stage2")
        self.assertEqual(metadata["parent_checkpoint"], "/tmp/out/checkpoints/epoch_1")
        self.assertEqual(metadata["input_fingerprints"], fingerprints)
        self.assertEqual(metadata["hyperparameters"]["advantage_alpha"], 0.0)
        self.assertEqual(metadata["hyperparameters"]["clip_range"], 0.2)
        self.assertEqual(metadata["hyperparameters"]["eval_max_new_tokens"], 128)
        self.assertNotIn("judge_model", metadata["hyperparameters"])

    def test_stage3_cli_keeps_full_defaults_and_adds_experiment_resume_controls(self):
        parser = build_stage3_parser()
        args = parser.parse_args(
            [
                "--model_path", "/tmp/model",
                "--checkpoint_path", "/tmp/stage2",
                "--sks_name", "<Sheldon>",
                "--train_json", "/tmp/train.json",
                "--test_json", "/tmp/test.json",
                "--output_dir", "/tmp/output",
            ]
        )

        self.assertEqual(args.algorithm, "token_grpo")
        self.assertEqual(args.num_epochs, 3)
        self.assertEqual(args.max_steps_per_epoch, 0)
        self.assertFalse(args.no_resume)
        self.assertEqual(args.identity_margin, 1.0)
        self.assertEqual(args.icd_soft_clip, 0.5)
        self.assertEqual(args.icd_conflict_threshold, 0.25)
        self.assertEqual(args.ig_identity_margin, 1.0)
        self.assertEqual(args.ig_soft_clip, 0.25)
        self.assertEqual(args.ca_semantic_threshold, 0.60)
        self.assertEqual(args.ca_informative_margin, 0.10)
        self.assertEqual(args.ca_sft_weight, 0.05)
        self.assertEqual(args.ca_fallback_sft_weight, 0.20)

    def test_formal_launcher_exposes_checkpoint_recomputation_toggle(self):
        script = (
            Path(__file__).resolve().parents[1] / "run_qwen35_stage3_ca_dynamic_gspo.sh"
        ).read_text(encoding="utf-8")

        self.assertIn(
            'DISABLE_GRADIENT_CHECKPOINTING="${DISABLE_GRADIENT_CHECKPOINTING:-1}"',
            script,
        )
        self.assertIn('performance_args+=(--disable_gradient_checkpointing)', script)

    def test_experiment_runner_rejects_missing_stage2_checkpoint_argument(self):
        args = Namespace(
            checkpoint_path=None,
            num_epochs=3,
            num_samples=4,
            advantage_eps=1e-6,
        )
        with self.assertRaisesRegex(ValueError, "common Stage 2"):
            _validate_experiment_args(args, get_algorithm_spec("gspo"))
        self.assertEqual(args.advantage_eps, 1e-6)

    def test_icd_variants_validate_their_declared_initial_candidate_count(self):
        common = dict(checkpoint_path="/tmp/stage2", num_epochs=1, advantage_eps=1e-6)
        _validate_experiment_args(
            Namespace(**common, num_samples=8, identity_margin=1.0, icd_soft_clip=0.5),
            get_algorithm_spec("icd_gspo_v0"),
        )
        _validate_experiment_args(
            Namespace(**common, num_samples=4, identity_margin=1.0, icd_soft_clip=0.5),
            get_algorithm_spec("icd_gspo_v1"),
        )
        with self.assertRaisesRegex(ValueError, "num_samples"):
            _validate_experiment_args(
                Namespace(**common, num_samples=4, identity_margin=1.0, icd_soft_clip=0.5),
                get_algorithm_spec("icd_gspo_v0"),
            )

    def test_ig_variant_validates_strict_finite_margin_contract(self):
        common = dict(
            checkpoint_path="/tmp/stage2",
            num_epochs=1,
            num_samples=4,
            advantage_eps=1e-6,
        )
        _validate_experiment_args(
            Namespace(**common, ig_identity_margin=1.0, ig_soft_clip=0.25),
            get_algorithm_spec("ig_dynamic_gspo"),
        )
        for margin, clip in ((0.5, 0.25), (float("nan"), 0.25), (1.0, float("inf"))):
            with self.subTest(margin=margin, clip=clip), self.assertRaises(ValueError):
                _validate_experiment_args(
                    Namespace(**common, ig_identity_margin=margin, ig_soft_clip=clip),
                    get_algorithm_spec("ig_dynamic_gspo"),
                )

    def test_ca_variant_validates_thresholds_and_anchor_weights(self):
        common = dict(
            checkpoint_path="/tmp/stage2",
            num_epochs=1,
            num_samples=4,
            advantage_eps=1e-6,
        )
        valid = dict(
            ca_semantic_threshold=0.60,
            ca_informative_margin=0.10,
            ca_sft_weight=0.05,
            ca_fallback_sft_weight=0.20,
        )
        _validate_experiment_args(
            Namespace(**common, **valid),
            get_algorithm_spec("ca_dynamic_gspo"),
        )

        for key, value in (
            ("ca_semantic_threshold", 1.1),
            ("ca_informative_margin", -0.1),
            ("ca_sft_weight", -0.1),
            ("ca_fallback_sft_weight", 0.01),
        ):
            invalid = dict(valid)
            invalid[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                _validate_experiment_args(
                    Namespace(**common, **invalid),
                    get_algorithm_spec("ca_dynamic_gspo"),
                )


if __name__ == "__main__":
    unittest.main()
