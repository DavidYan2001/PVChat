import itertools
import math
import unittest

from qwen35_pvchat.adaptive_policy import decide_dynamic_rollout
from qwen35_pvchat.identity_gated_dynamic_policy import (
    IDENTITY_CORRECT,
    IDENTITY_UNKNOWN,
    IDENTITY_WRONG,
    decide_identity_gated_rollout,
    identity_gate_value,
    identity_gated_advantages,
)
from qwen35_pvchat.identity_presence import (
    ABSENT,
    PRESENT,
    UNKNOWN,
    detect_identity_presence,
    detect_target_identity_presence,
)
from qwen35_pvchat.policy_objectives import normalize_group_advantages


class SharedIdentityEvidenceTest(unittest.TestCase):
    def test_legacy_metric_labels_are_preserved(self):
        person = "<Sheldon>"
        cases = {
            "<Sheldon> is in this video.": PRESENT,
            "Yes, <Sheldon> appears in the clip.": PRESENT,
            "<Sheldon> does not appear in the video.": ABSENT,
            "No, I cannot find <Sheldon> here.": ABSENT,
            "The footage does not include <Sheldon>.": ABSENT,
            "The lighting is dark.": UNKNOWN,
        }

        for answer, expected in cases.items():
            with self.subTest(answer=answer):
                self.assertEqual(detect_identity_presence(answer, person), expected)

    def test_training_evidence_requires_target_scope_outside_identity_questions(self):
        person = "<Sheldon>"

        self.assertEqual(
            detect_target_identity_presence("No.", person, is_identity_question=True),
            ABSENT,
        )
        self.assertEqual(
            detect_target_identity_presence("Yes.", person, is_identity_question=True),
            PRESENT,
        )
        self.assertEqual(
            detect_target_identity_presence("No hat is visible.", person, is_identity_question=False),
            UNKNOWN,
        )
        self.assertEqual(
            detect_target_identity_presence(
                "<Sheldon> wears no hat.", person, is_identity_question=False
            ),
            PRESENT,
        )
        self.assertEqual(
            detect_target_identity_presence(
                "<Sheldon> is not present.", person, is_identity_question=False
            ),
            ABSENT,
        )
        self.assertEqual(
            detect_target_identity_presence(
                "The person is not present.", person, is_identity_question=False
            ),
            UNKNOWN,
        )

    def test_standalone_bare_name_uses_word_boundaries(self):
        self.assertEqual(
            detect_target_identity_presence(
                "Sheldon is walking.", "<Sheldon>", is_identity_question=False
            ),
            PRESENT,
        )
        self.assertEqual(
            detect_target_identity_presence(
                "The shelldoned wall is visible.", "<Sheldon>", is_identity_question=False
            ),
            UNKNOWN,
        )


class IdentityGateValueTest(unittest.TestCase):
    def test_gate_value_is_relative_to_positive_or_negative_video_label(self):
        self.assertEqual(identity_gate_value(PRESENT, is_positive=True), IDENTITY_CORRECT)
        self.assertEqual(identity_gate_value(ABSENT, is_positive=True), IDENTITY_WRONG)
        self.assertEqual(identity_gate_value(ABSENT, is_positive=False), IDENTITY_CORRECT)
        self.assertEqual(identity_gate_value(PRESENT, is_positive=False), IDENTITY_WRONG)
        self.assertEqual(identity_gate_value(UNKNOWN, is_positive=True), IDENTITY_UNKNOWN)
        self.assertEqual(identity_gate_value(UNKNOWN, is_positive=False), IDENTITY_UNKNOWN)


class IdentityGatedRolloutTest(unittest.TestCase):
    def test_conflict_free_and_all_unknown_groups_exactly_match_dynamic_gspo(self):
        cases = (
            ([0.90, 0.90], [IDENTITY_CORRECT, IDENTITY_CORRECT]),
            ([0.50, 0.50], [IDENTITY_UNKNOWN, IDENTITY_UNKNOWN]),
            ([0.80, 0.65], [IDENTITY_CORRECT, IDENTITY_UNKNOWN]),
            ([0.50] * 4, [IDENTITY_UNKNOWN] * 4),
            ([0.90] * 8, [IDENTITY_UNKNOWN] * 8),
        )

        for rewards, states in cases:
            with self.subTest(rewards=rewards, states=states):
                expected = decide_dynamic_rollout(rewards)
                actual = decide_identity_gated_rollout(rewards, states)
                self.assertEqual(
                    (actual.target_count, actual.should_update, actual.reason),
                    (expected.target_count, expected.should_update, expected.reason),
                )
                self.assertFalse(actual.identity_triggered)

    def test_mixed_correct_and_wrong_stops_and_updates_immediately(self):
        decision = decide_identity_gated_rollout(
            [0.95, 0.10],
            [IDENTITY_WRONG, IDENTITY_CORRECT],
        )

        self.assertEqual(decision.target_count, 2)
        self.assertTrue(decision.should_update)
        self.assertFalse(decision.needs_more)
        self.assertTrue(decision.identity_triggered)
        self.assertEqual(decision.reason, "identity_mixed")

    def test_explicit_wrong_without_correct_expands_to_eight_then_skips(self):
        cases = (
            (2, 4, True),
            (4, 8, True),
            (8, 8, False),
        )

        for count, target, needs_more in cases:
            states = [IDENTITY_WRONG] + [IDENTITY_UNKNOWN] * (count - 1)
            decision = decide_identity_gated_rollout([0.5] * count, states)
            with self.subTest(count=count):
                self.assertEqual(decision.target_count, target)
                self.assertEqual(decision.needs_more, needs_more)
                self.assertFalse(decision.should_update)
                self.assertTrue(decision.identity_triggered)


class IdentityGatedAdvantageTest(unittest.TestCase):
    def test_conflict_free_advantages_are_bit_exact_dynamic_advantages(self):
        rewards = [0.15, 0.35, 0.80, 0.40]
        expected = normalize_group_advantages(rewards, alpha=1.0, eps=1e-6)

        for states in (
            [IDENTITY_CORRECT] * 4,
            [IDENTITY_UNKNOWN] * 4,
            [IDENTITY_CORRECT, IDENTITY_UNKNOWN] * 2,
        ):
            with self.subTest(states=states):
                self.assertEqual(
                    identity_gated_advantages(rewards, states, eps=1e-6),
                    expected,
                )

    def test_identity_tiers_cannot_be_reversed_by_soft_reward(self):
        reward_sets = (
            [1.0, -1.0, 0.5],
            [-1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        )
        state_permutations = set(
            itertools.permutations(
                (IDENTITY_CORRECT, IDENTITY_UNKNOWN, IDENTITY_WRONG)
            )
        )

        for rewards in reward_sets:
            for states in state_permutations:
                advantages = identity_gated_advantages(
                    rewards,
                    states,
                    identity_margin=1.0,
                    soft_clip=0.25,
                )
                by_state = dict(zip(states, advantages))
                with self.subTest(rewards=rewards, states=states):
                    self.assertGreater(by_state[IDENTITY_CORRECT], by_state[IDENTITY_UNKNOWN])
                    self.assertGreater(by_state[IDENTITY_UNKNOWN], by_state[IDENTITY_WRONG])
                    self.assertAlmostEqual(sum(advantages), 0.0, places=12)

    def test_soft_reward_keeps_weak_order_inside_one_identity_tier(self):
        rewards = [0.1, 0.9, 0.5, -0.2]
        states = [IDENTITY_CORRECT, IDENTITY_CORRECT, IDENTITY_WRONG, IDENTITY_WRONG]
        advantages = identity_gated_advantages(rewards, states)

        self.assertGreaterEqual(advantages[1], advantages[0])
        self.assertGreaterEqual(advantages[2], advantages[3])

    def test_margin_contract_rejects_equality_non_finite_and_negative_values(self):
        invalid = (
            (0.5, 0.25),
            (math.nan, 0.25),
            (math.inf, 0.25),
            (1.0, math.nan),
            (1.0, math.inf),
            (1.0, -0.1),
        )

        for margin, clip in invalid:
            with self.subTest(margin=margin, clip=clip), self.assertRaises(ValueError):
                identity_gated_advantages(
                    [0.0, 1.0],
                    [IDENTITY_CORRECT, IDENTITY_WRONG],
                    identity_margin=margin,
                    soft_clip=clip,
                )


if __name__ == "__main__":
    unittest.main()
