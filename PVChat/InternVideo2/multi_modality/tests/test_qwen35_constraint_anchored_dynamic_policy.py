import unittest

from qwen35_pvchat.constraint_anchored_dynamic_policy import (
    IDENTITY_CORRECT,
    IDENTITY_UNKNOWN,
    IDENTITY_WRONG,
    VALIDITY_INVALID,
    VALIDITY_VALID,
    constraint_anchored_advantages,
    decide_constraint_anchored_rollout,
    score_constraint_anchored_answer,
)


class ConstraintAnchoredRewardTest(unittest.TestCase):
    def test_location_fact_reward_prefers_left_over_a_right_contradiction(self):
        common = {
            "question": "Where is <Sheldon> positioned?",
            "gold_answer": "<Sheldon> is sitting to the left of another person.",
            "person_token": "<Sheldon>",
            "is_positive": True,
            "qa_type": "location",
        }

        correct = score_constraint_anchored_answer(
            candidate_answer="<Sheldon> is seated on the left beside another person.",
            **common,
        )
        contradictory = score_constraint_anchored_answer(
            candidate_answer="<Sheldon> is seated on the right beside another person.",
            **common,
        )

        self.assertGreater(correct.components["fact_coverage"], contradictory.components["fact_coverage"])
        self.assertGreater(correct.components["content_score"], contradictory.components["content_score"])
        self.assertGreater(correct.reward, contradictory.reward)

    def test_special_personal_token_loop_is_invalid_and_strongly_penalized(self):
        clean = score_constraint_anchored_answer(
            "What is <Sheldon> doing?",
            "<Sheldon> is sitting on a couch.",
            "<Sheldon> is sitting on a couch.",
            "<Sheldon>",
            True,
            "action",
        )
        leaked = score_constraint_anchored_answer(
            "What is <Sheldon> doing?",
            "<Sheldon> is sitting on a couch.",
            "<Sheldon> is sitting <sks_token2><sks_token2><sks_token2><sks_token2><sks_token2>.",
            "<Sheldon>",
            True,
            "action",
        )

        self.assertEqual(clean.components["validity"], VALIDITY_VALID)
        self.assertEqual(leaked.components["validity"], VALIDITY_INVALID)
        self.assertEqual(leaked.components["degeneration"], 1.0)
        self.assertLess(leaked.reward, clean.reward)

    def test_negative_identity_answer_distinguishes_absent_from_present(self):
        common = (
            "Does this clip contain <Sheldon>?",
            "<Sheldon> is not in this video.",
        )
        absent = score_constraint_anchored_answer(
            *common,
            "<Sheldon> is not present in this video.",
            "<Sheldon>",
            False,
            "identity",
        )
        present = score_constraint_anchored_answer(
            *common,
            "<Sheldon> is present in this video.",
            "<Sheldon>",
            False,
            "identity",
        )

        self.assertEqual(absent.components["identity_gate"], IDENTITY_CORRECT)
        self.assertEqual(present.components["identity_gate"], IDENTITY_WRONG)
        self.assertGreater(absent.reward, present.reward)

    def test_name_only_answer_cannot_outscore_matching_clothing_facts(self):
        common = {
            "question": "What is <Sheldon> wearing?",
            "gold_answer": "<Sheldon> is wearing a green shirt over an orange long-sleeved top.",
            "person_token": "<Sheldon>",
            "is_positive": True,
            "qa_type": "clothing",
        }
        detailed = score_constraint_anchored_answer(
            candidate_answer="<Sheldon> wears a green shirt over an orange long sleeve top.",
            **common,
        )
        name_only = score_constraint_anchored_answer(
            candidate_answer="<Sheldon> is visible in the video.",
            **common,
        )

        self.assertGreater(detailed.components["semantic"], name_only.components["semantic"])
        self.assertGreater(detailed.components["fact_coverage"], name_only.components["fact_coverage"])
        self.assertGreater(detailed.reward, name_only.reward)


class ConstraintAnchoredAdvantageTest(unittest.TestCase):
    def test_hierarchy_beats_content_but_preserves_content_order_inside_a_tier(self):
        rows = [
            {"validity": VALIDITY_INVALID, "identity_gate": IDENTITY_CORRECT, "content_score": 1.0},
            {"validity": VALIDITY_VALID, "identity_gate": IDENTITY_WRONG, "content_score": 1.0},
            {"validity": VALIDITY_VALID, "identity_gate": IDENTITY_CORRECT, "content_score": 0.2},
            {"validity": VALIDITY_VALID, "identity_gate": IDENTITY_CORRECT, "content_score": 0.9},
        ]

        advantages = constraint_anchored_advantages(rows)

        self.assertLess(advantages[0], advantages[1])
        self.assertLess(advantages[1], advantages[2])
        self.assertLess(advantages[2], advantages[3])
        self.assertAlmostEqual(sum(advantages), 0.0, places=12)

    def test_flat_rows_return_zero_advantages(self):
        rows = [
            {"validity": VALIDITY_VALID, "identity_gate": IDENTITY_CORRECT, "content_score": 0.8},
            {"validity": VALIDITY_VALID, "identity_gate": IDENTITY_CORRECT, "content_score": 0.8},
        ]

        self.assertEqual(constraint_anchored_advantages(rows), [0.0, 0.0])


class ConstraintAnchoredRolloutTest(unittest.TestCase):
    @staticmethod
    def row(content, identity=IDENTITY_CORRECT, validity=VALIDITY_VALID):
        return {
            "validity": validity,
            "identity_gate": identity,
            "content_score": content,
        }

    def test_two_high_consistent_candidates_are_skipped_as_easy(self):
        decision = decide_constraint_anchored_rollout([self.row(0.90), self.row(0.92)])

        self.assertFalse(decision.should_update)
        self.assertFalse(decision.needs_more)
        self.assertEqual(decision.reason, "easy_saturated")

    def test_identity_conflict_is_immediately_informative(self):
        decision = decide_constraint_anchored_rollout(
            [self.row(0.60), self.row(0.95, identity=IDENTITY_WRONG)]
        )

        self.assertTrue(decision.should_update)
        self.assertFalse(decision.needs_more)
        self.assertEqual(decision.reason, "constraint_informative")

    def test_all_poor_candidates_expand_to_eight_then_use_sft_fallback(self):
        two = [self.row(0.30, identity=IDENTITY_UNKNOWN) for _ in range(2)]
        four = [self.row(0.35, identity=IDENTITY_UNKNOWN) for _ in range(4)]
        eight = [self.row(0.40, identity=IDENTITY_UNKNOWN) for _ in range(8)]

        decision_two = decide_constraint_anchored_rollout(two)
        decision_four = decide_constraint_anchored_rollout(four)
        decision_eight = decide_constraint_anchored_rollout(eight)

        self.assertEqual(decision_two.target_count, 4)
        self.assertTrue(decision_two.needs_more)
        self.assertEqual(decision_four.target_count, 8)
        self.assertTrue(decision_four.needs_more)
        self.assertEqual(decision_eight.target_count, 8)
        self.assertTrue(decision_eight.should_update)
        self.assertTrue(decision_eight.use_sft_fallback)
        self.assertEqual(decision_eight.reason, "sft_fallback")

    def test_informative_content_difference_updates_without_fallback(self):
        decision = decide_constraint_anchored_rollout(
            [self.row(0.45), self.row(0.80), self.row(0.62), self.row(0.60)]
        )

        self.assertTrue(decision.should_update)
        self.assertFalse(decision.use_sft_fallback)
        self.assertEqual(decision.reason, "content_informative")


if __name__ == "__main__":
    unittest.main()
