import unittest

from qwen35_pvchat.identity_constrained_policy import (
    active_soft_weights,
    decide_fixed_icd_rollout,
    decide_vector_icd_rollout,
    identity_constrained_advantages,
    score_icd_answer,
    vector_reward_diagnostics,
)


def row(identity, semantic=0.0, specificity=0.0, coverage=0.0):
    return {
        "identity": float(identity),
        "semantic": float(semantic),
        "specificity": float(specificity),
        "coverage": float(coverage),
    }


class IdentityConstrainedRewardTest(unittest.TestCase):
    def test_active_soft_weights_are_normalized_and_negative_masks_details(self):
        positive = active_soft_weights("action", True)
        negative = active_soft_weights("action", False)

        self.assertEqual(set(positive), {"semantic", "specificity", "coverage"})
        self.assertAlmostEqual(sum(positive.values()), 1.0)
        self.assertEqual(negative, {"semantic": 1.0})

    def test_positive_and_negative_answers_use_presence_as_identity_feasibility(self):
        common = {
            "question": "What is <P> doing?",
            "person_token": "<P>",
            "qa_type": "action",
        }
        positive = score_icd_answer(
            gold_answer="<P> is running outside.",
            candidate_answer="<P> is running outside.",
            is_positive=True,
            **common,
        )
        positive_absence = score_icd_answer(
            gold_answer="<P> is running outside.",
            candidate_answer="<P> is not present in the video.",
            is_positive=True,
            **common,
        )
        negative_absence = score_icd_answer(
            gold_answer="<P> is not present in the video.",
            candidate_answer="<P> is not present in the video.",
            is_positive=False,
            **common,
        )
        negative_hallucination = score_icd_answer(
            gold_answer="<P> is not present in the video.",
            candidate_answer="<P> is running outside.",
            is_positive=False,
            **common,
        )

        self.assertEqual(positive.components["identity"], 1.0)
        self.assertEqual(positive.components["specificity"], 1.0)
        self.assertEqual(positive.components["coverage"], 1.0)
        self.assertEqual(positive_absence.components["identity"], -1.0)
        self.assertEqual(negative_absence.components["identity"], 1.0)
        self.assertEqual(negative_absence.components["specificity"], 0.0)
        self.assertEqual(negative_absence.components["coverage"], 0.0)
        self.assertEqual(negative_hallucination.components["identity"], -1.0)
        self.assertGreater(negative_absence.reward, negative_hallucination.reward)

    def test_short_person_name_is_not_matched_inside_a_generic_word(self):
        result = score_icd_answer(
            question="What is <P> doing?",
            gold_answer="<P> is running outside.",
            candidate_answer="The person is running outside.",
            person_token="<P>",
            is_positive=True,
            qa_type="action",
        )

        self.assertEqual(result.components["identity"], 0.0)
        self.assertEqual(result.components["specificity"], 0.3)


class IdentityMarginAdvantageTest(unittest.TestCase):
    def test_feasible_candidate_strictly_outranks_detailed_infeasible_candidate(self):
        rows = [
            row(-1, semantic=1, specificity=1, coverage=1),
            row(1, semantic=0, specificity=0, coverage=0),
            row(1, semantic=0.2, specificity=0.2, coverage=0.2),
            row(-1, semantic=0.8, specificity=0.8, coverage=0.8),
        ]

        advantages = identity_constrained_advantages(
            rows,
            qa_type="action",
            is_positive=True,
            identity_margin=1.0,
            soft_clip=0.5,
        )

        feasible = [value for value, item in zip(advantages, rows) if item["identity"] > 0]
        infeasible = [value for value, item in zip(advantages, rows) if item["identity"] <= 0]
        self.assertGreater(min(feasible), max(infeasible))

    def test_all_feasible_centers_soft_signal_and_all_infeasible_returns_zero(self):
        feasible = identity_constrained_advantages(
            [row(1, semantic=value, specificity=value, coverage=value) for value in (0.0, 0.3, 0.7, 1.0)],
            qa_type="action",
            is_positive=True,
        )
        infeasible = identity_constrained_advantages(
            [row(-1, semantic=value, specificity=value, coverage=value) for value in (0.0, 0.3, 0.7, 1.0)],
            qa_type="action",
            is_positive=True,
        )

        self.assertAlmostEqual(sum(feasible), 0.0, places=12)
        self.assertEqual(infeasible, [0.0, 0.0, 0.0, 0.0])

    def test_single_feasible_candidate_uses_only_identity_margin(self):
        advantages = identity_constrained_advantages(
            [row(1, semantic=1), row(-1, semantic=1), row(-1), row(-1)],
            qa_type="action",
            is_positive=True,
            identity_margin=1.0,
            soft_clip=0.5,
        )

        self.assertAlmostEqual(advantages[0], 0.75)
        self.assertEqual(advantages[1:], [-0.25, -0.25, -0.25])

    def test_identity_margin_must_cover_the_soft_clip(self):
        with self.assertRaisesRegex(ValueError, "identity_margin"):
            identity_constrained_advantages(
                [row(1), row(-1)],
                qa_type="action",
                is_positive=True,
                identity_margin=0.4,
                soft_clip=0.5,
            )


class VectorAdaptiveRolloutTest(unittest.TestCase):
    def test_v0_requires_eight_and_skips_an_all_infeasible_group(self):
        update = decide_fixed_icd_rollout(
            [row(1, semantic=value) for value in range(8)],
            qa_type="action",
            is_positive=True,
        )
        skip = decide_fixed_icd_rollout(
            [row(-1, semantic=1.0) for _ in range(8)],
            qa_type="action",
            is_positive=True,
        )

        self.assertTrue(update.should_update)
        self.assertEqual(update.target_count, 8)
        self.assertFalse(skip.should_update)
        self.assertEqual(skip.reason, "all_identity_infeasible")

    def test_v1_expands_identity_uncertainty_and_cross_component_conflict(self):
        mixed = [row(1, 1, 0, 1), row(1, 0, 1, 0), row(-1), row(-1)]
        conflict = [
            row(1, 1.0, 0.0, 0.5),
            row(1, 0.7, 0.3, 0.5),
            row(1, 0.3, 0.7, 0.5),
            row(1, 0.0, 1.0, 0.5),
        ]

        self.assertTrue(
            decide_vector_icd_rollout(mixed, "action", True, conflict_threshold=0.25).needs_more
        )
        conflict_decision = decide_vector_icd_rollout(
            conflict,
            "action",
            True,
            conflict_threshold=0.20,
        )
        self.assertTrue(conflict_decision.needs_more)
        self.assertEqual(conflict_decision.target_count, 8)

    def test_v1_keeps_four_for_aligned_informative_rewards_and_skips_zero_signal(self):
        aligned = [
            row(1, value, value, value)
            for value in (0.0, 0.3, 0.7, 1.0)
        ]
        identical = [row(1, 0.5, 0.5, 0.5) for _ in range(4)]

        update = decide_vector_icd_rollout(
            aligned,
            "action",
            True,
            conflict_threshold=0.25,
        )
        skip = decide_vector_icd_rollout(
            identical,
            "action",
            True,
            conflict_threshold=0.25,
        )

        self.assertEqual((update.target_count, update.should_update), (4, True))
        self.assertEqual(skip.reason, "zero_advantage")
        self.assertFalse(skip.should_update)

    def test_vector_diagnostics_report_separate_signals(self):
        diagnostics = vector_reward_diagnostics(
            [row(1, 1, 0, 1), row(1, 0, 1, 0), row(-1), row(-1)],
            qa_type="action",
            is_positive=True,
        )

        self.assertAlmostEqual(diagnostics["feasibility_rate"], 0.5)
        self.assertGreater(diagnostics["identity_uncertainty"], 0.0)
        self.assertGreater(diagnostics["soft_dispersion"], 0.0)
        self.assertGreater(diagnostics["rank_disagreement"], 0.0)


if __name__ == "__main__":
    unittest.main()
