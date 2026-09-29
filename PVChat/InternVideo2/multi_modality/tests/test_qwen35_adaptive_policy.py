import math
import unittest

from qwen35_pvchat.adaptive_policy import (
    CATEGORIES,
    PersonalizedAdaptiveState,
    decide_dynamic_rollout,
)
from qwen35_pvchat.rewards import pa_component_weights, score_answer, score_pa_answer


class DynamicRolloutTest(unittest.TestCase):
    def test_all_dynamic_rollout_branches_and_boundaries(self):
        cases = [
            ([0.85, 0.85], (2, False, "easy_saturated")),
            ([0.85, 0.80], (4, False, "expand")),
            ([0.80, 0.65], (2, True, "informative")),
            ([0.50, 0.50, 0.50, 0.50], (8, False, "expand_hard")),
            ([0.60, 0.60, 0.60, 0.46], (4, False, "low_dispersion")),
            ([0.80, 0.65, 0.80, 0.80], (4, True, "informative")),
            ([0.90] * 7 + [0.75], (8, True, "informative")),
            ([0.90] * 8, (8, False, "zero_advantage")),
        ]
        for rewards, expected in cases:
            decision = decide_dynamic_rollout(rewards)
            self.assertEqual(
                (decision.target_count, decision.should_update, decision.reason), expected
            )
        self.assertTrue(decide_dynamic_rollout([0.5, 0.5]).needs_more)
        self.assertFalse(decide_dynamic_rollout([0.8, 0.65]).needs_more)
        self.assertEqual(decide_dynamic_rollout([0.55, 0.50, 0.50, 0.45]).reason, "expand_hard")
        self.assertEqual(decide_dynamic_rollout([0.80, 0.65, 0.80, 0.80]).reason, "informative")

    def test_only_supported_rollout_lengths_are_accepted(self):
        with self.assertRaises(ValueError):
            decide_dynamic_rollout([0.1])
        with self.assertRaises(ValueError):
            decide_dynamic_rollout([0.1] * 3)

    def test_non_finite_rewards_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                decide_dynamic_rollout([0.0, value])

    def test_easy_dispersion_boundary_uses_exact_comparison(self):
        below = math.nextafter(0.85, math.inf)
        at = 0.85
        above = math.nextafter(0.85, -math.inf)

        self.assertLess(0.90 - below, 0.05)
        self.assertGreaterEqual(0.90 - at, 0.05)
        self.assertGreater(0.90 - above, 0.05)
        self.assertEqual(decide_dynamic_rollout([0.90, below]).reason, "easy_saturated")
        self.assertEqual(decide_dynamic_rollout([0.90, at]).reason, "expand")
        self.assertEqual(decide_dynamic_rollout([0.90, above]).reason, "expand")

    def test_informative_dispersion_boundary_uses_exact_comparison(self):
        below = math.nextafter(0.15, -math.inf)
        at = 0.15
        above = math.nextafter(0.15, math.inf)

        self.assertEqual(decide_dynamic_rollout([below, 0.0]).reason, "expand")
        self.assertEqual(decide_dynamic_rollout([at, 0.0]).reason, "informative")
        self.assertEqual(decide_dynamic_rollout([above, 0.0]).reason, "informative")

    def test_easy_mean_boundary_uses_exact_comparison(self):
        below = math.nextafter(0.85, -math.inf)
        at = 0.85
        above = math.nextafter(0.85, math.inf)

        self.assertEqual(decide_dynamic_rollout([below, below]).reason, "expand")
        self.assertEqual(decide_dynamic_rollout([at, at]).reason, "easy_saturated")
        self.assertEqual(decide_dynamic_rollout([above, above]).reason, "easy_saturated")


class PersonalizedAdaptiveStateTest(unittest.TestCase):
    def test_multiplier_math_is_normalized_and_tokens_are_normalized(self):
        state = PersonalizedAdaptiveState()
        state.observe("Sheldon", "identity", 1.0)
        weights = state.multipliers("<Sheldon>")
        self.assertEqual(set(weights), set(CATEGORIES))
        self.assertAlmostEqual(sum(weights.values()) / len(CATEGORIES), 1.0)
        self.assertLess(weights["identity"], weights["action"])

    def test_observations_clip_and_apply_in_order(self):
        state = PersonalizedAdaptiveState()
        state.observe("p", "action", 99)
        self.assertAlmostEqual(state.profiles["<p>"]["action"], 0.55)
        state.observe_many([
            ("p", "action", -99),
            ("p", "action", 1),
        ])
        self.assertAlmostEqual(state.profiles["<p>"]["action"], 0.5455)

    def test_unknown_categories_are_rejected(self):
        state = PersonalizedAdaptiveState()
        with self.assertRaises(ValueError):
            state.observe("p", "unknown", 0.0)
        with self.assertRaises(ValueError):
            state.multiplier("p", "unknown")
        with self.assertRaises(ValueError):
            PersonalizedAdaptiveState.from_state_dict({"profiles": {"<p>": {"unknown": 0.5}}})

    def test_state_round_trip_includes_hyperparameters_and_profiles(self):
        state = PersonalizedAdaptiveState(initial_ema=0.4, decay=0.8, raw_multiplier_min=0.6, raw_multiplier_max=1.4)
        state.observe("<A>", "emotion", 0.2)
        restored = PersonalizedAdaptiveState.from_state_dict(state.state_dict())
        self.assertEqual(restored.state_dict(), state.state_dict())
        self.assertEqual(restored.multipliers("<A>"), state.multipliers("<A>"))

    def test_invalid_hyperparameters_are_rejected(self):
        invalid_kwargs = [
            {"initial_ema": math.nan},
            {"initial_ema": math.inf},
            {"initial_ema": -0.01},
            {"initial_ema": 1.01},
            {"decay": math.nan},
            {"decay": math.inf},
            {"decay": -0.01},
            {"decay": 1.01},
            {"raw_multiplier_min": math.nan},
            {"raw_multiplier_max": math.inf},
            {"raw_multiplier_min": 1.6, "raw_multiplier_max": 1.5},
        ]
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PersonalizedAdaptiveState(**kwargs)

    def test_non_finite_observations_are_rejected(self):
        state = PersonalizedAdaptiveState()
        for reward in (math.nan, math.inf, -math.inf):
            with self.subTest(reward=reward), self.assertRaises(ValueError):
                state.observe("P", "action", reward)

    def test_loaded_profile_values_must_be_finite_probabilities(self):
        for value in (math.nan, math.inf, -math.inf, -0.01, 1.01):
            profile = {category: 0.5 for category in CATEGORIES}
            profile["action"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                PersonalizedAdaptiveState.from_state_dict({"profiles": {"P": profile}})

    def test_loaded_hyperparameters_are_validated(self):
        for state in (
            {"initial_ema": math.nan},
            {"decay": 1.01},
            {"raw_multiplier_min": -math.inf},
        ):
            with self.subTest(state=state), self.assertRaises(ValueError):
                PersonalizedAdaptiveState.from_state_dict(state)

    def test_duplicate_people_after_normalization_are_rejected(self):
        profile = {category: 0.5 for category in CATEGORIES}
        with self.assertRaises(ValueError):
            PersonalizedAdaptiveState.from_state_dict(
                {"profiles": {"P": profile, "<P>": profile}}
            )


class PaRewardTest(unittest.TestCase):
    def test_pa_weights_are_exact(self):
        self.assertEqual(pa_component_weights("identity", True), {
            "identity_correctness": 0.80, "identity_name": 0.10, "conciseness": 0.05, "format": 0.05,
        })
        self.assertEqual(pa_component_weights("action", True), {
            "semantic": 0.65, "presence_consistency": 0.15, "identity_name": 0.10,
            "conciseness": 0.05, "format": 0.05,
        })
        self.assertEqual(pa_component_weights("action", False), {
            "absence_consistency": 0.60, "semantic": 0.25, "identity_name": 0.05,
            "conciseness": 0.05, "format": 0.05,
        })

    def test_positive_and_negative_pa_consistency(self):
        common = dict(question="What is <P> doing?", gold_answer="<P> is running.", person_token="<P>", qa_type="action")
        positive = score_pa_answer(candidate_answer="<P> is running.", is_positive=True, **common)
        negative_absence = score_pa_answer(
            question="What is <P> doing?", gold_answer="<P> is absent.", candidate_answer="<P> is not present.",
            person_token="<P>", is_positive=False, qa_type="action",
        )
        negative_hallucination = score_pa_answer(
            question="What is <P> doing?", gold_answer="<P> is absent.", candidate_answer="<P> is running.",
            person_token="<P>", is_positive=False, qa_type="action",
        )
        self.assertEqual(positive.components["presence_consistency"], 1.0)
        self.assertEqual(negative_absence.components["absence_consistency"], 1.0)
        self.assertEqual(negative_hallucination.components["absence_consistency"], -1.0)
        self.assertGreater(negative_absence.reward, negative_hallucination.reward)

    def test_legacy_nonidentity_score_is_unchanged(self):
        args = dict(question="What is <P> doing?", gold_answer="<P> is running.", candidate_answer="<P> is running.", person_token="<P>", is_positive=True, qa_type="action")
        legacy = score_answer(**args)
        self.assertAlmostEqual(legacy.reward, 1.0)
        self.assertEqual(legacy.components, {"semantic": 1.0, "identity_name": 1.0, "conciseness": 1.0, "format": 1.0})

    def test_pa_short_bare_name_requires_word_boundaries(self):
        for candidate in ("The video is dark.", "The person is running."):
            result = score_pa_answer(
                question="What is <P> doing?",
                gold_answer="<P> is running.",
                candidate_answer=candidate,
                person_token="<P>",
                is_positive=True,
                qa_type="action",
            )
            with self.subTest(candidate=candidate):
                self.assertEqual(result.components["presence_consistency"], 0.0)

    def test_pa_accepts_exact_token_and_standalone_bare_name(self):
        for candidate in ("<P> is running.", "P is running."):
            result = score_pa_answer(
                question="What is <P> doing?",
                gold_answer="<P> is running.",
                candidate_answer=candidate,
                person_token="<P>",
                is_positive=True,
                qa_type="action",
            )
            with self.subTest(candidate=candidate):
                self.assertEqual(result.components["presence_consistency"], 1.0)

    def test_short_token_substring_no_longer_counts_as_presence(self):
        """训练奖励改用评测分类器后, "P"⊂"person"的子串巧合不再算提到人物。

        旧实现因此给这种答非所问的回答+1; 新语义与评测accuracy一致:
        无法判定存在性的回答按错误处理。
        """
        result = score_answer(
            question="Is <P> present?",
            gold_answer="Yes.",
            candidate_answer="The person is running.",
            person_token="<P>",
            is_positive=True,
            qa_type="identity",
        )
        self.assertEqual(result.components["identity_correctness"], -1.0)


if __name__ == "__main__":
    unittest.main()
