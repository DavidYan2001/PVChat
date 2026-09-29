import math
import unittest

import torch

from qwen35_pvchat.grpo import normalize_advantages, token_grpo_loss
from qwen35_pvchat.policy_objectives import (
    connected_zero_loss,
    decoupled_component_advantages,
    masked_sequence_logprobs,
    normalize_group_advantages,
    sequence_policy_loss,
    supervised_anchor_loss,
    token_policy_loss,
)


class NormalizeGroupAdvantagesTest(unittest.TestCase):
    def test_alpha_zero_half_and_one_follow_population_std_formula(self):
        rewards = [1.0, 2.0, 5.0]
        mean = 8.0 / 3.0
        std = math.sqrt(26.0) / 3.0

        for alpha in (0.0, 0.5, 1.0):
            with self.subTest(alpha=alpha):
                expected = [
                    (reward - mean) / (std + 1e-6) ** alpha
                    for reward in rewards
                ]
                actual = normalize_group_advantages(rewards, alpha=alpha)

                self.assertEqual(len(actual), len(expected))
                for actual_value, expected_value in zip(actual, expected):
                    self.assertAlmostEqual(actual_value, expected_value, places=12)

    def test_empty_and_zero_variance_groups_return_zeros(self):
        self.assertEqual(normalize_group_advantages([], alpha=0.5), [])
        self.assertEqual(
            normalize_group_advantages([3.25, 3.25, 3.25], alpha=0.0),
            [0.0, 0.0, 0.0],
        )
        self.assertEqual(
            normalize_group_advantages([3.25, 3.25, 3.25], alpha=1.0),
            [0.0, 0.0, 0.0],
        )

    def test_token_grpo_alpha_one_keeps_the_existing_tiny_std_behavior(self):
        rewards = [1.0, 1.0 + 1e-8, 1.0 - 1e-8]
        self.assertEqual(
            normalize_group_advantages(rewards, alpha=1.0),
            normalize_advantages(rewards),
        )

    def test_alpha_outside_closed_unit_interval_is_rejected(self):
        for alpha in (-0.01, 1.01):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    normalize_group_advantages([1.0, 2.0], alpha=alpha)


class MaskedSequenceLogprobsTest(unittest.TestCase):
    def test_rows_with_unequal_answer_lengths_average_only_valid_tokens(self):
        token_logprobs = torch.tensor(
            [
                [-1.0, 100.0, 200.0],
                [-2.0, -4.0, -6.0],
            ],
            dtype=torch.float64,
        )
        mask = torch.tensor(
            [
                [True, False, False],
                [True, True, True],
            ]
        )

        actual = masked_sequence_logprobs(token_logprobs, mask)

        self.assertTrue(
            torch.equal(actual, torch.tensor([-1.0, -4.0], dtype=torch.float64))
        )

    def test_row_without_valid_completion_token_is_rejected(self):
        token_logprobs = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
        mask = torch.tensor([[True, False], [False, False]])

        with self.assertRaises(ValueError):
            masked_sequence_logprobs(token_logprobs, mask)


class PolicyLossTest(unittest.TestCase):
    def test_supervised_anchor_loss_averages_only_gold_completion_tokens(self):
        logprobs = torch.tensor(
            [[-1.0, -3.0, 100.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        mask = torch.tensor([[True, True, False]])

        loss = supervised_anchor_loss(logprobs, mask)

        self.assertAlmostEqual(loss.item(), 2.0, places=12)
        loss.backward()
        self.assertTrue(torch.equal(logprobs.grad, torch.tensor([[-0.5, -0.5, 0.0]], dtype=torch.float64)))

    def test_supervised_anchor_loss_rejects_an_empty_gold_completion(self):
        with self.assertRaisesRegex(ValueError, "valid completion token"):
            supervised_anchor_loss(
                torch.zeros((1, 2)),
                torch.zeros((1, 2), dtype=torch.bool),
            )

    def test_token_policy_loss_matches_current_token_grpo_loss_exactly(self):
        actual_new = torch.tensor(
            [[-0.8, -1.3, -4.0], [-2.0, -0.2, -3.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        expected_new = actual_new.detach().clone().requires_grad_(True)
        old_logprobs = torch.tensor(
            [[-1.0, -1.0, -5.0], [-1.5, -0.4, -2.5]],
            dtype=torch.float64,
        )
        advantages = torch.tensor([1.25, -0.75], dtype=torch.float64)
        mask = torch.tensor(
            [[True, True, False], [True, False, False]],
        )

        actual_total, actual_metrics = token_policy_loss(
            actual_new,
            old_logprobs,
            advantages,
            mask,
            clip_range=0.2,
            drift_beta=0.02,
        )
        expected_total, expected_metrics = token_grpo_loss(
            expected_new,
            old_logprobs,
            advantages,
            mask,
            clip_range=0.2,
            drift_beta=0.02,
        )

        self.assertTrue(torch.equal(actual_total, expected_total))
        self.assertTrue(
            torch.equal(actual_metrics["policy_loss"], expected_metrics["policy_loss"])
        )
        self.assertTrue(
            torch.equal(actual_metrics["drift_loss"], expected_metrics["drift_loss"])
        )
        self.assertFalse(actual_metrics["policy_loss"].requires_grad)
        self.assertFalse(actual_metrics["drift_loss"].requires_grad)

        actual_total.backward()
        expected_total.backward()
        self.assertTrue(torch.equal(actual_new.grad, expected_new.grad))

    def test_sequence_policy_loss_uses_one_geometric_mean_ratio_per_candidate(self):
        old_logprobs = torch.tensor(
            [
                [-2.0, -2.0, 50.0],
                [-3.0, -3.0, -3.0],
            ],
            dtype=torch.float64,
        )
        new_logprobs = torch.tensor(
            [
                [-2.0 + math.log(1.21), -2.0, -50.0],
                [-3.0 + math.log(0.729), -3.0, -3.0],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        advantages = torch.tensor([2.0, -1.0], dtype=torch.float64)
        mask = torch.tensor(
            [[True, True, False], [True, True, True]],
        )

        total, metrics = sequence_policy_loss(
            new_logprobs,
            old_logprobs,
            advantages,
            mask,
            clip_range=0.2,
            drift_beta=0.02,
        )

        expected_policy = -0.65
        expected_drift = (math.log(1.1) ** 2 + math.log(0.9) ** 2) / 2.0
        expected_total = expected_policy + 0.02 * expected_drift
        self.assertAlmostEqual(total.item(), expected_total, places=12)
        self.assertAlmostEqual(metrics["policy_loss"].item(), expected_policy, places=12)
        self.assertAlmostEqual(metrics["drift_loss"].item(), expected_drift, places=12)
        self.assertFalse(metrics["policy_loss"].requires_grad)
        self.assertFalse(metrics["drift_loss"].requires_grad)

    def test_sequence_clipping_branches_have_zero_policy_gradient(self):
        old_logprobs = torch.full((3, 1), -2.0, dtype=torch.float64)
        new_logprobs = torch.tensor(
            [
                [-2.0 + math.log(1.5)],
                [-2.0 + math.log(0.5)],
                [-2.0],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        advantages = torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64)
        mask = torch.ones_like(new_logprobs, dtype=torch.bool)

        total, metrics = sequence_policy_loss(
            new_logprobs,
            old_logprobs,
            advantages,
            mask,
            clip_range=0.2,
            drift_beta=0.0,
        )
        total.backward()

        self.assertAlmostEqual(metrics["policy_loss"].item(), -1.4 / 3.0, places=12)
        self.assertAlmostEqual(new_logprobs.grad[0, 0].item(), 0.0, places=12)
        self.assertAlmostEqual(new_logprobs.grad[1, 0].item(), 0.0, places=12)
        self.assertAlmostEqual(new_logprobs.grad[2, 0].item(), -1.0 / 3.0, places=12)


class DecoupledComponentAdvantagesTest(unittest.TestCase):
    def test_components_are_normalized_independently_with_spec_weights(self):
        component_rows = [
            {
                "identity_correctness": 1.0,
                "identity_name": 1.0,
                "conciseness": 0.5,
                "format": 1.0,
            },
            {
                "identity_correctness": 0.0,
                "identity_name": 1.0,
                "conciseness": 1.0,
                "format": 0.0,
            },
            {
                "identity_correctness": -1.0,
                "conciseness": 0.0,
                "format": 1.0,
            },
        ]
        component_weights = {
            "identity_correctness": 0.80,
            "identity_name": 0.10,
            "conciseness": 0.05,
            "format": 0.05,
        }

        actual = decoupled_component_advantages(
            component_rows,
            component_weights,
            alpha=1.0,
        )

        columns = {
            key: [float(row.get(key, 0.0)) for row in component_rows]
            for key in sorted(component_weights)
        }
        expected = [0.0 for _ in component_rows]
        for key in sorted(component_weights):
            values = columns[key]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            std = math.sqrt(variance)
            normalized = (
                [0.0 for _ in values]
                if std == 0.0
                else [(value - mean) / (std + 1e-6) for value in values]
            )
            for index, value in enumerate(normalized):
                expected[index] += component_weights[key] * value

        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, places=12)

    def test_missing_components_are_zero_and_weight_order_is_deterministic(self):
        component_rows = [
            {"semantic": 1.0, "presence_consistency": 1.0},
            {"semantic": 0.0},
            {"presence_consistency": -1.0},
        ]
        forward_weights = {
            "semantic": 0.65,
            "presence_consistency": 0.15,
            "identity_name": 0.10,
            "conciseness": 0.05,
            "format": 0.05,
        }
        reverse_weights = dict(reversed(list(forward_weights.items())))

        forward = decoupled_component_advantages(
            component_rows,
            forward_weights,
            alpha=0.5,
        )
        reverse = decoupled_component_advantages(
            component_rows,
            reverse_weights,
            alpha=0.5,
        )

        self.assertEqual(forward, reverse)
        self.assertAlmostEqual(sum(forward), 0.0, places=12)


class ConnectedZeroLossTest(unittest.TestCase):
    def test_backward_produces_exact_zero_gradient_on_connected_tensor(self):
        tensor = torch.tensor([2.0, -3.0, 7.0], requires_grad=True)

        loss = connected_zero_loss(tensor)

        self.assertEqual(loss.shape, torch.Size([]))
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertTrue(torch.equal(tensor.grad, torch.zeros_like(tensor)))


if __name__ == "__main__":
    unittest.main()
