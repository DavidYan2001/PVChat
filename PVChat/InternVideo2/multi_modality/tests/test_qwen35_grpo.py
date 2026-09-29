import unittest
from collections import Counter
from types import SimpleNamespace

import torch

from qwen35_pvchat.grpo import (
    build_completion_labels,
    completion_token_logprobs,
    normalize_advantages,
    select_causal_positions,
    selected_causal_cross_entropy,
    selected_token_logprobs,
    token_grpo_loss,
)
from qwen35_pvchat.rewards import score_answer


class Qwen35RewardTest(unittest.TestCase):
    def test_positive_identity_statement_is_not_penalized_for_missing_yes_word(self):
        result = score_answer(
            question="Does this footage include <Sheldon>?",
            gold_answer="The footage shows <Sheldon>.",
            candidate_answer="The footage shows <Sheldon>.",
            person_token="<Sheldon>",
            is_positive=True,
            qa_type="identity",
        )

        self.assertEqual(result.components["identity_correctness"], 1.0)
        self.assertGreaterEqual(result.reward, 0.9)

    def test_negative_identity_rejects_person_hallucination(self):
        result = score_answer(
            question="Is <Sheldon> here?",
            gold_answer="No, <Sheldon> is absent.",
            candidate_answer="<Sheldon> is visible.",
            person_token="<Sheldon>",
            is_positive=False,
            qa_type="identity",
        )

        self.assertLess(result.reward, 0.0)

    def test_natural_negative_phrasings_score_positive_on_negative_video(self):
        """旧白名单漏掉的自然否定表述必须拿到+1（与评测分类器一致）。"""
        for candidate in (
            "<Sheldon> is not in this video.",
            "There's no <Sheldon> present in this video.",
            "The video does not contain <Sheldon>.",
            "I see no sign of <Sheldon>.",
            "No, I cannot find <Sheldon> here.",
        ):
            result = score_answer(
                question="Can you spot <Sheldon> in this clip?",
                gold_answer="<Sheldon> is not in this video.",
                candidate_answer=candidate,
                person_token="<Sheldon>",
                is_positive=False,
                qa_type="identity",
            )
            self.assertEqual(
                result.components["identity_correctness"], 1.0, msg=candidate
            )

    def test_affirmative_answers_still_rewarded_on_positive_video(self):
        for candidate in (
            "Yes, <Sheldon> is present.",
            "The footage shows <Sheldon>.",
            "Yes.",
        ):
            result = score_answer(
                question="Is <Sheldon> in this video?",
                gold_answer="<Sheldon> is present in this recording.",
                candidate_answer=candidate,
                person_token="<Sheldon>",
                is_positive=True,
                qa_type="identity",
            )
            self.assertEqual(
                result.components["identity_correctness"], 1.0, msg=candidate
            )


class Qwen35GrpoLossTest(unittest.TestCase):
    def test_selected_answer_logits_match_full_causal_math(self):
        torch.manual_seed(11)
        full_logits = torch.randn(2, 5, 13)
        input_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5],
                [1, 2, 6, 7, 8],
            ]
        )
        labels = torch.tensor(
            [
                [-100, -100, 3, 4, -100],
                [-100, -100, -100, 7, 8],
            ]
        )

        positions, targets = select_causal_positions(labels)
        selected_logits = full_logits.index_select(1, positions)
        selected_loss = selected_causal_cross_entropy(selected_logits, targets)
        selected_logprobs, selected_mask = selected_token_logprobs(selected_logits, targets)

        shifted_labels = labels[:, 1:]
        expected_loss = torch.nn.functional.cross_entropy(
            full_logits[:, :-1].reshape(-1, full_logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=-100,
        )
        full_logprobs, full_mask = completion_token_logprobs(full_logits, input_ids, labels)

        self.assertEqual(positions.tolist(), [1, 2, 3])
        self.assertTrue(torch.allclose(selected_loss, expected_loss))
        self.assertTrue(torch.allclose(selected_logprobs, full_logprobs.index_select(1, positions)))
        self.assertTrue(torch.equal(selected_mask, full_mask.index_select(1, positions)))

    def test_completion_labels_mask_prompt_and_padding_after_first_stop_token(self):
        sequences = torch.tensor(
            [
                [1, 2, 3, 10, 11, 99, 99],
                [1, 2, 3, 20, 99, 99, 99],
            ]
        )

        labels = build_completion_labels(sequences, prompt_length=3, stop_token_ids=[99])

        self.assertEqual(labels[0].tolist(), [-100, -100, -100, 10, 11, 99, -100])
        self.assertEqual(labels[1].tolist(), [-100, -100, -100, 20, 99, -100, -100])

    def test_group_advantages_are_zero_mean(self):
        advantages = normalize_advantages([0.2, 0.5, 0.9, 0.4])

        self.assertAlmostEqual(sum(advantages), 0.0, places=6)
        self.assertGreater(max(advantages), 0.0)
        self.assertLess(min(advantages), 0.0)

    def test_token_grpo_loss_has_gradient_even_when_scalar_is_near_zero(self):
        new_logprobs = torch.tensor(
            [[-1.0, -1.0], [-1.0, -1.0]],
            requires_grad=True,
        )
        old_logprobs = new_logprobs.detach().clone()
        advantages = torch.tensor([1.0, -1.0])
        mask = torch.ones_like(new_logprobs, dtype=torch.bool)

        loss, _ = token_grpo_loss(
            new_logprobs,
            old_logprobs,
            advantages,
            mask,
            clip_range=0.2,
            drift_beta=0.02,
        )
        loss.backward()

        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        self.assertGreater(new_logprobs.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()


class ReferenceKlLossTest(unittest.TestCase):
    def test_zero_when_policies_match(self):
        from qwen35_pvchat.grpo import reference_kl_loss
        lp = torch.log(torch.tensor([[0.5, 0.25], [0.1, 0.9]]))
        mask = torch.ones_like(lp, dtype=torch.bool)
        self.assertAlmostEqual(float(reference_kl_loss(lp, lp.clone(), mask)), 0.0, places=6)

    def test_positive_and_masked(self):
        from qwen35_pvchat.grpo import reference_kl_loss
        new = torch.log(torch.tensor([[0.5, 0.5]]))
        ref = torch.log(torch.tensor([[0.9, 0.5]]))
        full = torch.ones_like(new, dtype=torch.bool)
        self.assertGreater(float(reference_kl_loss(new, ref, full)), 0.0)
        # mask掉分歧token后应为0
        partial = torch.tensor([[False, True]])
        self.assertAlmostEqual(float(reference_kl_loss(new, ref, partial)), 0.0, places=6)
