import unittest

import torch
import torch.nn as nn

from qwen35_pvchat.personalization import (
    PersonalizedEmbedding,
    PersonalizedLMHead,
    build_identity_prefix,
    build_personalized_tokens,
)


class PersonalizedTokenTest(unittest.TestCase):
    def test_builds_person_token_and_sixteen_detail_tokens(self):
        tokens = build_personalized_tokens("<Sheldon>", 16)

        self.assertEqual(tokens[0], "<Sheldon>")
        self.assertEqual(tokens[1], "<sks_token1>")
        self.assertEqual(tokens[-1], "<sks_token16>")
        self.assertEqual(len(tokens), 17)

    def test_identity_prefix_places_all_tokens_before_question(self):
        tokens = build_personalized_tokens("<Sheldon>", 2)

        prompt = build_identity_prefix("What is <Sheldon> doing?", tokens)

        self.assertTrue(prompt.startswith("<Sheldon><sks_token1><sks_token2>"))
        self.assertTrue(prompt.endswith("What is <Sheldon> doing?"))

    def test_embedding_only_trains_personalized_rows(self):
        base = nn.Embedding(10, 4)
        base.weight.requires_grad_(False)
        wrapped = PersonalizedEmbedding(base, token_ids=[8, 9])

        output = wrapped(torch.tensor([[1, 8, 2, 9]])).sum()
        output.backward()

        self.assertIsNone(base.weight.grad)
        self.assertIsNotNone(wrapped.personal_rows.grad)
        self.assertGreater(wrapped.personal_rows.grad.abs().sum().item(), 0.0)

    def test_lm_head_replaces_only_personalized_logits(self):
        base = nn.Linear(4, 10, bias=False)
        base.weight.requires_grad_(False)
        wrapped = PersonalizedLMHead(base, token_ids=[8, 9])
        hidden = torch.randn(2, 3, 4)

        logits = wrapped(hidden)
        base_logits = base(hidden)

        self.assertTrue(torch.equal(logits[..., :8], base_logits[..., :8]))
        logits[..., 8:].sum().backward()
        self.assertIsNone(base.weight.grad)
        self.assertGreater(wrapped.personal_rows.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
