import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from qwen35_pvchat.remoh_attention import (
    DEFAULT_REMOH_LAYER_INDICES,
    Qwen35ReMoHAttention,
    Qwen35ReMoHRouter,
    apply_head_weights,
    patch_qwen35_remoh_layers,
    remoh_generation_masks,
)


class Qwen35ReMoHRouterTest(unittest.TestCase):
    def test_identity_initialization_keeps_all_head_weights_at_one(self):
        router = Qwen35ReMoHRouter(
            hidden_size=8,
            num_attention_heads=16,
            routed_head_indices=(3, 7, 11, 15),
        )
        hidden_states = torch.randn(2, 5, 8)

        output = router(hidden_states)

        self.assertTrue(torch.equal(output.alpha, torch.full_like(output.alpha, 0.5)))
        self.assertTrue(torch.equal(output.routed_gates, torch.ones_like(output.routed_gates)))
        self.assertTrue(torch.equal(output.head_weights, torch.ones_like(output.head_weights)))

    def test_identity_initialized_relu_router_receives_gradients(self):
        router = Qwen35ReMoHRouter(
            hidden_size=8,
            num_attention_heads=4,
            routed_head_indices=(3,),
        )
        hidden_states = torch.randn(2, 3, 8)

        output = router(hidden_states)
        output.routed_head_weights.sum().backward()

        self.assertGreater(router.router.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(router.alpha_proj.weight.grad.abs().sum().item(), 0.0)

    def test_identity_weights_leave_context_unchanged(self):
        context = torch.randn(2, 5, 16, 4)
        weights = torch.ones(2, 5, 16)

        weighted = apply_head_weights(context, weights)

        self.assertTrue(torch.equal(weighted, context))


class _FakeAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim * 2,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()


class _FakeLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "full_attention":
            self.self_attn = _FakeAttention(config, layer_idx)


class Qwen35ReMoHPatchTest(unittest.TestCase):
    def test_generation_masks_are_temporarily_attached_and_then_cleared(self):
        config = SimpleNamespace(
            hidden_size=32,
            head_dim=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            _attn_implementation="eager",
        )
        wrapper = Qwen35ReMoHAttention(
            _FakeAttention(config, layer_idx=0),
            routed_head_indices=(3,),
        )
        video_mask = torch.tensor([[False, True, True]])
        text_mask = ~video_mask

        with remoh_generation_masks(wrapper, video_mask, text_mask):
            self.assertIs(wrapper.generation_video_token_mask, video_mask)
            self.assertIs(wrapper.generation_text_token_mask, text_mask)

        self.assertIsNone(wrapper.generation_video_token_mask)
        self.assertIsNone(wrapper.generation_text_token_mask)

    def test_new_router_inherits_pretrained_attention_dtype(self):
        config = SimpleNamespace(
            hidden_size=32,
            head_dim=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            _attn_implementation="eager",
        )
        original = _FakeAttention(config, layer_idx=0).to(dtype=torch.bfloat16)

        wrapper = Qwen35ReMoHAttention(original, routed_head_indices=(3,))

        self.assertEqual(wrapper.router.router.weight.dtype, torch.bfloat16)
        self.assertEqual(wrapper.router.alpha_proj.weight.dtype, torch.bfloat16)

    def test_identity_initialized_wrapper_matches_original_qwen_attention(self):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

        config = Qwen3_5TextConfig(
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_hidden_layers=1,
            layer_types=["full_attention"],
            attention_dropout=0.0,
        )
        config._attn_implementation = "eager"
        original = Qwen3_5Attention(config, layer_idx=0).eval()
        wrapper = Qwen35ReMoHAttention(
            original,
            routed_head_indices=(3,),
        ).eval()
        hidden_states = torch.randn(2, 6, 32)
        position_embeddings = (
            torch.ones(2, 6, 8),
            torch.zeros(2, 6, 8),
        )
        video_mask = torch.tensor(
            [
                [False, True, True, False, False, False],
                [False, True, True, True, False, False],
            ]
        )
        text_mask = ~video_mask

        expected, _ = original(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=None,
        )
        actual, _ = wrapper(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=None,
            pvchat_video_token_mask=video_mask,
            pvchat_text_token_mask=text_mask,
        )

        self.assertTrue(torch.equal(actual, expected))

    def test_default_patch_replaces_only_selected_full_attention_layers(self):
        layer_types = [
            "linear_attention" if (index + 1) % 4 else "full_attention"
            for index in range(32)
        ]
        config = SimpleNamespace(
            hidden_size=64,
            head_dim=4,
            num_attention_heads=16,
            num_key_value_heads=4,
            layer_types=layer_types,
            _attn_implementation="eager",
        )
        layers = nn.ModuleList([_FakeLayer(config, index) for index in range(32)])
        model = SimpleNamespace(
            model=SimpleNamespace(language_model=SimpleNamespace(layers=layers))
        )
        original_projection_ids = {
            index: id(layers[index].self_attn.q_proj)
            for index in DEFAULT_REMOH_LAYER_INDICES
        }

        patched = patch_qwen35_remoh_layers(model)

        self.assertEqual(tuple(patched), DEFAULT_REMOH_LAYER_INDICES)
        for index, layer in enumerate(layers):
            if index in DEFAULT_REMOH_LAYER_INDICES:
                self.assertIsInstance(layer.self_attn.router, Qwen35ReMoHRouter)
                self.assertEqual(id(layer.self_attn.q_proj), original_projection_ids[index])
            elif layer.block_type == "full_attention":
                self.assertIsInstance(layer.self_attn, _FakeAttention)


if __name__ == "__main__":
    unittest.main()
