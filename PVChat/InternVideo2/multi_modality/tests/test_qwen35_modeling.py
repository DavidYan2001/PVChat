import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from qwen35_pvchat.checkpoint import (
    load_optimizer_state,
    load_trainable_state,
    save_trainable_state,
)
from qwen35_pvchat.modeling import (
    LoRALinear,
    find_language_lora_targets,
    inject_language_lora,
    set_lora_dropout,
)
from qwen35_pvchat.remoh_losses import AdaptiveReMoHLoss


class _NestedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.q_proj = nn.Linear(4, 4)
        self.model.language_model.mlp = nn.Module()
        self.model.language_model.mlp.up_proj = nn.Linear(4, 8)
        self.model.visual = nn.Module()
        self.model.visual.q_proj = nn.Linear(4, 4)
        self.model.language_model.router = nn.Linear(4, 2)
        self.lm_head = nn.Linear(4, 10)


class Qwen35ModelingTest(unittest.TestCase):
    def test_lora_linear_starts_as_exact_identity_update(self):
        torch.manual_seed(7)
        base = nn.Linear(4, 3)
        inputs = torch.randn(2, 4)
        expected = base(inputs).detach()

        layer = LoRALinear(base, rank=2, alpha=4, dropout=0.0)
        actual = layer(inputs)

        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(layer.base_layer.weight.requires_grad)
        self.assertTrue(layer.lora_A.requires_grad)
        self.assertTrue(layer.lora_B.requires_grad)
        self.assertTrue(torch.count_nonzero(layer.lora_B) == 0)

        actual.sum().backward()
        self.assertIsNotNone(layer.lora_B.grad)
        self.assertGreater(layer.lora_B.grad.abs().sum().item(), 0.0)

    def test_lora_targets_only_language_linears(self):
        targets = find_language_lora_targets(_NestedModel())

        self.assertIn("model.language_model.q_proj", targets)
        self.assertIn("model.language_model.mlp.up_proj", targets)
        self.assertNotIn("model.visual.q_proj", targets)
        self.assertNotIn("model.language_model.router", targets)
        self.assertNotIn("lm_head", targets)

    def test_lora_injection_only_replaces_language_targets(self):
        model = _NestedModel()

        replaced = inject_language_lora(model, rank=2, alpha=4, dropout=0.0)

        self.assertEqual(
            replaced,
            ["model.language_model.mlp.up_proj", "model.language_model.q_proj"],
        )
        self.assertIsInstance(model.model.language_model.q_proj, LoRALinear)
        self.assertIsInstance(model.model.language_model.mlp.up_proj, LoRALinear)
        self.assertIsInstance(model.model.visual.q_proj, nn.Linear)
        self.assertIsInstance(model.model.language_model.router, nn.Linear)
        self.assertIsInstance(model.lm_head, nn.Linear)

    def test_stage3_can_disable_lora_dropout_without_changing_weights(self):
        model = _NestedModel()
        inject_language_lora(model, rank=2, alpha=4, dropout=0.05)

        count = set_lora_dropout(model, probability=0.0)

        modules = [module for module in model.modules() if isinstance(module, LoRALinear)]
        self.assertEqual(count, len(modules))
        self.assertTrue(all(module.dropout.p == 0.0 for module in modules))

    def test_trainable_checkpoint_ignores_frozen_parameters(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model[1].weight.requires_grad_(True)
        expected = model[1].weight.detach().clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trainable.pt"
            save_trainable_state(model, path)
            with torch.no_grad():
                model[1].weight.zero_()
            result = load_trainable_state(model, path)

        self.assertTrue(torch.equal(model[1].weight, expected))
        self.assertEqual(result["loaded_keys"], 1)

    def test_optimizer_checkpoint_restores_adam_moments_and_step(self):
        torch.manual_seed(17)
        source_model = nn.Linear(3, 2)
        source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=1e-4)
        source_model(torch.ones(2, 3)).sum().backward()
        source_optimizer.step()
        expected = source_optimizer.state_dict()

        target_model = nn.Linear(3, 2)
        target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=1e-4)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            checkpoint_dir.mkdir()
            torch.save(expected, checkpoint_dir / "optimizer.pt")

            result = load_optimizer_state(target_optimizer, checkpoint_dir)

        actual = target_optimizer.state_dict()
        self.assertEqual(result["state_entries"], len(expected["state"]))
        self.assertEqual(actual["param_groups"], expected["param_groups"])
        for parameter_id, expected_state in expected["state"].items():
            actual_state = actual["state"][parameter_id]
            self.assertTrue(torch.equal(actual_state["step"], expected_state["step"]))
            self.assertTrue(torch.equal(actual_state["exp_avg"], expected_state["exp_avg"]))
            self.assertTrue(torch.equal(actual_state["exp_avg_sq"], expected_state["exp_avg_sq"]))


class ReMoHLossTest(unittest.TestCase):
    def test_identity_gates_have_no_head_activation_enhancement_penalty(self):
        controller = AdaptiveReMoHLoss(target_active_ratio=0.5, initial_spr_weight=1e-8)
        logits = torch.ones(2, 3, 4, requires_grad=True)
        gates = torch.relu(logits)

        result = controller.from_gates([(logits, gates)])

        self.assertAlmostEqual(result.active_ratio.item(), 1.0, places=5)
        self.assertAlmostEqual(result.hae_loss.item(), 0.0, places=6)
        self.assertGreater(result.spr_loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
