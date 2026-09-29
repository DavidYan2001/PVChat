import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import run_pvchat_checkpoint_test as checkpoint_test


class CheckpointTestEntryPointTest(unittest.TestCase):
    def test_parser_registers_required_checkpoint_test_paths(self):
        argv = [
            "test_pvchat_checkpoint.py",
            "--model_path", "base_model",
            "--checkpoint_path", "policy_epoch_2",
            "--sks_name", "<Sheldon>",
            "--test_json", "test.json",
            "--output_dir", "evaluations/epoch_2",
            "--epoch", "2",
        ]

        with patch.object(sys, "argv", argv):
            args = checkpoint_test.get_args()

        self.assertEqual(args.checkpoint_path, "policy_epoch_2")
        self.assertEqual(args.epoch, 2)

    def test_full_test_uses_all_samples_and_epoch_output_directory(self):
        args = SimpleNamespace(
            model_path="base_model",
            checkpoint_path="policy_epoch_2",
            sks_name="<Sheldon>",
            test_json="test.json",
            output_dir="evaluations/epoch_2",
            epoch=2,
            eval_seed=42,
        )
        model = MagicMock()
        tokenizer = MagicMock()
        config = MagicMock()

        with (
            patch.object(
                checkpoint_test,
                "load_trainable_model",
                return_value=(model, tokenizer, config, ["<Sheldon>"], []),
            ),
            patch.object(checkpoint_test, "run_test", return_value=[]) as run_test,
            patch.object(checkpoint_test.torch.cuda, "empty_cache"),
        ):
            checkpoint_test.run_full_test(args)

        run_test.assert_called_once_with(
            model,
            tokenizer,
            "test.json",
            torch.device("cuda", 0),
            config=config,
            output_dir="evaluations/epoch_2",
            sample_count=None,
            stage_name="After GRPO epoch 2 full test",
            save_results=True,
            seed=42,
        )


if __name__ == "__main__":
    unittest.main()
