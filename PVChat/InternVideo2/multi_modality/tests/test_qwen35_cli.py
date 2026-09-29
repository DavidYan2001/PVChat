import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from qwen35_pvchat.cli import build_sft_parser, build_stage3_parser
from qwen35_pvchat import trainer


class Qwen35CliTest(unittest.TestCase):
    def test_evaluation_runs_by_default_and_can_be_explicitly_skipped(self):
        parser = build_sft_parser(stage=1)
        required = [
            "--model_path",
            "/tmp/model",
            "--sks_name",
            "<Sheldon>",
            "--train_json",
            "/tmp/train.json",
            "--test_json",
            "/tmp/test.json",
            "--output_dir",
            "/tmp/output",
        ]

        default_args = parser.parse_args(required)
        skipped_args = parser.parse_args(required + ["--skip_evaluation"])

        self.assertFalse(default_args.skip_evaluation)
        self.assertTrue(skipped_args.skip_evaluation)
        self.assertEqual(default_args.eval_batch_size, 4)

    def test_stage2_registers_optimizer_resume_and_epoch_offset(self):
        parser = build_sft_parser(stage=2)
        args = parser.parse_args(
            [
                "--model_path",
                "/tmp/model",
                "--checkpoint_path",
                "/tmp/stage2/checkpoint",
                "--sks_name",
                "<Sheldon>",
                "--train_json",
                "/tmp/train.json",
                "--test_json",
                "/tmp/test.json",
                "--output_dir",
                "/tmp/output",
                "--resume_optimizer",
                "--epoch_offset",
                "1",
            ]
        )

        self.assertTrue(args.resume_optimizer)
        self.assertEqual(args.epoch_offset, 1)

    def test_stage2_registers_evaluation_only_mode(self):
        parser = build_sft_parser(stage=2)
        args = parser.parse_args(
            [
                "--model_path",
                "/tmp/model",
                "--checkpoint_path",
                "/tmp/stage2/checkpoint",
                "--sks_name",
                "<Sheldon>",
                "--train_json",
                "/tmp/train.json",
                "--test_json",
                "/tmp/test.json",
                "--output_dir",
                "/tmp/output",
                "--eval_only",
            ]
        )

        self.assertTrue(args.eval_only)

    def test_evaluation_only_skips_dataset_optimizer_training_and_checkpoint_save(self):
        args = SimpleNamespace(
            stage=2,
            eval_only=True,
            epoch_offset=0,
            resume_optimizer=False,
            checkpoint_path="/tmp/stage2/checkpoint",
            model_path="/tmp/model",
            sks_name="<Sheldon>",
            num_detail_tokens=16,
            remoh_layers="7,11,15,19",
            routed_heads="3,7,11,15",
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            attn_implementation="sdpa",
            disable_gradient_checkpointing=False,
            video_min_tokens=4,
            video_max_tokens=768,
            output_dir="/tmp/output",
            test_json="/tmp/test.json",
            eval_max_new_tokens=96,
            eval_batch_size=4,
            skip_evaluation=False,
            skip_metrics=False,
            judge_backend="dashscope",
            judge_model="qwen3.7-max",
            judge_fallback_models="fallback",
            api_num_workers=10,
        )
        context = SimpleNamespace(device="cpu", is_main=True, local_rank=0)
        bundle = SimpleNamespace(
            model=object(),
            processor=object(),
            personalized_tokens=["<Sheldon>"],
        )

        with (
            patch.object(trainer, "load_qwen35_pvchat_model", return_value=bundle),
            patch.object(trainer, "trainable_parameter_summary", return_value={}),
            patch.object(trainer, "PVChatSFTDataset") as dataset,
            patch.object(trainer, "build_optimizer") as build_optimizer,
            patch.object(trainer, "save_checkpoint") as save_checkpoint,
            patch.object(trainer, "run_distributed_evaluation") as evaluate,
            patch.object(trainer, "run_metrics") as metrics,
            patch.object(trainer, "barrier"),
        ):
            result = trainer.run_sft_stage(args, context)

        dataset.assert_not_called()
        build_optimizer.assert_not_called()
        save_checkpoint.assert_not_called()
        evaluate.assert_called_once()
        self.assertEqual(evaluate.call_args.kwargs["batch_size"], 4)
        metrics.assert_called_once()
        self.assertEqual(result, Path(args.checkpoint_path))

    def test_stage3_defaults_to_full_dataset(self):
        parser = build_stage3_parser()
        args = parser.parse_args(
            [
                "--model_path",
                "/tmp/model",
                "--checkpoint_path",
                "/tmp/stage2/checkpoint",
                "--sks_name",
                "<Sheldon>",
                "--train_json",
                "/tmp/train.json",
                "--test_json",
                "/tmp/test.json",
                "--output_dir",
                "/tmp/output",
            ]
        )

        self.assertEqual(args.max_steps_per_epoch, 0)
        self.assertEqual(args.ig_identity_margin, 1.0)
        self.assertEqual(args.ig_soft_clip, 0.25)
        self.assertEqual(args.rollout_buffer_size, 4)
        self.assertEqual(args.drift_beta, 0.02)
        self.assertEqual(args.judge_backend, "local_qwen35")
        self.assertTrue(args.local_judge_model_path.endswith("models/Qwen3.5-35B-A3B"))
        self.assertEqual(args.local_judge_batch_size, 64)
        self.assertEqual(args.local_judge_es_items_per_prompt, 1)
        self.assertEqual(args.local_judge_dc_items_per_prompt, 10)


if __name__ == "__main__":
    unittest.main()
