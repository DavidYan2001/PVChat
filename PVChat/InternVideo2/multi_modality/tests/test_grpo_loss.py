import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler

import finetune_internvideo_REOMH_one_person2_stage as stage2
import finetune_internvideo_REOMH_one_person3_grpo as stage3
from finetune_internvideo_REOMH_one_person3_grpo import grpo_policy_loss, masked_sequence_logprobs


class GrpoLossTest(unittest.TestCase):
    def test_relative_rollout_video_uses_train_json_dataset_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "datasets" / "cekebv-hq"
            train_json = dataset_root / "Sheldon" / "train.json"
            video_path = dataset_root / "35666" / "sample.mp4"
            rollout_path = (
                dataset_root
                / "Sheldon"
                / "finetune_outputs_grpo"
                / "rollouts"
                / "epoch_2"
                / "train_rollouts.jsonl"
            )
            train_json.parent.mkdir(parents=True)
            train_json.write_text('{"videos": []}', encoding="utf-8")
            video_path.parent.mkdir(parents=True)
            video_path.touch()
            rollout_path.parent.mkdir(parents=True)

            resolved = stage3.resolve_video_path(
                "35666/sample.mp4",
                rollout_path,
                train_json=train_json,
            )

        self.assertEqual(resolved, str(video_path))

    def test_train_epoch_honors_max_steps_on_model_device(self):
        class TinyPolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.logit_bias = nn.Parameter(torch.zeros(3))
                self.calls = 0

            def forward(self, input_ids, **kwargs):
                self.calls += 1
                logits = self.logit_bias.view(1, 1, -1).expand(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    -1,
                )
                return SimpleNamespace(logits=logits)

        model = TinyPolicy()
        ref_model = TinyPolicy()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        batch = {
            "input_ids": torch.tensor([[0, 1, 2]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "video": torch.zeros((1, 1)),
            "labels": torch.tensor([[-100, 1, 2]]),
            "video_idx": torch.zeros((1, 3), dtype=torch.bool),
            "old_logprobs": torch.tensor([-1.0]),
            "advantages": torch.tensor([1.0]),
            "rewards": torch.tensor([1.0]),
        }
        args = SimpleNamespace(
            clip_range=0.2,
            kl_beta=0.02,
            max_grad_norm=1.0,
            max_train_steps=2,
        )
        context = stage3.DistributedContext(world_size=1, rank=1, local_rank=0)

        with patch.object(stage3, "restore_non_person_embeddings"):
            stage3.train_one_epoch(
                model,
                ref_model,
                [batch, batch, batch, batch],
                optimizer,
                tokenizer=MagicMock(),
                sks_tokens=["<Sheldon>"],
                prefix_tokens=[],
                orig_embeds=torch.empty(0),
                args=args,
                epoch=0,
                context=context,
            )

        self.assertEqual(model.calls, 2)
        self.assertEqual(ref_model.calls, 2)

    def test_four_gpu_context_is_parsed_from_torchrun_environment(self):
        context = stage3.distributed_context_from_env({
            "WORLD_SIZE": "4",
            "RANK": "2",
            "LOCAL_RANK": "2",
        })

        self.assertTrue(context.distributed)
        self.assertEqual(context.world_size, 4)
        self.assertEqual(context.rank, 2)
        self.assertEqual(context.local_rank, 2)
        self.assertFalse(context.is_main)

    def test_distributed_initialization_selects_local_gpu_and_long_nccl_timeout(self):
        context = stage3.DistributedContext(world_size=4, rank=3, local_rank=3)

        with (
            patch.object(stage3, "distributed_context_from_env", return_value=context),
            patch.object(stage3.torch.cuda, "set_device") as set_device,
            patch.object(stage3.dist, "init_process_group") as init_process_group,
        ):
            result = stage3.initialize_distributed()

        self.assertEqual(result, context)
        set_device.assert_called_once_with(3)
        self.assertEqual(init_process_group.call_args.kwargs["backend"], "nccl")
        self.assertEqual(init_process_group.call_args.kwargs["init_method"], "env://")
        self.assertEqual(init_process_group.call_args.kwargs["timeout"].total_seconds(), 7200)

    def test_distributed_loader_partitions_dataset_by_rank(self):
        context = stage3.distributed_context_from_env({
            "WORLD_SIZE": "4",
            "RANK": "1",
            "LOCAL_RANK": "1",
        })
        args = SimpleNamespace(batch_size=1, num_workers=0)

        loader, sampler = stage3.build_grpo_dataloader(list(range(12)), args, context)

        self.assertIsInstance(sampler, DistributedSampler)
        self.assertEqual(sampler.num_replicas, 4)
        self.assertEqual(sampler.rank, 1)
        self.assertEqual(len(loader), 3)

    def test_only_optimizer_owned_parameters_remain_trainable(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
        optimizer = torch.optim.AdamW([model[0].weight], lr=1e-3)

        trainable_count = stage3.limit_trainable_parameters_to_optimizer(model, optimizer)

        self.assertTrue(model[0].weight.requires_grad)
        self.assertFalse(model[0].bias.requires_grad)
        self.assertFalse(model[1].weight.requires_grad)
        self.assertFalse(model[1].bias.requires_grad)
        self.assertEqual(trainable_count, model[0].weight.numel())

    def test_get_args_registers_max_train_steps(self):
        argv = [
            "stage3.py",
            "--rollout_jsonl", "rollouts.jsonl",
            "--model_path", "model",
            "--sks_name", "<Sheldon>",
            "--output_dir", "output",
            "--max_train_steps", "200",
        ]

        with patch.object(sys, "argv", argv):
            args = stage3.get_args()

        self.assertEqual(args.max_train_steps, 200)

    def test_get_args_registers_iterative_epoch_and_reference_checkpoint(self):
        argv = [
            "stage3.py",
            "--rollout_jsonl", "rollouts.jsonl",
            "--model_path", "model",
            "--checkpoint_path", "policy_epoch_1",
            "--reference_checkpoint_path", "stage2_final",
            "--sks_name", "<Sheldon>",
            "--output_dir", "output",
            "--epoch_offset", "1",
        ]

        with patch.object(sys, "argv", argv):
            args = stage3.get_args()

        self.assertEqual(args.epoch_offset, 1)
        self.assertEqual(args.reference_checkpoint_path, "stage2_final")

    def test_reference_checkpoint_defaults_to_policy_checkpoint(self):
        explicit = SimpleNamespace(
            checkpoint_path="policy_epoch_1",
            reference_checkpoint_path="stage2_final",
        )
        fallback = SimpleNamespace(
            checkpoint_path="stage2_final",
            reference_checkpoint_path=None,
        )

        self.assertEqual(stage3.resolve_reference_checkpoint(explicit), "stage2_final")
        self.assertEqual(stage3.resolve_reference_checkpoint(fallback), "stage2_final")

    def test_checkpoint_saves_and_restores_optimizer_state(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            stage3.save_checkpoint(
                model,
                tokenizer=MagicMock(),
                config=MagicMock(),
                sks_tokens=["<Sheldon>"],
                prefix_tokens=[],
                output_dir=tmpdir,
                optimizer=optimizer,
            )
            self.assertTrue((Path(tmpdir) / "optimizer.pt").exists())

            restored_optimizer = torch.optim.AdamW(model.parameters(), lr=9e-3)
            restored = stage3.restore_optimizer_state(restored_optimizer, tmpdir)

        self.assertTrue(restored)
        self.assertEqual(restored_optimizer.param_groups[0]["lr"], 1e-3)

    def test_global_epoch_number_applies_offset(self):
        self.assertEqual(stage3.global_epoch_number(local_epoch=0, epoch_offset=2), 3)
        self.assertEqual(stage3.global_epoch_number(local_epoch=1, epoch_offset=2), 4)

    def test_masked_sequence_logprobs_averages_only_answer_tokens(self):
        logits = torch.tensor([[
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
            [5.0, 0.0, 0.0],
        ]])
        input_ids = torch.tensor([[0, 1, 2]])
        labels = torch.tensor([[-100, 1, 2]])

        seq_logprob, token_count = masked_sequence_logprobs(logits, input_ids, labels)

        self.assertEqual(token_count.item(), 2)
        self.assertLess(abs(seq_logprob.item()), 0.05)

    def test_grpo_policy_loss_prefers_positive_advantage_higher_logprob(self):
        logprobs = torch.tensor([-0.1, -2.0])
        old_logprobs = torch.tensor([-1.0, -1.0])
        advantages = torch.tensor([1.0, -1.0])

        loss = grpo_policy_loss(logprobs, old_logprobs, advantages, clip_range=0.2)

        self.assertLess(loss.item(), 0.0)

    def test_shared_test_function_accepts_explicit_config_and_output_dir(self):
        model = MagicMock()
        config = SimpleNamespace(sks_name="<Sheldon>")
        test_path = "/tmp/Sheldon/<Sheldon>test.json"

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(stage2, "PersonalizedVideoDataset", return_value=[]) as dataset_class:
                results = stage2.test(
                    model,
                    tokenizer=MagicMock(),
                    test_path=test_path,
                    device=torch.device("cpu"),
                    config=config,
                    output_dir=tmpdir,
                    sample_count=0,
                    save_results=True,
                )

            self.assertTrue((Path(tmpdir) / "test_results_RE_MOH_Sheldon_1person.json").exists())

        self.assertEqual(results, [])
        dataset_class.assert_called_once_with(
            json_path=test_path,
            tokenizer=unittest.mock.ANY,
            device=torch.device("cpu"),
            config=config,
            split="test",
        )

    def test_grpo_saves_epoch_checkpoint_before_running_evaluation(self):
        events = []
        model = MagicMock()
        config = SimpleNamespace()
        args = SimpleNamespace(
            batch_size=1,
            eval_samples=20,
            eval_seed=42,
            num_epochs=1,
            num_workers=0,
            output_dir="",
            rollout_jsonl="rollouts.jsonl",
            save_epochs=1,
            sks_name="<Sheldon>",
            test_json="test.json",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            args.output_dir = tmpdir
            with (
                patch.object(stage3, "get_args", return_value=args),
                patch.object(
                    stage3,
                    "load_trainable_model",
                    return_value=(model, MagicMock(), config, ["<Sheldon>"], []),
                ),
                patch.object(stage3, "clone_reference_model", return_value=MagicMock()),
                patch.object(stage3, "GrpoRolloutDataset", return_value=[]),
                patch.object(stage3, "DataLoader", return_value=[]),
                patch.object(stage3, "build_optimizer", return_value=MagicMock()),
                patch.object(stage3, "train_one_epoch", return_value=0.0),
                patch.object(
                    stage3,
                    "save_checkpoint",
                    side_effect=lambda *call_args, **kwargs: events.append(
                        ("save", Path(call_args[-1]).name)
                    ),
                ),
                patch.object(
                    stage3,
                    "test",
                    side_effect=lambda *call_args, **kwargs: events.append(
                        ("test", kwargs.get("stage_name"))
                    ),
                ),
            ):
                stage3.main()

        self.assertLess(
            events.index(("save", "checkpoint_epoch_1")),
            events.index(("test", "After GRPO epoch 1")),
        )


if __name__ == "__main__":
    unittest.main()
