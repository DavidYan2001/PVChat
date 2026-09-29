import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from qwen35_pvchat.distributed import distributed_context_from_env
from qwen35_pvchat.identity_gated_dynamic_policy import decide_identity_gated_rollout
from qwen35_pvchat.stage3_experiment import _all_reduce_active_count, active_loss_scale


def _ig_gloo_worker(rank, init_path, queue):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
    )
    try:
        context = SimpleNamespace(
            device=torch.device("cpu"),
            distributed=True,
            world_size=2,
        )
        active = decide_identity_gated_rollout([0.1, 0.9], [-1.0, 1.0])
        skipped = decide_identity_gated_rollout([0.5] * 8, [-1.0] * 8)
        local_decisions = (active, skipped) if rank == 0 else (skipped, active)
        values = []
        for decision in local_decisions:
            active_count = _all_reduce_active_count(context, decision.should_update)
            values.append(
                (
                    decision.observed_count,
                    decision.should_update,
                    active_count,
                    active_loss_scale(2, active_count, decision.should_update),
                )
            )
        queue.put((rank, values))
    finally:
        dist.destroy_process_group()


class DistributedContextTest(unittest.TestCase):
    def test_reads_torchrun_rank_variables(self):
        context = distributed_context_from_env(
            {"WORLD_SIZE": "4", "RANK": "2", "LOCAL_RANK": "2"}
        )

        self.assertTrue(context.distributed)
        self.assertEqual(context.world_size, 4)
        self.assertEqual(context.rank, 2)
        self.assertEqual(context.local_rank, 2)
        self.assertFalse(context.is_main)

    def test_defaults_to_single_process(self):
        context = distributed_context_from_env({})

        self.assertFalse(context.distributed)
        self.assertTrue(context.is_main)
        self.assertEqual(context.world_size, 1)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is unavailable")
    def test_ig_different_rollout_and_active_branches_keep_two_ranks_synchronized(self):
        with tempfile.TemporaryDirectory() as temporary:
            init_path = str(Path(temporary) / "gloo_init")
            context = mp.get_context("spawn")
            queue = context.Queue()
            processes = [
                context.Process(target=_ig_gloo_worker, args=(rank, init_path, queue))
                for rank in range(2)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=30)
                self.assertFalse(process.is_alive(), "Gloo worker did not finish")
                self.assertEqual(process.exitcode, 0)
            results = dict(queue.get(timeout=5) for _ in range(2))

        self.assertEqual([item[0] for item in results[0]], [2, 8])
        self.assertEqual([item[0] for item in results[1]], [8, 2])
        for rank_values in results.values():
            self.assertEqual([item[2] for item in rank_values], [1, 1])
            self.assertEqual(sorted(item[3] for item in rank_values), [0.0, 2.0])


if __name__ == "__main__":
    unittest.main()
