"""torchrun/DDP的最小封装。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0

    @property
    def distributed(self):
        return self.world_size > 1

    @property
    def is_main(self):
        return self.rank == 0

    @property
    def device(self):
        return torch.device("cuda", self.local_rank) if torch.cuda.is_available() else torch.device("cpu")


def distributed_context_from_env(environment=None) -> DistributedContext:
    environment = os.environ if environment is None else environment
    return DistributedContext(
        world_size=int(environment.get("WORLD_SIZE", 1)),
        rank=int(environment.get("RANK", 0)),
        local_rank=int(environment.get("LOCAL_RANK", 0)),
    )


def initialize_distributed() -> DistributedContext:
    context = distributed_context_from_env()
    if torch.cuda.is_available():
        torch.cuda.set_device(context.local_rank)
    if context.distributed and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(hours=2),
        )
    return context


def barrier(context: DistributedContext):
    if context.distributed:
        dist.barrier()


def cleanup_distributed(context: DistributedContext):
    if context.distributed and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
