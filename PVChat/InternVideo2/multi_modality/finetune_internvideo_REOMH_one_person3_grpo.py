import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import rnn as rnn_utils
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from finetune_internvideo_REOMH_one_person2_stage import (
    IMG_TOKEN,
    VID_TOKEN,
    build_input_ids,
    ensure_complete_config,
    load_video,
    select_eval_indices,
    test,
)
from generate_grpo_rollouts import load_model_state_compat  # noqa: F401  # 兼容旧引用
from internvideo_compact_checkpoint import load_checkpoint_state, save_compact_checkpoint


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


def distributed_context_from_env(env=None):
    env = os.environ if env is None else env
    world_size = max(int(env.get("WORLD_SIZE", "1")), 1)
    rank = int(env.get("RANK", "0")) if world_size > 1 else 0
    local_rank = int(env.get("LOCAL_RANK", "0")) if world_size > 1 else 0
    return DistributedContext(world_size=world_size, rank=rank, local_rank=local_rank)


def initialize_distributed():
    context = distributed_context_from_env()
    if context.distributed:
        torch.cuda.set_device(context.local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(hours=2),
        )
    return context


def distributed_barrier(context):
    if context.distributed and dist.is_initialized():
        dist.barrier()


def cleanup_distributed(context):
    if context.distributed and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def global_epoch_number(local_epoch, epoch_offset=0):
    return int(epoch_offset) + int(local_epoch) + 1


def resolve_reference_checkpoint(args):
    return getattr(args, "reference_checkpoint_path", None) or getattr(args, "checkpoint_path", None)



class GroupBatchSampler(torch.utils.data.Sampler):
    """One batch = all candidates of one rollout group (standard GRPO: one update per group).

    Groups are shuffled each epoch; the candidates inside a group stay together so the
    group-relative advantages are applied within a single optimizer step.
    """

    def __init__(self, dataset):
        groups = {}
        for index, sample in enumerate(dataset.samples):
            groups.setdefault(sample["record_idx"], []).append(index)
        self.groups = list(groups.values())

    def __iter__(self):
        order = list(range(len(self.groups)))
        random.shuffle(order)
        for position in order:
            yield list(self.groups[position])

    def __len__(self):
        return len(self.groups)


def build_grpo_dataloader(dataset, args, context):
    sampler = None
    if getattr(args, "group_batch", False):
        if context.distributed:
            raise ValueError("--group_batch is implemented for single-process training only")
        loader = DataLoader(
            dataset,
            batch_sampler=GroupBatchSampler(dataset),
            collate_fn=collate_grpo,
            num_workers=args.num_workers,
        )
        return loader, None
    if context.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collate_grpo,
        num_workers=args.num_workers,
    )
    return loader, sampler


def limit_trainable_parameters_to_optimizer(model, optimizer):
    optimizer_param_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }
    trainable_count = 0
    for param in model.parameters():
        trainable = id(param) in optimizer_param_ids
        param.requires_grad_(trainable)
        if trainable:
            trainable_count += param.numel()
    return trainable_count


def wrap_distributed_model(model, context):
    if not context.distributed:
        return model
    return DistributedDataParallel(
        model,
        device_ids=[context.local_rank],
        output_device=context.local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
    )


def masked_sequence_logprobs(logits, input_ids, labels):
    """Average next-token log-prob over answer tokens only."""
    shifted_logits = logits[:, :-1, :].float()
    shifted_input_ids = input_ids[:, 1:].long()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe_targets = shifted_input_ids.masked_fill(~mask, 0)
    token_logprobs = F.log_softmax(shifted_logits, dim=-1).gather(
        -1, safe_targets.unsqueeze(-1)
    ).squeeze(-1)
    token_logprobs = token_logprobs.masked_fill(~mask, 0.0)
    token_count = mask.sum(dim=1).clamp_min(1)
    return token_logprobs.sum(dim=1) / token_count, token_count


def grpo_policy_loss(logprobs, old_logprobs, advantages, clip_range=0.2):
    ratio = torch.exp(logprobs - old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    return -torch.min(unclipped, clipped).mean()


def read_rollout_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_video_path(video_path, rollout_path, train_json=None):
    if os.path.isabs(video_path):
        return video_path
    base_dirs = []
    if train_json:
        train_json_dir = Path(train_json).resolve().parent
        base_dirs.extend((train_json_dir, train_json_dir.parent))
    rollout_dir = Path(rollout_path).resolve().parent
    base_dirs.extend((rollout_dir, rollout_dir.parent, rollout_dir.parent.parent))
    for base_dir in base_dirs:
        candidate = base_dir / video_path
        if candidate.exists():
            return str(candidate)
    return video_path


class GrpoRolloutDataset(Dataset):
    def __init__(self, rollout_jsonl, tokenizer, config, max_length=512, train_json=None):
        self.records = read_rollout_jsonl(rollout_jsonl)
        self.samples = []
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = max_length
        self.rollout_jsonl = rollout_jsonl

        for record_idx, record in enumerate(self.records):
            for candidate in record.get("candidates", []):
                self.samples.append({
                    "record_idx": record_idx,
                    "group_id": record.get("group_id"),
                    "video_path": resolve_video_path(
                        record.get("video_path", ""),
                        rollout_jsonl,
                        train_json=train_json,
                    ),
                    "question": record.get("question", ""),
                    "gold_answer": record.get("gold_answer", ""),
                    "answer": candidate.get("answer", ""),
                    "reward": float(candidate.get("reward", 0.0)),
                    "advantage": float(candidate.get("advantage", 0.0)),
                    "old_logprob": candidate.get("old_logprob"),
                })

    def __len__(self):
        return len(self.samples)

    def _build_conversation(self, video_tensor, question, answer):
        conversation = "[INST] "
        if video_tensor.shape[1] == 1:
            ilen = video_tensor.shape[0]
            conversation += ("<Image>" + IMG_TOKEN + "</Image>") * ilen
        else:
            ilen = video_tensor.shape[1]
            conversation += ("<Video>" + VID_TOKEN + "</Video>") * ilen
        conversation += "[/INST] "
        sks_and_tokens = (
            f"[{self.config.sks_name}]"
            + "is"
            + "".join([f"<token{i}>" for i in range(self.config.model_config["bridge"]["num_prefix_token"])])
            + " "
        )
        if self.config.model_config["bridge"]["num_prefix_token"] == 0:
            sks_and_tokens = f"[{self.config.sks_name}]"
        conversation += "[INST]" + sks_and_tokens + question + "[/INST]"
        conversation += answer + "</s>"
        return conversation

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_tensor = load_video(
            sample["video_path"],
            num_segments=8,
            return_msg=False,
            resolution=224,
            hd_num=6,
        )
        conversation = self._build_conversation(video_tensor, sample["question"], sample["answer"])
        tokenized = build_input_ids(
            self.tokenizer,
            conversation,
            max_length=self.max_length,
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            video_placeholder="[<VID_PLH>]",
        )
        labels = tokenized["input_ids"].clone()
        inst_tokens = self.tokenizer.encode("[/INST]")
        inst_end_indices = torch.where(labels == inst_tokens[-1])[0]
        if len(inst_end_indices) >= 2:
            labels[:inst_end_indices[1] + 1] = -100
        labels[tokenized["index"]] = -100
        old_logprob = sample["old_logprob"]
        return {
            "video": video_tensor,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "video_idx": tokenized["index"],
            "labels": labels.to(torch.int64),
            "advantage": torch.tensor(sample["advantage"], dtype=torch.float32),
            "reward": torch.tensor(sample["reward"], dtype=torch.float32),
            "old_logprob": torch.tensor(float(old_logprob), dtype=torch.float32) if old_logprob is not None else torch.tensor(float("nan")),
            "question": sample["question"],
            "gold_answer": sample["gold_answer"],
            "answer": sample["answer"],
            "video_path": sample["video_path"],
        }


def collate_grpo(batch):
    input_ids = rnn_utils.pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=0)
    attention_mask = rnn_utils.pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
    video_idx = rnn_utils.pad_sequence([item["video_idx"] for item in batch], batch_first=True, padding_value=0)
    labels = rnn_utils.pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100).to(torch.int64)
    videos = torch.stack([item["video"].squeeze(0) if item["video"].shape[0] == 1 else item["video"] for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "video_idx": video_idx,
        "labels": labels,
        "video": videos,
        "advantages": torch.stack([item["advantage"] for item in batch]),
        "rewards": torch.stack([item["reward"] for item in batch]),
        "old_logprobs": torch.stack([item["old_logprob"] for item in batch]),
        "question": [item["question"] for item in batch],
        "gold_answer": [item["gold_answer"] for item in batch],
        "answer": [item["answer"] for item in batch],
        "video_path": [item["video_path"] for item in batch],
    }


def load_trainable_model(args):
    config_source = args.checkpoint_path if args.checkpoint_path else args.model_path
    config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
    config.sks_name = args.sks_name
    config.model_config["sks_name"] = args.sks_name
    config = ensure_complete_config(config, os.path.join(args.model_path, "config.json"))

    tokenizer = AutoTokenizer.from_pretrained(config_source, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda()

    sks_tokens = [args.sks_name]
    prefix_tokens = [f"<token{i}>" for i in range(config.model_config["bridge"]["num_prefix_token"])]
    tokenizer.add_tokens(sks_tokens + prefix_tokens)
    model.tokenizer = tokenizer
    model.lm.resize_token_embeddings(len(tokenizer))
    if hasattr(model.lm, "lm_head"):
        old_lm_head = model.lm.lm_head
        if old_lm_head.out_features != len(tokenizer):
            model.lm.lm_head = nn.Linear(old_lm_head.in_features, len(tokenizer), bias=False).to(model.lm.device)
            model.lm.lm_head.weight.data[:old_lm_head.out_features] = old_lm_head.weight.data
    model._pvchat_delta_spec = None
    if args.checkpoint_path:
        # 紧凑delta与旧式完整state自动分派；spec挂在模型上供保存时提取同一键集。
        model._pvchat_delta_spec = load_checkpoint_state(model, args.checkpoint_path)
    return model, tokenizer, config, sks_tokens, prefix_tokens



def install_reference_vision_cache(ref_model, max_entries=256):
    """Exact per-video memo of the frozen reference model's vision path.

    The reference model is in eval() mode with every parameter frozen, so its
    encode_vision output is a deterministic function of the input frames. The
    same video appears once per candidate (about 100x per epoch); caching its
    encoder output (keyed by a fingerprint, verified with torch.equal) removes
    that repeated ViT+QFormer compute without changing any value the loss sees.
    """
    import torch

    orig_encode = ref_model.encode_vision
    store = {}
    order = []

    def fingerprint(image):
        flat = image.reshape(-1)
        stride = max(1, flat.numel() // 4096)
        probe = flat[::stride].float()
        return (tuple(image.shape), str(image.dtype), round(float(probe.sum()), 3), round(float(probe[::7].sum()), 3))

    def _clone(value):
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, (tuple, list)):
            return type(value)(_clone(v) for v in value)
        return value

    def cached_encode_vision(image, instruction=None):
        if instruction is not None:
            return orig_encode(image, instruction)
        key = fingerprint(image)
        hit = store.get(key)
        if hit is not None and torch.equal(hit["input"], image):
            if os.environ.get("GRPO_REF_CACHE_VERIFY"):
                fresh = orig_encode(image, instruction)
                a = fresh[0] if isinstance(fresh, (tuple, list)) else fresh
                b = hit["output"][0] if isinstance(hit["output"], (tuple, list)) else hit["output"]
                diff = float((a.float() - b.float()).abs().max())
                logging.info(f"[ref-cache verify] hit key={key[0]} max|diff|={diff:.3e}")
                assert diff < 1e-3, f"reference vision cache mismatch: {diff}"
            return _clone(hit["output"])
        out = orig_encode(image, instruction)
        store[key] = {"input": image.detach().clone(), "output": _clone(out)}
        order.append(key)
        while len(order) > max_entries:
            store.pop(order.pop(0), None)
        return out

    ref_model.encode_vision = cached_encode_vision
    return store


def clone_reference_model(args, config, tokenizer, sks_tokens, prefix_tokens):
    ref_model = AutoModel.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda()
    ref_model.tokenizer = tokenizer
    ref_model.lm.resize_token_embeddings(len(tokenizer))
    if hasattr(ref_model.lm, "lm_head"):
        old_lm_head = ref_model.lm.lm_head
        if old_lm_head.out_features != len(tokenizer):
            ref_model.lm.lm_head = nn.Linear(old_lm_head.in_features, len(tokenizer), bias=False).to(ref_model.lm.device)
            ref_model.lm.lm_head.weight.data[:old_lm_head.out_features] = old_lm_head.weight.data
    reference_checkpoint = resolve_reference_checkpoint(args)
    if reference_checkpoint:
        load_checkpoint_state(ref_model, reference_checkpoint)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    return ref_model


def build_optimizer(model, config, tokenizer, sks_tokens, prefix_tokens, args):
    moh_params = []
    qformer_params = []
    for name, param in model.named_parameters():
        if "router" in name or "alpha_proj" in name:
            moh_params.append(param)
    for name, param in model.qformer.named_parameters():
        if "router" not in name and "alpha_proj" not in name:
            qformer_params.append(param)

    trainable_params = [
        {"params": model.personal_query_tokens, "lr": args.personal_token_lr},
        {"params": [model.get_input_embeddings().weight], "lr": args.token_lr},
        {"params": qformer_params, "lr": args.qformer_lr},
        {"params": moh_params, "lr": args.moh_lr},
    ]
    if config.model_config["llm"].get("use_lora"):
        trainable_params.append({
            "params": [p for n, p in model.named_parameters() if "lora" in n.lower()],
            "lr": args.lora_lr,
        })
    return torch.optim.AdamW(trainable_params)


def restore_non_person_embeddings(model, orig_embeds, tokenizer, sks_tokens, prefix_tokens):
    with torch.no_grad():
        special_token_ids = tokenizer.convert_tokens_to_ids(sks_tokens + prefix_tokens)
        keep_indices = torch.ones(model.get_input_embeddings().weight.size(0), dtype=torch.bool, device=model.get_input_embeddings().weight.device)
        keep_indices[special_token_ids] = False
        model.get_input_embeddings().weight.data[keep_indices] = orig_embeds[keep_indices]


def restore_optimizer_state(optimizer, checkpoint_path):
    if not checkpoint_path:
        return False
    state_path = Path(checkpoint_path) / "optimizer.pt"
    if not state_path.exists():
        return False
    optimizer.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=False))
    return True


def train_one_epoch(
        model,
        ref_model,
        loader,
        optimizer,
        tokenizer,
        sks_tokens,
        prefix_tokens,
        orig_embeds,
        args,
        epoch,
        context=None,
):
    context = context or DistributedContext()
    base_model = unwrap_model(model)
    device = next(base_model.parameters()).device
    model.train()
    total_loss = 0.0
    completed_steps = 0
    total_steps = len(loader)
    if args.max_train_steps > 0:
        total_steps = min(total_steps, args.max_train_steps)
    with tqdm(
            total=total_steps,
            desc=f"GRPO Epoch {epoch}",
            disable=not context.is_main,
    ) as pbar:
        for batch in loader:
            if args.max_train_steps > 0 and completed_steps >= args.max_train_steps:
                break
            batch = {
                key: (value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value)
                for key, value in batch.items()
            }
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                video=batch["video"],
                labels=None,
                video_idx=batch["video_idx"],
            )
            logprobs, _ = masked_sequence_logprobs(outputs.logits, batch["input_ids"], batch["labels"])

            with torch.no_grad():
                ref_outputs = ref_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    video=batch["video"],
                    labels=None,
                    video_idx=batch["video_idx"],
                )
                ref_logprobs, _ = masked_sequence_logprobs(ref_outputs.logits, batch["input_ids"], batch["labels"])

            old_logprobs = batch["old_logprobs"]
            old_logprobs = torch.where(torch.isnan(old_logprobs), logprobs.detach(), old_logprobs)
            advantages = batch["advantages"]
            policy_loss = grpo_policy_loss(logprobs, old_logprobs, advantages, clip_range=args.clip_range)
            kl_loss = (logprobs - ref_logprobs.detach()).pow(2).mean()
            loss = policy_loss + args.kl_beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.max_grad_norm)
            optimizer.step()
            restore_non_person_embeddings(base_model, orig_embeds, tokenizer, sks_tokens, prefix_tokens)

            total_loss += float(loss.detach().cpu())
            completed_steps += 1
            pbar.update(1)
            pbar.set_postfix({
                "loss": f"{float(loss.detach().cpu()):.4f}",
                "pg": f"{float(policy_loss.detach().cpu()):.4f}",
                "kl": f"{float(kl_loss.detach().cpu()):.4f}",
                "reward": f"{float(batch['rewards'].mean().detach().cpu()):.3f}",
            })
    stats = torch.tensor(
        [total_loss, completed_steps],
        dtype=torch.float64,
        device=device,
    )
    if context.distributed:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float((stats[0] / stats[1].clamp_min(1)).cpu())


def save_checkpoint(
        model,
        tokenizer,
        config,
        sks_tokens,
        prefix_tokens,
        output_dir,
        epoch=None,
        optimizer=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    delta_spec = getattr(model, "_pvchat_delta_spec", None)
    if delta_spec is not None:
        # 紧凑模式：只保存个性化部分（与Stage 2 delta同构，约百MB级），
        # 且不保存optimizer——完整state和optimizer moments会按人物数量
        # 线性撑爆磁盘。
        save_compact_checkpoint(
            model,
            delta_spec,
            output_dir,
            tokenizer=tokenizer,
            config=config,
            training_info={
                "sks_tokens": sks_tokens,
                "prefix_tokens": prefix_tokens,
            },
        )
    else:
        torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
        tokenizer.save_pretrained(output_dir)
        config.save_pretrained(output_dir)
        torch.save({
            "sks_tokens": sks_tokens,
            "prefix_tokens": prefix_tokens,
        }, output_dir / "training_info.bin")
        if optimizer is not None:
            torch.save(optimizer.state_dict(), output_dir / "optimizer.pt")
    if epoch is not None:
        print(f"[GRPO] Saved checkpoint epoch {epoch} to {output_dir}")
    else:
        print(f"[GRPO] Saved final model to {output_dir}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_jsonl", required=True)
    parser.add_argument(
        "--train_json",
        default=None,
        help="Source training JSON used to resolve relative rollout video paths.",
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument(
        "--reference_checkpoint_path",
        default=None,
        help="Fixed Stage 2 checkpoint used by the KL reference model.",
    )
    parser.add_argument("--sks_name", required=True)
    parser.add_argument("--test_json", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument(
        "--epoch_offset",
        type=int,
        default=0,
        help="Completed outer GRPO epochs before this one-epoch invocation.",
    )
    parser.add_argument("--save_epochs", type=int, default=1)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--kl_beta", type=float, default=0.02)
    parser.add_argument("--personal_token_lr", type=float, default=1e-6)
    parser.add_argument("--token_lr", type=float, default=1e-6)
    parser.add_argument("--qformer_lr", type=float, default=1e-6)
    parser.add_argument("--moh_lr", type=float, default=1e-6)
    parser.add_argument("--lora_lr", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--group_batch",
        action="store_true",
        help="一组(同一问题的全部候选)作为一个batch、一次优化步(标准GRPO, 与Qwen管线一致); 默认为每个候选一步。",
    )
    parser.add_argument(
        "--disable_ref_vision_cache",
        action="store_true",
        help="关闭参照模型视觉路径的精确缓存(同一视频只编码一次); 仅影响速度。",
    )
    parser.add_argument("--eval_samples", type=int, default=20)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=0,
        help="Maximum optimizer steps per rank and epoch; 0 runs the full epoch.",
    )
    return parser.parse_args()


def main():
    context = initialize_distributed()
    try:
        if not context.is_main:
            logging.disable(logging.INFO)
        args = get_args()
        person_dir_name = args.sks_name.strip("<>")
        output_root = Path(args.output_dir)
        checkpoint_root = output_root / "checkpoints" / person_dir_name
        if context.is_main:
            checkpoint_root.mkdir(parents=True, exist_ok=True)
        distributed_barrier(context)

        model, tokenizer, config, sks_tokens, prefix_tokens = load_trainable_model(args)
        ref_model = clone_reference_model(args, config, tokenizer, sks_tokens, prefix_tokens)
        if not getattr(args, "disable_ref_vision_cache", False):
            install_reference_vision_cache(ref_model)
            logging.info("[GRPO] reference-model vision cache enabled (exact; eval-mode frozen path)")
        dataset = GrpoRolloutDataset(
            args.rollout_jsonl,
            tokenizer=tokenizer,
            config=config,
            train_json=getattr(args, "train_json", None),
        )
        loader, sampler = build_grpo_dataloader(dataset, args, context)
        optimizer = build_optimizer(model, config, tokenizer, sks_tokens, prefix_tokens, args)
        optimizer_restored = restore_optimizer_state(optimizer, getattr(args, "checkpoint_path", None))
        trainable_count = limit_trainable_parameters_to_optimizer(model, optimizer)
        orig_embeds = model.get_input_embeddings().weight.data.clone()
        model = wrap_distributed_model(model, context)
        if context.is_main:
            print(
                f"[GRPO] world_size={context.world_size}, candidates={len(dataset)}, "
                f"steps_per_rank={len(loader)}, trainable_parameters={trainable_count:,}, "
                f"optimizer_restored={optimizer_restored}",
                flush=True,
            )

        for epoch in range(args.num_epochs):
            global_epoch = global_epoch_number(epoch, getattr(args, "epoch_offset", 0))
            if sampler is not None:
                sampler.set_epoch(global_epoch - 1)
            avg_loss = train_one_epoch(
                model,
                ref_model,
                loader,
                optimizer,
                tokenizer,
                sks_tokens,
                prefix_tokens,
                orig_embeds,
                args,
                global_epoch,
                context=context,
            )
            distributed_barrier(context)
            if context.is_main:
                base_model = unwrap_model(model)
                print(f"[GRPO] Epoch {global_epoch}: loss={avg_loss:.4f}")
                if (epoch + 1) % args.save_epochs == 0:
                    save_checkpoint(
                        base_model,
                        tokenizer,
                        config,
                        sks_tokens,
                        prefix_tokens,
                        checkpoint_root / f"checkpoint_epoch_{global_epoch}",
                        epoch=global_epoch,
                        optimizer=optimizer,
                    )
                if args.test_json and args.eval_samples > 0:
                    test(
                        base_model,
                        tokenizer,
                        args.test_json,
                        torch.device("cuda", context.local_rank),
                        config=config,
                        output_dir=args.output_dir,
                        sample_count=args.eval_samples,
                        stage_name=f"After GRPO epoch {global_epoch}",
                        save_results=False,
                        seed=args.eval_seed + global_epoch - 1,
                    )
            distributed_barrier(context)

        base_model = unwrap_model(model)
        if context.is_main:
            final_dir = output_root / "final_model"
            save_checkpoint(
                base_model,
                tokenizer,
                config,
                sks_tokens,
                prefix_tokens,
                final_dir,
            )
        distributed_barrier(context)
        cleanup_distributed(context)
        if context.is_main and args.test_json:
            del ref_model
            torch.cuda.empty_cache()
            test(
                base_model,
                tokenizer,
                args.test_json,
                torch.device("cuda", context.local_rank),
                config=config,
                output_dir=args.output_dir,
                sample_count=None,
                stage_name=(
                    f"After GRPO epoch "
                    f"{getattr(args, 'epoch_offset', 0) + args.num_epochs} full test"
                ),
                save_results=True,
                seed=args.eval_seed,
            )
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
