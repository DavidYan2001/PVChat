"""Stage 1和Stage 2共用的DDP监督训练循环。"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from pvchat_dataset_validation import describe_training_summary, validate_training_json_classes

from .checkpoint import load_optimizer_state, save_checkpoint
from .data import DynamicVideoBudget, PVChatSFTDataset, collate_qwen35_features, video_overrides_from_token_budget
from .distributed import barrier, move_batch_to_device, unwrap_model
from .evaluation import run_distributed_evaluation, run_metrics
from .grpo import select_causal_positions, selected_causal_cross_entropy
from .modeling import build_optimizer, load_qwen35_pvchat_model, trainable_parameter_summary
from .remoh_losses import AdaptiveReMoHLoss


def _forward_inputs(batch, include_labels=True):
    excluded = {"records"}
    if not include_labels:
        excluded.add("labels")
    values = {key: value for key, value in batch.items() if key not in excluded}
    mm_types = values["mm_token_type_ids"]
    values["pvchat_video_token_mask"] = mm_types.eq(2)
    values["pvchat_text_token_mask"] = mm_types.eq(0)
    return values


def build_sft_loader(dataset, processor, args, context):
    sampler = (
        DistributedSampler(dataset, context.world_size, context.rank, shuffle=True)
        if context.distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=partial(
            collate_qwen35_features,
            pad_token_id=processor.tokenizer.pad_token_id,
        ),
    )
    return loader, sampler


def train_sft_epoch(model, loader, sampler, optimizer, remoh_loss, args, context, epoch):
    if sampler is not None:
        sampler.set_epoch(epoch)
    model.train()
    total_loss = 0.0
    completed = 0
    progress = tqdm(loader, desc=f"Stage {args.stage} Epoch {epoch + 1}", disable=not context.is_main)
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(progress):
        if args.max_train_steps > 0 and completed >= args.max_train_steps:
            break
        batch = move_batch_to_device(batch, context.device)
        # Qwen视频prompt可能有数千token，但真正需要监督的只有末尾答案。
        # 只计算答案位置的词表logits，交叉熵与完整计算完全等价。
        positions, targets = select_causal_positions(batch["labels"])
        forward_inputs = _forward_inputs(batch, include_labels=False)
        forward_inputs["logits_to_keep"] = positions
        outputs = model(**forward_inputs)
        ce_loss = selected_causal_cross_entropy(outputs.logits, targets)
        base_model = unwrap_model(model)
        auxiliary = remoh_loss.from_model(base_model)
        loss = ce_loss + auxiliary.total_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().cpu())
        completed += 1
        if context.is_main:
            progress.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                ce=f"{float(ce_loss.detach()):.4f}",
                active=f"{float(auxiliary.active_ratio.detach()):.3f}",
            )

    stats = torch.tensor([total_loss, completed], dtype=torch.float64, device=context.device)
    if context.distributed:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float((stats[0] / stats[1].clamp_min(1)).cpu())


def run_sft_evaluation(model, bundle, args, context, video_overrides):
    result_json = Path(args.output_dir) / "evaluation" / "test_results.json"
    run_distributed_evaluation(
        model,
        bundle.processor,
        args.test_json,
        bundle.personalized_tokens,
        result_json,
        context,
        stage=2,
        max_new_tokens=args.eval_max_new_tokens,
        batch_size=args.eval_batch_size,
        video_overrides=video_overrides,
    )
    if context.is_main and not args.skip_metrics:
        run_metrics(
            result_json,
            Path(args.output_dir) / "metrics",
            bundle.personalized_tokens[0],
            judge_backend=args.judge_backend,
            judge_model=args.judge_model,
            judge_fallback_models=args.judge_fallback_models,
            api_num_workers=args.api_num_workers,
            bertscore_device=f"cuda:{context.local_rank}" if torch.cuda.is_available() else "cpu",
            local_judge_model_path=getattr(args, "local_judge_model_path", None),
            local_judge_device=getattr(args, "local_judge_device", "cuda:0"),
            local_judge_batch_size=getattr(args, "local_judge_batch_size", 64),
            local_judge_es_items_per_prompt=getattr(
                args, "local_judge_es_items_per_prompt", 1
            ),
            local_judge_dc_items_per_prompt=getattr(
                args, "local_judge_dc_items_per_prompt", 10
            ),
            local_judge_max_new_tokens=getattr(args, "local_judge_max_new_tokens", 64),
        )
    barrier(context)


def run_sft_stage(args, context):
    if args.epoch_offset < 0:
        raise ValueError("--epoch_offset不能小于0。")
    if args.resume_optimizer and not args.checkpoint_path:
        raise ValueError("--resume_optimizer必须与--checkpoint_path一起使用。")
    if args.eval_only and not args.checkpoint_path:
        raise ValueError("--eval_only必须通过--checkpoint_path指定待评测checkpoint。")
    if args.eval_only and args.skip_evaluation:
        raise ValueError("--eval_only不能与--skip_evaluation同时使用。")

    budget_per_10s = getattr(args, "video_budget_per_10_seconds", None)
    if budget_per_10s:
        # 逐视频按实际时长缩放的视觉预算（768 * ceil(秒数/10)），训练与测试共用。
        video_overrides = DynamicVideoBudget(int(budget_per_10s), min_tokens=int(args.video_min_tokens))
        if context.is_main:
            print(f"[Qwen3.5] 逐视频动态视觉预算 {video_overrides.describe()}（video_max_tokens 不再作为固定预算）", flush=True)
    else:
        video_overrides = video_overrides_from_token_budget(
            args.video_min_tokens,
            args.video_max_tokens,
        )
    bundle = load_qwen35_pvchat_model(
        model_path=args.model_path,
        person_token=args.sks_name,
        device=context.device,
        checkpoint_path=args.checkpoint_path,
        num_detail_tokens=args.num_detail_tokens,
        remoh_layers=args.remoh_layers,
        routed_heads=args.routed_heads,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        attn_implementation=args.attn_implementation,
        gradient_checkpointing=not args.disable_gradient_checkpointing,
    )
    if context.is_main:
        print("[Qwen3.5] 可训练参数:", trainable_parameter_summary(bundle.model), flush=True)

    if args.eval_only:
        run_sft_evaluation(bundle.model, bundle, args, context, video_overrides)
        return Path(args.checkpoint_path)

    data_summary = validate_training_json_classes(
        args.train_json,
        f"Qwen3.5 Stage {args.stage}",
    )
    if context.is_main:
        print(describe_training_summary(data_summary), flush=True)

    dataset = PVChatSFTDataset(
        args.train_json,
        bundle.processor,
        bundle.personalized_tokens,
        stage=args.stage,
        video_overrides=video_overrides,
    )
    loader, sampler = build_sft_loader(dataset, bundle.processor, args, context)
    optimizer = build_optimizer(
        bundle.model,
        token_lr=args.token_lr,
        remoh_lr=args.remoh_lr,
        lora_lr=args.lora_lr,
        weight_decay=args.weight_decay,
    )
    optimizer_resume = None
    if args.resume_optimizer:
        optimizer_resume = load_optimizer_state(optimizer, args.checkpoint_path)
        if context.is_main:
            print(
                "[Qwen3.5] 已恢复optimizer: "
                f"state={optimizer_resume['state_entries']} "
                f"groups={optimizer_resume['param_groups']}",
                flush=True,
            )
    remoh_loss = AdaptiveReMoHLoss(
        target_active_ratio=args.target_active_ratio,
        initial_spr_weight=args.initial_spr_weight,
        hae_weight=args.hae_weight,
    ).to(context.device)

    model = bundle.model
    if context.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )

    for local_epoch in range(args.num_epochs):
        epoch = args.epoch_offset + local_epoch
        average_loss = train_sft_epoch(
            model,
            loader,
            sampler,
            optimizer,
            remoh_loss,
            args,
            context,
            epoch,
        )
        if context.is_main:
            print(f"[Stage {args.stage}] Epoch {epoch + 1} loss={average_loss:.6f}", flush=True)

    output_root = Path(args.output_dir)
    checkpoint_dir = output_root / "checkpoint"
    base_model = unwrap_model(model)
    if context.is_main:
        metadata = dict(bundle.metadata)
        metadata.update(
            {
                "stage": args.stage,
                "train_json": str(Path(args.train_json).resolve()),
                "test_json": str(Path(args.test_json).resolve()),
                "num_epochs": args.epoch_offset + args.num_epochs,
                "epochs_this_run": args.num_epochs,
                "epoch_offset": args.epoch_offset,
                "parent_checkpoint": (
                    str(Path(args.checkpoint_path).resolve())
                    if args.checkpoint_path
                    else None
                ),
                "optimizer_resumed": optimizer_resume is not None,
                "video_min_tokens": args.video_min_tokens,
                "video_max_tokens": args.video_max_tokens,
                "video_budget_per_10_seconds": int(budget_per_10s) if budget_per_10s else None,
            }
        )
        save_checkpoint(checkpoint_dir, base_model, bundle.processor, metadata, optimizer=optimizer)
    barrier(context)

    if args.skip_evaluation:
        if context.is_main:
            print(f"[Stage {args.stage}] 已按参数跳过测试，仅用于smoke test。", flush=True)
        return checkpoint_dir

    run_sft_evaluation(model, bundle, args, context, video_overrides)
    return checkpoint_dir
