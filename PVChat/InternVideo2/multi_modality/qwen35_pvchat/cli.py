"""三个Stage共用的命令行参数。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_FALLBACK_MODELS = (
    "qwen3.7-max-preview,qwen3.7-max-2026-06-08,"
    "qwen3.7-max-2026-05-17,qwen3.6-max-preview"
)
DEFAULT_LOCAL_JUDGE_MODEL_PATH = os.environ.get(
    "PVCHAT_LOCAL_JUDGE_MODEL_PATH",
    str(Path(__file__).resolve().parents[4] / "models" / "Qwen3.5-35B-A3B"),
)


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--model_path", required=True, help="本地Qwen3.5-9B基础权重目录。")
    parser.add_argument("--checkpoint_path", default=None, help="上一Stage的轻量checkpoint目录。")
    parser.add_argument("--sks_name", required=True, help="人物token，例如<Sheldon>。")
    parser.add_argument("--num_detail_tokens", type=int, default=16)
    parser.add_argument("--remoh_layers", default="7,11,15,19")
    parser.add_argument("--routed_heads", default="3,7,11,15")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--attn_implementation", default="sdpa", choices=("eager", "sdpa", "flash_attention_2"))
    parser.add_argument(
        "--video_min_tokens",
        type=int,
        default=4,
        help="视频动态分辨率的最小视觉预算。",
    )
    parser.add_argument(
        "--video_max_tokens",
        type=int,
        default=768,
        help="视频动态分辨率的最大视觉预算；默认768以控制训练显存。",
    )
    parser.add_argument(
        "--video_budget_per_10_seconds",
        type=int,
        default=None,
        help="逐视频动态视觉预算：每10秒实际时长给这么多token（新版Stage 2用768）；"
        "设置后video_max_tokens不再作为固定预算，仅video_min_tokens仍生效。",
    )
    parser.add_argument("--disable_gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--disable_batched_rollout",
        action="store_true",
        help="buffered模式下退回逐组生成；批量生成只改墙钟时间，不改算法语义。",
    )


def add_optimizer_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--token_lr", type=float, default=1e-4)
    parser.add_argument("--remoh_lr", type=float, default=1e-6)
    parser.add_argument("--lora_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--target_active_ratio", type=float, default=0.5)
    parser.add_argument("--initial_spr_weight", type=float, default=1e-8)
    parser.add_argument("--hae_weight", type=float, default=0.5)


def add_evaluation_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--eval_max_new_tokens", type=int, default=96)
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="本地Qwen测试一次并行生成的QA数量；RTX 6000 Ada默认使用4。",
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="仅用于smoke test；默认每个Stage都会执行完整测试。",
    )
    parser.add_argument("--skip_metrics", action="store_true", help="仍生成测试JSON，但不计算五指标。")
    parser.add_argument(
        "--judge_backend",
        default="local_qwen35",
        choices=("none", "dashscope", "local_qwen35"),
    )
    parser.add_argument("--judge_model", default="qwen3.7-max")
    parser.add_argument("--judge_fallback_models", default=DEFAULT_FALLBACK_MODELS)
    parser.add_argument("--api_num_workers", type=int, default=20)
    parser.add_argument("--local_judge_model_path", default=DEFAULT_LOCAL_JUDGE_MODEL_PATH)
    parser.add_argument("--local_judge_device", default="cuda:0")
    parser.add_argument("--local_judge_batch_size", type=int, default=64)
    parser.add_argument("--local_judge_es_items_per_prompt", type=int, default=1)
    parser.add_argument("--local_judge_dc_items_per_prompt", type=int, default=10)
    parser.add_argument("--local_judge_max_new_tokens", type=int, default=64)


def build_sft_parser(stage: int):
    parser = argparse.ArgumentParser(description=f"Qwen3.5 PVChat Stage {stage}")
    add_model_arguments(parser)
    add_optimizer_arguments(parser)
    add_evaluation_arguments(parser)
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--test_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=0, help="0表示完整遍历训练集。")
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="只加载--checkpoint_path执行完整测试和指标，不训练或重写checkpoint。",
    )
    parser.add_argument(
        "--resume_optimizer",
        action="store_true",
        help="从--checkpoint_path恢复AdamW动量与step；真正续训时必须启用。",
    )
    parser.add_argument(
        "--epoch_offset",
        type=int,
        default=0,
        help="此前已经完成的epoch数；用于续训时的shuffle种子和累计轮数。",
    )
    parser.set_defaults(stage=stage)
    return parser


def build_stage3_parser():
    parser = argparse.ArgumentParser(description="Qwen3.5 PVChat Stage 3 Online GRPO")
    add_model_arguments(parser)
    add_optimizer_arguments(parser)
    add_evaluation_arguments(parser)
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--test_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument(
        "--max_steps_per_epoch",
        type=int,
        default=0,
        help="0表示每轮使用完整训练集；正数仅用于smoke test。",
    )
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--drift_beta", type=float, default=0.02)
    parser.add_argument(
        "--ref_kl_beta",
        type=float,
        default=0.0,
        help="标准GRPO的参照KL系数: 锚向冻结Stage 2参照模型(k3估计); 0为关闭。"
        "开启时训练期常驻一份冻结参照模型(约+19GB显存)。",
    )
    parser.add_argument(
        "--greedy_anchor_candidate",
        action="store_true",
        help="每组rollout的第一个候选改用greedy解码(=stage2确定性行为), "
        "保证全组皆错时仍有可强化的正确样本。",
    )
    parser.add_argument(
        "--negative_group_weight",
        type=float,
        default=1.0,
        help="gold为'不在/拒答'的组的advantage放大系数, 对冲正负样本组更新失衡; 1.0为关闭。",
    )
    parser.add_argument(
        "--algorithm",
        default="token_grpo",
        choices=(
            "token_grpo",
            "gspo",
            "dr_gspo",
            "dynamic_gspo",
            "pa_gspo",
            "icd_gspo_v0",
            "icd_gspo_v1",
            "ig_dynamic_gspo",
            "ca_dynamic_gspo",
            "ca_dynamic_gspo_buffered",
        ),
        help="共享实验运行器使用的固定Stage 3策略；旧入口默认保持token GRPO。",
    )
    parser.add_argument(
        "--rollout_buffer_size",
        type=int,
        default=4,
        help="CA-Dynamic-GSPO固定旧策略rollout窗口大小；必须大于1。",
    )
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument(
        "--identity_threshold",
        type=float,
        default=0.5,
        help="ICD候选通过身份可行性约束的阈值。",
    )
    parser.add_argument(
        "--identity_margin",
        type=float,
        default=1.0,
        help="ICD身份可行候选与不可行候选之间的组内margin。",
    )
    parser.add_argument(
        "--icd_soft_clip",
        type=float,
        default=0.5,
        help="ICD可行候选软advantage的绝对值上限；不得大于identity_margin。",
    )
    parser.add_argument(
        "--icd_conflict_threshold",
        type=float,
        default=0.25,
        help="ICD-GSPO V1从4个扩展到8个候选的向量不确定性阈值。",
    )
    parser.add_argument(
        "--ig_identity_margin",
        type=float,
        default=1.0,
        help="IG-Dynamic-GSPO身份层级margin，必须严格大于2倍ig_soft_clip。",
    )
    parser.add_argument(
        "--ig_soft_clip",
        type=float,
        default=0.25,
        help="IG-Dynamic-GSPO保留的原Dynamic-GSPO软优势截断值。",
    )
    parser.add_argument(
        "--ca_semantic_threshold",
        type=float,
        default=0.60,
        help="CA-Dynamic-GSPO在8候选阶段接受最佳内容候选的最低分。",
    )
    parser.add_argument(
        "--ca_informative_margin",
        type=float,
        default=0.10,
        help="CA-Dynamic-GSPO认为候选内容差异可用于策略更新的最小跨度。",
    )
    parser.add_argument(
        "--ca_sft_weight",
        type=float,
        default=0.05,
        help="CA-Dynamic-GSPO普通更新组的GT答案锚定权重。",
    )
    parser.add_argument(
        "--ca_fallback_sft_weight",
        type=float,
        default=0.20,
        help="8个候选仍不合格时使用的GT答案锚定权重。",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="忽略phase marker并从本算法的Stage 2父checkpoint重新运行。",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser
