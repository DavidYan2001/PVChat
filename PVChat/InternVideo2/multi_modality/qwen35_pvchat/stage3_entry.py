"""五组Stage 3实验共用的最小命令行入口。"""

from __future__ import annotations

from .cli import build_stage3_parser
from .distributed import cleanup_distributed, initialize_distributed
from .stage3_experiment import run_stage3_experiment


def run_fixed_stage3(algorithm: str, argv=None):
    args = build_stage3_parser().parse_args(argv)
    if not args.checkpoint_path:
        raise ValueError("Stage 3必须通过--checkpoint_path指定统一的Stage 2 checkpoint。")
    # 入口文件固定算法，不允许命令行参数把实验写入错误的结果目录。
    args.algorithm = algorithm
    context = initialize_distributed()
    try:
        return run_stage3_experiment(args, context, algorithm=algorithm)
    finally:
        cleanup_distributed(context)
