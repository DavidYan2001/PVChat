#!/usr/bin/env python3
"""Stage 2：按2 FPS读取完整视频继续训练。"""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "transformers-qwen35" / "src"))

from qwen35_pvchat.cli import build_sft_parser
from qwen35_pvchat.distributed import cleanup_distributed, initialize_distributed
from qwen35_pvchat.trainer import run_sft_stage


def main():
    args = build_sft_parser(stage=2).parse_args()
    if not args.checkpoint_path:
        raise ValueError("Stage 2必须通过--checkpoint_path指定Stage 1或上一轮Stage 2 checkpoint。")
    context = initialize_distributed()
    try:
        run_sft_stage(args, context)
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
