#!/usr/bin/env python3
"""Stage 1：首帧复制短视频的4帧初始训练。"""

from pathlib import Path
import sys


# 优先使用下载在项目内、明确支持Qwen3.5的Transformers源码。
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "transformers-qwen35" / "src"))

from qwen35_pvchat.cli import build_sft_parser
from qwen35_pvchat.distributed import cleanup_distributed, initialize_distributed
from qwen35_pvchat.trainer import run_sft_stage


def main():
    args = build_sft_parser(stage=1).parse_args()
    context = initialize_distributed()
    try:
        run_sft_stage(args, context)
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
