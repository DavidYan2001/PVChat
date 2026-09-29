#!/usr/bin/env python3
"""Qwen3.5 PVChat Stage 3: Buffered Constraint-Anchored Dynamic-GSPO."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "transformers-qwen35" / "src"))

from qwen35_pvchat.stage3_entry import run_fixed_stage3

ALGORITHM = "ca_dynamic_gspo_buffered"


def main(argv=None):
    return run_fixed_stage3(ALGORITHM, argv)


if __name__ == "__main__":
    main()
