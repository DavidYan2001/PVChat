import argparse
import gc

import torch

from finetune_internvideo_REOMH_one_person2_stage import test as run_test
from finetune_internvideo_REOMH_one_person3_grpo import load_trainable_model


def get_args():
    parser = argparse.ArgumentParser(description="Run the full PVChat test set for one checkpoint.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--sks_name", required=True)
    parser.add_argument("--test_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--eval_seed", type=int, default=42)
    return parser.parse_args()


def run_full_test(args):
    model, tokenizer, config, _, _ = load_trainable_model(args)
    results = run_test(
        model,
        tokenizer,
        args.test_json,
        torch.device("cuda", 0),
        config=config,
        output_dir=args.output_dir,
        sample_count=None,
        stage_name=f"After GRPO epoch {args.epoch} full test",
        save_results=True,
        seed=args.eval_seed,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    run_full_test(get_args())


if __name__ == "__main__":
    main()
