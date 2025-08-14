#!/usr/bin/env python3
# remove_generated_answer.py

import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Remove 'generated_answer' fields from a JSON file.")
    parser.add_argument("--input_json", type=str,default="/root/autodl-tmp/yufei/Qwen2.5/data_result/Ch/input/pvchat_Ch.json",
                        help="Path to the original input JSON file.")
    parser.add_argument("--output_json", type=str,default="/root/autodl-tmp/yufei/Qwen2.5/data_result/Ch/input/internvideo_Ch-1person-kong.json",
                        help="Path to save the new JSON file without 'generated_answer' fields.")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 读取旧的 JSON
    with open(args.input_json, "r") as f:
        data = json.load(f)

    # 2. 删除所有的 generated_answer 字段
    if "results" in data:
        for result in data["results"]:
            if "qa_pairs" in result:
                for qa_pair in result["qa_pairs"]:
                    if "generated_answer" in qa_pair:
                        del qa_pair["generated_answer"]

    # 3. 保存到新的 JSON 文件，不覆盖原文件
    with open(args.output_json, "w") as f:
        json.dump(data, f, indent=2)

    print(f"All 'generated_answer' fields have been removed.\nNew file saved to: {args.output_json}")


if __name__ == "__main__":
    main()
