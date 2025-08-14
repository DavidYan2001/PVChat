#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

def process_file(input_path, output_path):
    """
    读取 input_path json，修改 "data" -> "videos"
    并对每条 qa_pair 中 question/answer 做 <sks1> -> <Ro>, <sks2> -> <Ra> 的替换。
    然后把结果保存到 output_path
    """

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # 原格式: {"data": [ ... ] }

    # 1) 将 data => videos
    if "data" in data:
        data["videos"] = data.pop("data")
    else:
        print(f"警告: {input_path} 中没有发现 'data' 字段，可能已是正确格式?")

    # 2) 遍历每个 video 下的 qa_pairs, 做替换
    for video_item in data.get("videos", []):
        qa_list = video_item.get("qa_pairs", [])
        for qa in qa_list:
            # question
            if "question" in qa:
                qa["question"] = qa["question"].replace("<sks1>", "<Ro>").replace("<sks2>", "<Ra>")
            # answer
            if "answer" in qa:
                qa["answer"]   = qa["answer"].replace("<sks1>", "<Ro>").replace("<sks2>", "<Ra>")

    # 3) 写回到 output_path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已处理并保存: {output_path}")


if __name__ == "__main__":

    # 输入文件
    train_input = "/root/autodl-tmp/yufei/datasets/cekebv-hq/train_all_video_updated.json"
    test_input  = "/root/autodl-tmp/yufei/datasets/cekebv-hq/test_all_video_updated.json"

    # 输出目录
    out_dir = "/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality"
    os.makedirs(out_dir, exist_ok=True)

    # 输出文件
    train_output = os.path.join(out_dir, "<Ra&Ro>train.json")
    test_output  = os.path.join(out_dir, "<Ra&Ro>test.json")

    # 处理 train
    process_file(train_input, train_output)
    # 处理 test
    process_file(test_input, test_output)
