#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

def process_file(input_path, output_path):
    """
    Read the input_path json and modify "data" -> "videos"
And replace the question/answer in each qa_pair with <sks1> -> <Ro>, <sks2> -> <Ra>.
Then save the result to output_path
    """

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # 原格式: {"data": [ ... ] }

    # 1) 将 data => videos
    if "data" in data:
        data["videos"] = data.pop("data")
    else:
        print(f"Warning: The 'data' field was not found in {input_path}. Could it be in the correct format?")

    # 2) Traverse the qa pairs under each video and make replacements
    for video_item in data.get("videos", []):
        qa_list = video_item.get("qa_pairs", [])
        for qa in qa_list:
            # question
            if "question" in qa:
                qa["question"] = qa["question"].replace("<sks1>", "<Ro>").replace("<sks2>", "<Ra>")
            # answer
            if "answer" in qa:
                qa["answer"]   = qa["answer"].replace("<sks1>", "<Ro>").replace("<sks2>", "<Ra>")

    # 3) Write back to the output path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Processed and saved: {output_path}")


if __name__ == "__main__":

    # input file
    train_input = "/root/autodl-tmp/yufei/datasets/cekebv-hq/train_all_video_updated.json"
    test_input  = "/root/autodl-tmp/yufei/datasets/cekebv-hq/test_all_video_updated.json"

    # output file
    out_dir = "/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality"
    os.makedirs(out_dir, exist_ok=True)

    # output file
    train_output = os.path.join(out_dir, "<Ra&Ro>train.json")
    test_output  = os.path.join(out_dir, "<Ra&Ro>test.json")

    # process train
    process_file(train_input, train_output)
    # process test
    process_file(test_input, test_output)
