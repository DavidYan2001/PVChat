#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os


def replace_ro_with_ti(file_path):
    """
    Read the file_path json, then replace all "<Ro>" in the json content with "<Ti>", and save it back to the original file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1) # 1) Recursive substitution
    #   The outermost layer of # data is a dict, which contains "videos" (list), and each contains "qa_pairs" (list)...
    #   Here, whenever we encounter a string, we replace("<Ro>", "<Ti>").
    def recurse_replace(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                obj[k] = recurse_replace(v)
        elif isinstance(obj, list):
            for i in range(len(obj)):
                obj[i] = recurse_replace(obj[i])
        elif isinstance(obj, str):
            return obj.replace("<Ro>", "<Ti>")
        return obj

    data = recurse_replace(data)

    # 2) 保存回文件
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"All <Ro> has been replaced with <Ti> in {file_path} and overwritten for saving")


if __name__ == "__main__":
    out_dir = "/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality"
    train_file = os.path.join(out_dir, "<Ra&Ro>train.json")
    test_file = os.path.join(out_dir, "<Ra&Ro>test.json")

    # Replace train.json first
    replace_ro_with_ti(train_file)
    # Replace test.json
    replace_ro_with_ti(test_file)
