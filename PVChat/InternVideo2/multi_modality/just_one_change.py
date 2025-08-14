#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os


def replace_ro_with_ti(file_path):
    """
    读取 file_path json，然后将 json 内容中所有 "<Ro>" 替换成 "<Ti>"，再保存回原文件。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1) 递归替换
    #   data 最外层是个 dict，里面有 "videos" (list)，每个里有 "qa_pairs" (list)...
    #   这里我们只要遇到 string 就 replace("<Ro>", "<Ti>")
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
    print(f"已在 {file_path} 中把 <Ro> 全部替换为 <Ti> 并覆盖保存。")


if __name__ == "__main__":
    out_dir = "/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality"
    train_file = os.path.join(out_dir, "<Ra&Ro>train.json")
    test_file = os.path.join(out_dir, "<Ra&Ro>test.json")

    # 先替换 train.json
    replace_ro_with_ti(train_file)
    # 再替换 test.json
    replace_ro_with_ti(test_file)
