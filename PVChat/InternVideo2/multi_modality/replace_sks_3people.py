#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os


def get_args():
    parser = argparse.ArgumentParser(description="Replace <sks1>, <sks2>, <sks3> with actual names for three people.")
    parser.add_argument('--sks1', type=str, default='Cl', help='Name of first person (default: Cl)')
    parser.add_argument('--sks2', type=str, default='Xo', help='Name of second person (default: Xo)')
    parser.add_argument('--sks3', type=str, default='Ja', help='Name of third person (default: Ja)')
    return parser.parse_args()


def replace_sks_in_json(input_path, output_path, sks_name1, sks_name2, sks_name3):
    """
    1. Read the JSON file of input_path. If there is a" data" key in it, change it to "videos".
    2. Traverse data["videos"] :
    Set sks_present1, sks_present2, sks_present3 = <sks_name1>, <sks_name2>, <sks_name3>
    - traverse qa_pairs will '< sks1 >' = > < sks_name1 >, '< sks2 >' = > < sks_name2 >, '< sks3 >' = > < sks_name3 >
    3. Write the processing result to output_path.
    """
    if not os.path.exists(input_path):
        print(f"[Warning] {input_path} does not exist.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Change the "data" key to "videos". If "videos" already exists, you can decide for yourself whether to retain or overwrite it
    if "data" in data:
        data["videos"] = data.pop("data")

    # If videos do not exist, it indicates that the file structure does not meet expectations
    if "videos" not in data or not isinstance(data["videos"], list):
        print(f"[Warning] {input_path} => No 'videos' list found. Skipping.")
        return

    for video in data["videos"]:
        # updata sks_present
        video["sks_present1"] = f"<{sks_name1}>"
        video["sks_present2"] = f"<{sks_name2}>"
        video["sks_present3"] = f"<{sks_name3}>"

        # process QA pairs
        qa_pairs = video.get("qa_pairs", [])
        for qa_pair in qa_pairs:
            # question
            qa_pair["question"] = qa_pair["question"].replace("<sks1>", f"<{sks_name1}>")
            qa_pair["question"] = qa_pair["question"].replace("<sks2>", f"<{sks_name2}>")
            qa_pair["question"] = qa_pair["question"].replace("<sks3>", f"<{sks_name3}>")

            # answer
            qa_pair["answer"] = qa_pair["answer"].replace("<sks1>", f"<{sks_name1}>")
            qa_pair["answer"] = qa_pair["answer"].replace("<sks2>", f"<{sks_name2}>")
            qa_pair["answer"] = qa_pair["answer"].replace("<sks3>", f"<{sks_name3}>")

    # Save the modified file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[Info] Processed => {input_path}, output => {output_path}")


def clean_old_files(sks_name1, sks_name2, sks_name3):
    """
    Example of cleaning up old files:
      - test_Cl_Xo_Ja.json
      - train_Cl_Xo_Ja.json
      - test_all_video_Cl_Xo_Ja.json
      - train_all_video_Cl_Xo_Ja.json
    It can be modified or commented out as needed
    """
    base_path = "/root/autodl-tmp/yufei/datasets/cekebv-hq"
    files_to_remove = [
        f"test_{sks_name1}_{sks_name2}_{sks_name3}.json",
        f"train_{sks_name1}_{sks_name2}_{sks_name3}.json",
        f"test_all_video_{sks_name1}_{sks_name2}_{sks_name3}.json",
        f"train_all_video_{sks_name1}_{sks_name2}_{sks_name3}.json"
    ]

    for file_name in files_to_remove:
        file_path = os.path.join(base_path, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed: {file_path}")


def main():
    args = get_args()

    sks_name1 = args.sks1
    sks_name2 = args.sks2
    sks_name3 = args.sks3

    base_dir = "/root/autodl-tmp/yufei/datasets/cekebv-hq"
    input_train = os.path.join(base_dir, f"train_all_video_updated_{args.sks1}_{args.sks2}_{args.sks3}.json")
    input_test  = os.path.join(base_dir, f"test_all_video_updated_{args.sks1}_{args.sks2}_{args.sks3}.json")


    output_dir = "/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality"
    os.makedirs(output_dir, exist_ok=True)

    output_train = os.path.join(output_dir, f"<{sks_name1}>_<{sks_name2}>_<{sks_name3}>.json")
    output_test  = os.path.join(output_dir, f"<{sks_name1}>_<{sks_name2}>_<{sks_name3}>_test.json")

    replace_sks_in_json(input_train, output_train, sks_name1, sks_name2, sks_name3)
    replace_sks_in_json(input_test,  output_test,  sks_name1, sks_name2, sks_name3)

    print("[Done] Replaced sks in JSON files.")


if __name__ == "__main__":
    main()
