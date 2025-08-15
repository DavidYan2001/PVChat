import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Replace <sks> with custom name in JSON files')
    parser.add_argument('--sks_name', type=str, required=True, help='Name to replace <sks> with (e.g., <Mi>)')
    return parser.parse_args()


def replace_sks_in_json(input_path, output_path, sks_name):
    # Read the JSON file
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Change the "data" key to "videos"
    if "data" in data:
        data["videos"] = data.pop("data")

    # Traverse all video data
    for video in data["videos"]:
        # updata sks_present
        video["sks_present"] = sks_name

        # Update all QA pairs
        for qa_pair in video["qa_pairs"]:
            qa_pair["question"] = qa_pair["question"].replace("<sks>", sks_name)
            qa_pair["answer"] = qa_pair["answer"].replace("<sks>", sks_name)

    # Save the modified file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def clean_old_files():
    base_path = "/root/autodl-tmp/yufei/datasets/cekebv-hq"
    files_to_remove = [
        "test.json",
        "train.json",
        "test_all_video.json",
        "train_all_video.json"
    ]

    for file_name in files_to_remove:
        file_path = os.path.join(base_path, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed: {file_path}")


def main():
    args = parse_args()
    sks_name = args.sks_name

    # clean old file
    clean_old_files()

    # input file path
    input_train = "/root/autodl-tmp/yufei/datasets/cekebv-hq/train_all_video_updated.json"
    input_test = "/root/autodl-tmp/yufei/datasets/cekebv-hq/test_all_video_updated.json"

    # Output file path (use the full name directly, including Angle brackets)
    output_dir = "/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality"
    output_train = os.path.join(output_dir, f"{sks_name}.json")
    output_test = os.path.join(output_dir, f"{sks_name}test.json")

    # Handle training and test files
    replace_sks_in_json(input_train, output_train, sks_name)
    replace_sks_in_json(input_test, output_test, sks_name)

    print(f"Files processed successfully:")
    print(f"Train file saved as: {output_train}")
    print(f"Test file saved as: {output_test}")


if __name__ == "__main__":
    main()