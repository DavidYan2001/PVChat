import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Replace <sks> with custom name in JSON files')
    parser.add_argument('--sks_name', type=str, required=True, help='Name to replace <sks> with (e.g., <Mi>)')
    return parser.parse_args()


def replace_sks_in_json(input_path, output_path, sks_name):
    # 读取 JSON 文件
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 将 "data" 键改为 "videos"
    if "data" in data:
        data["videos"] = data.pop("data")

    # 遍历所有视频数据
    for video in data["videos"]:
        # 更新 sks_present
        video["sks_present"] = sks_name

        # 更新所有 QA 对
        for qa_pair in video["qa_pairs"]:
            qa_pair["question"] = qa_pair["question"].replace("<sks>", sks_name)
            qa_pair["answer"] = qa_pair["answer"].replace("<sks>", sks_name)

    # 保存修改后的文件
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

    # 清理旧文件
    clean_old_files()

    # 输入文件路径
    input_train = "/root/autodl-tmp/yufei/datasets/cekebv-hq/train_all_video_updated.json"
    input_test = "/root/autodl-tmp/yufei/datasets/cekebv-hq/test_all_video_updated.json"

    # 输出文件路径 (直接使用完整的名称，包括尖括号)
    output_dir = "/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality"
    output_train = os.path.join(output_dir, f"{sks_name}.json")
    output_test = os.path.join(output_dir, f"{sks_name}test.json")

    # 处理训练和测试文件
    replace_sks_in_json(input_train, output_train, sks_name)
    replace_sks_in_json(input_test, output_test, sks_name)

    print(f"Files processed successfully:")
    print(f"Train file saved as: {output_train}")
    print(f"Test file saved as: {output_test}")


if __name__ == "__main__":
    main()