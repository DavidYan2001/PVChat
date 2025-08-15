import argparse
import json
import os


def get_args():
    parser = argparse.ArgumentParser(description="Build dataset for two people and both cases")

    parser.add_argument('--sks1', type=str, default='Sh',
                        help='Name of first person (default: Sh)')
    parser.add_argument('--sks2', type=str, default='Ho',
                        help='Name of second person (default: Ho)')
    return parser.parse_args()

def replace_sks_in_json(input_path, output_path, sks_name1,sks_name2):
    # Read the JSON file
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Change the "data" key to "videos"
    if "data" in data:
        data["videos"] = data.pop("data")

    # Traverse all video data
    for video in data["videos"]:
        # updata sks_present
        video["sks_present1"] = sks_name1
        video["sks_present2"] = sks_name2
        # updata QA pairs
        for qa_pair in video["qa_pairs"]:
            qa_pair["question"] = qa_pair["question"].replace("<sks1>", sks_name1)
            qa_pair["answer"] = qa_pair["answer"].replace("<sks1>", sks_name1)
            qa_pair["question"] = qa_pair["question"].replace("<sks2>", sks_name2)
            qa_pair["answer"] = qa_pair["answer"].replace("<sks2>", sks_name2)

    # Save the modified file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def clean_old_files(sks_name1,sks_name2):
    base_path = "/root/autodl-tmp/yufei/datasets/cekebv-hq"
    files_to_remove = [
        "test_"+ sks_name1 + '_' + sks_name2 + '.json',
         "train_"+ sks_name1 + '_' + sks_name2 + '.json'
        "test_all_video_"+ sks_name1 + '_' + sks_name2 + '.json',
        "train_all_video_"+ sks_name1 + '_' + sks_name2 + '.json'
    ]

    for file_name in files_to_remove:
        file_path = os.path.join(base_path, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed: {file_path}")


def main():
    args = get_args()
    sks_name1 = '<'+args.sks1+'>'
    sks_name2 = '<'+args.sks2+'>'

    # clean old files
    clean_old_files(args.sks1,args.sks2)

    # Input file path
    input_train = "/root/autodl-tmp/yufei/datasets/cekebv-hq/train_all_video_updated_" + args.sks1 + '_' + args.sks2 + '.json'
    input_test = "/root/autodl-tmp/yufei/datasets/cekebv-hq/test_all_video_updated_" + args.sks1 + '_' + args.sks2 + '.json'

    # Output file path (use the full name directly, including Angle brackets)
    output_dir = "/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality"
    output_train = os.path.join(output_dir, f"{sks_name1}_{sks_name2}.json")
    output_test = os.path.join(output_dir, f"{sks_name1}_{sks_name2}_test.json")

    # Handle training and test files
    replace_sks_in_json(input_train, output_train, sks_name1,sks_name2)
    replace_sks_in_json(input_test, output_test, sks_name1,sks_name2)

    print(f"Files processed successfully:")
    print(f"Train file saved as: {output_train}")
    print(f"Test file saved as: {output_test}")


if __name__ == "__main__":
    main()