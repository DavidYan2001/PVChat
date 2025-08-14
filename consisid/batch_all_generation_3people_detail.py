import os
import json
import random
import subprocess
from glob import glob
from multiprocessing import Process
import argparse
def load_cekebv_data(train_json, test_json):
    """
    Load train.json + test.json and merge them to return a list:
    [
      {
        "video_name": "...",
        "sample_type": "both"/"person1"/"person2"/"random"/...,
        "gender": "male"/"female",
        "age": "child"/"young"/"middle-aged"/"elderly",
        "gender2": "male"/"female",
        "age2": "young"/"middle-aged"/...,
        ...
      },
      ...
    ]
    """
    combined_data = []
    for fpath in [train_json, test_json]:
        if not os.path.exists(fpath):
            print(f"[Warning] File not found: {fpath}")
            continue

        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "data" in data:
                combined_data.extend(data["data"])
            else:
                print(f"[Warning] 'data' field not found in {fpath}, please check the file structure.")

    return combined_data


def split_workload(total_items, num_gpus):
    """Divide the workload based on the number of available Gpus and return [(start1, end1), (start2, end2), ...]"""
    base_size = total_items // num_gpus
    remainder = total_items % num_gpus

    ranges = []
    start = 0
    for i in range(num_gpus):
        size = base_size + (1 if i < remainder else 0)
        end = start + size
        ranges.append((start, end))
        start = end

    return ranges

def get_available_gpus():
    """The detection system is available with GPU and returns the list of ids."""
    import subprocess
    try:
        nvidia_smi = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        available_gpus = []
        for line in nvidia_smi.stdout.strip().split('\n'):
            index_str, used_str, total_str = line.split(',')
            index = int(float(index_str))
            used = float(used_str)
            total = float(total_str)
            if total == 0:
                continue
            # If the video memory usage rate is less than 20%, it is considered usable
            if (used / total) < 0.2:
                available_gpus.append(index)
        return available_gpus
    except subprocess.SubprocessError:
        print("[Warning] Failed to run nvidia-smi. Defaulting to GPU 0")
        return [0]

def load_class_data(class_data_path):
    """加载 class_data_modified.json"""
    with open(class_data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def collect_leaf_texts(subtree):
    "Collect all the lowest-level strings (leaf nodes) from the nested dictionary subtree and return a list." ""
    results = []
    if isinstance(subtree, dict):
        for _, v in subtree.items():
            results.extend(collect_leaf_texts(v))
    elif isinstance(subtree, list):
        for item in subtree:
            results.extend(collect_leaf_texts(item))
    elif isinstance(subtree, str):
        results.append(subtree)
    return results

def normalize_age(age_str):
    """"middle-aged" -> "middle_aged" 。"""
    mapping = {
        "child": "child",
        "young": "young",
        "middle-aged": "middle_aged",
        "middle_aged": "middle_aged",
        "elderly": "elderly"
    }
    return mapping.get(age_str, age_str)  # 如果没有映射到，就保持原样

def get_prompts_for_gender_age(class_data, gender, age, num_prompts=10):
    """
    Find the corresponding subtree in class_data based on gender + age, collect all leaf descriptions, and randomly select num_prompts
    """
    if not gender or not age:
        return []
    if gender not in class_data:
        print(f"[Info] Gender={gender} not found in class_data, return empty list.")
        return []

    gender_data = class_data[gender]
    if age not in gender_data:
        print(f"[Info] Age={age} not found in class_data[{gender}], return empty list.")
        return []

    subtree = gender_data[age]
    all_leaf_texts = collect_leaf_texts(subtree)
    if not all_leaf_texts:
        return []

    random.shuffle(all_leaf_texts)
    return all_leaf_texts[:num_prompts]


def generate_videos_for_person(
    face_images,
    prompts,
    consisid_dir,
    gpu_id,
    video_name,
    person_suffix=""
):
    """
    For a certain person's face + prompt, invoke consisid_script to generate a video.
person_suffix is used to distinguish output files, such as "_2".
    """
    for idx_p, prompt in enumerate(prompts, start=1):
        face_image = random.choice(face_images)
        cmd = [
            "python",
            consisid_script_path,  #
            "--img_file_path", face_image,
            "--prompt", prompt,
            "--output_path", consisid_dir,
            "--seed", "42",
            "--is_upscale",
            "--is_frame_interpolation"
        ]

        print(f"[GPU {gpu_id}] Generating {idx_p}/{len(prompts)} for {video_name}{person_suffix} with {os.path.basename(face_image)} | {prompt}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[GPU {gpu_id}] Generation failed: {str(e)}.")
            continue

def generate_videos_for_range(
    video_info_list,
    deepfacelab_output_path,
    class_data,
    consisid_script_path_,
    gpu_id,
    start_idx,
    end_idx
):
    """
    Process the video within the range of video_info_list[start_idx:end_idx] on the specified GPU.
    """
    # 设置 CUDA 设备
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    global consisid_script_path
    consisid_script_path = consisid_script_path_

    total_videos = len(video_info_list)
    end_idx = min(end_idx, total_videos)

    for i in range(start_idx, end_idx):
        video_data = video_info_list[i]
        video_name = video_data.get("video_name", "")
        if not video_name:
            print(f"[GPU {gpu_id}] No video_name at index {i}, skip.")
            continue

        sample_type = video_data.get("sample_type", "single")


        base_name = os.path.splitext(video_name)[0]


        video_output_dir = os.path.join(deepfacelab_output_path, base_name)
        if not os.path.isdir(video_output_dir):
            print(f"[GPU {gpu_id}] Directory not found for video: {video_output_dir}, skip.")
            continue

        # 读取第一人的 gender / age
        gender = video_data.get("gender")
        raw_age = video_data.get("age")
        age = normalize_age(raw_age)

        # 读取第二人的 gender2 / age2
        gender2 = video_data.get("gender2")
        raw_age2 = video_data.get("age2")
        age2 = normalize_age(raw_age2)

        print(f"\n[GPU {gpu_id}] Processing {video_name} ({i+1}/{end_idx}) | sample_type={sample_type}")
        print(f" - Person1(gender={gender}, age={age}), Person2(gender={gender2}, age={age2})")

        # HQ_face (person1)
        hq_face_dir = os.path.join(video_output_dir, "HQ_face")
        face_images_1 = []
        if os.path.isdir(hq_face_dir):
            face_images_1 = glob(os.path.join(hq_face_dir, "*.jpg")) + glob(os.path.join(hq_face_dir, "*.png"))

        # HQ_face2 (person2)
        hq_face2_dir = os.path.join(video_output_dir, "HQ_face2")
        face_images_2 = []
        if os.path.isdir(hq_face2_dir):
            face_images_2 = glob(os.path.join(hq_face2_dir, "*.jpg")) + glob(os.path.join(hq_face2_dir, "*.png"))

        # 准备 prompts
        prompts_1 = get_prompts_for_gender_age(class_data, gender, age, num_prompts=10)
        prompts_2 = get_prompts_for_gender_age(class_data, gender2, age2, num_prompts=10)

        # 如果不是 both，就清空第二人的 prompts
        if sample_type != "both":
            prompts_2 = []

        # =============== 新增：分开两个输出目录 ===============
        # 单人或第一人 => consisid_generation
        consisid_dir_1 = os.path.join(video_output_dir, "consisid_generation")
        os.makedirs(consisid_dir_1, exist_ok=True)

        # 第二人 => consisid_generation2
        consisid_dir_2 = os.path.join(video_output_dir, "consisid_generation2")
        os.makedirs(consisid_dir_2, exist_ok=True)

        # =============== Skip Check ===============

        existing_videos_p1 = glob(os.path.join(consisid_dir_1, "*.mp4"))
        skip_person1 = (len(existing_videos_p1) >= 10)

        existing_videos_p2 = glob(os.path.join(consisid_dir_2, "*.mp4"))
        skip_person2 = (len(existing_videos_p2) >= 10)

        # ============ 生成第一人 ============
        if not skip_person1:
            if gender and age and face_images_1 and prompts_1:
                print(f"[GPU {gpu_id}] Person1 => {consisid_dir_1}")
                print(f"   {len(face_images_1)} faces, {len(prompts_1)} prompts.")
                generate_videos_for_person(
                    face_images=face_images_1,
                    prompts=prompts_1,
                    consisid_dir=consisid_dir_1,
                    gpu_id=gpu_id,
                    video_name=video_name,
                    person_suffix=""
                )
            else:
                print(f"[GPU {gpu_id}] Person1 data incomplete or no prompts/faces, skip Person1.")
        else:
            print(f"[GPU {gpu_id}] Already >=10 videos in {consisid_dir_1}, skip Person1.")

        # ============ 生成第二人 (only if sample_type == 'both') ============
        if sample_type == "both":
            if not skip_person2:
                if gender2 and age2 and face_images_2 and prompts_2:
                    print(f"[GPU {gpu_id}] Person2 => {consisid_dir_2}")
                    print(f"   {len(face_images_2)} faces, {len(prompts_2)} prompts.")
                    generate_videos_for_person(
                        face_images=face_images_2,
                        prompts=prompts_2,
                        consisid_dir=consisid_dir_2,
                        gpu_id=gpu_id,
                        video_name=video_name,
                        person_suffix="_2"
                    )
                else:
                    print(f"[GPU {gpu_id}] Person2 data or directory incomplete, skip Person2.")
            else:
                print(f"[GPU {gpu_id}] Already >=10 videos in {consisid_dir_2}, skip Person2.")


def run_parallel_processing(
    train_json,
    test_json,
    class_data_path,
    deepfacelab_output_path,
    consisid_script
):
    """
    Main entry function
    Read train.json + test.json and merge them
    2) Read the class_data_modified.json
    3) Randomly shuffle the video list
    4) Multi-process processing
    """
    global consisid_script_path
    consisid_script_path = consisid_script

    available_gpus = get_available_gpus()
    if not available_gpus:
        print("[Error] No available GPUs found. Exiting...")
        return

    print(f"[Main] Found {len(available_gpus)} available GPU(s): {available_gpus}")

    video_info_list = load_cekebv_data(train_json, test_json)
    if not video_info_list:
        print("[Main] No videos found in train/test data, exit.")
        return

    class_data = load_class_data(class_data_path)

    total_videos = len(video_info_list)
    print(f"[Main] Total videos: {total_videos}")


    random.shuffle(video_info_list)


    workload_ranges = split_workload(total_videos, len(available_gpus))
    processes = []
    for gpu_idx, (start, end) in zip(available_gpus, workload_ranges):
        p = Process(
            target=generate_videos_for_range,
            args=(
                video_info_list,
                deepfacelab_output_path,
                class_data,
                consisid_script_path,
                gpu_idx,
                start,
                end
            )
        )
        print(f"[Main] Starting process on GPU {gpu_idx} for range [{start}:{end}]...")
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("[Main] All processes completed.")
def get_args():
    parser = argparse.ArgumentParser(description="Build dataset for two people and both cases")

    parser.add_argument('--sks1', type=str, default='Sh',
                        help='Name of first person (default: Sh)')
    parser.add_argument('--sks2', type=str, default='Ho',
                        help='Name of second person (default: Ho)')
    parser.add_argument('--sks3', type=str, default='Ho',
                        help='Name of second person (default: Ho)')
    return parser.parse_args()

if __name__ == "__main__":

    args = get_args()
    train_json = "/root/autodl-tmp/yufei/datasets/cekebv-hq/train_"+args.sks1+'_'+args.sks2+args.sks3+'.json'
    test_json = "/root/autodl-tmp/yufei/datasets/cekebv-hq/test_"+args.sks1+'_'+args.sks2+args.sks3+'.json'
    class_data_path = "/root/autodl-tmp/yufei/ConsisID/class_data_modified.json"

    deepfacelab_output_path = "/root/autodl-tmp/yufei/DeepFaceLab/output"
    consisid_script_path = "/root/autodl-tmp/yufei/ConsisID/infer.py"

    print("Starting parallel processing...")
    run_parallel_processing(
        train_json,
        test_json,
        class_data_path,
        deepfacelab_output_path,
        consisid_script_path
    )
    print("Done.")
