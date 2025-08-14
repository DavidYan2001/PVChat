# batch_generation.py
import os
import json
import random
import subprocess
from glob import glob
from multiprocessing import Process

def load_cekebv_data(train_json, test_json):
    """
    Load and merge train.json + test.json, return a list:
    [
      {
        "video_name": "...",
        "gender": "male" / "female",
        "age": "child" / "young" / "middle-aged" / "elderly",
        ...
      },
      ...
    ]
    """
    combined_data = []

    #for fpath in [train_json, test_json]:
    for fpath in [train_json]:
        if not os.path.exists(fpath):
            print(f"[Warning] File not found: {fpath}")
            continue

        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # data is usually {"data": [ {...}, {...} ]}
            if "data" in data:
                combined_data.extend(data["data"])
            else:
                print(f"[Warning] 'data' field not found in {fpath}, please check the file structure.")

    return combined_data


def split_workload(total_items, num_gpus):
    """
    Split workload according to available GPU count
    Return processing range list for each GPU [(start1, end1), (start2, end2), ...]
    """
    base_size = total_items // num_gpus
    remainder = total_items % num_gpus

    ranges = []
    start = 0
    for i in range(num_gpus):
        # If there's a remainder, front GPUs process one more
        size = base_size + (1 if i < remainder else 0)
        end = start + size
        ranges.append((start, end))
        start = end

    return ranges

def get_available_gpus():
    """
    Detect available GPUs in the system
    Return list of available GPU IDs
    """
    try:
        # Use nvidia-smi to get GPU information
        nvidia_smi = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )

        available_gpus = []
        for line in nvidia_smi.stdout.strip().split('\n'):
            index, used_memory, total_memory = map(float, line.split(','))
            # If memory usage is less than 20%, consider the GPU available
            if (used_memory / total_memory) < 0.2:
                available_gpus.append(int(index))

        return available_gpus
    except subprocess.SubprocessError:
        print("[Warning] Failed to run nvidia-smi. Defaulting to GPU 0")
        return [0]

def load_class_data(class_data_path):
    """
    Load class_data_modified.json, return nested dictionary, for example:
    {
      "male": {
        "child": { ... },
        "young": { ... },
        "middle_aged": { ... },
        "elderly": { ... }
      },
      "female": { ... }
    }
    """
    with open(class_data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def collect_leaf_texts(subtree):
    """
    Collect all bottom-level strings (leaf nodes) from nested dictionary subtree, return list.
    """
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
    """
    In train.json / test.json, age might be "middle-aged", but in class_data_modified.json it's "middle_aged".
    Do a simple mapping conversion.
    """
    mapping = {
        "child": "child",
        "young": "young",
        "middle-aged": "middle_aged",
        "middle_aged": "middle_aged",
        "elderly": "elderly"
    }
    return mapping.get(age_str, age_str)  # If not mapped, keep original


def get_prompts_for_gender_age(class_data, gender, age, num_prompts=10):
    """
    Based on gender("male"/"female") and age("child"/"young"/"middle_aged"/"elderly"),
    find corresponding subtree in class_data, collect all leaf descriptions, and randomly select num_prompts.
    If less than num_prompts, return all.
    """
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


def generate_videos_for_range(
    video_info_list,
    deepfacelab_output_path,
    class_data,
    consisid_script_path,
    gpu_id,
    start_idx,
    end_idx
):
    """
    Process videos from video_info_list[start_idx:end_idx] on specified GPU.
    Each element in video_info_list contains:
    {
      "video_name": "xxx.mp4",
      "gender": "male"/"female",
      "age": "child"/"young"/"middle-aged"/"elderly",
      ...
    }
    """
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    total_videos = len(video_info_list)
    end_idx = min(end_idx, total_videos)

    for i in range(start_idx, end_idx):
        video_data = video_info_list[i]
        video_name = video_data.get("video_name", "")
        if not video_name:
            print(f"[GPU {gpu_id}] No video_name, skip.")
            continue

        # Remove suffix, e.g. "d82LKPNesE8_112.mp4" -> "d82LKPNesE8_112"
        base_name = os.path.splitext(video_name)[0]

        # Check if /root/autodl-tmp/yufei/DeepFaceLab/output/<base_name> exists
        video_output_dir = os.path.join(deepfacelab_output_path, base_name)
        if not os.path.isdir(video_output_dir):
            print(f"[GPU {gpu_id}] Directory not found for video: {video_output_dir}, skip.")
            continue

        # Get gender and age, and convert (middle-aged -> middle_aged)
        gender = video_data.get("gender", "male")  # Default to male if not specified
        raw_age = video_data.get("age", "young")   # Default to young if not specified
        age = normalize_age(raw_age)

        print(f"[GPU {gpu_id}] Processing {video_name} ({i+1}/{end_idx}), gender={gender}, age={age}.")

        # HQ_face directory
        hq_face_dir = os.path.join(video_output_dir, "HQ_face")
        if not os.path.isdir(hq_face_dir):
            print(f"[GPU {gpu_id}] {hq_face_dir} not found, skip.")
            continue

        # Get all face images
        face_images = glob(os.path.join(hq_face_dir, "*.jpg")) + glob(os.path.join(hq_face_dir, "*.png"))
        if not face_images:
            print(f"[GPU {gpu_id}] No images found in {hq_face_dir}, skip.")
            continue

        # Get random prompts corresponding to (gender, age) from class_data
        prompts = get_prompts_for_gender_age(class_data, gender, age, num_prompts=10)
        if not prompts:
            print(f"[GPU {gpu_id}] No prompts for gender={gender}, age={age}, skip.")
            continue

        # Only create consisid_generation directory when really needed here
        consisid_dir = os.path.join(video_output_dir, "consisid_generation")
        os.makedirs(consisid_dir, exist_ok=True)

        existing_videos = glob(os.path.join(consisid_dir, "*.mp4"))
        if existing_videos:
            print(f"[GPU {gpu_id}] Already found generated videos in {consisid_dir}, skip this video.")
            continue
        # For each prompt, randomly select a face
        for idx_p, prompt in enumerate(prompts, start=1):
            face_image = random.choice(face_images)
            cmd = [
                "python",
                consisid_script_path,
                "--img_file_path", face_image,
                "--prompt", prompt,
                "--output_path", consisid_dir,
                "--seed", "42",
                "--is_upscale",
                "--is_frame_interpolation"
            ]

            print(f"[GPU {gpu_id}] Generating {idx_p}/{len(prompts)} for {video_name} with {os.path.basename(face_image)}.")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[GPU {gpu_id}] Generation failed: {str(e)}.")
                continue


def run_parallel_processing(
    train_json,
    test_json,
    class_data_path,
    deepfacelab_output_path,
    consisid_script_path
):
    """
    Main entry function:
    1) Read and merge train.json + test.json
    2) Read class_data_modified.json
    3) Randomly shuffle video list
    4) Split into parts for GPU=0 and GPU=1
    """

    available_gpus = get_available_gpus()
    if not available_gpus:
        print("[Error] No available GPUs found. Exiting...")
        return

    print(f"[Main] Found {len(available_gpus)} available GPU(s): {available_gpus}")
    # 1) Load train.json + test.json
    video_info_list = load_cekebv_data(train_json, test_json)
    if not video_info_list:
        print("[Main] No videos found in train/test data, exit.")
        return

    # 2) Load class_data_modified.json
    class_data = load_class_data(class_data_path)

    total_videos = len(video_info_list)
    print(f"[Main] Total videos: {total_videos}")

    # Randomly shuffle list for more even distribution to GPUs
    random.shuffle(video_info_list)
    # 3) Assign tasks based on available GPU count
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

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("[Main] All processes completed.")


if __name__ == "__main__":
    # Example paths, modify according to actual situation
    train_json = "/root/autodl-tmp/yufei/datasets/cekebv-hq/train.json"
    test_json = "/root/autodl-tmp/yufei/datasets/cekebv-hq/test.json"
    class_data_path = "/root/autodl-tmp/yufei/ConsisID/class_data_modified.json"

    deepfacelab_output_path = "/root/autodl-tmp/yufei/DeepFaceLab/output"
    consisid_script_path = "/root/autodl-tmp/yufei/ConsisID/infer.py"

    print("Starting parallel processing on 2 GPUs...")
    run_parallel_processing(
        train_json,
        test_json,
        class_data_path,
        deepfacelab_output_path,
        consisid_script_path
    )
    print("Done.")
