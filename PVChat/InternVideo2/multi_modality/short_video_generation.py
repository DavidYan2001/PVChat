#!/usr/bin/env python
import argparse
import json
import os
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='Extract first 8 frames of each video and update JSON')
    parser.add_argument('--sks_name', type=str, required=True,
                        help='Identifier for videos (e.g., "<Pe>")')
    return parser.parse_args()


def create_short_video1(input_video_path, output_video_path, num_frames=8):
    """
    读取输入视频，截取前 num_frames 帧，保存为新视频
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        return False

    frames = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        print("No frames extracted.")
        return False

    # 获取第一帧尺寸，尝试获取原视频FPS，否则默认为30
    height, width, _ = frames[0].shape
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    return True

def create_short_video(input_video_path, output_video_path, num_frames=8):
    """
    从输入视频跳过前9帧，读取第10帧，然后把它重复 num_frames 次，保存为新视频
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        return False

    # 跳过前9帧
    for _ in range(50):
        ret, _ = cap.read()
        if not ret:
            print("Failed to skip 9 frames before reading the 10th frame.")
            cap.release()
            return False

    # 读取第10帧
    ret, tenth_frame = cap.read()
    cap.release()  # 不再读取后续帧，直接释放

    if not ret or tenth_frame is None:
        print("Failed to read the 10th frame.")
        return False

    # 准备写出：将这第 10 帧重复 num_frames 次
    height, width, _ = tenth_frame.shape
    fps = 30  # 固定为30，也可以尝试 cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for _ in range(num_frames):
        out.write(tenth_frame)

    out.release()
    return True

def process_videos(input_json_path, sks_name):
    # 读取 JSON 文件
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    videos = data.get("videos", [])
    if not videos:
        print("No video entries found in JSON.")
        return []

    # 构造新输出文件夹（去除尖括号后，加上 _short_video）
    folder_name = sks_name.strip("<>") + "_short_video"
    output_folder = os.path.join("/root/autodl-tmp/yufei/DeepFaceLab/output", folder_name)
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder created: {output_folder}")

    # 可以先指定一个公共父目录，用于获取相对路径
    base_dir = "/root/autodl-tmp/yufei/DeepFaceLab/output"

    new_videos = []
    for video in videos:
        orig_video_path = video.get("video_path")
        if not orig_video_path or not os.path.exists(orig_video_path):
            # 忽略不存在的文件
            continue

        # 转为绝对路径
        abs_video_path = os.path.abspath(orig_video_path)

        # 将路径拆分成各级目录和文件名的列表
        parts = abs_video_path.split(os.sep)
        # 例: ['', 'root', 'autodl-tmp', 'yufei', 'datasets', 'cekebv-hq', '35666', 'Cixf80-WZX0_3.mp4']

        # 如果想保证一定有至少3级（倒数第3、倒数第2、倒数第1）可用，可以做一个简单判断：
        if len(parts) < 3:
            print(f"Warning: path {abs_video_path} has less than 3 segments.")
            continue

        # 取末尾3个
        #   倒数第3 => parts[-3]  (如 "cekebv-hq")
        #   倒数第2 => parts[-2]  (如 "35666")
        #   倒数第1 => parts[-1]  (如 "Cixf80-WZX0_3.mp4")
        dir_part_1 = parts[-3]
        dir_part_2 = parts[-2]
        base_filename = parts[-1]  # "Cixf80-WZX0_3.mp4"

        # 分离文件名与扩展名
        filename_no_ext, ext = os.path.splitext(base_filename)
        # filename_no_ext => "Cixf80-WZX0_3"
        # ext             => ".mp4"

        # 拼接成新的文件名: "cekebv-hq_35666_Cixf80-WZX0_3_short_video.mp4"
        new_video_name = f"{dir_part_1}_{dir_part_2}_{filename_no_ext}_short_video{ext}"

        # 拼接得到完整输出路径
        new_video_path = os.path.join(output_folder, new_video_name)
        print(f"Processing: {orig_video_path} => {new_video_path}")

        # 调用你的 create_short_video 函数(已改为取第10帧、重复8次)
        success = create_short_video(orig_video_path, new_video_path, num_frames=8)
        if not success:
            continue

        # 更新视频条目
        new_video = video.copy()
        new_video["video_name"] = new_video_name
        new_video["video_path"] = new_video_path

        # 仅保留 qa_pairs 中 "is_special" 为 true 的问答对，以及服装相关问答
        special_substrings = [
            "wearing in this video?",
            "outfit in this footage?",
            "What color and style of clothing",
            "How would you describe",
            "What notable features can you see"
        ]
        original_qa = video.get("qa_pairs", [])
        filtered_qa = [
            qa for qa in original_qa
            if qa.get("is_special", False)
               or any(sub in qa.get("question", "") for sub in special_substrings)
        ]
        new_video["qa_pairs"] = filtered_qa

        new_videos.append(new_video)

    return new_videos



def save_new_json(new_videos, input_json_path, sks_name):
    input_dir = os.path.dirname(input_json_path)
    output_json_name = f"{sks_name}_short_train.json"
    output_json_path = os.path.join(input_dir, output_json_name)

    new_data = {"videos": new_videos}
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    print(f"New JSON saved: {output_json_path}")
    return output_json_path


def main():
    args = parse_args()
    sks_name = args.sks_name

    # 输入 JSON 文件路径（位于 multi_modality 目录下）
    input_json_path = os.path.join("/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality", f"{sks_name}.json")
    if not os.path.exists(input_json_path):
        print(f"Input JSON file not found: {input_json_path}")
        return

    new_videos = process_videos(input_json_path, sks_name)
    if not new_videos:
        print("No videos processed.")
        return

    save_new_json(new_videos, input_json_path, sks_name)


if __name__ == "__main__":
    main()
