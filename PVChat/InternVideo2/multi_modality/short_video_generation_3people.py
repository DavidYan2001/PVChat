#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract the 10th frame of each video and repeat it to form a short clip. '
                    'Then update JSON (3-people version).'
    )
    parser.add_argument('--sks1', type=str, default='Cl',
                        help='Name of first person (default: Cl)')
    parser.add_argument('--sks2', type=str, default='Xo',
                        help='Name of second person (default: Xo)')
    parser.add_argument('--sks3', type=str, default='Ja',
                        help='Name of third person (default: Ja)')
    return parser.parse_args()


def create_short_video(input_video_path, output_video_path, num_frames=8):
    """
    从输入视频里【跳过前9帧，读取第10帧】，然后把这帧复制 num_frames 次，保存为新视频。
    如果你想改为读取第一帧，或者其它帧，可自行修改此函数。
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        return False

    # 跳过前 9 帧
    for _ in range(9):
        ret, _ = cap.read()
        if not ret:
            print("[Warning] Not enough frames to skip 9. Video too short?")
            cap.release()
            return False

    # 读取第 10 帧
    ret, tenth_frame = cap.read()
    cap.release()

    if not ret or tenth_frame is None:
        print("[Warning] Failed to read the 10th frame.")
        return False

    # 根据该帧的尺寸/通道数写视频
    height, width, _ = tenth_frame.shape
    fps = 30  # 固定FPS为30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for _ in range(num_frames):
        out.write(tenth_frame)

    out.release()
    return True


def process_videos(input_json_path, sks_name1, sks_name2, sks_name3):
    """
    1) 读取 input_json_path 对应的 JSON
    2) 对每个 video, 取其 video_path 并检查是否存在
    3) 在 /root/autodl-tmp/yufei/DeepFaceLab/output 下建立新文件夹 {sks1}_{sks2}_{sks3}_short_video
    4) 为每个视频调用 create_short_video => 生成短视频
    5) 更新 video_name 和 video_path
    6) 仅保留 qa_pairs 里 is_special=True 或衣着描述相关的问答
    7) 返回更新后的视频列表
    """
    if not os.path.exists(input_json_path):
        print(f"[Warning] JSON file not found: {input_json_path}")
        return []

    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    videos = data.get("videos", [])
    if not videos:
        print("No 'videos' list found in JSON.")
        return []

    # 创建输出文件夹
    folder_name = f"{sks_name1}_{sks_name2}_{sks_name3}_short_video"
    output_folder = os.path.join("/root/autodl-tmp/yufei/DeepFaceLab/output", folder_name)
    os.makedirs(output_folder, exist_ok=True)
    print(f"[Info] Output folder created: {output_folder}")

    new_videos = []

    # 准备过滤 QA 的关键词
    special_substrings = [
        "wearing in this video?",
        "outfit in this footage?",
        "What color and style of clothing",
        "How would you describe",
        "What notable features can you see"
    ]

    for video in videos:
        orig_video_path = video.get("video_path")
        if not orig_video_path or not os.path.exists(orig_video_path):
            print(f"[Warning] video path not found: {orig_video_path}")
            continue

        abs_video_path = os.path.abspath(orig_video_path)
        parts = abs_video_path.split(os.sep)
        if len(parts) < 3:
            print(f"[Warning] Path {abs_video_path} has fewer than 3 segments.")
            continue

        # 取末尾3个
        dir_part_1 = parts[-3]  # 如 "cekebv-hq"
        dir_part_2 = parts[-2]  # 如 "35666"
        base_filename = parts[-1]  # "Cixf80-WZX0_3.mp4"

        filename_no_ext, ext = os.path.splitext(base_filename)
        # 构造新的短视频文件名
        new_video_name = f"{dir_part_1}_{dir_part_2}_{filename_no_ext}_short_video{ext}"
        new_video_path = os.path.join(output_folder, new_video_name)
        print(f"[Info] Processing: {orig_video_path} => {new_video_path}")

        success = create_short_video(orig_video_path, new_video_path, num_frames=8)
        if not success:
            continue

        new_video = video.copy()
        new_video["video_name"] = new_video_name
        new_video["video_path"] = new_video_path

        # 保留 QA
        original_qa = video.get("qa_pairs", [])
        filtered_qa = [
            qa for qa in original_qa
            if qa.get("is_special", False)
               or any(sub in qa.get("question", "") for sub in special_substrings)
        ]
        new_video["qa_pairs"] = filtered_qa

        new_videos.append(new_video)

    return new_videos


def save_new_json(new_videos, input_json_path, sks_name1, sks_name2, sks_name3):
    """
    将 new_videos 写回 JSON (仍然使用 "videos" 键).
    输出文件名:  <sks1>_<sks2>_<sks3>_short_train.json
    存放在 input_json_path 同目录
    """
    input_dir = os.path.dirname(input_json_path)
    output_json_name = f"<{sks_name1}>_<{sks_name2}>_<{sks_name3}>_short_train.json"
    output_json_path = os.path.join(input_dir, output_json_name)

    new_data = {"videos": new_videos}
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print(f"[Info] New JSON saved => {output_json_path}")
    return output_json_path


def main():
    args = parse_args()
    sks_name1 = args.sks1
    sks_name2 = args.sks2
    sks_name3 = args.sks3

    # 假设输入 JSON 位于 multi_modality 目录:
    base_dir = "/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality"
    input_json_path = os.path.join(base_dir, f"<{sks_name1}>_<{sks_name2}>_<{sks_name3}>.json")

    if not os.path.exists(input_json_path):
        print(f"[Error] Input JSON not found: {input_json_path}")
        return

    new_videos = process_videos(input_json_path, sks_name1, sks_name2, sks_name3)
    if not new_videos:
        print("[Info] No videos processed or no valid entries.")
        return

    save_new_json(new_videos, input_json_path, sks_name1, sks_name2, sks_name3)


if __name__ == "__main__":
    main()
