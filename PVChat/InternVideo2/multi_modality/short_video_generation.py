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
    Read input video, extract first num_frames frames, save as new video
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

    # Get first frame dimensions, try to get original video FPS, otherwise default to 30
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
    From input video skip first 9 frames, read the 10th frame, then repeat it num_frames times, save as new video
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        return False

    # Skip first 9 frames
    for _ in range(50):
        ret, _ = cap.read()
        if not ret:
            print("Failed to skip 9 frames before reading the 10th frame.")
            cap.release()
            return False

    # Read the 10th frame
    ret, tenth_frame = cap.read()
    cap.release()  # No longer reading subsequent frames, release directly

    if not ret or tenth_frame is None:
        print("Failed to read the 10th frame.")
        return False

    # Prepare to write: repeat this 10th frame num_frames times
    height, width, _ = tenth_frame.shape
    fps = 30  # Fixed at 30, could also try cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for _ in range(num_frames):
        out.write(tenth_frame)

    out.release()
    return True

def process_videos(input_json_path, sks_name):
    # Read JSON file
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    videos = data.get("videos", [])
    if not videos:
        print("No video entries found in JSON.")
        return []

    # Construct new output folder (remove angle brackets, add _short_video)
    folder_name = sks_name.strip("<>") + "_short_video"
    output_folder = os.path.join("/root/autodl-tmp/yufei/DeepFaceLab/output", folder_name)
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder created: {output_folder}")

    # Can specify a common parent directory first to get relative paths
    base_dir = "/root/autodl-tmp/yufei/DeepFaceLab/output"

    new_videos = []
    for video in videos:
        orig_video_path = video.get("video_path")
        if not orig_video_path or not os.path.exists(orig_video_path):
            # Ignore non-existent files
            continue

        # Convert to absolute path
        abs_video_path = os.path.abspath(orig_video_path)

        # Split path into list of directories and filename
        parts = abs_video_path.split(os.sep)
        # Example: ['', 'root', 'autodl-tmp', 'yufei', 'datasets', 'cekebv-hq', '35666', 'Cixf80-WZX0_3.mp4']

        # If want to ensure at least 3 levels available (3rd from last, 2nd from last, last), do a simple check:
        if len(parts) < 3:
            print(f"Warning: path {abs_video_path} has less than 3 segments.")
            continue

        # Take last 3
        #   3rd from last => parts[-3]  (e.g. "cekebv-hq")
        #   2nd from last => parts[-2]  (e.g. "35666")
        #   last => parts[-1]  (e.g. "Cixf80-WZX0_3.mp4")
        dir_part_1 = parts[-3]
        dir_part_2 = parts[-2]
        base_filename = parts[-1]  # "Cixf80-WZX0_3.mp4"

        # Separate filename and extension
        filename_no_ext, ext = os.path.splitext(base_filename)
        # filename_no_ext => "Cixf80-WZX0_3"
        # ext             => ".mp4"

        # Concatenate into new filename: "cekebv-hq_35666_Cixf80-WZX0_3_short_video.mp4"
        new_video_name = f"{dir_part_1}_{dir_part_2}_{filename_no_ext}_short_video{ext}"

        # Concatenate to get complete output path
        new_video_path = os.path.join(output_folder, new_video_name)
        print(f"Processing: {orig_video_path} => {new_video_path}")

        # Call your create_short_video function (modified to take 10th frame, repeat 8 times)
        success = create_short_video(orig_video_path, new_video_path, num_frames=8)
        if not success:
            continue

        # Update video entry
        new_video = video.copy()
        new_video["video_name"] = new_video_name
        new_video["video_path"] = new_video_path

        # Only keep qa_pairs where "is_special" is true, and clothing-related Q&As
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

    # Input JSON file path (located in multi_modality directory)
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