#!/usr/bin/env python
import argparse
import json
import os
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='Extract first 8 frames of each video and update JSON')
    parser.add_argument('--sks1', type=str, default='Sh',
                        help='Name of first person (default: Sh)')
    parser.add_argument('--sks2', type=str, default='Ho',
                        help='Name of second person (default: Ho)')
    return parser.parse_args()


def create_short_video2(input_video_path, output_video_path, num_frames=8):
    """
    Read the input video, extract the first num_frames, and save them as a new video
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        return False

    frames = []
    # Skip the first 9 frames
    for _ in range(9):
        ret, _ = cap.read()
        if not ret:
            print("Failed to skip frames before the 10th.")
            cap.release()
            return False


    ret, tenth_frame = cap.read()
    cap.release()  # No longer read subsequent frames; release directly

    if not ret or tenth_frame is None:
        print("Failed to read the 10th frame.")
        return False
    # Get the first frame size and try to obtain the FPS of the original video; otherwise, the default is 30
    height, width, _ = tenth_frame.shape
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for _ in range(num_frames):
        out.write(tenth_frame)
    out.release()
    return True

def create_short_video(input_video_path, output_video_path, num_frames=8):
    """
    Read only the first frame from the input video, then copy it num_frames several times and save it as a new video
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        return False

    # read first frames
    ret, first_frame = cap.read()
    cap.release()  # No longer read subsequent frames; release directly

    if not ret or first_frame is None:
        print("Failed to read the first frame.")
        return False

    # Repeat this first frame num frames times
    height, width, _ = first_frame.shape
    fps = 30  # Here you can fix it at 30, or you can try cap.get(cv2.CAP_PROP_FPS) to see if it works
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for _ in range(num_frames):
        out.write(first_frame)

    out.release()
    return True


def process_videos(input_json_path, sks_name1, sks_name2):
    # read JSON file
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    videos = data.get("videos", [])
    if not videos:
        print("No video entries found in JSON.")
        return []

    # Construct a new output folder
    folder_name = sks_name1 + "_" + sks_name2 + "_short_video"
    output_folder = os.path.join("/root/autodl-tmp/yufei/DeepFaceLab/output", folder_name)
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder created: {output_folder}")

    # Set a public parent directory to serve as the benchmark for relative paths
    base_dir = "/root/autodl-tmp/yufei/DeepFaceLab/output"

    new_videos = []
    for video in videos:
        orig_video_path = video.get("video_path")
        if not orig_video_path or not os.path.exists(orig_video_path):

            continue

        # Switch to the absolute path
        abs_video_path = os.path.abspath(orig_video_path)

        # Split the path into a list of various levels of directories and file names
        parts = abs_video_path.split(os.sep)

        # If you want to ensure that at least three levels (the third from the bottom, the second from the bottom, and the first from the bottom) are available, you can make a simple judgment:
        if len(parts) < 3:
            print(f"Warning: path {abs_video_path} has less than 3 segments.")
            continue

        # Take the last three
        # The third from the bottom => parts[-3] (such as "cekebv-hq")
        # The second to last => parts[-2] (e.g. "35666")
        # The last one => parts[-1] (e.g. "Cixf80-WZX0_3.mp4")
        dir_part_1 = parts[-3]
        dir_part_2 = parts[-2]
        base_filename = parts[-1]  # "Cixf80-WZX0_3.mp4"

        # Separate file names from extensions
        filename_no_ext, ext = os.path.splitext(base_filename)
        # filename_no_ext => "Cixf80-WZX0_3"
        # ext             => ".mp4"

        # Concatenate them into new file names: "cekebv-hq_35666_Cixf80-WZX0_3_short_video.mp4"
        new_video_name = f"{dir_part_1}_{dir_part_2}_{filename_no_ext}_short_video{ext}"

        # The complete output path is obtained by splicing
        new_video_path = os.path.join(output_folder, new_video_name)
        print(f"Processing: {orig_video_path} => {new_video_path}")

        # Call your create_short_video function (it has been changed to take the 10th frame and repeat 8 times)
        success = create_short_video(orig_video_path, new_video_path, num_frames=8)
        if not success:
            continue

        # Update video entries
        new_video = video.copy()
        new_video["video_name"] = new_video_name
        new_video["video_path"] = new_video_path

        # Keep only questions in qa_pairs where "is_special" is True or where the clothing description is involved
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


def save_new_json(new_videos, input_json_path, sks_name1,sks_name2):
    input_dir = os.path.dirname(input_json_path)

    output_json_name = f"<{sks_name1}>_<{sks_name2}>_short_train.json"
    output_json_path = os.path.join(input_dir, output_json_name)

    new_data = {"videos": new_videos}
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    print(f"New JSON saved: {output_json_path}")
    return output_json_path


def main():
    args = parse_args()
    sks_name1 = args.sks1
    sks_name2 = args.sks2

    # Enter the JSON file path (located in the multi_modality directory)
    input_json_path = os.path.join("/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality", f"<{sks_name1}>_<{sks_name2}>.json")
    if not os.path.exists(input_json_path):
        print(f"Input JSON file not found: {input_json_path}")
        return

    new_videos = process_videos(input_json_path, sks_name1,sks_name2)
    if not new_videos:
        print("No videos processed.")
        return

    save_new_json(new_videos, input_json_path, sks_name1,sks_name2)


if __name__ == "__main__":
    main()
