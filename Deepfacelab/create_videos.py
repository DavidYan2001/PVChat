import cv2
import sys
import os
from pathlib import Path

def create_group_videos(filtered_dir, video_group_dir):
    FPS = 30
    FRAME_SIZE = (512, 512)

    for group_dir in Path(filtered_dir).glob("*"):
        if not group_dir.is_dir():
            continue

        group_name = group_dir.name
        video_path = Path(video_group_dir) / f"{group_name}.mp4"

        images = sorted(group_dir.glob("*.jpg"))
        if not images:
            print(f"No images found in {group_dir}. Skipping...")
            continue

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, FPS, FRAME_SIZE)

        for image_path in images:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            resized_image = cv2.resize(image, FRAME_SIZE)
            video_writer.write(resized_image)

        video_writer.release()
        print(f"Video saved to {video_path}")

if __name__ == "__main__":
    create_group_videos(sys.argv[1], sys.argv[2])
