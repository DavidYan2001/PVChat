import os
import subprocess
from pathlib import Path
import argparse


def ensure_dir(directory):
    """Ensure directory exists, create it if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def process_single_image(source_image, output_dir, ref_video_dir, liveportrait_path, device_id):
    """Process single image with all reference videos"""
    # Get all reference videos
    ref_videos = [
        "angry_512.mp4",
        "no_512.mp4",
        "query_512.mp4",
        "smile_512.mp4",
        "yes_512.mp4"
    ]

    # Ensure output directory exists
    ensure_dir(output_dir)

    total_tasks = len(ref_videos)
    print(f"Source image: {source_image}")
    print(f"Output directory: {output_dir}")
    print(f"Found {total_tasks} reference videos")

    for i, video in enumerate(ref_videos, 1):
        print(f"\nProcessing progress: {i}/{total_tasks}")
        print(f"Reference video: {video}")

        # Create subdirectory named after the video
        video_output_dir = os.path.join(output_dir, os.path.splitext(video)[0])
        ensure_dir(video_output_dir)

        # Build the complete command
        cmd = [
            "python", "inference.py",
            "-s", source_image,
            "-d", os.path.join(ref_video_dir, video),
            "--output-dir", video_output_dir,
            "--device-id", str(device_id)
        ]

        # Switch to LivePortrait directory and execute command
        try:
            os.chdir(liveportrait_path)
            print(f"Executing command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print(f"Successfully processed image {source_image} with video {video}")
        except subprocess.CalledProcessError as e:
            print(f"Processing failed: {e}")
            print(f"Skipping current task and continuing to next")
        except Exception as e:
            print(f"Error occurred: {e}")
            print(f"Skipping current task and continuing to next")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate animations based on single image and reference videos')
    parser.add_argument('--source', '-s', required=True, help='Path to source image')
    parser.add_argument('--output-dir', '-o', required=True, help='Output directory path')
    parser.add_argument('--base-path', '-b', default='/root/autodl-tmp/yufei',
                        help='Base path containing LivePortrait and reference video directories')
    parser.add_argument('--device-id', '-d', type=int, default=1, help='Device ID')

    args = parser.parse_args()

    # Set paths
    base_path = args.base_path
    ref_video_dir = f"{base_path}/DeepFaceLab/ref_video"
    liveportrait_path = f"{base_path}/LivePortrait"

    print("Starting image processing...")
    process_single_image(
        source_image=args.source,
        output_dir=args.output_dir,
        ref_video_dir=ref_video_dir,
        liveportrait_path=liveportrait_path,
        device_id=args.device_id
    )
    print("\nProcessing completed!")


if __name__ == "__main__":
    main()