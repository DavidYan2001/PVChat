import os
import subprocess
from pathlib import Path


def ensure_dir(directory):
    """Make sure the directory exists; if not, create it"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def process_images(base_path, device_id):
    # Basic path
    deeplabf_output = f"{base_path}/DeepFaceLab/output"
    ref_video_dir = f"{base_path}/DeepFaceLab/ref_video"
    liveportrait_path = f"{base_path}/LivePortrait"

    # Get all reference videos
    ref_videos = [
        "angry_512.mp4",
        "no_512.mp4",
        "query_512.mp4",
        "smile_512.mp4",
        "yes_512.mp4"
    ]

    # Get all output directories
    output_dirs = [d for d in os.listdir(deeplabf_output)
                   if os.path.isdir(os.path.join(deeplabf_output, d))]

    total_tasks = len(output_dirs) * len(ref_videos)
    completed_tasks = 0

    print(f"Find the output directories {len(output_dirs)} and the reference videos {len(ref_videos)}")
    print(f"A total of {total_tasks} tasks need to be handled")

    for dir_name in output_dirs:
        hq_face_dir = os.path.join(deeplabf_output, dir_name, "HQ_face")

        # Check whether the "HQ face" directory exists
        if not os.path.exists(hq_face_dir):
            print(f"Warning: {hq_face_dir} does not exist. Skip this directory")
            continue

        # Get all the jpg files in this directory
        face_images = [f for f in os.listdir(hq_face_dir) if f.endswith('.jpg')]

        if not face_images:
            print(f"Warning: The jpg file was not found in {hq_face_dir}. Skip this directory")
            continue

        # Use the first image as the source image
        source_image = os.path.join(hq_face_dir, face_images[0])

        # Process each reference video
        for video in ref_videos:
            completed_tasks += 1
            print(f"\nprocessing progress: {completed_tasks}/{total_tasks}")
            print(f"Processing the directory: {dir_name}")
            print(f"Reference video: {video}")

            # Create an output directory
            output_dir = os.path.join(deeplabf_output, dir_name, "moved_image")
            ensure_dir(output_dir)

            # Build a complete command
            cmd = [
                "python", "inference.py",
                "-s", source_image,
                "-d", os.path.join(ref_video_dir, video),
                "--output-dir", output_dir,
                "--device-id", str(device_id)
            ]

            # Switch to the LivePortrait directory and execute the command
            try:
                os.chdir(liveportrait_path)
                print(f"executive command: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
                print(f"Successfully processed the image of {dir_name} and the video {video}")
            except subprocess.CalledProcessError as e:
                print(f"Failed processing: {e}")
                print(f"Skip the current task and proceed to the next one")
            except Exception as e:
                print(f"go wrong: {e}")
                print(f"Skip the current task and proceed to the next one")


def main():
    # Basic path and device ID
    base_path = "/root/autodl-tmp/yufei"
    device_id = 1

    print("Start batch processing of images...")
    process_images(base_path, device_id)
    print("\nAll processing is completed.!")


if __name__ == "__main__":
    main()