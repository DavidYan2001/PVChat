import os
import subprocess
import random
import json
from pathlib import Path

# Limit the number of threads
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"


def ensure_dir(directory):
    """Ensure directory exists, create it if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_latest_mp4_file(directory):
    """
    Get the path of the newly generated mp4 file from the specified directory.
    If inference.py only generates a single mp4 file, you can use this function to locate and rename it.
    If multiple files are generated, further filtering is required based on the actual situation.
    """
    mp4_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mp4')]
    if not mp4_files:
        return None, None
    # Classify video files
    concat_video = None
    normal_video = None
    for video in mp4_files:
        if '_concat.mp4' in video:
            concat_video = video
        else:
            normal_video = video
    return normal_video, concat_video


def process_images(base_path, device_id, train_json_path):
    # DeepFaceLab/output directory
    deeplabf_output = f"{base_path}/DeepFaceLab/output"
    # Reference video directory
    ref_video_dir = f"{base_path}/DeepFaceLab/ref_video"
    # LivePortrait script directory (need to execute inference.py in this directory)
    liveportrait_path = f"{base_path}/LivePortrait"

    # 5 reference videos
    ref_videos = [
        "angry_512.mp4",
        "no_512.mp4",
        "query_512.mp4",
        "smile_512.mp4",
        "yes_512.mp4"
    ]

    # Read train.json file, filter out videos with is_positive as True
    try:
        with open(train_json_path, 'r') as f:
            train_data = json.load(f)
    except Exception as e:
        print(f"[Error] Unable to read {train_json_path}: {e}")
        return

    positive_entries = [entry for entry in train_data.get("data", []) if entry.get("is_positive")]
    if not positive_entries:
        print("[Info] No videos with is_positive as True in train.json.")
        return

    # Get base filenames of positive videos (without extension) to match directory names under DeepFaceLab/output
    positive_bases = [os.path.splitext(entry["video_name"])[0] for entry in positive_entries]
    print(f"[Info] Found {len(positive_bases)} positive videos from train.json: {positive_bases}")

    # Get all output directories (e.g.: d82LKPNesE8_100 etc.)
    all_output_dirs = [
        d for d in os.listdir(deeplabf_output)
        if os.path.isdir(os.path.join(deeplabf_output, d))
    ]
    # Keep only folders whose directory names contain positive video base names
    output_dirs = []
    for d in all_output_dirs:
        for base in positive_bases:
            if base in d:
                output_dirs.append(d)
                break

    if not output_dirs:
        print("[Info] No DeepFaceLab/output directories matching positive videos in train.json found.")
        return

    print(f"Found {len(output_dirs)} output directories to process: {output_dirs}")

    # Iterate through each output directory for processing
    for dir_name in output_dirs:
        current_output_dir = os.path.join(deeplabf_output, dir_name)
        hq_face_dir = os.path.join(current_output_dir, "HQ_face")
        photomaker_dir = os.path.join(current_output_dir, "photomaker")

        # --- 1) Get the first image from HQ_face directory (if exists) ---
        hq_face_image = None
        if os.path.exists(hq_face_dir):
            face_images = [f for f in os.listdir(hq_face_dir) if f.endswith('.jpg')]
            face_images.sort()  # Ensure order
            if face_images:
                hq_face_image = os.path.join(hq_face_dir, face_images[0])
            else:
                print(f"[Warning] No jpg file found in {hq_face_dir}, skipping HQ_face.")

        # --- 2) Randomly select one image from each photomaker/scene_0~5 ---
        photomaker_images = []
        if os.path.exists(photomaker_dir):
            for scene_i in range(6):
                scene_dir = os.path.join(photomaker_dir, f"scene_{scene_i}")
                if os.path.isdir(scene_dir):
                    candidate_imgs = [f for f in os.listdir(scene_dir) if f.endswith('.jpg')]
                    if candidate_imgs:
                        chosen_img = random.choice(candidate_imgs)
                        photomaker_images.append(os.path.join(scene_dir, chosen_img))
                    else:
                        print(f"[Warning] No jpg file in {scene_dir}, skipping this scene")
                else:
                    print(f"[Info] {scene_dir} does not exist, skipping")
        else:
            print(f"[Info] photomaker directory does not exist: {photomaker_dir}")

        # Merge HQ_face image and selected photomaker images
        source_images = []
        if hq_face_image:
            source_images.append(hq_face_image)
        source_images.extend(photomaker_images)
        similar_image_dir = os.path.join(current_output_dir, "similar_image")
        if os.path.isdir(similar_image_dir):
            sim_imgs = [f for f in os.listdir(similar_image_dir) if f.lower().endswith('.jpg')]
            sim_imgs.sort()
            sim_imgs_fullpath = [os.path.join(similar_image_dir, x) for x in sim_imgs]
            if sim_imgs_fullpath:
                source_images.extend(sim_imgs_fullpath)
                print(f"[Info] Found {len(sim_imgs_fullpath)} images in similar_image of {dir_name}, processing together.")
            else:
                print(f"[Info] No jpg file found in similar_image of {dir_name}, skipping.")
        if not source_images:
            print(f"[Warning] No images found from HQ_face or photomaker in {dir_name}, skipping this directory.")
            continue

        # --- 3) For each source image, generate results using 5 reference videos in sequence ---
        final_video_dir = os.path.join(current_output_dir, "5_movie_video")
        ensure_dir(final_video_dir)

        print(f"\nProcessing directory: {dir_name}")
        print(f"Total {len(source_images)} source images to process (from HQ_face + photomaker)")
        print(f"Output directory: {final_video_dir}")

        for source_image in source_images:
            source_basename = os.path.splitext(os.path.basename(source_image))[0]
            for video in ref_videos:
                ref_basename = os.path.splitext(video)[0]
                temp_output_dir = os.path.join(current_output_dir, "moved_image_temp")
                ensure_dir(temp_output_dir)
                cmd = [
                    "python", "inference.py",
                    "-s", source_image,
                    "-d", os.path.join(ref_video_dir, video),
                    "--output-dir", temp_output_dir,
                    "--device-id", str(device_id)
                ]
                try:
                    os.chdir(liveportrait_path)
                    print(f"\n[Executing command] {' '.join(cmd)}")
                    subprocess.run(cmd, check=True)
                    generated_mp4, _ = get_latest_mp4_file(temp_output_dir)
                    if generated_mp4 is None:
                        print(f"[Warning] No new mp4 file found in {temp_output_dir}, skipping rename.")
                    else:
                        final_name = f"{source_basename}_{ref_basename}.mp4"
                        final_path = os.path.join(final_video_dir, final_name)
                        os.rename(generated_mp4, final_path)
                        print(f"Generated video: {final_path}")
                except subprocess.CalledProcessError as e:
                    print(f"[Error] inference.py processing failed: {e}")
                except Exception as e:
                    print(f"[Exception] Error occurred: {e}")
                finally:
                    # Add code here if you need to clean up the temporary directory
                    pass

    print("\nAll processing completed!")


def main():
    base_path = "/root/autodl-tmp/yufei"
    device_id = 0
    # Specify train.json file path
    train_json_path = f"{base_path}/datasets/cekebv-hq/train_Jo.json"
    print("Starting batch image processing...")
    process_images(base_path, device_id, train_json_path)
    print("\nAll processing completed!")


if __name__ == "__main__":
    main()