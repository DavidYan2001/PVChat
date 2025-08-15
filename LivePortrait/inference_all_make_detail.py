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
    """Make sure the directory exists; if not, create it"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_latest_mp4_file(directory):
    """
    Obtain the path of the newly generated mp4 file from the specified directory.
    If Infers.py only generates a single mp4 file, you can use this function to locate and rename it.
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
    # Reference Video Directory
    ref_video_dir = f"{base_path}/DeepFaceLab/ref_video"
    # The directory where the LivePortrait script is located (inference.py needs to be executed in this directory)
    liveportrait_path = f"{base_path}/LivePortrait"

    # 5 refer video
    ref_videos = [
        "angry_512.mp4",
        "no_512.mp4",
        "query_512.mp4",
        "smile_512.mp4",
        "yes_512.mp4"
    ]

    # Read the train.json file and filter out the videos with is_positive as True
    try:
        with open(train_json_path, 'r') as f:
            train_data = json.load(f)
    except Exception as e:
        print(f"[Error] can't read {train_json_path}: {e}")
        return

    positive_entries = [entry for entry in train_data.get("data", []) if entry.get("is_positive")]
    if not positive_entries:
        print("There are no videos with is_positive set to True in train.json.")
        return

    # Obtain the base file name of the forward video (remove the extension) to match the directory name under DeepFaceLab/output
    positive_bases = [os.path.splitext(entry["video_name"])[0] for entry in positive_entries]
    print(f"Find {len(positive_bases)} forward videos from train.json: {positive_bases}")

    # Obtain all output directories (for example: d82LKPNesE8100, etc.)
    all_output_dirs = [
        d for d in os.listdir(deeplabf_output)
        if os.path.isdir(os.path.join(deeplabf_output, d))
    ]
    # Only keep the folders whose directory names contain the base names of the positive videos
    output_dirs = []
    for d in all_output_dirs:
        for base in positive_bases:
            if base in d:
                output_dirs.append(d)
                break

    if not output_dirs:
        print("[Hind] The DeepFaceLab/output directory that matches the forward video in train.json was not found")
        return

    print(f"Find the {len(output_dirs)} output directories to be processed:: {output_dirs}")

    # Traverse each output directory for processing
    for dir_name in output_dirs:
        current_output_dir = os.path.join(deeplabf_output, dir_name)
        hq_face_dir = os.path.join(current_output_dir, "HQ_face")
        photomaker_dir = os.path.join(current_output_dir, "photomaker")

        # --- 1) Obtain the first image (if it exists) from the HQ_face directory ---
        hq_face_image = None
        if os.path.exists(hq_face_dir):
            face_images = [f for f in os.listdir(hq_face_dir) if f.endswith('.jpg')]
            face_images.sort()  # Ensure Sequence
            if face_images:
                hq_face_image = os.path.join(hq_face_dir, face_images[0])
            else:
                print(f"[Warning] If the jpg file is not found in {hq_face_dir}, HQ_face will be skipped.")

        # --- 2) 从 photomaker/scene_0~5 中各随机选一张图片 ---
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
                        print(f"[Warning] There is no jpg file in {scene_dir}. Skip this scene")
                else:
                    print(f"[Hint]{scene_dir} does not exist. Skip")
        else:
            print(f"[Hint]  {photomaker_dir}")

        # Merge the HQ face image and the image selected by photomaker
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
                print(f"[Hint] Find the {len(sim_imgs_fullpath)} image in the similar_image of {dir_name} and process them together.")
            else:
                print(f"[Hint] No jpg file was found in the similar_image of {dir_name}, skip it.")
        if not source_images:
            print(f"[Warning] If neither the image of HQ_face nor the image of photomaker is found in {dir_name}, skip this directory.")
            continue

        # --- 3) For each source image, generate the result using five reference videos in sequence ---
        final_video_dir = os.path.join(current_output_dir, "5_movie_video")
        ensure_dir(final_video_dir)

        print(f"\nProcessing catalogue: {dir_name}")
        print(f"A total of {len(source_images)} source images to be processed (from HQ_face + photomaker) were found.")
        print(f"Output catalogue: {final_video_dir}")

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
                    print(f"\n[executive command] {' '.join(cmd)}")
                    subprocess.run(cmd, check=True)
                    generated_mp4, _ = get_latest_mp4_file(temp_output_dir)
                    if generated_mp4 is None:
                        print(f"[Warning] The newly generated mp4 file was not found in {temp_output_dir}. Skip and rename.")
                    else:
                        final_name = f"{source_basename}_{ref_basename}.mp4"
                        final_path = os.path.join(final_video_dir, final_name)
                        os.rename(generated_mp4, final_path)
                        print(f"Generate Video: {final_path}")
                except subprocess.CalledProcessError as e:
                    print(f"[Error] inference.py processing failed: {e}")
                except Exception as e:
                    print(f"[异常] 发生错误: {e}")
                finally:
                    # If you need to clean up the temporary directory, you can add code here
                    pass

    print("\n所有处理全部完成！")


def main():
    base_path = "/root/autodl-tmp/yufei"
    device_id = 0
    # Specify the path of the train.json file
    train_json_path = f"{base_path}/datasets/cekebv-hq/train.json"
    print("Start batch processing of images...")
    process_images(base_path, device_id, train_json_path)
    print("\nAll processing is completed.!")


if __name__ == "__main__":
    main()
