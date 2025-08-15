import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import subprocess
import random
from pathlib import Path


def ensure_dir(directory):
    """Make sure the directory exists; if not, create it"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_latest_mp4_file(directory):
    """
    Obtain the path of the newly generated mp4 file from a certain directory.
    If your Infers.py only generates a single mp4, you can use this function to locate and rename it.
    If multiple files are generated, further filtering is required based on actual needs.
    """
    mp4_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mp4')]
    if not mp4_files:
        return None, None

    concat_video = None
    normal_video = None
    for video in mp4_files:
        if '_concat.mp4' in video:
            concat_video = video
        else:
            normal_video = video

    return normal_video, concat_video


def process_images(base_path, device_id):
    # DeepFaceLab/output catalogue
    deeplabf_output = f"{base_path}/DeepFaceLab/output"
    # Refer video catalogue
    ref_video_dir = f"{base_path}/DeepFaceLab/ref_video"
    # The directory where the LivePortrait script is located (python inference.py needs to be executed in this directory)
    liveportrait_path = f"{base_path}/LivePortrait"

    # 5 ref video
    ref_videos = [
        "angry_512.mp4",
        "no_512.mp4",
        "query_512.mp4",
        "smile_512.mp4",
        "yes_512.mp4"
    ]

    # Obtain all output directories (such as d82LKPNesE8100, etc.)
    output_dirs = [
        d for d in os.listdir(deeplabf_output)
        if os.path.isdir(os.path.join(deeplabf_output, d))
    ]

    print(f"find {len(output_dirs)} output catalogue: {output_dirs}")


    for dir_name in output_dirs:
        current_output_dir = os.path.join(deeplabf_output, dir_name)
        hq_face_dir = os.path.join(current_output_dir, "HQ_face")
        photomaker_dir = os.path.join(current_output_dir, "photomaker")

        # --- 1) Get the first image (if it exists) from HQ face ---
        hq_face_image = None
        if os.path.exists(hq_face_dir):
            face_images = [f for f in os.listdir(hq_face_dir) if f.endswith('.jpg')]
            face_images.sort()  # Ensure the sequence
            if face_images:
                # Use the first one as the source image
                hq_face_image = os.path.join(hq_face_dir, face_images[0])
            else:
                print(f"[warning] If the jpg file is not found in {hq_face_dir}, HQ_face will be skipped")

        # --- 2) Randomly select one photo from photomaker/scene 0 to 5 ---
        photomaker_images = []
        if os.path.exists(photomaker_dir):
            for scene_i in range(6):
                scene_dir = os.path.join(photomaker_dir, f"scene_{scene_i}")
                if os.path.isdir(scene_dir):
                    candidate_imgs = [
                        f for f in os.listdir(scene_dir) if f.endswith('.jpg')
                    ]
                    if candidate_imgs:
                        chosen_img = random.choice(candidate_imgs)  # 随机选一张
                        photomaker_images.append(os.path.join(scene_dir, chosen_img))
                    else:
                        print(f"[warning] There is no jpg file in {scene_dir}. Skip this scene")
                else:
                    print(f"[Hint] {scene_dir} does not exist. Skip")
        else:
            print(f"[Hint] photomaker catelogue is not exist: {photomaker_dir}")

        # Merge the one from HQ_face with the six selected ones from photomaker
        source_images = []
        if hq_face_image:
            source_images.append(hq_face_image)
        source_images.extend(photomaker_images)
        similar_image_dir = os.path.join(current_output_dir, "similar_image")
        if os.path.isdir(similar_image_dir):
            sim_imgs = [f for f in os.listdir(similar_image_dir) if f.lower().endswith('.jpg')]
            sim_imgs.sort()
            # 拼接完整路径
            sim_imgs_fullpath = [os.path.join(similar_image_dir, x) for x in sim_imgs]
            if sim_imgs_fullpath:
                source_images.extend(sim_imgs_fullpath)
                print(f"[Hint]  Find the {len(sim_imgs_fullpath)} image in the similar_image of {dir_name} and process them together")
            else:
                print(f"[Hint]  No jpg file was found in the similar_image of {dir_name}, skip it.")
        if not source_images:
            print(f"[Warning] Neither the graph of HQ_face nor that of photomaker was found in {dir_name}. Skip this directory.")
            continue

        # --- 3) For each source image, run five reference videos ---
        # Put the results uniformly in the "5 movie video" directory
        final_video_dir = os.path.join(current_output_dir, "5_movie_video")
        ensure_dir(final_video_dir)

        print(f"\nProcess catalogue: {dir_name}")
        print(f"A total of {len(source_images)} source images to be processed (HQ_face + photomaker) were found.")
        print(f"Outputcatalogue : {final_video_dir}")

        # 遍历每张源图
        for source_image in source_images:
            source_basename = os.path.splitext(os.path.basename(source_image))[0]
            # For example, scene 0 output 2 or HQ face 0

            for video in ref_videos:
                ref_basename = os.path.splitext(video)[0]
                # such as angry_512, no_512

                # To prevent different task files from overwriting each other, a temporary output directory is created for each inference
                temp_output_dir = os.path.join(current_output_dir, "moved_image_temp")
                ensure_dir(temp_output_dir)

                cmd = [
                    "python", "inference.py",
                    "-s", source_image,
                    "-d", os.path.join(ref_video_dir, video),
                    "--output-dir", temp_output_dir,
                    "--device-id", str(device_id)
                ]

                # Enter the LivePortrait environment and execute
                try:
                    os.chdir(liveportrait_path)
                    print(f"\n[executive command] {' '.join(cmd)}")
                    subprocess.run(cmd, check=True)

                    # Suppose inference.py generates an mp4 file under temp_output_dir
                    generated_mp4,_ = get_latest_mp4_file(temp_output_dir)
                    if generated_mp4 is None:
                        print(f"[Warning] The newly generated mp4 was not found in {temp_output_dir}. Skip and rename.")
                    else:
                        # Construct a new name,for example: scene_0_output_2_no_512.mp4
                        final_name = f"{source_basename}_{ref_basename}.mp4"
                        final_path = os.path.join(final_video_dir, final_name)

                        # Rename and move to 5 movie video
                        os.rename(generated_mp4, final_path)
                        print(f"Generate Video: {final_path}")

                except subprocess.CalledProcessError as e:
                    print(f"[Error] inference.py executive unsuccessful: {e}")
                except Exception as e:
                    print(f"[Exception] An error occurred: {e}")
                finally:
                    # If need to clean up the temporary directory, you can do it here
                    pass

    print("\nAll processing has been completed！")


def main():
    # 基础路径和设备ID
    base_path = "/root/autodl-tmp/yufei"
    device_id = 1

    print("Start batch processing of images...")
    process_images(base_path, device_id)
    print("\nAll processing completed!")


if __name__ == "__main__":
    main()
