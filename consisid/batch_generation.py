import os
import json
import random
from glob import glob
import subprocess
from multiprocessing import Process
import numpy as np


def generate_videos_for_prompts(qa_dataset_path, deepfacelab_output_path, consisid_script_path, gpu_id, start_idx,
                                end_idx):
    """
    Process video subset on specified GPU
    """
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Load QA dataset
    with open(qa_dataset_path, 'r', encoding='utf-8') as f:
        qa_dataset = json.load(f)

    # Only process videos assigned to this GPU
    for video_idx in range(start_idx, end_idx):
        if video_idx >= len(qa_dataset['videos']):
            break

        video_data = qa_dataset['videos'][video_idx]
        video_name = video_data['video_name']
        base_name = os.path.splitext(video_name)[0]
        prompts = video_data.get('same_face_prompt', [])

        print(f"\n[GPU {gpu_id}] Processing video {video_name} ({video_idx + 1}/{end_idx})")

        # Get face image path
        face_dir = os.path.join(deepfacelab_output_path, base_name, 'filtered', 'human_0')
        if not os.path.exists(face_dir):
            print(f"[GPU {gpu_id}] Warning: No face images found for {video_name}")
            continue

        # Get all .jpg or .png images
        face_images = glob(os.path.join(face_dir, '*.[jp][pn][g]'))
        if not face_images:
            print(f"[GPU {gpu_id}] Warning: No images found in {face_dir}")
            continue

        # Create a video for each prompt
        output_dir = os.path.join('output', base_name)
        os.makedirs(output_dir, exist_ok=True)

        for i, prompt in enumerate(prompts):
            # Randomly select a face image
            face_image = random.choice(face_images)

            # Build command
            cmd = [
                'python',
                consisid_script_path,
                '--img_file_path',
                face_image,
                '--prompt',
                f'"{prompt}"',
                '--output_path',
                output_dir,
                '--seed',
                str(42),
                '--is_upscale',
                '--is_frame_interpolation'
            ]

            print(f"[GPU {gpu_id}] Generating video for {video_name} with prompt {i + 1}/{len(prompts)}")
            print(f"[GPU {gpu_id}] Using face image: {face_image}")

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[GPU {gpu_id}] Error generating video for {video_name}, prompt {i + 1}: {str(e)}")
                continue


def run_parallel_processing(qa_dataset_path, deepfacelab_output_path, consisid_script_path):
    """
    Process videos in parallel on two GPUs
    """
    # Load dataset to get total number of videos
    with open(qa_dataset_path, 'r', encoding='utf-8') as f:
        qa_dataset = json.load(f)
    total_videos = len(qa_dataset['videos'])

    # Calculate number of videos per GPU
    videos_per_gpu = total_videos // 2

    # Create two processes
    p1 = Process(target=generate_videos_for_prompts,
                 args=(qa_dataset_path, deepfacelab_output_path, consisid_script_path,
                       0, 0, videos_per_gpu))

    p2 = Process(target=generate_videos_for_prompts,
                 args=(qa_dataset_path, deepfacelab_output_path, consisid_script_path,
                       1, videos_per_gpu, total_videos))

    # Start processes
    print("Starting GPU 0 process...")
    p1.start()
    print("Starting GPU 1 process...")
    p2.start()

    # Wait for processes to complete
    p1.join()
    p2.join()


if __name__ == "__main__":
    qa_dataset_path = "/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/qa_dataset4.json"
    deepfacelab_output_path = "/root/autodl-tmp/yufei/DeepFaceLab/output"
    consisid_script_path = "infer.py"

    print("Starting parallel video generation on 2 GPUs...")
    run_parallel_processing(qa_dataset_path, deepfacelab_output_path, consisid_script_path)
    print("All processes completed!")