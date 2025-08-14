#!/bin/bash

# Increase system file handle limit
ulimit -n 65535

# Limit OpenBLAS thread count
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# Define paths
DATA_SRC_DIR="/root/autodl-tmp/yufei/DeepFaceLab/data_src"
OUTPUT_DIR="/root/autodl-tmp/yufei/DeepFaceLab/output"
WORKSPACE_DIR="/root/autodl-tmp/yufei/DeepFaceLab/workspace"

# Create necessary folders
mkdir -p "$WORKSPACE_DIR"
mkdir -p "$OUTPUT_DIR"

# Set GPU environment variables, default to GPU 0 and 1
export CUDA_VISIBLE_DEVICES=0,1

# First build the collection of videos to process
videos_to_process=()
echo "Scanning $DATA_SRC_DIR for videos to process..."
for VIDEO_PATH in "$DATA_SRC_DIR"/*.mp4; do
    [ -e "$VIDEO_PATH" ] || continue
    VIDEO_NAME=$(basename "$VIDEO_PATH" .mp4)
    # Assume the corresponding output folder format is "${VIDEO_NAME}"
    CHECK_DIR="$OUTPUT_DIR/${VIDEO_NAME}"
    if [ -d "$CHECK_DIR/aligned" ] && [ -d "$CHECK_DIR/filtered" ]; then
        echo "Skipping $VIDEO_NAME: processed output exists in $CHECK_DIR (aligned & filtered found)."
    else
        videos_to_process+=("$VIDEO_PATH")
    fi
done

echo "Total videos to process: ${#videos_to_process[@]}"

# Process videos in the collection sequentially
for VIDEO_PATH in "${videos_to_process[@]}"; do
    VIDEO_NAME=$(basename "$VIDEO_PATH" .mp4)
    # Construct output directory as "$OUTPUT_DIR/${VIDEO_NAME}"
    VIDEO_OUTPUT_DIR="$OUTPUT_DIR/${VIDEO_NAME}"
    IMAGE_DIR="$VIDEO_OUTPUT_DIR/images"
    ALIGNED_DIR="$VIDEO_OUTPUT_DIR/aligned"
    FILTERED_DIR="$VIDEO_OUTPUT_DIR/filtered"
    VIDEO_GROUP_DIR="$VIDEO_OUTPUT_DIR/groups"

    mkdir -p "$IMAGE_DIR" "$ALIGNED_DIR" "$FILTERED_DIR" "$VIDEO_GROUP_DIR"

    echo "Processing video: $VIDEO_NAME"

    # Step 1: Extract video frames
    echo "png" | python main.py videoed extract-video \
        --input-file "$VIDEO_PATH" \
        --output-dir "$IMAGE_DIR" \
        --fps 8

    # Step 2: Extract faces
    echo "y" | python main.py extract \
        --input-dir "$IMAGE_DIR" \
        --output-dir "$ALIGNED_DIR" \
        --detector s3fd \
        --face-type whole_face \
        --max-faces-from-image 2 \
        --image-size 1024 \
        --no-output-debug \
        --jpeg-quality 100 \
        --force-gpu-idxs 0,1

    # Step 3: Classify by similarity (save results to FILTERED_DIR)
    python - <<EOF
import os
import shutil
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import torch
from torchvision import transforms
from sklearn.cluster import DBSCAN
import numpy as np

aligned_dir = "$ALIGNED_DIR"
output_base_dir = "$FILTERED_DIR"
os.makedirs(output_base_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

embeddings = []
file_paths = []

for file_name in os.listdir(aligned_dir):
    file_path = os.path.join(aligned_dir, file_name)
    try:
        img = Image.open(file_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        embedding = model(img_tensor).detach().cpu().numpy().flatten()
        embeddings.append(embedding)
        file_paths.append(file_path)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if not embeddings:
    print(f"No valid face images found in {aligned_dir}. Skipping...")
else:
    embeddings = np.array(embeddings)

    # First try clustering with relaxed parameters
    clustering = DBSCAN(eps=0.6, min_samples=3, metric='cosine').fit(embeddings)
    labels = clustering.labels_

    # Check if all labels are noise
    if np.all(labels == -1):
        print("All faces classified as noise. Trying more lenient parameters...")
        # Use more lenient parameters
        clustering = DBSCAN(eps=0.7, min_samples=2, metric='cosine').fit(embeddings)
        labels = clustering.labels_

    # If still all noise, assign all samples to a single group
    if np.all(labels == -1):
        print("Still all noise. Assigning all faces to a single group.")
        labels = np.zeros(len(embeddings), dtype=int)

    # Calculate how many different clusters exist
    unique_labels = set(labels)
    unique_labels_without_noise = [label for label in unique_labels if label != -1]
    print(f"Found {len(unique_labels_without_noise)} groups of faces")

    # Process normal clusters
    for label in unique_labels:
        # Create a special group for noise points
        if label == -1:
            group_dir = os.path.join(output_base_dir, "human_noise")
        else:
            group_dir = os.path.join(output_base_dir, f"human_{label}")

        os.makedirs(group_dir, exist_ok=True)

        # Copy all files corresponding to this label
        for i, (curr_label, file_path) in enumerate(zip(labels, file_paths)):
            if curr_label == label:
                shutil.copy(file_path, os.path.join(group_dir, os.path.basename(file_path)))

print("Face grouping completed.")

# Check if any files were copied to the output directory
any_files_copied = False
for root, dirs, files in os.walk(output_base_dir):
    if files:
        any_files_copied = True
        break

if not any_files_copied:
    print("WARNING: No files were copied to the filtered directory. Copying all aligned faces to a single group...")
    # If no files were copied, copy all aligned faces to a single group
    default_group_dir = os.path.join(output_base_dir, "human_0")
    os.makedirs(default_group_dir, exist_ok=True)
    for file_path in file_paths:
        shutil.copy(file_path, os.path.join(default_group_dir, os.path.basename(file_path)))
EOF

    # Step 4: Synthesize classified videos, save generated videos in VIDEO_GROUP_DIR
    for GROUP_DIR in "$FILTERED_DIR"/*; do
        GROUP_NAME=$(basename "$GROUP_DIR")
        NEW_VIDEO_PATH="$VIDEO_GROUP_DIR/${GROUP_NAME}.mp4"

        python - <<EOF
import cv2
from pathlib import Path

image_dir = "$GROUP_DIR"
output_video_path = "$NEW_VIDEO_PATH"
FPS = 30
FRAME_SIZE = (512, 512)

images = sorted(Path(image_dir).glob("*.jpg"))
if not images:
    print(f"No images found in {image_dir}. Skipping...")
else:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, FPS, FRAME_SIZE)

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        resized_image = cv2.resize(image, FRAME_SIZE)
        video_writer.write(resized_image)

    video_writer.release()
    print(f"Video saved to {output_video_path}")
EOF
    done

done

echo "Batch processing completed."
