#!/bin/bash

# Enable error checking
set -e

# Define variables
WORKSPACE="/root/autodl-tmp/yufei/DeepFaceLab"
MODEL_DIR="$WORKSPACE/model"
ALIGNED_DIR="$WORKSPACE/output/one-person-clip/aligned_selected_from_FFHQ/human0/1_images_aligned"
FILTERED_DIR="$WORKSPACE/output/one-person-clip/filtered/human_0"

# Create new model name
MODEL_NAME="test_quick96_$(date +%s)"
echo "Using model name: $MODEL_NAME"

# Train model
echo "Starting training... (Press Ctrl+C to stop)"
expect -c "
set timeout -1
spawn python main.py train \
    --training-data-src-dir \"$ALIGNED_DIR\" \
    --training-data-dst-dir \"$FILTERED_DIR\" \
    --model-dir \"$MODEL_DIR\" \
    --model Quick96
expect \"Choose one of saved models, or enter a name to create a new model.\"
send \"$MODEL_NAME\r\"
expect \"Choose one or several GPU idxs (separated by comma).\"
send \"0,1\r\"
expect \"Loading.*\"
expect \"Starting..\"
interact
"