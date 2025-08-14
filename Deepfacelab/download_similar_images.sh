#!/bin/bash

# This foundation catalogue
LOCAL_BASE="/root/autodl-tmp/yufei/DeepFaceLab/output"

# Remote base directory
REMOTE_SIMILAR_BASE="/mnt/hdd1/yufei/img2dataset/similar_results"

# Remote server information
REMOTE_USER="xxxx"
REMOTE_HOST="xxxx"
REMOTE_PORT=xxx

# Traverse each video directory
find "$LOCAL_BASE" -mindepth 1 -maxdepth 1 -type d | while read VIDEO_DIR; do
    VIDEO_NAME=$(basename "$VIDEO_DIR")

    # Target local directory
    LOCAL_SIMILAR_DIR="$VIDEO_DIR/similar_laion_picture"
    mkdir -p "$LOCAL_SIMILAR_DIR"

    # Remote similar image directory
    REMOTE_DIR="$REMOTE_SIMILAR_BASE/$VIDEO_NAME"

    # Download similar pictures using rsync
    rsync -avz -e "ssh -p $REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_SIMILAR_DIR/"

    echo "Donw finish: $VIDEO_NAME"
done
