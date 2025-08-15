#!/bin/bash

# the local catalogue
LOCAL_BASE="/root/autodl-tmp/yufei/DeepFaceLab/output"

# the remote catalogue
REMOTE_BASE="/mnt/hdd1/yufei/img2dataset/tmp_picture"

# the remote server informayion
REMOTE_USER="yufei"
REMOTE_HOST="xxx.xxx.xxx"
REMOTE_PORT=xxxxx

# Traverse each video directory
find "$LOCAL_BASE" -mindepth 2 -maxdepth 2 -type d -name "HQ_face" | while read HQ_DIR; do
    VIDEO_DIR=$(dirname "$HQ_DIR")
    VIDEO_NAME=$(basename "$VIDEO_DIR")

    #Target remote directory
    REMOTE_DIR="$REMOTE_BASE/$VIDEO_NAME/HQ_face"

    # Create a remote directory
    ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST "mkdir -p '$REMOTE_DIR'"

    #Upload pictures using rsync
    rsync -avz -e "ssh -p $REMOTE_PORT" "$HQ_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"

    echo "upload finish: $VIDEO_NAME"
done
