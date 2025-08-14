#!/bin/bash

# 本地基目录
LOCAL_BASE="/root/autodl-tmp/yufei/DeepFaceLab/output"

# 远程基目录
REMOTE_BASE="/mnt/hdd1/yufei/img2dataset/tmp_picture"

# 远程服务器信息
REMOTE_USER="yufei"
REMOTE_HOST="gj03.ezfrp.com"
REMOTE_PORT=22101

# 遍历每个视频目录
find "$LOCAL_BASE" -mindepth 2 -maxdepth 2 -type d -name "HQ_face" | while read HQ_DIR; do
    VIDEO_DIR=$(dirname "$HQ_DIR")
    VIDEO_NAME=$(basename "$VIDEO_DIR")

    # 目标远程目录
    REMOTE_DIR="$REMOTE_BASE/$VIDEO_NAME/HQ_face"

    # 创建远程目录
    ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST "mkdir -p '$REMOTE_DIR'"

    # 使用rsync上传图片
    rsync -avz -e "ssh -p $REMOTE_PORT" "$HQ_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"

    echo "上传完成: $VIDEO_NAME"
done
