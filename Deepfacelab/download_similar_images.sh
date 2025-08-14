#!/bin/bash

# 本地基目录
LOCAL_BASE="/root/autodl-tmp/yufei/DeepFaceLab/output"

# 远程基目录
REMOTE_SIMILAR_BASE="/mnt/hdd1/yufei/img2dataset/similar_results"

# 远程服务器信息
REMOTE_USER="yufei"
REMOTE_HOST="gj03.ezfrp.com"
REMOTE_PORT=22101

# 遍历每个视频目录
find "$LOCAL_BASE" -mindepth 1 -maxdepth 1 -type d | while read VIDEO_DIR; do
    VIDEO_NAME=$(basename "$VIDEO_DIR")

    # 目标本地目录
    LOCAL_SIMILAR_DIR="$VIDEO_DIR/similar_laion_picture"
    mkdir -p "$LOCAL_SIMILAR_DIR"

    # 远程相似图片目录
    REMOTE_DIR="$REMOTE_SIMILAR_BASE/$VIDEO_NAME"

    # 使用rsync下载相似图片
    rsync -avz -e "ssh -p $REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_SIMILAR_DIR/"

    echo "下载完成: $VIDEO_NAME"
done
