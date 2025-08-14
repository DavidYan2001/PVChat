#!/usr/bin/env bash

# 定义变量
WORKSPACE="/root/autodl-tmp/yufei/DeepFaceLab"
BASE_ALIGNED_DIR="$WORKSPACE/output/one-person-clip/aligned_selected_from_FFHQ/human0"
DATA_SRC_VIDEO="$WORKSPACE/data_src/one-person-clip.mp4"
OUTPUT_DIR="$WORKSPACE/output/one-person-clip"
RESULT_DIR="$OUTPUT_DIR/results/FFHQ2video/human1"
MODEL_DIR="$WORKSPACE/model"
LOG_FILE="$WORKSPACE/training.log"

# 确保目录存在
mkdir -p "$RESULT_DIR"
mkdir -p "$MODEL_DIR"

# 训练模型函数
train_model() {
    local img_num=$1
    local src_dir="${BASE_ALIGNED_DIR}/${img_num}_images_aligned"
    echo "Starting training process for image set ${img_num}..."
    echo "Using source directory: ${src_dir}"
    expect <<EOF
spawn python main.py train \
    --training-data-src-dir "${src_dir}" \
    --training-data-dst-dir "$OUTPUT_DIR/filtered/human_0" \
    --model-dir "$MODEL_DIR" \
    --model Quick96 \
    --no-preview
expect "Choose one of saved models, or enter a name to create a new model."
send "64075_human${img_num}\r"
expect "Choose one or several GPU idxs (separated by comma)."
send "0,1\r"
expect eof
EOF
    echo "Training process completed for image set ${img_num}."
}

# 合并人脸函数
merge_faces() {
    local img_num=$1
    echo "Starting merge process for image set ${img_num}..."
    expect <<EOF
spawn python main.py merge \
    --input-dir "$OUTPUT_DIR/images" \
    --output-dir "$OUTPUT_DIR/merged_human_0_${img_num}" \
    --output-mask-dir "$OUTPUT_DIR/merged_human_${img_num}_mask" \
    --aligned-dir "$OUTPUT_DIR/filtered/human_0" \
    --model-dir "$MODEL_DIR" \
    --model Quick96
expect "Choose one of saved models, or enter a name to create a new model."
send "64075_human${img_num}\r"
expect "Choose one or several GPU idxs (separated by comma)."
send "0,1\r"
expect "Use interactive merger? ( y/n ) :"
send "n\r"
expect "Choose mode:"
send "4\r"
expect "Hist match threshold ( 0..255 ) :"
send "255\r"
expect "Choose mask mode:"
send "4\r"
expect "Choose erode mask modifier ( -400..400 ) :"
send "0\r"
expect "Choose blur mask modifier ( 0..400 ) :"
send "0\r"
expect "Choose motion blur power ( 0..100 ) :"
send "0\r"
expect "Choose output face scale modifier ( -50..50 ) :"
send "0\r"
expect "Color transfer to predicted face"
send "\r"
expect "Choose sharpen mode:"
send "0\r"
expect "Choose super resolution power ( 0..100 ?:help ) :"
send "0\r"
expect "Choose image degrade by denoise power ( 0..500 ) :"
send "0\r"
expect "Choose image degrade by bicubic rescale power ( 0..100 ) :"
send "0\r"
expect "Degrade color power of final image ( 0..100 ) :"
send "0\r"
expect "Number of workers? ( 1-180 ?:help ) :"
send "8\r"
expect eof
EOF
    echo "Merge process completed for image set ${img_num}."
}

# 生成视频函数
generate_video() {
    local img_num=$1
    echo "Starting video generation process for image set ${img_num}..."
    expect <<EOF
spawn python main.py videoed video-from-sequence \
    --input-dir "$OUTPUT_DIR/merged_human_0_${img_num}" \
    --output-file "$RESULT_DIR/result${img_num}_video.mp4" \
    --reference-file "$DATA_SRC_VIDEO"
expect "Bitrate of output file in MB/s :"
send "16\r"
expect eof
EOF
    echo "Video generation completed: $RESULT_DIR/result${img_num}_video.mp4"
}

# 清理文件函数
cleanup_files() {
    local img_num=$1
    echo "Cleaning up files..."
    # 清理模型文件
    rm -rf "$MODEL_DIR"/*
    mkdir -p "$MODEL_DIR"

    # 清理合并后的文件
    rm -rf "$OUTPUT_DIR/merged_human_0_${img_num}"
    rm -rf "$OUTPUT_DIR/merged_human_0_mask_${img_num}"

    echo "Cleanup completed."
}

# 主循环执行5次处理
i=2
while [ $i -le 2 ]
do
    echo "Processing image set $i of 5..."
    train_model "$i"
    merge_faces "$i"
    generate_video "$i"
   # cleanup_files "$i"
    echo "Completed processing image set $i"
    echo "----------------------------------------"
    i=$((i + 1))
done

echo "All processing completed. Generated 5 videos in $RESULT_DIR"