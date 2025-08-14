#!/usr/bin/env bash

original_video_name="one-person-clip"
video_name="man2"
# 定义变量
WORKSPACE="/root/autodl-tmp/yufei/DeepFaceLab"
SRC_DIR="$WORKSPACE/output/$original_video_name/filtered/human_0"
DST_DIR="$WORKSPACE/output/$video_name/filtered/human_0"
OUTPUT_DIR="$WORKSPACE/output/$video_name"
RESULT_DIR="$OUTPUT_DIR/results/face_swap"
MODEL_DIR="$WORKSPACE/model"
DATA_DST_VIDEO="$WORKSPACE/data_src/$video_name.mp4"

# 确保目录存在
mkdir -p "$RESULT_DIR"
mkdir -p "$MODEL_DIR"

# 训练模型函数
train_model() {
    echo "Starting training process..."
   # expect <<EOF
#spawn
 python main.py train \
    --training-data-src-dir "$SRC_DIR" \
    --training-data-dst-dir "$DST_DIR" \
    --model-dir "$MODEL_DIR" \
    --model Quick96 \
    --no-preview
    echo "Training process completed."
}

# 合并人脸函数
merge_faces() {
    echo "Starting merge process..."
    expect <<EOF
spawn python main.py merge \
    --input-dir "$OUTPUT_DIR/images" \
    --output-dir "$OUTPUT_DIR/merged" \
    --output-mask-dir "$OUTPUT_DIR/merged_mask" \
    --aligned-dir "$DST_DIR" \
    --model-dir "$MODEL_DIR" \
    --model Quick96
expect "Choose one of saved models, or enter a name to create a new model."
send "new\r"
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
    echo "Merge process completed."
}

# 生成视频函数


generate_video() {
    local input_dir="$1"        # 合并后的图片序列目录
    local output_file="$2"      # 输出视频文件路径
    local reference_file="$3"   # 原始参考视频文件路径

    # 获取原始视频的参数
    local orig_fps=$(ffprobe -v quiet -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate "$reference_file" | bc -l | xargs printf "%.0f")
    local orig_duration=$(ffprobe -v quiet -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries format=duration "$reference_file")
    local orig_bitrate=$(ffprobe -v quiet -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=bit_rate "$reference_file")

    # 将比特率转换为MB/s
    local bitrate_mbps=$(echo "scale=0; $orig_bitrate / 1000000" | bc)

    # 如果无法获取原始参数，使用默认值
    if [ -z "$orig_fps" ]; then
        echo "Warning: Could not detect original FPS, using default value of 30"
        orig_fps=30
    fi

    if [ -z "$bitrate_mbps" ] || [ "$bitrate_mbps" -lt 1 ]; then
        echo "Warning: Could not detect original bitrate or it's too low, using default value of 16"
        bitrate_mbps=16
    fi

    echo "Original video parameters:"
    echo "FPS: $orig_fps"
    echo "Duration: $orig_duration seconds"
    echo "Bitrate: $bitrate_mbps MB/s"

    # 计算需要的总帧数
    local total_frames_needed=$(echo "$orig_fps * $orig_duration" | bc | xargs printf "%.0f")
    local current_frames=$(ls "$input_dir"/*.png 2>/dev/null | wc -l)
    if [ "$current_frames" -eq 0 ]; then
        current_frames=$(ls "$input_dir"/*.jpg 2>/dev/null | wc -l)
    fi

    echo "Total frames needed: $total_frames_needed"
    echo "Current frames: $current_frames"

    # 如果需要更多帧，创建临时目录并复制帧
    if [ "$current_frames" -lt "$total_frames_needed" ]; then
        local temp_dir="${input_dir}_temp"
        mkdir -p "$temp_dir"

        # 复制并重命名帧
        local repeats=$((total_frames_needed / current_frames))
        local remaining_frames=$((total_frames_needed % current_frames))
        local frame_count=0

        for file in "$input_dir"/*; do
            i=0
            while [ $i -lt $repeats ]; do
                cp "$file" "$temp_dir/frame_$(printf "%08d" $frame_count).png"
                frame_count=$((frame_count + 1))
                i=$((i + 1))
            done
            # 处理剩余帧
            if [ "$remaining_frames" -gt 0 ]; then
                cp "$file" "$temp_dir/frame_$(printf "%08d" $frame_count).png"
                frame_count=$((frame_count + 1))
                remaining_frames=$((remaining_frames - 1))
            fi
        done

        # 使用临时目录替换原输入目录
        input_dir="$temp_dir"
    fi

    # 执行视频生成
    python main.py videoed video-from-sequence \
        --input-dir "$input_dir" \
        --output-file "$output_file" \
        --reference-file "$reference_file" \
        --fps $orig_fps \
        --bitrate "$bitrate_mbps"

    # 如果使用了临时目录，清理它
    if [ -d "${input_dir}_temp" ]; then
        rm -rf "${input_dir}_temp"
    fi

    # 验证生成的视频
    local new_duration=$(ffprobe -v quiet -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries format=duration "$output_file")
    local duration_diff=$(echo "scale=2; ($new_duration - $orig_duration)^2" | bc)

    echo "New video duration: $new_duration seconds"
    echo "Duration difference: $duration_diff seconds"
}
# 在主脚本中使用这个函数
echo "Starting video generation process..."

# 清理文件函数
cleanup_files() {
    echo "Cleaning up files..."
    # 清理模型文件
    rm -rf "$MODEL_DIR"/*
    mkdir -p "$MODEL_DIR"

    # 清理合并后的文件
    rm -rf "$OUTPUT_DIR/merged"
    rm -rf "$OUTPUT_DIR/merged_mask"

    echo "Cleanup completed."
}

# 主执行流程
echo "Starting face swap process..."
train_model
merge_faces
generate_video \
    "$OUTPUT_DIR/merged" \
    "$RESULT_DIR/result_video.mp4" \
    "$DATA_DST_VIDEO"
cleanup_files
echo "Face swap process completed. Output video: $RESULT_DIR/result_video.mp4"