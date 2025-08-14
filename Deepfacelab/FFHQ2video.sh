#!/usr/bin/env bash

# Define variables
WORKSPACE="/root/autodl-tmp/yufei/DeepFaceLab"
video_name="yufei2"
DATA_SRC_VIDEO="$WORKSPACE/data_src/yufei2.mp4"
OUTPUT_DIR="$WORKSPACE/output/$video_name"
FFHQ_DIR="$OUTPUT_DIR/aligned_cai"
RESULT_DIR="$OUTPUT_DIR/results/FFHQ2video/yufei2"
MODEL_DIR="$WORKSPACE/model"
LOG_FILE="$WORKSPACE/training.log"
DATA_DST_VIDEO="$WORKSPACE/data_src/$video_name.mp4"

# CPU optimization settings
export OMP_NUM_THREADS=8      # Limit OpenMP thread count
export MKL_NUM_THREADS=8      # Limit MKL thread count
export NUMEXPR_NUM_THREADS=8  # Limit NumExpr thread count
export OPENBLAS_NUM_THREADS=8 # Limit OpenBLAS thread count
set timeout 300

# Ensure directories exist
mkdir -p "$RESULT_DIR"
mkdir -p "$MODEL_DIR"

# Function to extract and align faces from images
extract_face_from_image() {
    echo "Starting face extraction from image..."

python main.py extract \
    --input-dir "$WORKSPACE/data_dst" \
    --output-dir "$FFHQ_DIR" \
    --detector s3fd \
     --jpeg-quality 100 \
     --image-size 512 \
     --force-gpu-idxs "0,1" \
     --face-type 'whole_face' \
     --max-faces-from-image  1 \
     --no-output-debug


    echo "Face extraction completed."
}


# Automated training and monitor iteration



train_model_old() {
    echo "Starting training process..."
python main.py train \
    --training-data-src-dir "$FFHQ_DIR" \
    --training-data-dst-dir "$OUTPUT_DIR/filtered/human_0" \
    --model-dir "$MODEL_DIR" \
    --model SAEHD \
    --no-preview \
    --force-gpu-idxs "0,1" \
    --silent-start


    echo "Training process completed."
}

train_model() {
    echo "Starting training process..."
     # Create log directory
    mkdir -p "$WORKSPACE/logs"
    #nice -n 19 expect <<EOF
    expect <<EOF 2>&1 | tee "$WORKSPACE/logs/training_error.log"
spawn python main.py train \
    --training-data-src-dir "$FFHQ_DIR" \
    --training-data-dst-dir "$OUTPUT_DIR/filtered/human_0" \
    --model-dir "$MODEL_DIR" \
    --model SAEHD \
    --no-preview \
    --force-gpu-idxs "0,1" \
    --silent-start
expect "Autobackup every N hour ( 0..24 ?:help ) :"
send "0\r"
expect "Write preview history ( y/n ?:help ) :"
send "y\r"
expect "Choose image for the preview history ( y/n ) :"
send "y\r"
expect "Target iteration"
send "100000\r"
expect "Flip SRC faces randomly"
send "n\r"
expect "Flip DST faces randomly"
send "n\r"
expect "Batch_size"
send "4\r"
expect "Resolution ( 64-640 ?:help ) :"
send "256\r"
expect "Face type ( h/mf/f/wf/head ?:help ) :"
send "wf\r"
expect "AE architecture"
send "df\r"
expect "AutoEncoder dimensions"
send "256\r"
expect "Encoder dimensions"
send "64\r"
expect "Decoder dimensions"
send "64\r"
expect "Decoder mask dimensions"
send "32\r"
expect "Masked training"
send "y\r"
expect "Eyes and mouth priority"
send "y\r"
expect "Uniform yaw distribution of samples"
send "y\r"
expect "Blur out mask"
send "y\r"
expect "Place models and optimizer on GPU"
send "y\r"
expect "Use AdaBelief optimizer"
send "y\r"
expect "Use learning rate dropout"
send "n\r"
expect "Enable random warp of samples"
send "y\r"
expect "Random hue/saturation/light intensity"
send "0.0\r"
expect "GAN power"
send "0.0\r"
expect "GAN patch size"
send "64\r"
expect "GAN dimensions"
send "32\r"
expect "'True face' power"
send "0.2\r"
expect "Face style power"
send "5.0\r"
expect "Background style power"
send "2.0\r"
expect "Color transfer for src faceset"
send "lct\r"
expect "Enable gradient clipping"
send "y\r"
expect "Enable pretraining mode"
send "n\r"
# Wait for training
# No need to expect specific output next, directly enter interact
expect {
    "Initializing models" {
        # After capturing initialization prompt, wait for "Loading samples" to appear
        # Use regex to match "Loading samples" prefix, not requiring 100% exact match
        expect -re "Loading samples.*"
        interact
    }
    timeout {
        puts "Timeout waiting for initialization"
    }
}
EOF
    sleep 50
    # Check if training started successfully

}
# Automated merge
merge_faces_old() {
    echo "Starting merge process..."
    expect <<EOF
spawn python main.py merge \
    --input-dir "$OUTPUT_DIR/images" \
    --output-dir "$OUTPUT_DIR/merged_human_0" \
    --output-mask-dir "$OUTPUT_DIR/merged_human_0_mask" \
    --aligned-dir "$OUTPUT_DIR/filtered/human_0" \
    --model-dir "$MODEL_DIR" \
    --model Quick96
expect "Choose one of saved models, or enter a name to create a new model."
send "new\r"
expect "Choose one or several GPU idxs (separated by comma)."
send "0\r"
expect "Use interactive merger? ( y/n ) :"
send "n\r"
expect "Choose mode:"
send "4\r"
expect "Hist match threshold ( 0..255 ) :"
send "200\r"
expect "Choose mask mode:"
send "4\r"
expect "Choose erode mask modifier ( -400..400 ) :"
send "-50\r"
expect "Choose blur mask modifier ( 0..400 ) :"
send "100\r"
expect "Choose motion blur power ( 0..100 ) :"
send "40\r"
expect "Choose output face scale modifier ( -50..50 ) :"
send "0\r"
expect "Color transfer to predicted face"
send "y\r"
expect "Choose sharpen mode:"
send "1\r"
expect "Choose super resolution power ( 0..100 ?:help ) :"
send "40\r"
expect "Choose image degrade by denoise power ( 0..500 ) :"
send "20\r"
expect "Choose image degrade by bicubic rescale power ( 0..100 ) :"
send "0\r"
expect "Degrade color power of final image ( 0..100 ) :"
send "20\r"
expect "Number of workers? ( 1-180 ?:help ) :"
send "8\r"
expect eof
EOF
    echo "Merge process completed."
}

merge_faces() {
    echo "Starting merge process..."
    expect <<EOF
spawn python main.py merge \
    --input-dir "$OUTPUT_DIR/images" \
    --output-dir "$OUTPUT_DIR/merged_human_0" \
    --output-mask-dir "$OUTPUT_DIR/merged_human_0_mask" \
    --aligned-dir "$OUTPUT_DIR/filtered/human_0" \
    --model-dir "$MODEL_DIR" \
    --model SAEHD
expect "No saved models found. Enter a name of a new model"
send "new\r"
expect "Choose one or several GPU idxs (separated by comma)."
send "0\r"
expect "Autobackup every N hour ( 0..24 ?:help ) :"
send "2\r"
expect "Write preview history ( y/n ?:help ) :"
send "y\r"
expect "Choose image for the preview history ( y/n ) :"
send "y\r"
expect "Target iteration"
send "300000\r"
expect "Flip SRC faces randomly"
send "y\r"
expect "Flip DST faces randomly"
send "y\r"
expect "Batch_size"
send "16\r"
expect "Resolution ( 64-640 ?:help ) :"
send "512\r"
expect "Face type ( h/mf/f/wf/head ?:help ) :"
send "wf\r"
expect "AE architecture"
send "df\r"
expect "AutoEncoder dimensions"
send "512\r"
expect "Encoder dimensions"
send "128\r"
expect "Decoder dimensions"
send "128\r"
expect "Decoder mask dimensions"
send "64\r"
expect "Masked training"
send "y\r"
expect "Eyes and mouth priority"
send "y\r"
expect "Uniform yaw distribution of samples"
send "y\r"
expect "Blur out mask"
send "y\r"
expect "Place models and optimizer on GPU"
send "y\r"
expect "Use AdaBelief optimizer"
send "y\r"
expect "Use learning rate dropout"
send "n\r"
expect "Enable random warp of samples"
send "y\r"
expect "Random hue/saturation/light intensity"
send "0.2\r"
expect "GAN power"
send "0.1\r"
expect "GAN patch size"
send "128\r"
expect "GAN dimensions"
send "64\r"
expect "'True face' power"
send "0.2\r"
expect "Face style power"
send "5.0\r"
expect "Background style power"
send "2.0\r"
expect "Color transfer for src faceset"
send "lct\r"
expect "Enable gradient clipping"
send "y\r"
expect "Enable pretraining mode"
send "n\r"
expect eof
EOF
    echo "Merge process completed."
}

# 自动化视频生成
generate_video() {
    echo "Starting video generation process..."
    expect <<EOF
spawn python main.py videoed video-from-sequence \
    --input-dir "$OUTPUT_DIR/merged_human_0" \
    --output-file "$RESULT_DIR/result6_video.mp4" \
    --reference-file "$DATA_SRC_VIDEO"
expect "Bitrate of output file in MB/s :"
send "16\r"
expect eof
EOF
    echo "Video generation completed: $RESULT_DIR/result6_video1.mp4"
}
generate_video_new() {
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
# 执行流程
#extract_face_from_image  # First extract faces from images
train_model
#merge_faces
#generate_video_new \
#    "$OUTPUT_DIR/merged_human_0" \
#    "$RESULT_DIR/result_video.mp4" \
#    "$DATA_DST_VIDEO"
#cleanup_files
echo "Face swap process completed. Output video: $RESULT_DIR/result_video.mp4"