#!/usr/bin/expect -f
## 设置环境变量
#ulimit -n 65535
#
## 限制 OpenBLAS 线程数
#export OMP_NUM_THREADS=4
#export OPENBLAS_NUM_THREADS=4
#export MKL_NUM_THREADS=4
#export VECLIB_MAXIMUM_THREADS=4
#export NUMEXPR_NUM_THREADS=4
#export CUDA_VISIBLE_DEVICES=0,1
#WORKSPACE_DIR="/root/autodl-tmp/yufei/DeepFaceLab/workspace"
#FACE_SWAP_WORKSPACE="$WORKSPACE_DIR/face_swap"
#
## 确保输出目录存在
#mkdir -p "$FACE_SWAP_WORKSPACE/data_dst/merged"
#mkdir -p "$FACE_SWAP_WORKSPACE/data_dst/merged_mask"
#
## 运行合并命令
## 回答顺序：
## 1. model1          （模型名称）
## 2. y               （确认模型）
## 3. 20              （工作进程数）
## 4. n               （不使用交互式合并）
## 第一阶段：初始化模型的输入
#echo -e "model1\n0\n0\nn\n0\nn\ny\n8\n128\nf\nliae-ud\n256\n64\n64\n22\nn\nn\nn\ny\ny\nn\ny\n0.0\n0.0\n0.0\n0.0\nnone\nn\nn\nn\nn\n4\n" | python main.py merge \
#    --input-dir "$FACE_SWAP_WORKSPACE/data_dst" \
#    --output-dir "$FACE_SWAP_WORKSPACE/data_dst/merged" \
#    --output-mask-dir "$FACE_SWAP_WORKSPACE/data_dst/merged_mask" \
#    --aligned-dir "$FACE_SWAP_WORKSPACE/data_dst/aligned" \
#    --model-dir "$FACE_SWAP_WORKSPACE/model" \
#    --model SAEHD
#
#
#
#
#
#
## 检查结果
#echo "Checking merged results..."
#ls -l "$FACE_SWAP_WORKSPACE/data_dst/merged"



set timeout -1
set workspace "/root/autodl-tmp/yufei/DeepFaceLab/workspace"
set face_swap_workspace "$workspace/face_swap"

spawn python main.py merge \
    --input-dir "$face_swap_workspace/data_dst" \
    --output-dir "$face_swap_workspace/data_dst/merged" \
    --output-mask-dir "$face_swap_workspace/data_dst/merged_mask" \
    --aligned-dir "$face_swap_workspace/data_dst/aligned" \
    --model-dir "$face_swap_workspace/model" \
    --model SAEHD

# 自动回答交互问题
expect "Enter a name of a new model :"
send "model1\r"

expect "Which GPU indexes to choose? :"
send "0\r"

expect "Autobackup every N hour"
send "0\r"

expect "Write preview history"
send "n\r"

expect "Target iteration"
send "0\r"

expect "Flip SRC faces randomly"
send "n\r"

expect "Flip DST faces randomly"
send "y\r"

expect "Batch_size"
send "128\r"

expect "Resolution"
send "256\r"

expect "Face type"
send "f\r"

expect "AE architecture"
send "liae-ud\r"

expect "AutoEncoder dimensions"
send "256\r"

expect "Encoder dimensions"
send "64\r"

expect "Decoder dimensions"
send "64\r"

expect "Decoder mask dimensions"
send "22\r"

expect "Eyes and mouth priority"
send "n\r"

expect "Uniform yaw distribution of samples"
send "n\r"

expect "Blur out mask"
send "n\r"

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

expect "Face style power"
send "0.0\r"

expect "Background style power"
send "0.0\r"

expect "Color transfer for src faceset"
send "none\r"

expect "Enable gradient clipping"
send "n\r"

expect "Enable pretraining mode"
send "n\r"

# 最后两个问题需要特别设置
expect "Use interactive merger? ( y/n )"
send "n\r"

expect "Number of workers?"
send "16\r"

expect "Choose mode:"
send "0\r"

# 等待程序运行完成
expect eof
