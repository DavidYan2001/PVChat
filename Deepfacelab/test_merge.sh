#!/usr/bin/expect -f



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

# Automatically answer interactive questions
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

# The last two questions need to be specially set
expect "Use interactive merger? ( y/n )"
send "n\r"

expect "Number of workers?"
send "16\r"

expect "Choose mode:"
send "0\r"

# Wait for the program to finish running
expect eof
