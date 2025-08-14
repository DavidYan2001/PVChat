#!/bin/bash

# Define common parameters
SCRIPT_PATH="/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/finetune_internvideo_REOMH_one_person2_stage.py"
BASE_PATH="/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality"
RESULT_BASE_DIR="${BASE_PATH}/different_data_result/Ch"
SKS_NAME="<Ch>"
MODEL_PATH="/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/Internvideo2_chat_8B_HD_finetune_REMOH"
BATCH_SIZE=2
NUM_EPOCHS=4
SAVE_EPOCHS=4

# Create base results directory if it doesn't exist
mkdir -p "$RESULT_BASE_DIR"

# Function to run training with specific parameters
run_training() {
    local version=$1
    local output_dir="${RESULT_BASE_DIR}/Ch${version}"
    local train_json="${BASE_PATH}/${SKS_NAME}_${version}.json"
    local short_train_json="${BASE_PATH}/${SKS_NAME}_short_train_${version}.json"
    local test_json="${BASE_PATH}/${SKS_NAME}test.json"

    echo "========================================================"
    echo "Starting training run for version ${version}"
    echo "Train JSON: ${train_json}"
    echo "Short Train JSON: ${short_train_json}"
    echo "Test JSON: ${test_json}"
    echo "Output Directory: ${output_dir}"
    echo "========================================================"

    # Create output directory
    mkdir -p "$output_dir"

    # Run the training script
    python "$SCRIPT_PATH" \
      --mode train \
      --model_path "$MODEL_PATH" \
      --sks_name "$SKS_NAME" \
      --batch_size "$BATCH_SIZE" \
      --num_epochs "$NUM_EPOCHS" \
      --save_epochs "$SAVE_EPOCHS" \
      --train_json "$train_json" \
      --short_train_json "$short_train_json" \
      --test_json "$test_json" \
      --output_dir "$output_dir"

    # Log completion
    if [ $? -eq 0 ]; then
        echo "Training for version ${version} completed successfully"
    else
        echo "ERROR: Training for version ${version} failed with exit code $?"
    fi

    echo "========================================================"
    echo ""
}

# Run training for each version in reverse order (2, 1, 0)
run_training "2"
run_training "1"
run_training "0"

echo "All training runs completed!"