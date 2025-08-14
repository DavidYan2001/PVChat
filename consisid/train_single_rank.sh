# export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE="offline"
# export WANDB_MODE="online"

export MODEL_PATH="BestWishYsh/ConsisID-preview"
export CONFIG_PATH="BestWishYsh/ConsisID-preview"
export TYPE="i2v"
export DATASET_PATH="asserts/demo_train_data/total_train_data.txt"
export OUTPUT_PATH="consisid_finetune_single_rank"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export PDSH_RCMD_TYPE=ssh
# # NCCL setting
# export GLOO_SOCKET_IFNAME=bond0
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_TC=162
# export NCCL_IB_TIMEOUT=25
# export NCCL_PXN_DISABLE=0
# export NCCL_IB_QPS_PER_CONNECTION=4
# export NCCL_ALGO=Ring
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export NCCL_IB_RETRY_CNT=32
# export NCCL_ALGO=Tree

# if you are not using wth 8 gus, change `accelerate_config_machine_single.yaml` num_processes as your gpu number
accelerate launch --config_file util/deepspeed_configs/accelerate_config_machine_single.yaml \
  train.py \
  --config_path $CONFIG_PATH \
  --dataloader_num_workers 8 \
  --gradient_checkpointing \
  --pretrained_model_name_or_path $MODEL_PATH \
  --instance_data_root $DATASET_PATH \
  --validation_prompt "The video features a woman standing next to an airplane, engaged in a conversation on her cell phone. She is wearing sunglasses and a black top, and she appears to be talking seriously. The airplane has a green stripe running along its side, and there is a large engine visible behind her. The woman seems to be standing near the entrance of the airplane, possibly preparing to board or just having disembarked. The setting suggests that she might be at an airport or a private airfield. The overall atmosphere of the video is professional and focused, with the woman's attire and the presence of the airplane indicating a business or travel context." \
  --validation_images "asserts/example_images/4.png" \
  --validation_prompt_separator ::: \
  --num_validation_videos 1 \
  --validation_epochs 1000 \
  --seed 42 \
  --mixed_precision bf16 \
  --output_dir $OUTPUT_PATH \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 49 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --checkpointing_steps 250 \
  --num_train_epochs 15 \
  --learning_rate 3e-6 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 200 \
  --lr_num_cycles 1 \
  --gradient_checkpointing \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --resume_from_checkpoint="latest" \
  --report_to wandb \
  --sample_stride 3 \
  --skip_frames_start 7 \
  --skip_frames_end 7 \
  --miss_tolerance 6 \
  --min_distance 3 \
  --min_frames 1 \
  --max_frames 1 \
  --LFE_num_tokens 32 \
  --LFE_output_dim 768 \
  --LFE_heads 12 \
  --cross_attn_interval 2 \
  --is_train_face \
  --is_single_face \
  --enable_mask_loss \
  --is_accelerator_state_dict \
  --is_validation \
  --is_align_face \
  --train_type $TYPE \
  --is_shuffle_data
  
  # --is_kps \
  # --pretrained_weight "checkpoint-1250" \
  # --is_diff_lr \
  # --low_vram \
  # --is_cross_face
  # --enable_slicing \
  # --enable_tiling \
  # --use_ema