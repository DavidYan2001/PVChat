#!/usr/bin/env bash
set -euo pipefail
# Qwen3.5-9B backbone, Stage 3 (CA-Dynamic-GSPO) for one person, starting from a Stage 2
# checkpoint. Runs training -> full test decode -> five metrics in one process and resumes
# per phase (training / evaluation / metrics markers) when rerun with the same output_dir.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 PERSON=Sheldon bash run_qwen35_stage3_ca_dynamic_gspo.sh
#
# Environment variables (all optional unless marked):
#   PERSON        (required) person name without angle brackets
#   DATA_DIR      directory holding <PERSON>.json / <PERSON>test.json
#                 (default: <repo>/PVChat/datasets/cekebv-hq/$PERSON)
#   TRAIN_JSON / TEST_JSON   override the JSON paths individually
#   STAGE2_CKPT   Stage 2 checkpoint dir (default: $DATA_DIR/qwen35_outputs/stage2/checkpoint)
#   MODEL_PATH    base Qwen3.5-9B weights (default: <repo>/Qwen3.5-9B)
#   OUTPUT_DIR    (default: $DATA_DIR/qwen35_outputs/stage3_ca_dynamic_gspo)
#   QWEN_PYTHON   python of the Qwen3.5 environment (default: python)
#   NPROC         number of GPUs for DDP (default: number of entries in CUDA_VISIBLE_DEVICES, else 1)
#   JUDGE_BACKEND local_qwen35 (default) | dashscope | none
#   LOCAL_JUDGE_MODEL_PATH  (default: <repo>/models/Qwen3.5-35B-A3B)
#   VIDEO_MAX_TOKENS        fixed visual budget in 32x32 pixel blocks (default: 768)
#   VIDEO_BUDGET_PER_10S    per-video budget = value * ceil(seconds / 10); overrides VIDEO_MAX_TOKENS
#   FFMPEG_LIB_DIR          if set, prepended to LD_LIBRARY_PATH together with the pip NPP libs
#   DISABLE_GRADIENT_CHECKPOINTING  1 (default, faster; needs ~45 GB) or 0 (recompute activations, fits 48 GB cards)
#   EXTRA_ARGS              extra flags appended to the command line

PERSON="${PERSON:?PERSON is required, e.g. PERSON=Sheldon}"
SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SD}/../../.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/PVChat/datasets/cekebv-hq/$PERSON}"
TRAIN_JSON="${TRAIN_JSON:-$DATA_DIR/<$PERSON>.json}"
TEST_JSON="${TEST_JSON:-$DATA_DIR/<$PERSON>test.json}"
STAGE2_CKPT="${STAGE2_CKPT:-$DATA_DIR/qwen35_outputs/stage2/checkpoint}"
MODEL_PATH="${MODEL_PATH:-$ROOT/Qwen3.5-9B}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_DIR/qwen35_outputs/stage3_ca_dynamic_gspo}"
QWEN_PYTHON="${QWEN_PYTHON:-python}"
JUDGE_BACKEND="${JUDGE_BACKEND:-local_qwen35}"
LOCAL_JUDGE_MODEL_PATH="${LOCAL_JUDGE_MODEL_PATH:-$ROOT/models/Qwen3.5-35B-A3B}"
VIDEO_MAX_TOKENS="${VIDEO_MAX_TOKENS:-768}"
DISABLE_GRADIENT_CHECKPOINTING="${DISABLE_GRADIENT_CHECKPOINTING:-1}"
if [[ -z "${NPROC:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then NPROC=$(awk -F',' '{print NF}' <<<"$CUDA_VISIBLE_DEVICES"); else NPROC=1; fi
fi

test -s "$TRAIN_JSON" || { echo "[Stage3][Error] missing $TRAIN_JSON" >&2; exit 1; }
test -s "$TEST_JSON" || { echo "[Stage3][Error] missing $TEST_JSON" >&2; exit 1; }
test -s "$STAGE2_CKPT/pvchat_trainable.pt" || { echo "[Stage3][Error] missing Stage 2 checkpoint $STAGE2_CKPT/pvchat_trainable.pt" >&2; exit 1; }
test -d "$MODEL_PATH" || { echo "[Stage3][Error] missing base weights $MODEL_PATH" >&2; exit 1; }

# TorchCodec dlopens FFmpeg (4-7) and CUDA NPP at runtime; point it at them when the
# system library path does not already provide them.
if [[ -n "${FFMPEG_LIB_DIR:-}" ]]; then
  NPP_LIB_DIR="${NPP_LIB_DIR:-$("$QWEN_PYTHON" -c 'import sysconfig, os; print(os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "npp", "lib"))')}"
  export LD_LIBRARY_PATH="${FFMPEG_LIB_DIR}:${NPP_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export PYTHONPATH="$ROOT/transformers-qwen35/src:$SD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" TOKENIZERS_PARALLELISM=false

VIDEO_ARGS=(--video_min_tokens 4 --video_max_tokens "$VIDEO_MAX_TOKENS")
if [[ -n "${VIDEO_BUDGET_PER_10S:-}" ]]; then
  VIDEO_ARGS+=(--video_budget_per_10_seconds "$VIDEO_BUDGET_PER_10S")
fi
JUDGE_ARGS=(--judge_backend "$JUDGE_BACKEND")
if [[ "$JUDGE_BACKEND" == "local_qwen35" ]]; then
  JUDGE_ARGS+=(
    --local_judge_model_path "$LOCAL_JUDGE_MODEL_PATH"
    --local_judge_device cuda:0
    --local_judge_batch_size 64
    --local_judge_es_items_per_prompt 1
    --local_judge_dc_items_per_prompt 10
    --local_judge_max_new_tokens 512
  )
fi
read -r -a EXTRA <<<"${EXTRA_ARGS:-}"
performance_args=()
if [[ "$DISABLE_GRADIENT_CHECKPOINTING" == "1" ]]; then
  performance_args+=(--disable_gradient_checkpointing)
fi

mkdir -p "$OUTPUT_DIR"
echo "[Stage3] person=$PERSON gpus=${CUDA_VISIBLE_DEVICES:-unset} nproc=$NPROC out=$OUTPUT_DIR"
"$QWEN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node "$NPROC" \
  "$SD/finetune_qwen35_pvchat_stage3_ca_dynamic_gspo_buffered.py" \
  --model_path "$MODEL_PATH" \
  --checkpoint_path "$STAGE2_CKPT" \
  --sks_name "<$PERSON>" \
  --train_json "$TRAIN_JSON" \
  --test_json "$TEST_JSON" \
  --output_dir "$OUTPUT_DIR" \
  --num_epochs 1 --max_steps_per_epoch 0 \
  --num_samples 4 --max_new_tokens 96 \
  --temperature 0.5 --top_p 0.9 --top_k 50 \
  --negative_group_weight 2.0 \
  --clip_range 0.2 --drift_beta 0.02 --ref_kl_beta 0.0 \
  --rollout_buffer_size 4 \
  --ca_semantic_threshold 0.60 --ca_informative_margin 0.10 \
  --ca_sft_weight 0.05 --ca_fallback_sft_weight 0.20 \
  --advantage_eps 1e-6 --seed 42 \
  --token_lr 5e-5 --remoh_lr 1e-6 --lora_lr 1e-6 \
  --eval_batch_size 48 --eval_max_new_tokens 96 \
  "${VIDEO_ARGS[@]}" "${JUDGE_ARGS[@]}" ${performance_args[@]+"${performance_args[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}

echo "[Stage3] DONE person=$PERSON -> $OUTPUT_DIR/metrics/epoch_1/metrics_summary.json"
