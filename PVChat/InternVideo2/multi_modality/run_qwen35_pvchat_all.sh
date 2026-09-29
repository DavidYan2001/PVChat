#!/usr/bin/env bash
set -euo pipefail
# Qwen3.5-9B PVChat-R1: Stage 1 -> Stage 2 -> Stage 3 (CA-Dynamic-GSPO) for one person.
# Usually only PERSON_NAME needs to change; every other variable can be overridden on the
# command line, e.g.  GPU_IDS=0,1 STAGE2_EPOCHS=3 PERSON_NAME=Sheldon bash run_qwen35_pvchat_all.sh
#
#   PERSON_NAME   person name without angle brackets (default: Sheldon)
#   DATA_DIR      directory holding <P>_short_train.json, <P>.json, <P>test.json
#                 (default: <repo>/PVChat/datasets/cekebv-hq/$PERSON_NAME)
#   MODEL_PATH    base Qwen3.5-9B weights (default: <repo>/Qwen3.5-9B)
#   OUTPUT_ROOT   (default: $DATA_DIR/qwen35_outputs)
#   PYTHON_BIN    python of the Qwen3.5 environment (default: python)
#   GPU_IDS       comma-separated GPU indices; default = GPUs with <20% memory in use
#   RUN_STAGE1 / RUN_STAGE2 / RUN_STAGE3   set to 0 to skip a stage (default 1)
#   STAGE1_EPOCHS / STAGE2_EPOCHS          (default 1 / 1)
#   JUDGE_BACKEND local_qwen35 (default) | dashscope | none ; SKIP_METRICS=1 skips ES/DC/BERTScore
#   VIDEO_MAX_TOKENS (default 768) / VIDEO_BUDGET_PER_10S (per-video budget, optional)
#   DISABLE_GRADIENT_CHECKPOINTING  Stage 3: 1 (default) or 0 to recompute activations on 48 GB cards

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PERSON_NAME="${PERSON_NAME:-Sheldon}"
PERSON_TOKEN="<${PERSON_NAME}>"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/PVChat/datasets/cekebv-hq/${PERSON_NAME}}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/Qwen3.5-9B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_DIR}/qwen35_outputs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRANSFORMERS_SRC="${REPO_ROOT}/transformers-qwen35/src"

TRAIN_JSON="${DATA_DIR}/${PERSON_TOKEN}.json"
SHORT_TRAIN_JSON="${DATA_DIR}/${PERSON_TOKEN}_short_train.json"
TEST_JSON="${DATA_DIR}/${PERSON_TOKEN}test.json"

RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE2="${RUN_STAGE2:-1}"
RUN_STAGE3="${RUN_STAGE3:-1}"
JUDGE_BACKEND="${JUDGE_BACKEND:-local_qwen35}"
LOCAL_JUDGE_MODEL_PATH="${LOCAL_JUDGE_MODEL_PATH:-${REPO_ROOT}/models/Qwen3.5-35B-A3B}"
LOCAL_JUDGE_BATCH_SIZE="${LOCAL_JUDGE_BATCH_SIZE:-64}"
LOCAL_JUDGE_MAX_NEW_TOKENS="${LOCAL_JUDGE_MAX_NEW_TOKENS:-512}"
VIDEO_MIN_TOKENS="${VIDEO_MIN_TOKENS:-4}"
VIDEO_MAX_TOKENS="${VIDEO_MAX_TOKENS:-768}"
DISABLE_GRADIENT_CHECKPOINTING="${DISABLE_GRADIENT_CHECKPOINTING:-1}"   # Stage 3 only; 0 = recompute activations

# Pick GPUs with less than 20% memory in use when GPU_IDS is not given.
if [[ -z "${GPU_IDS:-}" ]]; then
  GPU_IDS="$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | \
    awk -F',' '{gsub(/ /,"",$0); if (($3 / $2) < 0.20) print $1}' | paste -sd, -)"
fi
if [[ -z "${GPU_IDS}" ]]; then
  echo "[Error] No GPU with <20% memory in use was found. Set GPU_IDS=0,1 explicitly." >&2
  exit 1
fi
NPROC_PER_NODE="$(awk -F',' '{print NF}' <<<"${GPU_IDS}")"

export PYTHONPATH="${TRANSFORMERS_SRC}:${SCRIPT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

METRIC_ARGS=(--judge_backend "${JUDGE_BACKEND}")
if [[ "${JUDGE_BACKEND}" == "local_qwen35" ]]; then
  METRIC_ARGS+=(
    --local_judge_model_path "${LOCAL_JUDGE_MODEL_PATH}"
    --local_judge_device cuda:0
    --local_judge_batch_size "${LOCAL_JUDGE_BATCH_SIZE}"
    --local_judge_es_items_per_prompt 1
    --local_judge_dc_items_per_prompt 10
    --local_judge_max_new_tokens "${LOCAL_JUDGE_MAX_NEW_TOKENS}"
  )
fi
if [[ "${SKIP_METRICS:-0}" == "1" ]]; then
  METRIC_ARGS+=(--skip_metrics)
fi
VIDEO_ARGS=(--video_min_tokens "${VIDEO_MIN_TOKENS}" --video_max_tokens "${VIDEO_MAX_TOKENS}")
if [[ -n "${VIDEO_BUDGET_PER_10S:-}" ]]; then
  VIDEO_ARGS+=(--video_budget_per_10_seconds "${VIDEO_BUDGET_PER_10S}")
fi

performance_args=()
if [[ "${DISABLE_GRADIENT_CHECKPOINTING}" == "1" ]]; then
  performance_args+=(--disable_gradient_checkpointing)
fi

run_distributed() {
  local script="$1"
  shift
  echo "[Run] GPUs=${GPU_IDS} script=${script}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/${script}" "$@"
}

if [[ "${RUN_STAGE1}" == "1" ]]; then
  run_distributed finetune_qwen35_pvchat_stage1.py \
    --model_path "${MODEL_PATH}" \
    --sks_name "${PERSON_TOKEN}" \
    --train_json "${SHORT_TRAIN_JSON}" \
    --test_json "${TEST_JSON}" \
    --output_dir "${OUTPUT_ROOT}/stage1" \
    --num_epochs "${STAGE1_EPOCHS:-1}" \
    --batch_size "${BATCH_SIZE:-1}" \
    "${VIDEO_ARGS[@]}" \
    "${METRIC_ARGS[@]}"
fi

if [[ "${RUN_STAGE2}" == "1" ]]; then
  run_distributed finetune_qwen35_pvchat_stage2.py \
    --model_path "${MODEL_PATH}" \
    --checkpoint_path "${OUTPUT_ROOT}/stage1/checkpoint" \
    --sks_name "${PERSON_TOKEN}" \
    --train_json "${TRAIN_JSON}" \
    --test_json "${TEST_JSON}" \
    --output_dir "${OUTPUT_ROOT}/stage2" \
    --num_epochs "${STAGE2_EPOCHS:-1}" \
    --batch_size "${BATCH_SIZE:-1}" \
    "${VIDEO_ARGS[@]}" \
    "${METRIC_ARGS[@]}"
fi

if [[ "${RUN_STAGE3}" == "1" ]]; then
  run_distributed finetune_qwen35_pvchat_stage3_ca_dynamic_gspo_buffered.py \
    --model_path "${MODEL_PATH}" \
    --checkpoint_path "${OUTPUT_ROOT}/stage2/checkpoint" \
    --sks_name "${PERSON_TOKEN}" \
    --train_json "${TRAIN_JSON}" \
    --test_json "${TEST_JSON}" \
    --output_dir "${OUTPUT_ROOT}/stage3_ca_dynamic_gspo" \
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
    "${VIDEO_ARGS[@]}" \
    "${METRIC_ARGS[@]}" ${performance_args[@]+"${performance_args[@]}"}
fi

echo "[Done] Qwen3.5 PVChat-R1 outputs: ${OUTPUT_ROOT}"
