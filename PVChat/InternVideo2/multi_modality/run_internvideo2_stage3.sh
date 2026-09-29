#!/usr/bin/env bash
set -euo pipefail
# InternVideo2 backbone, Stage 3 for one person on one GPU:
#   rollout generation -> GRPO (1 epoch, compact checkpoint) -> full test decode -> five metrics
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 PERSON=Sheldon STAGE2_CKPT=/path/to/stage2/checkpoint_epoch_N \
#     bash run_internvideo2_stage3.sh
#
# Environment variables (all optional unless marked):
#   PERSON        (required) person name without angle brackets, e.g. Sheldon
#   DATA_DIR      directory holding <PERSON>.json and <PERSON>test.json
#                 (default: <repo>/PVChat/datasets/cekebv-hq/$PERSON)
#   TRAIN_JSON / TEST_JSON   override the JSON paths individually
#   STAGE2_CKPT   (required) Stage 2 checkpoint directory (pvchat_delta.pt or pytorch_model.bin)
#   MODEL_PATH    InternVideo2 ReMoH model directory (default: Internvideo2_chat_8B_HD_finetune_REMOH)
#   IV2_PYTHON    python of the InternVideo2 environment (default: python)
#   QWEN_PYTHON   python of the Qwen3.5 environment used for metrics (default: $IV2_PYTHON)
#   JUDGE_MODEL   local judge weights for ES/DC (default: <repo>/models/Qwen3.5-35B-A3B)
#   OUT_ROOT      output root (default: $DATA_DIR/internvideo2_stage3_outputs)
#   TRAIN_EXTRA_ARGS  extra trainer flags, e.g. "--group_batch" (one optimizer step per group)
#
# Resume: each phase writes $OUT/status/<phase>.done and is skipped on rerun; the rollout
# phase also resumes from the groups already present in rollouts_epoch1.jsonl.

PERSON="${PERSON:?PERSON is required, e.g. PERSON=Sheldon}"
SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SD}/../../.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/PVChat/datasets/cekebv-hq/$PERSON}"
TRAIN_JSON="${TRAIN_JSON:-$DATA_DIR/<$PERSON>.json}"
TEST_JSON="${TEST_JSON:-$DATA_DIR/<$PERSON>test.json}"
STAGE2_CKPT="${STAGE2_CKPT:?STAGE2_CKPT is required (InternVideo2 Stage 2 checkpoint directory)}"
MODEL_PATH="${MODEL_PATH:-$SD/Internvideo2_chat_8B_HD_finetune_REMOH}"
IV_PY="${IV2_PYTHON:-python}"
QW_PY="${QWEN_PYTHON:-$IV_PY}"
JUDGE_MODEL="${JUDGE_MODEL:-$ROOT/models/Qwen3.5-35B-A3B}"
OUT_ROOT="${OUT_ROOT:-$DATA_DIR/internvideo2_stage3_outputs}"
read -r -a TRAIN_EXTRA <<<"${TRAIN_EXTRA_ARGS:-}"
OUT=$OUT_ROOT/$PERSON
M=$OUT/status

test -s "$TRAIN_JSON" || { echo "[IV2-S3][Error] missing $TRAIN_JSON" >&2; exit 1; }
test -s "$TEST_JSON" || { echo "[IV2-S3][Error] missing $TEST_JSON" >&2; exit 1; }
if [[ ! -s "$STAGE2_CKPT/pvchat_delta.pt" && ! -s "$STAGE2_CKPT/pytorch_model.bin" ]]; then
  echo "[IV2-S3][Error] $STAGE2_CKPT has neither pvchat_delta.pt nor pytorch_model.bin" >&2; exit 1
fi
test -d "$MODEL_PATH" || { echo "[IV2-S3][Error] missing model dir $MODEL_PATH" >&2; exit 1; }

mkdir -p "$OUT/rollouts" "$M"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# transformers 4.38 copies only one level of relative imports for a local
# trust_remote_code directory (model_config.py etc. would be missing). Pre-warm the
# whole directory into the dynamic-module cache; idempotent.
CACHE_DIR=$HOME/.cache/huggingface/modules/transformers_modules/$(basename "$MODEL_PATH")
mkdir -p "$CACHE_DIR"
touch "$CACHE_DIR/__init__.py"
cp -u "$MODEL_PATH"/*.py "$CACHE_DIR/"
echo "[IV2-S3] person=$PERSON out=$OUT gpu=${CUDA_VISIBLE_DEVICES:-unset}"

# 1) rollouts
ROLLOUT_JSONL=$OUT/rollouts/rollouts_epoch1.jsonl
if [[ ! -f $M/rollout.done ]]; then
  "$IV_PY" "$SD/generate_grpo_rollouts.py" \
    --train_json "$TRAIN_JSON" \
    --output_jsonl "$ROLLOUT_JSONL" \
    --model_path "$MODEL_PATH" \
    --checkpoint_path "$STAGE2_CKPT" \
    --sks_name "<$PERSON>" \
    --num_samples 4 --temperature 0.7 --top_p 0.9 --top_k 50 \
    --max_new_tokens 64 --keyword_backend rule --seed 42
  test -s "$ROLLOUT_JSONL"
  touch "$M/rollout.done"
fi

# 2) GRPO, 1 epoch; a compact Stage 2 input yields a compact checkpoint (no optimizer state)
CKPT_DIR=$OUT/checkpoints/$PERSON/checkpoint_epoch_1
if [[ ! -f $M/train.done ]]; then
  echo "${TRAIN_EXTRA_ARGS:-}" > "$M/train_mode.txt"
  "$IV_PY" "$SD/finetune_internvideo_REOMH_one_person3_grpo.py" \
    --rollout_jsonl "$ROLLOUT_JSONL" \
    --train_json "$TRAIN_JSON" \
    --model_path "$MODEL_PATH" \
    --checkpoint_path "$STAGE2_CKPT" \
    --reference_checkpoint_path "$STAGE2_CKPT" \
    --sks_name "<$PERSON>" \
    --test_json "$TEST_JSON" \
    --output_dir "$OUT" \
    --batch_size 1 --num_epochs 1 --save_epochs 1 --kl_beta 0.02 \
    --eval_samples 0 ${TRAIN_EXTRA[@]+"${TRAIN_EXTRA[@]}"}
  # compact input -> pvchat_delta.pt ; full (pytorch_model.bin) input -> full checkpoint
  test -s "$CKPT_DIR/pvchat_delta.pt" || test -s "$CKPT_DIR/pytorch_model.bin"
  touch "$M/train.done"
fi

# 3) full test-set decode
EVAL_DIR=$OUT/evaluation_epoch_1
if [[ ! -f $M/test.done ]]; then
  "$IV_PY" "$SD/run_pvchat_checkpoint_test.py" \
    --model_path "$MODEL_PATH" \
    --checkpoint_path "$CKPT_DIR" \
    --sks_name "<$PERSON>" \
    --test_json "$TEST_JSON" \
    --output_dir "$EVAL_DIR" \
    --epoch 1
  compgen -G "$EVAL_DIR/test_results_RE_MOH_*_1person.json" > /dev/null
  touch "$M/test.done"
fi
RESULT_JSON=$(compgen -G "$EVAL_DIR/test_results_RE_MOH_*_1person.json" | head -1)

# 4) metrics: Acc / BLEU-1 / BERTScore / ES / DC (local Qwen3.5-35B-A3B judge)
if [[ ! -f $M/metrics.done ]]; then
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  "$QW_PY" "$SD/evaluate_pvchat_metrics.py" \
    --input_json "$RESULT_JSON" \
    --output_dir "$OUT/metrics" \
    --sks_name "$PERSON" \
    --compute_bertscore --bertscore_model roberta-large \
    --judge_backend local_qwen35 --require_complete_judge \
    --local_judge_model_path "$JUDGE_MODEL" \
    --local_judge_device cuda:0 \
    --local_judge_batch_size 64 \
    --local_judge_es_items_per_prompt 1 \
    --local_judge_dc_items_per_prompt 10 \
    --local_judge_max_new_tokens 512
  test -s "$OUT/metrics/metrics_summary.json"
  touch "$M/metrics.done"
fi

echo "[IV2-S3] DONE person=$PERSON -> $OUT/metrics/metrics_summary.json"
