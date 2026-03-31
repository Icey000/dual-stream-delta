#!/bin/bash
set -euo pipefail

# Official caption test on a chosen checkpoint.
# This runs decode stage on the requested split (default: test) using the
# explicit checkpoint path, so it does not depend on phase-best checkpoint
# auto-discovery.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_NAME="${MODEL_NAME:-Dual-QFormer-Qwen}"
SOCCERNET_PATH="${SOCCERNET_PATH:-/path/to/caption-2024}"
AUDIO_ROOT="${AUDIO_ROOT:-/path/to/SoccerNet-audio}"
LLM_MODEL_PATH="${LLM_MODEL_PATH:-Qwen/Qwen2.5-7B}"
TEST_SPLIT="${TEST_SPLIT:-test}"
BATCH_SIZE_CAPTION="${BATCH_SIZE_CAPTION:-1}"
MAX_NUM_WORKER="${MAX_NUM_WORKER:-2}"
CAPTION_TEST_MAX_SAMPLES="${CAPTION_TEST_MAX_SAMPLES:-0}"
DECODE_NUM_EXAMPLES="${DECODE_NUM_EXAMPLES:-5}"
WANDB_PROJECT="${WANDB_PROJECT:-DVC-SoccerNet}"
WANDB_MODE="${WANDB_MODE:-online}"

# Default to the best checkpoint from the compare set, but allow the caller to
# override with CHECKPOINT_PATH directly.
CHECKPOINT_KIND="${CHECKPOINT_KIND:-rl}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

case "$CHECKPOINT_KIND" in
  rl)
    DEFAULT_CHECKPOINT="models/${MODEL_NAME}/rl/best_rl.pth.tar"
    ;;
  joint)
    DEFAULT_CHECKPOINT="models/${MODEL_NAME}/joint/best_joint.pth.tar"
    ;;
  caption)
    DEFAULT_CHECKPOINT="models/${MODEL_NAME}/caption/best_caption_ce.pth.tar"
    ;;
  *)
    echo "Invalid CHECKPOINT_KIND: ${CHECKPOINT_KIND} (expected rl, joint, or caption)" >&2
    exit 1
    ;;
esac

if [ -z "$CHECKPOINT_PATH" ]; then
  CHECKPOINT_PATH="$DEFAULT_CHECKPOINT"
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "Checkpoint not found: $CHECKPOINT_PATH" >&2
  exit 1
fi

BEST_DECODE_CONFIG="models/${MODEL_NAME}/caption/decoding_sweeps/best_decode_config.json"

SWEEP_MAX_NEW_TOKENS="${SWEEP_MAX_NEW_TOKENS:-30}"
SWEEP_NO_REPEAT_NGRAM_SIZE="${SWEEP_NO_REPEAT_NGRAM_SIZE:-2}"
SWEEP_NUM_BEAMS="${SWEEP_NUM_BEAMS:-3}"
SWEEP_TEMPERATURE="${SWEEP_TEMPERATURE:-0.6}"

timestamp="$(date +%Y-%m-%d_%H-%M-%S)"
log_dir="models/${MODEL_NAME}/official_test_logs"
mkdir -p "$log_dir"
run_log="${log_dir}/official_test_${timestamp}.log"

echo "Running official test on split=${TEST_SPLIT}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Log: ${run_log}"

caption_config_flag=()
if [ -f "$BEST_DECODE_CONFIG" ]; then
  caption_config_flag=(--caption_generation_config_json "$BEST_DECODE_CONFIG")
fi

python main.py \
  --stage decode \
  --SoccerNet_path "$SOCCERNET_PATH" \
  --audio_root "$AUDIO_ROOT" \
  --model_name "$MODEL_NAME" \
  --use_dual_stream \
  --llm_model_path "$LLM_MODEL_PATH" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --split_valid "$TEST_SPLIT" \
  --batch_size_caption "$BATCH_SIZE_CAPTION" \
  --max_num_worker "$MAX_NUM_WORKER" \
  --caption_valid_max_samples "$CAPTION_TEST_MAX_SAMPLES" \
  "${caption_config_flag[@]}" \
  --sweep_max_new_tokens "$SWEEP_MAX_NEW_TOKENS" \
  --sweep_no_repeat_ngram_size "$SWEEP_NO_REPEAT_NGRAM_SIZE" \
  --sweep_num_beams "$SWEEP_NUM_BEAMS" \
  --sweep_temperature "$SWEEP_TEMPERATURE" \
  --decode_num_examples "$DECODE_NUM_EXAMPLES" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_name "${MODEL_NAME}_${CHECKPOINT_KIND}_${TEST_SPLIT}_${timestamp}" \
  --wandb_group "official_test_${timestamp}" \
  --wandb_mode "$WANDB_MODE" \
  --loglevel INFO 2>&1 | tee "$run_log"

echo
echo "Done. Review ${run_log} for the full official test output."
