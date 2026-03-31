#!/bin/bash
set -euo pipefail

# Quick compare for three checkpoints on the validation split.
# This is intentionally much smaller than a formal test run.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_NAME="${MODEL_NAME:-Dual-QFormer-Qwen}"
SOCCERNET_PATH="${SOCCERNET_PATH:-/path/to/caption-2024}"
AUDIO_ROOT="${AUDIO_ROOT:-/path/to/SoccerNet-audio}"
LLM_MODEL_PATH="${LLM_MODEL_PATH:-Qwen/Qwen2.5-7B}"
BEST_DECODE_CONFIG="models/${MODEL_NAME}/caption/decoding_sweeps/best_decode_config.json"

# Quick mode defaults.
# Set VALID_MAX_SAMPLES=0 (or MAX_SAMPLE=0) before running if you want full validation.
VALID_SPLIT="valid"
VALID_MAX_SAMPLES="${VALID_MAX_SAMPLES:-${MAX_SAMPLE:-100}}"
BATCH_SIZE_CAPTION="${BATCH_SIZE_CAPTION:-1}"
MAX_NUM_WORKER="${MAX_NUM_WORKER:-2}"
DECODE_NUM_EXAMPLES="${DECODE_NUM_EXAMPLES:-3}"
WANDB_PROJECT="${WANDB_PROJECT:-DVC-SoccerNet}"
WANDB_MODE="${WANDB_MODE:-online}"

# One fixed decode setting keeps the comparison fair and fast.
SWEEP_MAX_NEW_TOKENS="${SWEEP_MAX_NEW_TOKENS:-30}"
SWEEP_NO_REPEAT_NGRAM_SIZE="${SWEEP_NO_REPEAT_NGRAM_SIZE:-2}"
SWEEP_NUM_BEAMS="${SWEEP_NUM_BEAMS:-3}"
SWEEP_TEMPERATURE="${SWEEP_TEMPERATURE:-0.6}"

declare -a CHECKPOINTS=(
  "rl|models/${MODEL_NAME}/rl/best_rl.pth.tar"
  "joint|models/${MODEL_NAME}/joint/best_joint.pth.tar"
  "caption|models/${MODEL_NAME}/caption/best_caption_ce.pth.tar"
)

timestamp="$(date +%Y-%m-%d_%H-%M-%S)"
log_dir="models/${MODEL_NAME}/compare_logs"
mkdir -p "$log_dir"
master_log="${log_dir}/compare_ckpts_${timestamp}.log"

echo "Comparing checkpoints on split=${VALID_SPLIT}, caption_valid_max_samples=${VALID_MAX_SAMPLES}"
echo "Master log: ${master_log}"

print_latest_summary() {
  local json_path="$1"
  python - "$json_path" <<'PY'
import json
import sys

path = sys.argv[1]
data = json.load(open(path))
best = data.get("best_by_cider") or {}
metrics = best.get("metrics") or {}
examples = best.get("examples") or []
ckpt = data.get("checkpoint_path", "<unknown>")
print(f"Summary for {ckpt}")
print(
    "  CIDEr={CIDEr:.6f} Bleu_4={Bleu_4:.6f} METEOR={METEOR:.6f} ROUGE_L={ROUGE_L:.6f} SPICE={SPICE:.6f}".format(
        CIDEr=float(metrics.get("CIDEr", 0.0) or 0.0),
        Bleu_4=float(metrics.get("Bleu_4", 0.0) or 0.0),
        METEOR=float(metrics.get("METEOR", 0.0) or 0.0),
        ROUGE_L=float(metrics.get("ROUGE_L", 0.0) or 0.0),
        SPICE=float(metrics.get("SPICE", 0.0) or 0.0),
    )
)
if examples:
    first = examples[0]
    print(f"  Example REF: {first.get('reference', '')}")
    print(f"  Example PRED: {first.get('prediction', '')}")
PY
}

for item in "${CHECKPOINTS[@]}"; do
  IFS='|' read -r name ckpt <<< "$item"
  if [ ! -f "$ckpt" ]; then
    echo "[skip] ${name}: checkpoint not found -> ${ckpt}"
    continue
  fi

  run_log="${log_dir}/${name}_${timestamp}.log"
  echo
  echo "============================================================" | tee -a "$master_log"
  echo "[${name}] checkpoint: ${ckpt}" | tee -a "$master_log"
  echo "[${name}] log: ${run_log}" | tee -a "$master_log"
  echo "============================================================" | tee -a "$master_log"

  python decode_sweep.py \
    --SoccerNet_path "$SOCCERNET_PATH" \
    --audio_root "$AUDIO_ROOT" \
    --model_name "$MODEL_NAME" \
    --checkpoint_path "$ckpt" \
    --use_dual_stream \
    --llm_model_path "$LLM_MODEL_PATH" \
    --split_valid "$VALID_SPLIT" \
    --batch_size_caption "$BATCH_SIZE_CAPTION" \
    --max_num_worker "$MAX_NUM_WORKER" \
    --caption_valid_max_samples "$VALID_MAX_SAMPLES" \
    --caption_generation_config_json "$BEST_DECODE_CONFIG" \
    --sweep_max_new_tokens "$SWEEP_MAX_NEW_TOKENS" \
    --sweep_no_repeat_ngram_size "$SWEEP_NO_REPEAT_NGRAM_SIZE" \
    --sweep_num_beams "$SWEEP_NUM_BEAMS" \
    --sweep_temperature "$SWEEP_TEMPERATURE" \
    --decode_num_examples "$DECODE_NUM_EXAMPLES" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_name "${name}_${timestamp}" \
    --wandb_group "compare_ckpts_${timestamp}" \
    --wandb_mode "$WANDB_MODE" \
    --loglevel INFO 2>&1 | tee "$run_log" | tee -a "$master_log" >/dev/null

  latest_json="$(find "models/${MODEL_NAME}/caption/decoding_sweeps" -maxdepth 1 -name 'decode_sweep_*.json' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
  if [ -n "${latest_json:-}" ] && [ -f "$latest_json" ]; then
    print_latest_summary "$latest_json" | tee -a "$master_log"
  fi
done

echo
echo "Done. Check ${log_dir} for per-checkpoint logs."
