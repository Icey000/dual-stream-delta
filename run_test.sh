#!/bin/bash
set -euo pipefail

SOCCERNET_PATH="${SOCCERNET_PATH:-/path/to/caption-2024}"
AUDIO_ROOT="${AUDIO_ROOT:-/path/to/SoccerNet-audio}"
MODEL_NAME="${MODEL_NAME:-Dual-QFormer-Qwen}"
LLM_MODEL_PATH="${LLM_MODEL_PATH:-Qwen/Qwen2.5-7B}"

python main.py \
--SoccerNet_path "$SOCCERNET_PATH" \
--audio_root "$AUDIO_ROOT" \
--model_name "$MODEL_NAME" \
--use_dual_stream \
--GPU 0 \
--pool QFormer \
--NMS_threshold 0.3 \
--max_epochs 20 \
--teacher_forcing_ratio 1 \
--batch_size 48 \
--pretrain \
--window_size_caption 30 \
--max_num_worker 2 \
--model_type qwen \
--llm_model_path "$LLM_MODEL_PATH" \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.05 \
--weight_decay 0.05 \
--encoder_dropout 0.3 \
--epochs_classify 15 \
--epochs_caption 20 \
--test_only


#--teacher_forcing_ratio 0.9 \
#--continue_training \
#--pool TRANS \
#--window_size_caption 45
