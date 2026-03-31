#!/bin/bash
# ==========================================
# 🌍 训练环境配置
# ==========================================
# 解决 PyTorch 显存碎片化的警告 (必须在导入 torch 前设置环境变量)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ==========================================
# 🧪 可调超参数（集中管理）
# ==========================================
# 差异化微调：caption/spotting 默认开启（不冻结 Q-Former）
DISCRIMINATIVE_FT_CAPTION=true
DISCRIMINATIVE_FT_SPOTTING=true
STAGE=rl

# Caption 分组学习率
# 当前配置切到“caption 阶段同步微调 Q-Former/encoder”。
# 保持 Q-Former 学习率显著低于 LoRA/proj，避免一上来把已学到的表征冲坏。
LR_CAPTION_BASE=1e-5
LR_CAPTION_LORA=2e-5
LR_CAPTION_PROJ=1e-5
LR_CAPTION_QFORMER=7e-6
LR_JOINT=1e-5
LR_RL=5e-6

# Spotting 分组学习率
LR_SPOTTING_BASE=5e-5
LR_SPOTTING_PROJ_HEAD=3e-5
LR_SPOTTING_QFORMER=2e-6
SPOTTING_LOSS=ce
SPOTTING_TARGET_MODE=soft_window_multiclass
SPOTTING_SOFT_WINDOW_RADIUS=3
SPOTTING_SOFT_WINDOW_SIGMA=1.5
SPOTTING_USE_CENTER_REGRESSION=true
SPOTTING_CENTER_REGRESSION_WEIGHT=0.5
SPOTTING_CENTER_POSITIVE_THRESHOLD=0.5
FOCAL_ALPHA=0.75
FOCAL_GAMMA=2.0
JOINT_LAMBDA_CAPTION=2.0
JOINT_WARM_START_SPOTTING_HEAD=true
JOINT_SPOTTING_CHECKPOINT_PATH=
RL_REWARD=cider
RL_WEIGHT=1.0
RL_INIT_STAGE=joint
RL_SAMPLE_TEMPERATURE=0.6
RL_SAMPLE_TOP_P=1.0
RL_SAMPLE_MAX_NEW_TOKENS=30

# 梯度裁剪（全员防护）
MAX_GRAD_NORM=0.5
MAX_GRAD_NORM_CLASSIFY=0.5
MAX_GRAD_NORM_CAPTION=0.5
MAX_GRAD_NORM_SPOTTING=0.5
MAX_GRAD_NORM_JOINT=0.5
MAX_GRAD_NORM_RL=0.5

# DeepSpeed 显存/通信开关（方案1: 最小代价省显存）
DS_OVERLAP_COMM=false
DS_ROUND_ROBIN_GRADIENTS=True

# Smoke 测试（每个 epoch 只跑前 N 个 batch；0 表示关闭）
SMOKE_STEPS=0
SMOKE_STEPS_CLASSIFY=$SMOKE_STEPS
SMOKE_STEPS_CAPTION=$SMOKE_STEPS
SMOKE_STEPS_SPOTTING=$SMOKE_STEPS
SMOKE_STEPS_JOINT=$SMOKE_STEPS
SMOKE_STEPS_RL=$SMOKE_STEPS
ACCUMULATION_STEPS_JOINT=1
ACCUMULATION_STEPS_RL=1

# 阶段冻结控制（caption 现在显式不冻结 encoder/Q-Former）
FREEZE_ENCODER_CLASSIFY=false
FREEZE_ENCODER_CAPTION=false
FREEZE_ENCODER_SPOTTING=false
FREEZE_ENCODER_JOINT=false
START_STAGE=spotting
CONTINUE_TRAINING=False
LOAD_BEST_METRIC_CHECKPOINT=False
RUN_DVC=false
# Caption 评估控制
SKIP_CAPTION_EVAL=false
CAPTION_VALID_MAX_SAMPLES=100
CAPTION_TEST_MAX_SAMPLES=1000
# Spotting 评估控制
SPOTTING_VALID_MAX_SAMPLES=300
SPOTTING_TEST_MAX_SAMPLES=1000
# Validation 指标评估频率：每 N 个 epoch 跑一次 BLEU/CIDEr/SPICE/mAP
EVALUATION_FREQUENCY=10
EVALUATION_FREQUENCY_CLASSIFY=5
EVALUATION_FREQUENCY_CAPTION=1
EVALUATION_FREQUENCY_SPOTTING=1
EVALUATION_FREQUENCY_JOINT=1
EVALUATION_FREQUENCY_RL=1
# DataLoader worker（测试阶段可适度增大）
MAX_NUM_WORKER=2

# Caption decoding defaults
CAPTION_MAX_NEW_TOKENS=48
CAPTION_NO_REPEAT_NGRAM_SIZE=3
CAPTION_NUM_BEAMS=1
CAPTION_LENGTH_PENALTY=0.9
CAPTION_DO_SAMPLE=false
CAPTION_TEMPERATURE=1.0
CAPTION_TOP_P=1.0
CAPTION_REPETITION_PENALTY=1.15
BEST_DECODE_CONFIG_JSON=models/Dual-QFormer-Qwen/caption/decoding_sweeps/best_decode_config.json
CAPTION_GENERATION_CONFIG_JSON=$BEST_DECODE_CONFIG_JSON
RL_EVAL_GENERATION_CONFIG_JSON=$BEST_DECODE_CONFIG_JSON
DECODE_NUM_EXAMPLES=5

SWEEP_MAX_NEW_TOKENS="20 30 40"
SWEEP_NO_REPEAT_NGRAM_SIZE="2 3 4"
SWEEP_NUM_BEAMS="1 3 5"
SWEEP_TEMPERATURE="0.6 0.7 0.8"

is_true() {
    case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|y|on) return 0 ;;
        *) return 1 ;;
    esac
}

if is_true "$DISCRIMINATIVE_FT_CAPTION"; then
    DISCRIMINATIVE_FT_CAPTION_FLAG="--discriminative_ft_caption"
else
    DISCRIMINATIVE_FT_CAPTION_FLAG="--no-discriminative_ft_caption"
fi

if is_true "$DISCRIMINATIVE_FT_SPOTTING"; then
    DISCRIMINATIVE_FT_SPOTTING_FLAG="--discriminative_ft_spotting"
else
    DISCRIMINATIVE_FT_SPOTTING_FLAG="--no-discriminative_ft_spotting"
fi

FREEZE_ENCODER_CLASSIFY_FLAG=""
FREEZE_ENCODER_CAPTION_FLAG=""
FREEZE_ENCODER_SPOTTING_FLAG=""
if is_true "$FREEZE_ENCODER_CLASSIFY"; then
    FREEZE_ENCODER_CLASSIFY_FLAG="--freeze_encoder_classify"
fi
if is_true "$FREEZE_ENCODER_CAPTION"; then
    FREEZE_ENCODER_CAPTION_FLAG="--freeze_encoder_caption"
fi
if is_true "$FREEZE_ENCODER_SPOTTING"; then
    FREEZE_ENCODER_SPOTTING_FLAG="--freeze_encoder_spotting"
fi
FREEZE_ENCODER_JOINT_FLAG="--freeze_encoder_joint"
if ! is_true "$FREEZE_ENCODER_JOINT"; then
    FREEZE_ENCODER_JOINT_FLAG="--no-freeze_encoder_joint"
fi

SMOKE_STEPS_CLASSIFY_FLAG=""
SMOKE_STEPS_CAPTION_FLAG=""
SMOKE_STEPS_SPOTTING_FLAG=""
SMOKE_STEPS_JOINT_FLAG=""
SMOKE_STEPS_RL_FLAG=""
RUN_DVC_FLAG=""
SKIP_CAPTION_EVAL_FLAG=""
SPOTTING_VALID_MAX_SAMPLES_FLAG=""
SPOTTING_TEST_MAX_SAMPLES_FLAG=""
CONTINUE_TRAINING_FLAG=""
LOAD_BEST_METRIC_CHECKPOINT_FLAG=""
SPOTTING_USE_CENTER_REGRESSION_FLAG="--no-spotting_use_center_regression"
JOINT_WARM_START_SPOTTING_HEAD_FLAG="--joint_warm_start_spotting_head"
CAPTION_GENERATION_CONFIG_JSON_FLAG=""
RL_EVAL_GENERATION_CONFIG_JSON_FLAG=""
JOINT_SPOTTING_CHECKPOINT_PATH_FLAG=""
if [ -n "$SMOKE_STEPS_CLASSIFY" ]; then
    SMOKE_STEPS_CLASSIFY_FLAG="--smoke_steps_classify $SMOKE_STEPS_CLASSIFY"
fi
if [ -n "$SMOKE_STEPS_CAPTION" ]; then
    SMOKE_STEPS_CAPTION_FLAG="--smoke_steps_caption $SMOKE_STEPS_CAPTION"
fi
if [ -n "$SMOKE_STEPS_SPOTTING" ]; then
    SMOKE_STEPS_SPOTTING_FLAG="--smoke_steps_spotting $SMOKE_STEPS_SPOTTING"
fi
if [ -n "$SMOKE_STEPS_JOINT" ]; then
    SMOKE_STEPS_JOINT_FLAG="--smoke_steps_joint $SMOKE_STEPS_JOINT"
fi
if [ -n "$SMOKE_STEPS_RL" ]; then
    SMOKE_STEPS_RL_FLAG="--smoke_steps_rl $SMOKE_STEPS_RL"
fi
if is_true "$RUN_DVC"; then
    RUN_DVC_FLAG="--run_dvc"
fi
if is_true "$SKIP_CAPTION_EVAL"; then
    SKIP_CAPTION_EVAL_FLAG="--skip_caption_eval"
fi
CAPTION_DO_SAMPLE_FLAG="--no-caption_do_sample"
if is_true "$CAPTION_DO_SAMPLE"; then
    CAPTION_DO_SAMPLE_FLAG="--caption_do_sample"
fi
if [ -n "$SPOTTING_VALID_MAX_SAMPLES" ]; then
    SPOTTING_VALID_MAX_SAMPLES_FLAG="--spotting_valid_max_samples $SPOTTING_VALID_MAX_SAMPLES"
fi
if [ -n "$SPOTTING_TEST_MAX_SAMPLES" ]; then
    SPOTTING_TEST_MAX_SAMPLES_FLAG="--spotting_test_max_samples $SPOTTING_TEST_MAX_SAMPLES"
fi
if is_true "$CONTINUE_TRAINING"; then
    CONTINUE_TRAINING_FLAG="--continue_training"
fi
if is_true "$LOAD_BEST_METRIC_CHECKPOINT"; then
    LOAD_BEST_METRIC_CHECKPOINT_FLAG="--load_best_metric_checkpoint"
fi
if is_true "$SPOTTING_USE_CENTER_REGRESSION"; then
    SPOTTING_USE_CENTER_REGRESSION_FLAG="--spotting_use_center_regression"
fi
if ! is_true "$JOINT_WARM_START_SPOTTING_HEAD"; then
    JOINT_WARM_START_SPOTTING_HEAD_FLAG="--no-joint_warm_start_spotting_head"
fi
if [ -n "$CAPTION_GENERATION_CONFIG_JSON" ]; then
    CAPTION_GENERATION_CONFIG_JSON_FLAG="--caption_generation_config_json $CAPTION_GENERATION_CONFIG_JSON"
fi
if [ -n "$RL_EVAL_GENERATION_CONFIG_JSON" ]; then
    RL_EVAL_GENERATION_CONFIG_JSON_FLAG="--rl_eval_generation_config_json $RL_EVAL_GENERATION_CONFIG_JSON"
fi
if [ -n "$JOINT_SPOTTING_CHECKPOINT_PATH" ]; then
    JOINT_SPOTTING_CHECKPOINT_PATH_FLAG="--joint_spotting_checkpoint_path $JOINT_SPOTTING_CHECKPOINT_PATH"
fi

# 设置使用的显卡数量 (1 表示单卡，2 表示双卡分布式)
NUM_GPUS=2

# 显式指定你的 DeepSpeed 配置文件路径（假设就在当前目录下）
# ==========================================
# 🌍 硬件环境选择
# ==========================================
# 可选: V100 (默认 ZeRO-2 FP16), A6000 (默认 BF16)
GPU_TYPE="A6000"

if [ "$GPU_TYPE" == "V100" ]; then
    DS_CONFIG="configs/ds_v100_fp16.json"
    echo "⚡ 检测到 V100，使用 ZeRO-2 FP16 模式..."
else
    DS_CONFIG="configs/ds_a6000_bf16.json"
    echo "🔥 检测到 A6000，开启 BF16 满血模式..."
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "🚀 启动多卡分布式训练 (GPUs: $NUM_GPUS) ..."
    # 【改动在这里！】直接在 launch 后面加上 --use_deepspeed 和配置文件的路径
    CMD="accelerate launch --num_processes $NUM_GPUS --use_deepspeed --deepspeed_config_file $DS_CONFIG main.py --use_distributed"
else
    echo "🌟 启动单卡独立训练 ..."
    # 单卡模式：使用原生 python
    CMD="python main.py --GPU 0"
fi

# ==========================================
# ⚙️ 核心训练参数
# ==========================================
SPOTTING_EPOCHS=25
EPOCHS_JOINT=15
EPOCHS_RL=10
SOCCERNET_PATH="${SOCCERNET_PATH:-/path/to/caption-2024}"
AUDIO_ROOT="${AUDIO_ROOT:-/path/to/SoccerNet-audio}"
MODEL_NAME="${MODEL_NAME:-Dual-QFormer-Qwen}"
LLM_MODEL_PATH="${LLM_MODEL_PATH:-Qwen/Qwen2.5-7B}"

$CMD \
--SoccerNet_path "$SOCCERNET_PATH" \
--audio_root "$AUDIO_ROOT" \
--model_name "$MODEL_NAME" \
--use_dual_stream \
--pool QFormer \
--NMS_threshold 0.7 \
--stage $STAGE \
--spotting_epochs $SPOTTING_EPOCHS \
--epochs_joint $EPOCHS_JOINT \
--epochs_rl $EPOCHS_RL \
--start_stage $START_STAGE \
--batch_size_classify 128 \
--accumulation_steps_classify 1 \
--batch_size_caption 1 \
--accumulation_steps_caption 12 \
--batch_size_spotting 128 \
--accumulation_steps_spotting 1 \
--accumulation_steps_joint $ACCUMULATION_STEPS_JOINT \
--accumulation_steps_rl $ACCUMULATION_STEPS_RL \
--pretrain \
--window_size_caption 45 \
--evaluation_frequency $EVALUATION_FREQUENCY \
--evaluation_frequency_classify $EVALUATION_FREQUENCY_CLASSIFY \
--evaluation_frequency_caption $EVALUATION_FREQUENCY_CAPTION \
--evaluation_frequency_spotting $EVALUATION_FREQUENCY_SPOTTING \
--evaluation_frequency_joint $EVALUATION_FREQUENCY_JOINT \
--evaluation_frequency_rl $EVALUATION_FREQUENCY_RL \
--max_num_worker $MAX_NUM_WORKER \
--llm_model_path "$LLM_MODEL_PATH" \
--lora_r 8 \
--lora_alpha 8 \
--lora_dropout 0.1 \
--weight_decay 0.05 \
--encoder_dropout 0.3 \
--epochs_classify 20 \
--epochs_caption 20 \
--LR $LR_SPOTTING_BASE \
--LR_caption $LR_CAPTION_BASE \
--lr_caption_lora $LR_CAPTION_LORA \
--lr_caption_proj $LR_CAPTION_PROJ \
--lr_caption_qformer $LR_CAPTION_QFORMER \
--lr_joint $LR_JOINT \
--lr_rl $LR_RL \
--lr_spotting_proj_head $LR_SPOTTING_PROJ_HEAD \
--lr_spotting_qformer $LR_SPOTTING_QFORMER \
--spotting_loss $SPOTTING_LOSS \
--spotting_target_mode $SPOTTING_TARGET_MODE \
--spotting_soft_window_radius $SPOTTING_SOFT_WINDOW_RADIUS \
--spotting_soft_window_sigma $SPOTTING_SOFT_WINDOW_SIGMA \
--spotting_center_regression_weight $SPOTTING_CENTER_REGRESSION_WEIGHT \
--spotting_center_positive_threshold $SPOTTING_CENTER_POSITIVE_THRESHOLD \
--focal_alpha $FOCAL_ALPHA \
--focal_gamma $FOCAL_GAMMA \
--caption_max_new_tokens $CAPTION_MAX_NEW_TOKENS \
--caption_no_repeat_ngram_size $CAPTION_NO_REPEAT_NGRAM_SIZE \
--caption_num_beams $CAPTION_NUM_BEAMS \
--caption_length_penalty $CAPTION_LENGTH_PENALTY \
--caption_temperature $CAPTION_TEMPERATURE \
--caption_top_p $CAPTION_TOP_P \
--caption_repetition_penalty $CAPTION_REPETITION_PENALTY \
--decode_num_examples $DECODE_NUM_EXAMPLES \
--sweep_max_new_tokens $SWEEP_MAX_NEW_TOKENS \
--sweep_no_repeat_ngram_size $SWEEP_NO_REPEAT_NGRAM_SIZE \
--sweep_num_beams $SWEEP_NUM_BEAMS \
--sweep_temperature $SWEEP_TEMPERATURE \
--joint_lambda_caption $JOINT_LAMBDA_CAPTION \
--rl_reward $RL_REWARD \
--rl_weight $RL_WEIGHT \
--rl_init_stage $RL_INIT_STAGE \
--rl_sample_temperature $RL_SAMPLE_TEMPERATURE \
--rl_sample_top_p $RL_SAMPLE_TOP_P \
--rl_sample_max_new_tokens $RL_SAMPLE_MAX_NEW_TOKENS \
--max_grad_norm $MAX_GRAD_NORM \
--max_grad_norm_classify $MAX_GRAD_NORM_CLASSIFY \
--max_grad_norm_caption $MAX_GRAD_NORM_CAPTION \
--max_grad_norm_spotting $MAX_GRAD_NORM_SPOTTING \
--max_grad_norm_joint $MAX_GRAD_NORM_JOINT \
--max_grad_norm_rl $MAX_GRAD_NORM_RL \
--ds_overlap_comm $DS_OVERLAP_COMM \
--ds_round_robin_gradients $DS_ROUND_ROBIN_GRADIENTS \
--smoke_steps $SMOKE_STEPS \
$DISCRIMINATIVE_FT_CAPTION_FLAG \
$DISCRIMINATIVE_FT_SPOTTING_FLAG \
$FREEZE_ENCODER_CLASSIFY_FLAG \
$FREEZE_ENCODER_CAPTION_FLAG \
$FREEZE_ENCODER_SPOTTING_FLAG \
$FREEZE_ENCODER_JOINT_FLAG \
$SMOKE_STEPS_CLASSIFY_FLAG \
$SMOKE_STEPS_CAPTION_FLAG \
$SMOKE_STEPS_SPOTTING_FLAG \
$SMOKE_STEPS_JOINT_FLAG \
$SMOKE_STEPS_RL_FLAG \
$CAPTION_DO_SAMPLE_FLAG \
$SPOTTING_USE_CENTER_REGRESSION_FLAG \
$JOINT_WARM_START_SPOTTING_HEAD_FLAG \
$CAPTION_GENERATION_CONFIG_JSON_FLAG \
$RL_EVAL_GENERATION_CONFIG_JSON_FLAG \
$JOINT_SPOTTING_CHECKPOINT_PATH_FLAG \
$RUN_DVC_FLAG \
--caption_valid_max_samples $CAPTION_VALID_MAX_SAMPLES \
--caption_test_max_samples $CAPTION_TEST_MAX_SAMPLES \
$SPOTTING_VALID_MAX_SAMPLES_FLAG \
$SPOTTING_TEST_MAX_SAMPLES_FLAG \
$SKIP_CAPTION_EVAL_FLAG \
$LOAD_BEST_METRIC_CHECKPOINT_FLAG \
--wandb_resume never \
$CONTINUE_TRAINING_FLAG \
--split_test test
