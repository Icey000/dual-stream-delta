"""
main.py - SoccerNet Dense Video Captioning Main Entry Point

Three-phase training pipeline:
  Phase 1: Classification (pre-train encoder)
  Phase 2: Captioning (freeze encoder + fine-tune Qwen LoRA)
  Phase 3: Spotting (freeze encoder + train spotting head)
  Final:   Dense Video Captioning (end-to-end inference)
"""

import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, BooleanOptionalAction
import os
import atexit
import sys
import threading
import time as _time
import traceback

# 设置 expandable_segments 减少显存碎片 (建议在 import torch 之前)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import wandb
import logging
import time
from datetime import datetime
from utils import valid_probability
import spotting
import captioning
import classifying
import decoding_eval
import joint_training
import rl_scst

# Accelerate (多卡分布式训练)
try:
    from accelerate import Accelerator
    _ACCELERATE_AVAILABLE = True
except ImportError:
    _ACCELERATE_AVAILABLE = False


def _configure_worker_logging(is_main_process):
    """Keep console and file logging on rank0 only."""
    if is_main_process:
        return

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        try:
            handler.flush()
            handler.close()
        finally:
            root_logger.removeHandler(handler)

    root_logger.addHandler(logging.NullHandler())
    root_logger.setLevel(logging.ERROR)


class _ErrorSummaryHandler(logging.Handler):
    def __init__(self, max_entries=2000):
        super().__init__(level=logging.ERROR)
        self._max_entries = max_entries
        self._entries = {}
        self._order = []
        self._total = 0
        self._dumping = False

    def emit(self, record):
        if self._dumping or record.levelno < logging.ERROR:
            return
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        exc_text = None
        if record.exc_info:
            exc_text = "".join(traceback.format_exception(*record.exc_info)).strip()

        key = (msg, exc_text)
        self._total += 1
        if key not in self._entries:
            if len(self._order) >= self._max_entries:
                key = ("<error summary truncated>", None)
            if key not in self._entries:
                self._entries[key] = {"count": 0, "first": record.created, "last": record.created}
                self._order.append(key)
        entry = self._entries[key]
        entry["count"] += 1
        entry["last"] = record.created

    def dump_summary(self):
        if self._total == 0:
            return
        self._dumping = True
        try:
            logger = logging.getLogger()
            logger.error("========== Error Summary (unique=%d, total=%d) ==========", len(self._order), self._total)
            for msg, exc_text in self._order:
                entry = self._entries.get((msg, exc_text), {})
                count = entry.get("count", 0)
                first = entry.get("first", None)
                last = entry.get("last", None)
                first_str = _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(first)) if first else "unknown"
                last_str = _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(last)) if last else "unknown"
                logger.error("[count=%d, first=%s, last=%s] %s", count, first_str, last_str, msg)
                if exc_text:
                    for line in exc_text.splitlines():
                        logger.error("  %s", line)
            logger.error("========== End Error Summary ==========")
        finally:
            self._dumping = False


def _install_error_summary(is_main_process):
    if not is_main_process:
        return None
    handler = _ErrorSummaryHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    ))
    logging.getLogger().addHandler(handler)
    atexit.register(handler.dump_summary)

    def _handle_unhandled(exc_type, exc, tb):
        logging.getLogger().error("Unhandled exception", exc_info=(exc_type, exc, tb))
        if _old_excepthook is not None:
            _old_excepthook(exc_type, exc, tb)

    _old_excepthook = sys.excepthook
    sys.excepthook = _handle_unhandled

    if hasattr(threading, "excepthook"):
        _old_thread_excepthook = threading.excepthook

        def _thread_excepthook(args):
            logging.getLogger().error(
                "Unhandled thread exception",
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
            if _old_thread_excepthook is not None:
                _old_thread_excepthook(args)

        threading.excepthook = _thread_excepthook

    return handler


if __name__ == '__main__':

    parser = ArgumentParser(
        description='SoccerNet-Caption: Dense Video Captioning',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # ---- 数据路径 ----
    parser.add_argument('--SoccerNet_path', required=False, type=str, default="path/to/soccernet", help='Path for SoccerNet')
    parser.add_argument('--features', required=False, type=str, default="baidu_soccer_embeddings.npy", help='Video features')

    # ---- 训练控制 ----
    parser.add_argument(
        '--max_epochs',
        required=False,
        type=int,
        default=15,
        help='Legacy fallback for the spotting phase. Prefer --spotting_epochs.',
    )
    parser.add_argument(
        '--spotting_epochs',
        required=False,
        type=int,
        default=None,
        help='Number of epochs for the spotting phase (preferred name).',
    )
    parser.add_argument('--load_weights', required=False, type=str, default=None, help='weights to load')
    parser.add_argument('--model_name', required=False, type=str, default="QFormer-Qwen", help='named of the model to save')
    parser.add_argument('--test_only', required=False, action='store_true', help='Perform testing only')
    parser.add_argument('--run_dvc', required=False, action='store_true',
                        help='Run dense video captioning (DVC) after training; default is disabled')
    parser.add_argument(
        "--stage",
        required=False,
        type=str,
        choices=["pipeline", "decode", "joint", "rl", "dvc"],
        default="pipeline",
        help="Top-level stage runner. 'pipeline' keeps the original classify/caption/spotting flow.",
    )
    parser.add_argument(
        "--start_stage",
        required=False,
        type=str,
        choices=["classifying", "caption", "spotting"],
        default="classifying",
        help="Start pipeline from a specific stage (default: classifying).",
    )

    # ---- 数据分割 ----
    parser.add_argument('--split_train', nargs='+', default=["train"], help='list of split for training')
    parser.add_argument('--split_valid', nargs='+', default=["valid"], help='list of split for validation')
    parser.add_argument('--split_test', nargs='+', default=["test", "challenge"], help='list of split for testing')
    parser.add_argument('--skip_caption_eval', required=False, action='store_true',
                        help='Skip post-caption evaluation (validate_captioning on test splits)')
    parser.add_argument('--load_best_metric_checkpoint', action=BooleanOptionalAction, default=False,
                        help='Load best-metric checkpoint for evaluation instead of the default best-loss checkpoint')
    parser.add_argument('--caption_valid_max_samples', required=False, type=int, default=0,
                        help='If >0, limit caption validation to first N samples per split (0 means full split)')
    parser.add_argument('--caption_test_max_samples', required=False, type=int, default=0,
                        help='If >0, limit caption test/eval to first N samples per split (0 means full split)')
    parser.add_argument('--spotting_valid_max_samples', required=False, type=int, default=0,
                        help='If >0, limit spotting validation to first N samples per split (0 means full split)')
    parser.add_argument('--spotting_test_max_samples', required=False, type=int, default=0,
                        help='If >0, limit spotting test/eval to first N samples per split (0 means full split)')
    parser.add_argument('--checkpoint_path', required=False, type=str, default=None,
                        help='Optional explicit checkpoint path (used by decode stage)')

    # ---- 特征参数 ----
    parser.add_argument('--version', required=False, type=int, default=2, help='Version of the dataset')
    parser.add_argument('--feature_dim', required=False, type=int, default=None, help='Number of input features')
    parser.add_argument('--evaluation_frequency', required=False, type=int, default=15, help='Run metric validation every N epochs')
    parser.add_argument('--evaluation_frequency_classify', required=False, type=int, default=None,
                        help='Run classifying metric validation every N epochs (fallback: --evaluation_frequency)')
    parser.add_argument('--evaluation_frequency_caption', required=False, type=int, default=None,
                        help='Run caption metric validation every N epochs (fallback: --evaluation_frequency)')
    parser.add_argument('--evaluation_frequency_spotting', required=False, type=int, default=None,
                        help='Run spotting metric validation every N epochs (fallback: --evaluation_frequency)')
    parser.add_argument('--evaluation_frequency_joint', required=False, type=int, default=None,
                        help='Run joint stage validation every N epochs (fallback: caption/global frequency)')
    parser.add_argument('--evaluation_frequency_rl', required=False, type=int, default=None,
                        help='Run RL stage validation every N epochs (fallback: caption/global frequency)')
    parser.add_argument('--framerate', required=False, type=int, default=1, help='Framerate of the input features')
    parser.add_argument('--pool', required=False, type=str, default="QFormer", help='Pooling mode: QFormer or TRANS')
    parser.add_argument('--vlad_k', required=False, type=int, default=64, help='Size of the vocabulary for NetVLAD')
    parser.add_argument('--NMS_window', required=False, type=int, default=30, help='NMS window in second')
    parser.add_argument('--NMS_threshold', required=False, type=float, default=0.0, help='NMS threshold for positive results')

    # ---- 窗口大小 ----
    parser.add_argument('--window_size_spotting', required=False, type=int, default=30, help='Spotting window size (seconds)')
    parser.add_argument('--window_size_caption', required=False, type=int, default=30, help='Caption window size (seconds)')

    # ---- 编码器 ----
    parser.add_argument('--freeze_encoder', required=False, action='store_true', help='Freeze encoder weights')
    parser.add_argument('--pretrain', required=False, action='store_true', help='Use pre-trained encoder')
    parser.add_argument('--weights_encoder', required=False, type=str, default=None)

    # ---- 批量大小和学习率 ----
    parser.add_argument('--batch_size', required=False, type=int, default=32, help='Batch size')
    parser.add_argument('--batch_size_classify', required=False, type=int, default=None, help='Batch size for classifying')
    parser.add_argument('--batch_size_caption', required=False, type=int, default=None, help='Batch size for captioning')
    parser.add_argument('--batch_size_spotting', required=False, type=int, default=None, help='Batch size for spotting')
    parser.add_argument('--LR', required=False, type=float, default=5e-5, help='Learning Rate (AdamW initial LR)')
    parser.add_argument('--LR_caption', required=False, type=float, default=1e-5, help='Learning Rate for caption phase (default: same as --LR)')
    parser.add_argument('--discriminative_ft_caption', action=BooleanOptionalAction, default=True,
                        help='Enable discriminative fine-tuning param groups for captioning')
    parser.add_argument('--discriminative_ft_spotting', action=BooleanOptionalAction, default=True,
                        help='Enable discriminative fine-tuning param groups for spotting')
    parser.add_argument('--lr_caption_lora', required=False, type=float, default=None,
                        help='Caption LR for LoRA params (fallback: LR_caption)')
    parser.add_argument('--lr_caption_proj', required=False, type=float, default=None,
                        help='Caption LR for proj/head params (fallback: LR_caption)')
    parser.add_argument('--lr_caption_qformer', required=False, type=float, default=2e-6,
                        help='Caption LR for encoder/Q-Former params')
    parser.add_argument('--lr_spotting_proj_head', required=False, type=float, default=None,
                        help='Spotting LR for projection/head params (fallback: LR)')
    parser.add_argument('--lr_spotting_qformer', required=False, type=float, default=2e-6,
                        help='Spotting LR for encoder/Q-Former params')
    parser.add_argument('--lr_joint', required=False, type=float, default=None,
                        help='Joint stage LR (fallback: LR_caption then LR)')
    parser.add_argument('--lr_rl', required=False, type=float, default=None,
                        help='RL stage LR (fallback: LR_caption then LR)')
    parser.add_argument('--spotting_loss', required=False, type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='Loss for spotting phase: cross-entropy or focal loss')
    parser.add_argument('--spotting_target_mode', required=False, type=str, default='hard_multiclass',
                        choices=['hard_multiclass', 'soft_window_multiclass'],
                        help='Target construction for spotting training')
    parser.add_argument('--spotting_soft_window_radius', required=False, type=int, default=2,
                        help='Soft spotting label radius in clips (used by soft_window_multiclass)')
    parser.add_argument('--spotting_soft_window_sigma', required=False, type=float, default=1.0,
                        help='Gaussian sigma for soft spotting targets (used by soft_window_multiclass)')
    parser.add_argument('--spotting_use_center_regression', action=BooleanOptionalAction, default=False,
                        help='Enable center-offset regression head for spotting')
    parser.add_argument('--spotting_center_regression_weight', required=False, type=float, default=1.0,
                        help='Loss weight for spotting center-offset regression')
    parser.add_argument('--spotting_center_positive_threshold', required=False, type=float, default=0.5,
                        help='Event probability threshold for selecting positive soft-window clips in center regression')
    parser.add_argument('--focal_alpha', required=False, type=float, default=0.75,
                        help='Positive-class alpha for spotting focal loss')
    parser.add_argument('--focal_gamma', required=False, type=float, default=2.0,
                        help='Gamma for spotting focal loss')
    parser.add_argument('--freeze_encoder_classify', required=False, action='store_true',
                        help='Freeze encoder during classifying phase')
    parser.add_argument('--freeze_encoder_caption', required=False, action='store_true',
                        help='Freeze encoder during captioning phase')
    parser.add_argument('--freeze_encoder_spotting', required=False, action='store_true',
                        help='Freeze encoder during spotting phase')
    parser.add_argument('--freeze_encoder_joint', action=BooleanOptionalAction, default=True,
                        help='Freeze encoder/Q-Former during joint stage')
    parser.add_argument('--max_grad_norm', required=False, type=float, default=0.5,
                        help='Global grad clip max norm (all trainable params)')
    parser.add_argument('--max_grad_norm_classify', required=False, type=float, default=None,
                        help='Grad clip max norm override for classifying phase')
    parser.add_argument('--max_grad_norm_caption', required=False, type=float, default=None,
                        help='Grad clip max norm override for captioning phase')
    parser.add_argument('--max_grad_norm_spotting', required=False, type=float, default=None,
                        help='Grad clip max norm override for spotting phase')
    parser.add_argument('--max_grad_norm_joint', required=False, type=float, default=None,
                        help='Grad clip max norm override for joint phase')
    parser.add_argument('--max_grad_norm_rl', required=False, type=float, default=None,
                        help='Grad clip max norm override for RL phase')
    parser.add_argument('--smoke_steps', required=False, type=int, default=0,
                        help='If >0, run only first N batches per epoch for quick smoke test')
    parser.add_argument('--smoke_steps_classify', required=False, type=int, default=None,
                        help='Smoke steps override for classifying phase')
    parser.add_argument('--smoke_steps_caption', required=False, type=int, default=None,
                        help='Smoke steps override for captioning phase')
    parser.add_argument('--smoke_steps_spotting', required=False, type=int, default=None,
                        help='Smoke steps override for spotting phase')
    parser.add_argument('--smoke_steps_joint', required=False, type=int, default=None,
                        help='Smoke steps override for joint phase')
    parser.add_argument('--smoke_steps_rl', required=False, type=int, default=None,
                        help='Smoke steps override for RL phase')

    # =========================================================================
    # ==== 过拟合 / 正则化超参数 ====
    # =========================================================================
    parser.add_argument('--weight_decay', required=False, type=float, default=0.05,
                        help='AdamW weight decay (L2正则)')

    # --- Dropout ---
    parser.add_argument('--encoder_dropout', required=False, type=float, default=0.3,
                        help='编码器 Q-Former 内部的dropout概率')

    # --- 每个训练阶段的轮次 ---
    parser.add_argument('--epochs_classify', required=False, type=int, default=10,
                        help='Number of epochs for classifying phase')
    parser.add_argument('--epochs_caption', required=False, type=int, default=15,
                        help='Number of epochs for captioning phase')
    parser.add_argument('--epochs_joint', required=False, type=int, default=5,
                        help='Number of epochs for joint phase')
    parser.add_argument('--epochs_rl', required=False, type=int, default=3,
                        help='Number of epochs for RL phase')

    # --- 学习率调度器 ---
    parser.add_argument('--lr_tmax_classify', required=False, type=int, default=None,
                        help='CosineAnnealing T_max for classifying阶段')
    parser.add_argument('--lr_tmax_caption', required=False, type=int, default=None,
                        help='CosineAnnealing T_max for captioning阶段')
    parser.add_argument('--lr_tmax_spotting', required=False, type=int, default=None,
                        help='CosineAnnealing T_max for spotting阶段')

    # ---- 系统参数 ----
    parser.add_argument('--GPU', required=False, type=int, default=-1, help='ID of the GPU to use')
    parser.add_argument('--max_num_worker', required=False, type=int, default=4, help='number of worker to load data')
    parser.add_argument('--seed', required=False, type=int, default=0, help='seed for reproducibility')
    parser.add_argument('--loglevel', required=False, type=str, default='INFO', help='logging level')
    parser.add_argument('--top_k', required=False, type=int, default=1, help='Top k for generation')
    parser.add_argument('--caption_max_new_tokens', required=False, type=int, default=48,
                        help='Caption decoding max_new_tokens')
    parser.add_argument('--caption_no_repeat_ngram_size', required=False, type=int, default=3,
                        help='Caption decoding no_repeat_ngram_size (0 disables)')
    parser.add_argument('--caption_num_beams', required=False, type=int, default=1,
                        help='Caption decoding num_beams')
    parser.add_argument('--caption_length_penalty', required=False, type=float, default=0.9,
                        help='Caption decoding length_penalty')
    parser.add_argument('--caption_do_sample', action=BooleanOptionalAction, default=False,
                        help='Enable sampling-based caption decoding')
    parser.add_argument('--caption_temperature', required=False, type=float, default=1.0,
                        help='Caption decoding temperature (used when sampling)')
    parser.add_argument('--caption_top_p', required=False, type=float, default=1.0,
                        help='Caption decoding top_p (used when sampling)')
    parser.add_argument('--caption_repetition_penalty', required=False, type=float, default=1.15,
                        help='Caption decoding repetition_penalty')
    parser.add_argument('--caption_generation_config_json', required=False, type=str, default=None,
                        help='Optional JSON file overriding caption generation config for decode/joint/rl/dvc')
    parser.add_argument('--rl_eval_generation_config_json', required=False, type=str, default=None,
                        help='Optional JSON file overriding RL validation generation config')
    parser.add_argument('--sweep_max_new_tokens', nargs='+', type=int, default=[20, 30, 40],
                        help='Decode-stage sweep values for max_new_tokens')
    parser.add_argument('--sweep_no_repeat_ngram_size', nargs='+', type=int, default=[2, 3, 4],
                        help='Decode-stage sweep values for no_repeat_ngram_size')
    parser.add_argument('--sweep_num_beams', nargs='+', type=int, default=[1, 3, 5],
                        help='Decode-stage sweep values for num_beams')
    parser.add_argument('--sweep_temperature', nargs='+', type=float, default=[0.6, 0.7, 0.8],
                        help='Decode-stage sweep values for temperature')
    parser.add_argument('--decode_num_examples', required=False, type=int, default=5,
                        help='Number of generated examples to store per decode config')
    parser.add_argument('--joint_lambda_caption', required=False, type=float, default=1.0,
                        help='Caption CE weight inside joint loss')
    parser.add_argument('--joint_warm_start_spotting_head', action=BooleanOptionalAction, default=True,
                        help='Warm-start the joint spotting head from the best spotting checkpoint when available')
    parser.add_argument('--joint_spotting_checkpoint_path', required=False, type=str, default=None,
                        help='Optional explicit spotting checkpoint path for warming the joint spotting head')
    parser.add_argument('--accumulation_steps_joint', type=int, default=None, help='Accumulation for joint stage')
    parser.add_argument('--accumulation_steps_rl', type=int, default=None, help='Accumulation for RL stage')
    parser.add_argument('--rl_reward', required=False, type=str, default='cider', choices=['cider'],
                        help='Sequence-level reward for RL stage')
    parser.add_argument('--rl_weight', required=False, type=float, default=1.0,
                        help='Weight applied to SCST loss')
    parser.add_argument('--rl_init_stage', required=False, type=str, default='caption', choices=['caption', 'joint'],
                        help='Checkpoint source used to initialize RL stage')
    parser.add_argument('--rl_sample_temperature', required=False, type=float, default=0.7,
                        help='Sampling temperature for SCST sampled captions')
    parser.add_argument('--rl_sample_top_p', required=False, type=float, default=0.9,
                        help='Sampling top-p for SCST sampled captions')
    parser.add_argument('--rl_sample_max_new_tokens', required=False, type=int, default=30,
                        help='Sampling max_new_tokens for SCST sampled captions')
    parser.add_argument("--continue_training", required=False, action='store_true', help='Continue training from the last checkpoint')
    parser.add_argument("--wandb_run_id", required=False, type=str, default=None,
                        help="Explicit W&B run ID to resume/log into. Overrides models/<model_name>/wandb_id.txt")
    parser.add_argument("--wandb_resume", required=False, type=str, default="must",
                        choices=["allow", "must", "never"],
                        help="W&B resume policy when run_id is provided")

    # ---- 分布式训练 (Accelerate + DeepSpeed ZeRO-2) ----
    parser.add_argument("--use_distributed", action="store_true",
                        help="启用多卡分布式训练 (Accelerate + DeepSpeed ZeRO-2). "
                             "需先运行 'accelerate config' 配置，然后用 "
                             "'accelerate launch main.py ...' 启动")
    parser.add_argument("--ds_overlap_comm", required=False, type=str, default=None,
                        help="Override DeepSpeed zero_optimization.overlap_comm (true/false)")
    parser.add_argument("--ds_round_robin_gradients", required=False, type=str, default=None,
                        help="Override DeepSpeed zero_optimization.round_robin_gradients (true/false)")

    # ---- 双流模式参数 (Dual-Stream) ----
    parser.add_argument("--use_dual_stream", action="store_true", help="Enable dual-stream (video+audio) mode")
    parser.add_argument("--audio_root", type=str, default=None, help="Root path for audio features (CLAP .npy)")
    parser.add_argument("--video_input_dim", type=int, default=8576, help="Dimension of video features")
    parser.add_argument("--audio_input_dim", type=int, default=512, help="Dimension of audio features")

    # ---- LoRA / LLM 参数 ----
    parser.add_argument("--llm_model_path", type=str, default="Qwen/Qwen2.5-7B",
                        help="HuggingFace model path for Qwen LLM backbone")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")

    # ---- 梯度累积 (Gradient Accumulation) ----
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Fallback accumulation steps")
    parser.add_argument("--accumulation_steps_classify", type=int, default=None, help="Accumulation for classifying")
    parser.add_argument("--accumulation_steps_caption", type=int, default=None, help="Accumulation for captioning")
    parser.add_argument("--accumulation_steps_spotting", type=int, default=None, help="Accumulation for spotting")
    parser.add_argument("--lora_alpha", type=int, default=8, help="LoRA alpha (建议 V100 FP16 下设为 r 的 1~2 倍)")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # ---- 编码器隐藏层维度 ----
    parser.add_argument("--hidden_dim", type=int, default=3584, help="Hidden dimension for the encoder (should match LLM's hidden_size)")

    args = parser.parse_args()

    def _parse_optional_bool(v):
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid boolean value: {v}. Use true/false.")

    args.ds_overlap_comm = _parse_optional_bool(args.ds_overlap_comm)
    args.ds_round_robin_gradients = _parse_optional_bool(args.ds_round_robin_gradients)

    if args.spotting_epochs is None:
        args.spotting_epochs = args.max_epochs

    # for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

    os.makedirs(os.path.join("models", args.model_name), exist_ok=True)
    log_path = os.path.join("models", args.model_name,
                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))
                            
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    # 单卡模式下才设置 CUDA_VISIBLE_DEVICES (分布式模式下由 accelerate 管理 GPU)
    if args.GPU >= 0 and not getattr(args, 'use_distributed', False):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    # ---- 分布式训练初始化 ----
    # args.accelerator 会被各子模块 (classifying/captioning/spotting) 读取
    args.accelerator = None
    if getattr(args, 'use_distributed', False):
        if not _ACCELERATE_AVAILABLE:
            logging.warning("--use_distributed is set but accelerate is not installed; falling back to single-GPU training.")
            logging.warning("安装方式: pip install accelerate deepspeed")
        else:
            args.accelerator = Accelerator(log_with="wandb", gradient_accumulation_steps=args.accumulation_steps)
            ds_plugin = getattr(args.accelerator.state, "deepspeed_plugin", None)
            ds_config = getattr(ds_plugin, "deepspeed_config", None) if ds_plugin is not None else None
            zero_cfg = ds_config.get("zero_optimization") if isinstance(ds_config, dict) else None
            if isinstance(zero_cfg, dict):
                if args.ds_overlap_comm is not None:
                    zero_cfg["overlap_comm"] = bool(args.ds_overlap_comm)
                if args.ds_round_robin_gradients is not None:
                    zero_cfg["round_robin_gradients"] = bool(args.ds_round_robin_gradients)

    is_main = (args.accelerator is None) or args.accelerator.is_main_process
    _configure_worker_logging(is_main)
    _install_error_summary(is_main)

    if args.accelerator is not None and is_main:
        logging.info(f"Accelerate 初始化完成. 设备: {args.accelerator.device}, "
                     f"总进程数: {args.accelerator.num_processes}, "
                     f"主进程: {args.accelerator.is_main_process}")

    def _resolve_wandb_run_id(wandb_id_file):
        # 仅在 continue_training 时才尝试续 W&B run。
        if not getattr(args, "continue_training", False):
            return None, None
        if getattr(args, "wandb_run_id", None):
            return str(args.wandb_run_id).strip(), "--wandb_run_id"
        if os.path.exists(wandb_id_file):
            with open(wandb_id_file, 'r') as f:
                saved = f.read().strip()
            if saved:
                return saved, wandb_id_file
        return None, None

    if args.accelerator is not None:
        init_kwargs = {}
        wandb_id_file = os.path.join("models", args.model_name, "wandb_id.txt")
        run_id, run_id_src = _resolve_wandb_run_id(wandb_id_file)

        if run_id:
            if args.wandb_resume == "never":
                logging.info(
                    f"W&B run ID provided from {run_id_src}: {run_id}, but wandb_resume=never, so starting a new run."
                )
            else:
                logging.info(
                    f"Using W&B run ID from {run_id_src}: {run_id} (resume={args.wandb_resume})"
                )
                # 传给 wandb.init 的 kwargs 会塞在这里
                init_kwargs["wandb"] = {"id": run_id, "resume": args.wandb_resume}

        if getattr(args, "continue_training", False) and args.wandb_resume != "never" and not run_id:
            raise ValueError(
                "continue_training=True but no W&B run id found. "
                "Provide --wandb_run_id or ensure models/<model_name>/wandb_id.txt exists."
            )

        # 一键启动 Tracker！这行代码会自动把 args 转为字典并等效于 wandb.config.update()
        try:
            args.accelerator.init_trackers(
                project_name="DVC-SoccerNet",
                config=vars(args),
                init_kwargs=init_kwargs
            )
        except Exception as e:
            if run_id and not (getattr(args, "continue_training", False) and args.wandb_resume == "must"):
                logging.error(
                    "W&B init failed with run_id=%s resume=%s: %s. Falling back to a NEW W&B run.",
                    run_id, args.wandb_resume, e
                )
                args.accelerator.init_trackers(
                    project_name="DVC-SoccerNet",
                    config=vars(args),
                    init_kwargs={}
                )
            else:
                raise

        # 只有主进程负责把新的 Run ID 存入到文件里
        if args.accelerator.is_main_process:
            import wandb
            if wandb.run is not None:
                with open(wandb_id_file, 'w') as f:
                    f.write(wandb.run.id)
                    
    else:
        if is_main:
            wandb_id_file = os.path.join("models", args.model_name, "wandb_id.txt")
            run_id, run_id_src = _resolve_wandb_run_id(wandb_id_file)

            if getattr(args, "continue_training", False) and args.wandb_resume != "never" and not run_id:
                raise ValueError(
                    "continue_training=True but no W&B run id found. "
                    "Provide --wandb_run_id or ensure models/<model_name>/wandb_id.txt exists."
                )

            try:
                if run_id and args.wandb_resume != "never":
                    logging.info(
                        f"Using W&B run ID from {run_id_src}: {run_id} (resume={args.wandb_resume})"
                    )
                    run = wandb.init(project="DVC-SoccerNet", id=run_id, resume=args.wandb_resume)
                else:
                    if run_id and args.wandb_resume == "never":
                        logging.info(
                            f"W&B run ID provided from {run_id_src}: {run_id}, but wandb_resume=never, so starting a new run."
                        )
                    run = wandb.init(project="DVC-SoccerNet")
            except Exception as e:
                if run_id and not (getattr(args, "continue_training", False) and args.wandb_resume == "must"):
                    logging.error(
                        "W&B init failed with run_id=%s resume=%s: %s. Falling back to a NEW W&B run.",
                        run_id, args.wandb_resume, e
                    )
                    run = wandb.init(project="DVC-SoccerNet")
                else:
                    raise

            # Save run ID so it can be resumed later
            with open(wandb_id_file, 'w') as f:
                f.write(run.id)

            wandb.config.update(args, allow_val_change=True)

    start = time.time()

    if args.stage == "pipeline":
        stage_order = {"classifying": 0, "caption": 1, "spotting": 2}
        start_rank = stage_order[getattr(args, "start_stage", "classifying")]

        if start_rank <= stage_order["classifying"]:
            args.freeze_encoder = bool(getattr(args, "freeze_encoder_classify", False))
            logging.info('Starting classifying function')
            classifying.main(args)
            logging.info(f'Total Execution Time is {time.time()-start} seconds')

            if getattr(args, "accelerator", None) is not None:
                args.accelerator.free_memory()

            logging.info(
                "Phase 1 (Classify) complete. "
                f"Caption freeze_encoder={bool(getattr(args, 'freeze_encoder_caption', False))}, "
                f"Spotting freeze_encoder={bool(getattr(args, 'freeze_encoder_spotting', False))}."
            )
        else:
            logging.info(f"Skipping classifying stage due to --start_stage={args.start_stage}")

        if start_rank <= stage_order["caption"]:
            args.weights_encoder = f"models/{args.model_name}/classifying/model.pth.tar" if args.pretrain else None
            args.freeze_encoder = bool(getattr(args, "freeze_encoder_caption", False))
            logging.info('Starting caption function')
            captioning.main(args)
            logging.info(f'Total Execution Time is {time.time()-start} seconds')

            if getattr(args, "accelerator", None) is not None:
                args.accelerator.free_memory()
        else:
            logging.info(f"Skipping caption stage due to --start_stage={args.start_stage}")

        if start_rank <= stage_order["spotting"]:
            args.weights_encoder = f"models/{args.model_name}/classifying/model.pth.tar" if args.pretrain else None
            args.freeze_encoder = bool(getattr(args, "freeze_encoder_spotting", False))
            logging.info('Starting spotting function')
            spotting.main(args)
            logging.info(f'Total Execution Time is {time.time()-start} seconds')
        else:
            logging.info(f"Skipping spotting stage due to --start_stage={args.start_stage}")
    elif args.stage == "decode":
        decoding_eval.run(args)
        logging.info(f'Total Execution Time is {time.time()-start} seconds')
    elif args.stage == "joint":
        joint_training.main(args)
        logging.info(f'Total Execution Time is {time.time()-start} seconds')
    elif args.stage == "rl":
        rl_scst.main(args)
        logging.info(f'Total Execution Time is {time.time()-start} seconds')
    elif args.stage == "dvc":
        captioning.dvc(args)
        logging.info(f'Total Execution Time is {time.time()-start} seconds')

    if getattr(args, "accelerator", None) is not None:
        args.accelerator.free_memory()

    args.weights_encoder = None
    if args.stage == "pipeline" and args.run_dvc:
        captioning.dvc(args)
        logging.info(f'Total Execution Time is {time.time()-start} seconds')
    elif args.stage == "pipeline":
        logging.info("Skipping DVC stage (default). Use --run_dvc to enable post-training dense captioning.")

    if args.accelerator is not None:
        args.accelerator.end_training()
