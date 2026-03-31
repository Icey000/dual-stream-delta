"""
captioning.py - Dense Video Captioning training & evaluation

Supports:
  - Single-stream: model_qwen.Video2CaptionQwen (QFormer/TRANS + Qwen LoRA)
  - Dual-stream:   dual_qformer.DualVideo2CaptionLLM (Dual Q-Former + Qwen LoRA)
"""

import os
import logging
import re
import json
from collections import Counter
from datetime import datetime
import time
import numpy as np

import torch

from dataset import SoccerNetCaptions, PredictionCaptions
from train import trainer, test_captioning, validate_captioning, _resolve_best_checkpoint_path

import wandb
import copy


def _log_optimizer_groups(phase, named_groups):
    for group_name, params, lr in named_groups:
        num_params = sum(p.numel() for p in params)
        logging.info(
            f"[{phase}] optimizer_group={group_name}, tensors={len(params)}, params={num_params}, lr={lr}"
        )


def _build_caption_param_groups(model, args, fallback_lr):
    lr_lora = getattr(args, "lr_caption_lora", None) or fallback_lr
    lr_proj = getattr(args, "lr_caption_proj", None) or fallback_lr
    lr_qformer = getattr(args, "lr_caption_qformer", 2e-6)

    lora_params = []
    proj_head_params = []
    qformer_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "llm" in name and "lora_" in name:
            lora_params.append(param)
        elif name.startswith("encoder.") or ".encoder." in name:
            qformer_params.append(param)
        elif name.startswith("proj") or name.startswith("proj_norm") or ".proj." in name or ".proj_norm." in name:
            proj_head_params.append(param)
        else:
            proj_head_params.append(param)

    named_groups = [
        ("llm_lora", lora_params, lr_lora),
        ("proj_heads", proj_head_params, lr_proj),
        ("qformer_encoder", qformer_params, lr_qformer),
    ]
    _log_optimizer_groups("caption", named_groups)

    param_groups = []
    for group_name, params, lr in named_groups:
        if params:
            param_groups.append({"params": params, "lr": float(lr), "group_name": group_name})
    return param_groups


def get_caption_generation_config_from_args(args):
    config = {
        "max_new_tokens": int(getattr(args, "caption_max_new_tokens", 48)),
        "no_repeat_ngram_size": int(getattr(args, "caption_no_repeat_ngram_size", 3)),
        "num_beams": int(getattr(args, "caption_num_beams", 1)),
        "length_penalty": float(getattr(args, "caption_length_penalty", 0.9)),
        "do_sample": bool(getattr(args, "caption_do_sample", False)),
        "temperature": float(getattr(args, "caption_temperature", 1.0)),
        "top_p": float(getattr(args, "caption_top_p", 1.0)),
        "repetition_penalty": float(getattr(args, "caption_repetition_penalty", 1.15)),
        "top_k": int(getattr(args, "top_k", 1)),
    }
    config_json_path = getattr(args, "caption_generation_config_json", None)
    if config_json_path:
        with open(config_json_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"caption_generation_config_json must contain a JSON object: {config_json_path}")
        config.update({k: v for k, v in loaded.items() if v is not None})
    return config


def get_rl_eval_generation_config_from_args(args):
    config = get_caption_generation_config_from_args(args)
    config_json_path = getattr(args, "rl_eval_generation_config_json", None)
    if config_json_path:
        with open(config_json_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"rl_eval_generation_config_json must contain a JSON object: {config_json_path}")
        config.update({k: v for k, v in loaded.items() if v is not None})
    return config

def _add_module_prefix(state_dict):
    return {
        (k if k.startswith("module.") else f"module.{k}"): v
        for k, v in state_dict.items()
    }


_SIZE_MISMATCH_RE = re.compile(
    r"size mismatch for (?P<key>[^:]+): "
    r"copying a param with shape (?P<src>torch\.Size\([^)]+\)) from checkpoint, "
    r"the shape in current model is (?P<dst>torch\.Size\([^)]+\))\."
)


def _group_param_key(param_key):
    if ".layers." in param_key:
        head = param_key.split(".layers.", 1)[0]
        return f"{head}.layers.*"
    parts = param_key.split(".")
    if len(parts) <= 4:
        return param_key
    return ".".join(parts[:4]) + ".*"


def _log_incompatible_keys_summary(incompatible, source_name, max_show=6):
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if not missing and not unexpected:
        return
    if missing:
        shown = ", ".join(missing[:max_show])
        suffix = f" ... (+{len(missing) - max_show})" if len(missing) > max_show else ""
        logging.warning(
            "[%s] missing_keys=%d. sample: %s%s",
            source_name, len(missing), shown, suffix,
        )
    if unexpected:
        shown = ", ".join(unexpected[:max_show])
        suffix = f" ... (+{len(unexpected) - max_show})" if len(unexpected) > max_show else ""
        logging.warning(
            "[%s] unexpected_keys=%d. sample: %s%s",
            source_name, len(unexpected), shown, suffix,
        )


def _load_state_dict_with_compact_mismatch(
    model,
    state_dict,
    source_name,
    max_sample_lines=8,
    max_groups=8,
):
    try:
        incompatible = model.load_state_dict(state_dict, strict=False)
        _log_incompatible_keys_summary(incompatible, source_name)
        return
    except RuntimeError as exc:
        message = str(exc)
        if "size mismatch for " not in message:
            raise

        mismatch_lines = []
        grouped = Counter()
        zero_shape_count = 0

        for line in message.splitlines():
            if "size mismatch for " not in line:
                continue
            clean = line.strip()
            mismatch_lines.append(clean)
            matched = _SIZE_MISMATCH_RE.search(clean)
            if matched:
                key = matched.group("key")
                src = matched.group("src")
                dst = matched.group("dst")
                grouped[_group_param_key(key)] += 1
                if dst == "torch.Size([0])":
                    zero_shape_count += 1
            else:
                grouped["<unparsed>"] += 1

        if not mismatch_lines:
            raise

        summary = [
            f"[{source_name}] load_state_dict failed: {len(mismatch_lines)} size mismatches.",
        ]
        if zero_shape_count > 0:
            summary.append(
                "Detected model params with torch.Size([0]); this usually means sharded/empty init "
                "(e.g. ZeRO-3/meta-init path) before real parameter materialization."
            )

        summary.append("Top mismatch groups:")
        for group, count in grouped.most_common(max_groups):
            summary.append(f"  - {group}: {count}")

        summary.append("Mismatch samples:")
        for sample in mismatch_lines[:max_sample_lines]:
            summary.append(f"  - {sample}")
        omitted = len(mismatch_lines) - max_sample_lines
        if omitted > 0:
            summary.append(f"  - ... omitted {omitted} additional mismatch lines")

        raise RuntimeError("\n".join(summary)) from exc


def _load_optional_checkpoint(model, checkpoint_path, source_name):
    """Load a checkpoint if it exists; otherwise continue from scratch.

    This keeps smoke tests and brand-new model names from failing just because
    the phase-specific checkpoint has not been created yet.
    """
    if not os.path.exists(checkpoint_path):
        logging.warning("[%s][warm-start] checkpoint not found, skipping load: %s", source_name, checkpoint_path)
        return None

    logging.info("[%s][warm-start] loading checkpoint: %s", source_name, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    normalized_state_dict = {}
    prefix_hits = {"caption_model.": 0, "module.": 0}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("caption_model."):
            new_key = new_key[len("caption_model."):]
            prefix_hits["caption_model."] += 1
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
            prefix_hits["module."] += 1
        normalized_state_dict[new_key] = value

    if prefix_hits["caption_model."] or prefix_hits["module."]:
        logging.info(
            "[%s][warm-start] normalized checkpoint keys: caption_model=%d module=%d",
            source_name,
            prefix_hits["caption_model."],
            prefix_hits["module."],
        )

    _load_state_dict_with_compact_mismatch(
        model,
        normalized_state_dict,
        source_name=source_name,
    )
    ckpt_epoch = checkpoint.get("epoch", "unknown")
    logging.info("[%s][warm-start] loaded checkpoint epoch=%s", source_name, ckpt_epoch)
    del checkpoint
    return ckpt_epoch


def main(args):

    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(arg.rjust(15) + " : " + str(getattr(args, arg)))
    generation_config = get_caption_generation_config_from_args(args)
    evaluation_frequency = getattr(args, "evaluation_frequency_caption", None) or args.evaluation_frequency

    # ---- 判断是否启用双流模式 ----
    use_dual = getattr(args, "use_dual_stream", False)

    if use_dual:
        # ==================== 双流模式 ====================
        from dataset_dual import SoccerNetCaptionsDual, CollateGPTDual, collate_fn_padd_dual
        from dual_qformer import DualVideo2CaptionLLM

        audio_root = args.audio_root
        assert audio_root is not None, "Dual-stream mode requires --audio_root"

        if not args.test_only:
            dataset_Train = SoccerNetCaptionsDual(
                vision_root=args.SoccerNet_path, audio_root=audio_root,
                features=args.features, split=args.split_train,
                version=args.version, framerate=args.framerate,
                window_size=args.window_size_caption,
                llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B")
            )
            dataset_Valid = SoccerNetCaptionsDual(
                vision_root=args.SoccerNet_path, audio_root=audio_root,
                features=args.features, split=args.split_valid,
                version=args.version, framerate=args.framerate,
                window_size=args.window_size_caption,
                llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B")
            )
            dataset_Valid_metric = dataset_Valid

        if args.test_only:
            dataset_Test = SoccerNetCaptionsDual(
                vision_root=args.SoccerNet_path, audio_root=audio_root,
                features=args.features, split=args.split_test,
                version=args.version, framerate=args.framerate,
                window_size=args.window_size_caption,
                llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B")
            )

        if args.feature_dim is None:
            if not args.test_only and len(dataset_Train) > 0:
                args.feature_dim = dataset_Train[0][0].shape[-1]
            elif args.test_only and len(dataset_Test) > 0:
                args.feature_dim = dataset_Test[0][0].shape[-1]
            print("feature_dim found:", args.feature_dim)

        vocab_size = dataset_Test.vocab_size if args.test_only else dataset_Train.vocab_size
        model = DualVideo2CaptionLLM(
            vocab_size=vocab_size,
            video_input_dim=getattr(args, "video_input_dim", 1024),
            audio_input_dim=getattr(args, "audio_input_dim", 512),
            llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B"),
            lora_r=getattr(args, "lora_r", 8),
            lora_alpha=getattr(args, "lora_alpha", 16),
            lora_dropout=getattr(args, "lora_dropout", 0.05),
            weights=args.load_weights,
            weights_encoder=args.weights_encoder,
            freeze_encoder=args.freeze_encoder,
            top_k=args.top_k,
            max_new_tokens=generation_config["max_new_tokens"],
            no_repeat_ngram_size=generation_config["no_repeat_ngram_size"],
            num_beams=generation_config["num_beams"],
            length_penalty=generation_config["length_penalty"],
            do_sample=generation_config["do_sample"],
            temperature=generation_config["temperature"],
            top_p=generation_config["top_p"],
            repetition_penalty=generation_config["repetition_penalty"],
            encoder_dropout=getattr(args, "encoder_dropout", 0.1),
        )

        collate_fn = CollateGPTDual(llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B"))

    else:
        # ==================== 单流模式 (Qwen) ====================
        if not args.test_only:
            dataset_Train = SoccerNetCaptions(path=args.SoccerNet_path, features=args.features, split=args.split_train, version=args.version, framerate=args.framerate, window_size=args.window_size_caption, llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B"))
            dataset_Valid = SoccerNetCaptions(path=args.SoccerNet_path, features=args.features, split=args.split_valid, version=args.version, framerate=args.framerate, window_size=args.window_size_caption, llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B"))
            dataset_Valid_metric = dataset_Valid
        if args.test_only:
            dataset_Test = SoccerNetCaptions(path=args.SoccerNet_path, features=args.features, split=args.split_test, version=args.version, framerate=args.framerate, window_size=args.window_size_caption, llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B"))

        if args.feature_dim is None:
            if not args.test_only:
                args.feature_dim = dataset_Train[0][0].shape[-1]
            else:
                args.feature_dim = dataset_Test[0][0].shape[-1]
            print("feature_dim found:", args.feature_dim)

        from model_qwen import Video2CaptionQwen
        model = Video2CaptionQwen(
            input_size=args.feature_dim,
            vlad_k=args.vlad_k,
            window_size=args.window_size_caption,
            framerate=args.framerate,
            pool=args.pool,
            llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B"),
            lora_r=getattr(args, "lora_r", 8),
            lora_alpha=getattr(args, "lora_alpha", 16),
            lora_dropout=getattr(args, "lora_dropout", 0.05),
            weights_encoder=args.weights_encoder,
            freeze_encoder=args.freeze_encoder,
            top_k=args.top_k,
            max_new_tokens=generation_config["max_new_tokens"],
            no_repeat_ngram_size=generation_config["no_repeat_ngram_size"],
            num_beams=generation_config["num_beams"],
            length_penalty=generation_config["length_penalty"],
            do_sample=generation_config["do_sample"],
            temperature=generation_config["temperature"],
            top_p=generation_config["top_p"],
            repetition_penalty=generation_config["repetition_penalty"],
        )

        from dataset import CollateGPT
        collate_fn = CollateGPT(llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B"))



    logging.info(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total number of parameters: " + str(total_params))

    bs = getattr(args, 'batch_size_caption', None) or args.batch_size
    acc_steps = getattr(args, 'accumulation_steps_caption', None) or getattr(args, 'accumulation_steps', 1)
    smoke_steps = getattr(args, 'smoke_steps_caption', None)
    if smoke_steps is None:
        smoke_steps = getattr(args, 'smoke_steps', 0)
    caption_valid_max_samples = int(getattr(args, "caption_valid_max_samples", 0) or 0)
    if caption_valid_max_samples < 0:
        logging.warning(
            "[caption] caption_valid_max_samples=%d is invalid. Resetting to 0 (full split).",
            caption_valid_max_samples,
        )
        caption_valid_max_samples = 0
    logging.info(
        "[caption] generation_config=%s, evaluation_frequency=%s",
        generation_config,
        evaluation_frequency,
    )

    # create dataloader
    if not args.test_only:
        train_loader = torch.utils.data.DataLoader(dataset_Train,
            batch_size=bs, shuffle=True,
            num_workers=args.max_num_worker, pin_memory=True, collate_fn=collate_fn)

        val_loader = torch.utils.data.DataLoader(dataset_Valid,
            batch_size=bs, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True, collate_fn=collate_fn)

        if caption_valid_max_samples > 0 and len(dataset_Valid) > caption_valid_max_samples:
            logging.info(
                "[caption] validation split limiting evaluation samples: %d -> %d",
                len(dataset_Valid),
                caption_valid_max_samples,
            )
            dataset_Valid = torch.utils.data.Subset(dataset_Valid, range(caption_valid_max_samples))
            val_loader = torch.utils.data.DataLoader(dataset_Valid,
                batch_size=bs, shuffle=False,
                num_workers=args.max_num_worker, pin_memory=True, collate_fn=collate_fn)
            dataset_Valid_metric = dataset_Valid

        val_metric_loader = torch.utils.data.DataLoader(dataset_Valid_metric,
            batch_size=bs, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True, collate_fn=collate_fn)

    # training parameters
    if not args.test_only:
        criterion = torch.nn.CrossEntropyLoss()

        # 学习率: 优先使用 --LR_caption，否则回退到 --LR
        # 多卡训练时有效 batch size = batch_size * num_gpus * acc_steps
        # 通常需要降低学习率 (线性 scaling 的反向)
        caption_lr = getattr(args, 'LR_caption', None) or args.LR
        use_discriminative_ft = bool(getattr(args, "discriminative_ft_caption", True))
        logging.info(
            f"Caption phase LR base={caption_lr} (args.LR={args.LR}), "
            f"discriminative_ft_caption={use_discriminative_ft}"
        )

        accelerator = getattr(args, 'accelerator', None)
        # 兼容 DeepSpeed JSON config: 如果在 config 中已有 optimizer 定义，代码里创建 DummyOptim
        use_dummy_optimizer = False
        if accelerator is not None and getattr(accelerator.state, 'deepspeed_plugin', None) is not None:
            ds_plugin = accelerator.state.deepspeed_plugin
            ds_config = getattr(ds_plugin, 'deepspeed_config', None)
            if isinstance(ds_config, dict):
                world_size = getattr(accelerator, 'num_processes', 1)
                ds_config['gradient_accumulation_steps'] = int(acc_steps)
                ds_config['train_micro_batch_size_per_gpu'] = int(bs)
                ds_config['train_batch_size'] = int(bs) * int(acc_steps) * int(world_size)
                if use_discriminative_ft and ds_config.get('optimizer') is not None:
                    # Remove optimizer from DS config entirely to avoid Accelerate conflict.
                    ds_config.pop('optimizer', None)
                    logging.info(
                        "[DeepSpeed] DS optimizer disabled for discriminative FT, "
                        "using code AdamW param groups."
                    )
                elif ds_config.get('optimizer') is not None:
                    params = ds_config['optimizer'].setdefault('params', {})
                    params['lr'] = float(caption_lr)
                    params['betas'] = [0.9, 0.999]
                    params['eps'] = 1e-4
                    params['weight_decay'] = float(getattr(args, 'weight_decay', 0.01))
                    from accelerate.utils import DummyOptim
                    optimizer = DummyOptim(model.parameters())
                    scheduler = None
                    use_dummy_optimizer = True
                    logging.info(
                        f"[DeepSpeed] synced caption config: micro_bs={bs}, acc_steps={acc_steps}, train_bs={ds_config['train_batch_size']}, lr={caption_lr}"
                    )
                    logging.info("[DeepSpeed] optimizer is configured in JSON, using DummyOptim in code")

        if not use_dummy_optimizer:
            if use_discriminative_ft:
                param_groups = _build_caption_param_groups(model, args, fallback_lr=caption_lr)
            else:
                param_groups = model.parameters()
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=caption_lr,
                betas=(0.9, 0.999),
                eps=1e-4,          # FP16安全: 防止 epsilon 被截断为 0 导致 NaN
                weight_decay=getattr(args, "weight_decay", 0.01),
            )

            _epochs_caption = getattr(args, "epochs_caption", args.max_epochs)
            _tmax_caption = getattr(args, "lr_tmax_caption", None) or _epochs_caption
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_tmax_caption)
        else:
            _epochs_caption = getattr(args, "epochs_caption", args.max_epochs)

        # ZeRO-3 前加载权重，避免 prepare 后参数被 shard 成 size([0])
        caption_ckpt_path = _resolve_best_checkpoint_path(
            args.model_name,
            "caption",
            use_metric_best=getattr(args, "load_best_metric_checkpoint", False),
        )
        if getattr(args, "continue_training", False):
            logging.info(
                "[caption_train_checkpoint][warm-start] continue_training=True; "
                "optimizer/scheduler resume will be handled later by trainer()."
            )
        _load_optional_checkpoint(
            model,
            caption_ckpt_path,
            source_name="caption_train_checkpoint",
        )

        # 分布式模式: 同步 gradient_accumulation_steps 为 caption 阶段的实际值
        if accelerator is not None:
            accelerator.gradient_accumulation_steps = acc_steps
            logging.info(f"Accelerator gradient_accumulation_steps set to {acc_steps}")
            model, optimizer, train_loader, val_loader, val_metric_loader = accelerator.prepare(
                model, optimizer, train_loader, val_loader, val_metric_loader
            )
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

        trainer("caption", train_loader, val_loader, val_metric_loader,
                model, optimizer, scheduler, criterion,
                model_name=args.model_name,
                max_epochs=_epochs_caption, evaluation_frequency=evaluation_frequency,
                accumulation_steps=acc_steps,
                max_grad_norm=getattr(args, "max_grad_norm_caption", None)
                if getattr(args, "max_grad_norm_caption", None) is not None
                else getattr(args, "max_grad_norm", 0.5),
                smoke_steps=smoke_steps,
                continue_training=getattr(args, 'continue_training', False),
                accelerator=accelerator)

        del train_loader
        del val_loader
        del val_metric_loader
        del dataset_Train
        del dataset_Valid
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    if getattr(args, "skip_caption_eval", False):
        logging.info("[caption] skip_caption_eval=True, skipping post-training caption evaluation.")
        return

    caption_test_max_samples = int(getattr(args, "caption_test_max_samples", 0) or 0)
    if caption_test_max_samples < 0:
        logging.warning(
            "[caption] caption_test_max_samples=%d is invalid. Resetting to 0 (full split).",
            caption_test_max_samples,
        )
        caption_test_max_samples = 0

    # validate caption generation on groundtruth spots on multiple splits [test/challenge]
    for split in args.split_test:
        target_split = [split]

        if use_dual:
            from dataset_dual import SoccerNetCaptionsDual, collate_fn_padd_dual
            dataset_Test = SoccerNetCaptionsDual(
                vision_root=args.SoccerNet_path, audio_root=args.audio_root,
                features=args.features, split=target_split,
                version=args.version, framerate=args.framerate,
                window_size=args.window_size_caption,
            )
            test_collate = collate_fn_padd_dual
        else:
            dataset_Test = SoccerNetCaptions(
                path=args.SoccerNet_path,
                features=args.features,
                split=target_split,
                version=args.version,
                framerate=args.framerate,
                window_size=args.window_size_caption,
            )
            from dataset import CollateGPT
            test_collate = CollateGPT(llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B"))

        if len(dataset_Test) == 0:
            logging.warning(f"[caption] split={split} has 0 samples after filtering. Skipping validation.")
            continue

        if caption_test_max_samples > 0 and len(dataset_Test) > caption_test_max_samples:
            logging.info(
                "[caption] split=%s limiting evaluation samples: %d -> %d",
                split,
                len(dataset_Test),
                caption_test_max_samples,
            )
            dataset_Test = torch.utils.data.Subset(dataset_Test, range(caption_test_max_samples))

        test_loader = torch.utils.data.DataLoader(dataset_Test,
            batch_size=bs, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True, collate_fn=test_collate)

        results = validate_captioning(
            test_loader,
            model,
            args.model_name,
            smoke_steps=smoke_steps,
            generation_config=generation_config,
        )
        if results is None:
            continue

        logging.info("Best Performance at end of training in generating captions")
        logging.info(f'| Bleu_1: {results["Bleu_1"]}')
        logging.info(f'| Bleu_2: {results["Bleu_2"]}')
        logging.info(f'| Bleu_3: {results["Bleu_3"]}')
        logging.info(f'| Bleu_4: {results["Bleu_4"]}')
        logging.info(f'| METEOR: {results["METEOR"]}')
        logging.info(f'| ROUGE_L: {results["ROUGE_L"]}')
        logging.info(f'| CIDEr: {results["CIDEr"]}')
        logging.info(f'| SPICE: {results["SPICE"]}')

        log_dict = {f"{k}_{split}_gt": v for k, v in results.items()}
        if getattr(args, 'accelerator', None) is not None:
            args.accelerator.log(log_dict)
        else:
            wandb.log(log_dict)

    return


def dvc(args):

    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(arg.rjust(15) + " : " + str(getattr(args, arg)))
    logging.info("Starting DVC")
    generation_config = get_caption_generation_config_from_args(args)

    # dvc 由 main.py 只传入 args；accelerator 需要从 args 上取，避免局部变量未定义
    accelerator = getattr(args, "accelerator", None)
    device = accelerator.device if accelerator is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    smoke_steps = getattr(args, "smoke_steps_caption", None)
    if smoke_steps is None:
        smoke_steps = getattr(args, "smoke_steps", 0)

    use_dual = getattr(args, "use_dual_stream", False)

    if use_dual:
        from dataset_dual import SoccerNetCaptionsDual, PredictionCaptionsDual
        from dual_qformer import DualVideo2CaptionLLM

        audio_root = args.audio_root
        assert audio_root is not None, "Dual-stream DVC requires --audio_root"

        dataset_Test = SoccerNetCaptionsDual(
            vision_root=args.SoccerNet_path, audio_root=audio_root,
            features=args.features, split=args.split_test,
            version=args.version, framerate=args.framerate,
            window_size=args.window_size_caption,
            llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B")
        )

        if args.feature_dim is None:
            args.feature_dim = dataset_Test[0][0].shape[-1]
            print("feature_dim found:", args.feature_dim)

        model = DualVideo2CaptionLLM(
            vocab_size=dataset_Test.vocab_size,
            video_input_dim=getattr(args, "video_input_dim", 1024),
            audio_input_dim=getattr(args, "audio_input_dim", 512),
            llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B"),
            lora_r=getattr(args, "lora_r", 8),
            lora_alpha=getattr(args, "lora_alpha", 16),
            lora_dropout=getattr(args, "lora_dropout", 0.05),
            weights=args.load_weights,
            weights_encoder=args.weights_encoder,
            freeze_encoder=args.freeze_encoder,
            top_k=args.top_k,
            max_new_tokens=generation_config["max_new_tokens"],
            no_repeat_ngram_size=generation_config["no_repeat_ngram_size"],
            num_beams=generation_config["num_beams"],
            length_penalty=generation_config["length_penalty"],
            do_sample=generation_config["do_sample"],
            temperature=generation_config["temperature"],
            top_p=generation_config["top_p"],
            repetition_penalty=generation_config["repetition_penalty"],
            encoder_dropout=getattr(args, "encoder_dropout", 0.1),
        )
    else:
        dataset_Test = SoccerNetCaptions(path=args.SoccerNet_path, features=args.features, split=args.split_test, version=args.version, framerate=args.framerate, window_size=args.window_size_caption, llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B"))

        if args.feature_dim is None:
            args.feature_dim = dataset_Test[0][0].shape[-1]
            print("feature_dim found:", args.feature_dim)

        from model_qwen import Video2CaptionQwen
        model = Video2CaptionQwen(
            input_size=args.feature_dim,
            vlad_k=args.vlad_k,
            window_size=args.window_size_caption,
            framerate=args.framerate,
            pool=args.pool,
            llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B"),
            lora_r=getattr(args, "lora_r", 8),
            lora_alpha=getattr(args, "lora_alpha", 16),
            lora_dropout=getattr(args, "lora_dropout", 0.05),
            weights_encoder=args.weights_encoder,
            freeze_encoder=args.freeze_encoder,
            top_k=args.top_k,
            max_new_tokens=generation_config["max_new_tokens"],
            no_repeat_ngram_size=generation_config["no_repeat_ngram_size"],
            num_beams=generation_config["num_beams"],
            length_penalty=generation_config["length_penalty"],
            do_sample=generation_config["do_sample"],
            temperature=generation_config["temperature"],
            top_p=generation_config["top_p"],
            repetition_penalty=generation_config["repetition_penalty"],
        )

    logging.info(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total number of parameters: " + str(total_params))

    # For the best model only
    # ZeRO-3 时 model 可能是 DeepSpeedEngine，需先重建未 prepare 的模型再加载
    if accelerator is not None and getattr(accelerator.state, 'deepspeed_plugin', None) is not None:
        if use_dual:
            from dataset_dual import SoccerNetCaptionsDual, collate_fn_padd_dual
            from dual_qformer import DualVideo2CaptionLLM
            model = DualVideo2CaptionLLM(
                vocab_size=dataset_Test.vocab_size,
                video_input_dim=getattr(args, "video_input_dim", 1024),
                audio_input_dim=getattr(args, "audio_input_dim", 512),
                llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B"),
                lora_r=getattr(args, "lora_r", 8),
                lora_alpha=getattr(args, "lora_alpha", 16),
                lora_dropout=getattr(args, "lora_dropout", 0.05),
                weights=args.load_weights,
                weights_encoder=args.weights_encoder,
                freeze_encoder=args.freeze_encoder,
                top_k=args.top_k,
                max_new_tokens=generation_config["max_new_tokens"],
                no_repeat_ngram_size=generation_config["no_repeat_ngram_size"],
                num_beams=generation_config["num_beams"],
                length_penalty=generation_config["length_penalty"],
                do_sample=generation_config["do_sample"],
                temperature=generation_config["temperature"],
                top_p=generation_config["top_p"],
                repetition_penalty=generation_config["repetition_penalty"],
                encoder_dropout=getattr(args, "encoder_dropout", 0.1),
            )
        else:
            from model_qwen import Video2CaptionQwen
            model = Video2CaptionQwen(
                input_size=args.feature_dim,
                vlad_k=args.vlad_k,
                window_size=args.window_size_caption,
                framerate=args.framerate,
                pool=args.pool,
                llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B"),
                lora_r=getattr(args, "lora_r", 8),
                lora_alpha=getattr(args, "lora_alpha", 16),
                lora_dropout=getattr(args, "lora_dropout", 0.05),
                weights_encoder=args.weights_encoder,
                freeze_encoder=args.freeze_encoder,
                top_k=args.top_k,
                max_new_tokens=generation_config["max_new_tokens"],
                no_repeat_ngram_size=generation_config["no_repeat_ngram_size"],
                num_beams=generation_config["num_beams"],
                length_penalty=generation_config["length_penalty"],
                do_sample=generation_config["do_sample"],
                temperature=generation_config["temperature"],
                top_p=generation_config["top_p"],
                repetition_penalty=generation_config["repetition_penalty"],
            )

    caption_ckpt_path = _resolve_best_checkpoint_path(
        args.model_name,
        "caption",
        use_metric_best=getattr(args, "load_best_metric_checkpoint", False),
    )
    _load_optional_checkpoint(
        model,
        caption_ckpt_path,
        source_name="caption_test_checkpoint",
    )
    import gc
    gc.collect()
    model = model.to(device)

    # generate dense caption on multiple splits [test/challenge]
    for split in args.split_test:
        PredictionPath = os.path.join("models", args.model_name, f"outputs/{split}")

        if use_dual:
            from dataset_dual import PredictionCaptionsDual
            dataset_Test = PredictionCaptionsDual(
                vision_root=args.SoccerNet_path, audio_root=args.audio_root,
                PredictionPath=PredictionPath, features=args.features,
                split=[split], version=args.version,
                framerate=args.framerate, window_size=args.window_size_caption,
                llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B")
            )
        else:
            dataset_Test = PredictionCaptions(SoccerNetPath=args.SoccerNet_path, PredictionPath=PredictionPath, features=args.features, split=[split], version=args.version, framerate=args.framerate, window_size=args.window_size_caption, llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B"))

        bs_test = getattr(args, 'batch_size_caption', None) or args.batch_size
        test_loader = torch.utils.data.DataLoader(dataset_Test,
            batch_size=bs_test, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True)

        results = test_captioning(
            test_loader,
            model,
            args.model_name,
            smoke_steps=smoke_steps,
            generation_config=generation_config,
        )
        if results is None:
            continue

        logging.info("Best Performance at end of training in dense video captioning")
        logging.info(f'| Bleu_1_tight: {results["Bleu_1_tight"]}')
        logging.info(f'| Bleu_2_tight: {results["Bleu_2_tight"]}')
        logging.info(f'| Bleu_3_tight: {results["Bleu_3_tight"]}')
        logging.info(f'| Bleu_4_tight: {results["Bleu_4_tight"]}')
        logging.info(f'| METEOR_tight: {results["METEOR_tight"]}')
        logging.info(f'| ROUGE_L_tight: {results["ROUGE_L_tight"]}')
        logging.info(f'| CIDEr_tight: {results["CIDEr_tight"]}')
        logging.info(f'| Recall_tight: {results["Recall_tight"]}')
        logging.info(f'| Precision_tight: {results["Precision_tight"]}')

        logging.info(f'| Bleu_1_loose: {results["Bleu_1_loose"]}')
        logging.info(f'| Bleu_2_loose: {results["Bleu_2_loose"]}')
        logging.info(f'| Bleu_3_loose: {results["Bleu_3_loose"]}')
        logging.info(f'| Bleu_4_loose: {results["Bleu_4_loose"]}')
        logging.info(f'| METEOR_loose: {results["METEOR_loose"]}')
        logging.info(f'| ROUGE_L_loose: {results["ROUGE_L_loose"]}')
        logging.info(f'| CIDEr_loose: {results["CIDEr_loose"]}')
        logging.info(f'| Recall_loose: {results["Recall_loose"]}')
        logging.info(f'| Precision_loose: {results["Precision_loose"]}')

        log_dict = {f"{k}_{split}_pt": v for k, v in results.items()}
        if getattr(args, 'accelerator', None) is not None:
            args.accelerator.log(log_dict)
        else:
            wandb.log(log_dict)
