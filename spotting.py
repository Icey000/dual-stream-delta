from functools import partial
import os
import logging
from datetime import datetime
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, BooleanOptionalAction

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import SoccerNetClips, SoccerNetClipsTesting
from model import Video2Spot
from train import trainer, test_spotting, _resolve_best_checkpoint_path

import wandb


def _log_optimizer_groups(phase, named_groups):
    for group_name, params, lr in named_groups:
        num_params = sum(p.numel() for p in params)
        logging.info(
            f"[{phase}] optimizer_group={group_name}, tensors={len(params)}, params={num_params}, lr={lr}"
        )


def _build_spotting_param_groups(model, args, fallback_lr):
    lr_proj_head = getattr(args, "lr_spotting_proj_head", None) or fallback_lr
    lr_qformer = getattr(args, "lr_spotting_qformer", 2e-6)

    qformer_params = []
    proj_head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder.") or ".encoder." in name:
            qformer_params.append(param)
        else:
            proj_head_params.append(param)

    named_groups = [
        ("proj_heads", proj_head_params, lr_proj_head),
        ("qformer_encoder", qformer_params, lr_qformer),
    ]
    _log_optimizer_groups("spotting", named_groups)

    param_groups = []
    for group_name, params, lr in named_groups:
        if params:
            param_groups.append({"params": params, "lr": float(lr), "group_name": group_name})
    return param_groups


def _resolve_spotting_num_classes(dataset):
    dataset_num_classes = int(getattr(dataset, "num_classes", 0) or 0)
    dict_event = getattr(dataset, "dict_event", None) or {}
    dict_event_len = len(dict_event)

    if dataset_num_classes <= 0:
        dataset_num_classes = dict_event_len

    if dataset_num_classes <= 0:
        raise ValueError("Unable to determine spotting num_classes from dataset metadata.")

    if dict_event_len and dataset_num_classes != dict_event_len:
        logging.warning(
            "[spotting] dataset.num_classes=%d but len(dict_event)=%d. "
            "Using dataset.num_classes for the classifier head so it matches the label tensors.",
            dataset_num_classes,
            dict_event_len,
        )

    return dataset_num_classes


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)

        alpha_pos = torch.full_like(pt, self.alpha)
        alpha_neg = torch.full_like(pt, 1.0 - self.alpha)
        alpha_t = torch.where(targets > 0, alpha_pos, alpha_neg)

        loss = alpha_t * torch.pow(1.0 - pt, self.gamma) * ce

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):
    expects_prob_targets = True

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(targets * log_probs).sum(dim=1)
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def _build_spotting_criterion(args):
    target_mode = str(getattr(args, "spotting_target_mode", "hard_multiclass")).strip().lower()
    if target_mode == "soft_window_multiclass":
        logging.info("[spotting] Using SoftTargetCrossEntropy() for soft_window_multiclass targets")
        return SoftTargetCrossEntropy()

    spotting_loss = str(getattr(args, "spotting_loss", "ce")).strip().lower()
    if spotting_loss == "focal":
        alpha = float(getattr(args, "focal_alpha", 0.75))
        gamma = float(getattr(args, "focal_gamma", 2.0))
        logging.info(
            "[spotting] Using FocalLoss(alpha=%.4f, gamma=%.4f)",
            alpha,
            gamma,
        )
        return FocalLoss(alpha=alpha, gamma=gamma)

    logging.info("[spotting] Using CrossEntropyLoss()")
    return nn.CrossEntropyLoss()


def _log_spotting_label_distribution(split_name, dataset):
    labels = getattr(dataset, "game_labels", None)
    if labels is None or len(labels) == 0:
        logging.warning("[spotting][%s] label distribution unavailable.", split_name)
        return

    labels_np = np.asarray(labels)
    if labels_np.ndim != 2 or labels_np.shape[1] < 2:
        logging.warning(
            "[spotting][%s] unexpected label shape %s; skipping distribution log.",
            split_name,
            tuple(labels_np.shape),
        )
        return

    target_idx = labels_np.argmax(axis=1)
    counts = np.bincount(target_idx, minlength=labels_np.shape[1])
    total = int(counts.sum())
    if total <= 0:
        logging.warning("[spotting][%s] empty labels.", split_name)
        return

    bg_count = int(counts[0])
    event_count = int(counts[1:].sum())
    logging.info(
        "[spotting][%s] label_distribution total=%d bg=%d event=%d bg_ratio=%.4f event_ratio=%.4f",
        split_name,
        total,
        bg_count,
        event_count,
        bg_count / total,
        event_count / total,
    )


def _load_checkpoint_with_retry(path, accelerator=None, retries=5, wait_seconds=10):
    """Load a checkpoint safely after distributed training.

    On multi-rank jobs, some processes can reach the post-training load path
    before rank 0 has fully flushed the checkpoint to disk. A short barrier + retry
    loop avoids transient `failed finding central directory` errors on shared storage.
    """
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            if accelerator is not None:
                accelerator.wait_for_everyone()

            if not os.path.exists(path):
                raise FileNotFoundError(f"Checkpoint not found: {path}")

            file_size = os.path.getsize(path)
            if file_size <= 0:
                raise RuntimeError(f"Checkpoint exists but is empty: {path}")

            try:
                return torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                # 兼容旧版 PyTorch（不支持 weights_only 参数）
                return torch.load(path, map_location="cpu")
            except Exception as load_exc:
                logging.warning(
                    "Checkpoint safe-load failed for %s, fallback to legacy load: %s",
                    path,
                    load_exc,
                )
                return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            last_error = exc
            logging.warning(
                "Failed to load checkpoint %s (attempt %d/%d): %s",
                path,
                attempt,
                retries,
                exc,
            )
            if attempt < retries:
                time.sleep(wait_seconds)

    raise last_error


def main(args):

    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(arg.rjust(15) + " : " + str(getattr(args, arg)))

    accelerator = None
    use_dual = getattr(args, "use_dual_stream", False)
    use_center_regression = bool(getattr(args, "spotting_use_center_regression", False))
    center_positive_threshold = float(getattr(args, "spotting_center_positive_threshold", 0.5))

    if use_dual:
        # ==================== 双流模式 ====================
        from dataset_dual import SoccerNetClipsDual, SoccerNetClipsTestingDual
        from dual_qformer import DualVideo2Spot

        audio_root = args.audio_root
        assert audio_root is not None, "Dual-stream spotting requires --audio_root"

        if not args.test_only:
            dataset_Train = SoccerNetClipsDual(
                vision_root=args.SoccerNet_path, audio_root=audio_root,
                features=args.features, split=args.split_train,
                version=args.version, framerate=args.framerate,
                window_size=args.window_size_spotting,
                target_mode=getattr(args, "spotting_target_mode", "hard_multiclass"),
                soft_window_radius=getattr(args, "spotting_soft_window_radius", 2),
                soft_window_sigma=getattr(args, "spotting_soft_window_sigma", 1.0),
                build_center_targets=use_center_regression,
                center_positive_threshold=center_positive_threshold,
            )
            dataset_Valid = SoccerNetClipsDual(
                vision_root=args.SoccerNet_path, audio_root=audio_root,
                features=args.features, split=args.split_valid,
                version=args.version, framerate=args.framerate,
                window_size=args.window_size_spotting,
                target_mode=getattr(args, "spotting_target_mode", "hard_multiclass"),
                build_center_targets=use_center_regression,
                center_positive_threshold=center_positive_threshold,
            )
            dataset_Valid_metric = SoccerNetClipsTestingDual(
                vision_root=args.SoccerNet_path, audio_root=audio_root,
                features=args.features, split=args.split_valid,
                version=args.version, framerate=args.framerate,
                window_size=args.window_size_spotting,
            )

        if args.test_only:
            dataset_Test = SoccerNetClipsTestingDual(
                vision_root=args.SoccerNet_path, audio_root=audio_root,
                features=args.features, split=args.split_test,
                version=args.version, framerate=args.framerate,
                window_size=args.window_size_spotting,
            )

        dataset_for_classes = dataset_Test if args.test_only else dataset_Train
        spotting_num_classes = _resolve_spotting_num_classes(dataset_for_classes)
        model = DualVideo2Spot(
            num_classes=spotting_num_classes,
            video_input_dim=getattr(args, "video_input_dim", 1024),
            audio_input_dim=getattr(args, "audio_input_dim", 512),
            hidden_dim=getattr(args, "hidden_dim", 3584),
            dropout=getattr(args, "encoder_dropout", 0.1),
            weights=args.load_weights,
            weights_encoder=args.weights_encoder,
            freeze_encoder=args.freeze_encoder,
            use_center_regression=use_center_regression,
        )
        model.spotting_center_regression_weight = float(getattr(args, "spotting_center_regression_weight", 1.0))

    else:
        # ==================== 原有单流模式 ====================
        # create dataset
        if not args.test_only:
            dataset_Train = SoccerNetClips(
                path=args.SoccerNet_path,
                features=args.features,
                split=args.split_train,
                version=args.version,
                framerate=args.framerate,
                window_size=args.window_size_spotting,
                target_mode=getattr(args, "spotting_target_mode", "hard_multiclass"),
                soft_window_radius=getattr(args, "spotting_soft_window_radius", 2),
                soft_window_sigma=getattr(args, "spotting_soft_window_sigma", 1.0),
                build_center_targets=use_center_regression,
                center_positive_threshold=center_positive_threshold,
            )
            dataset_Valid = SoccerNetClips(
                path=args.SoccerNet_path,
                features=args.features,
                split=args.split_valid,
                version=args.version,
                framerate=args.framerate,
                window_size=args.window_size_spotting,
                target_mode=getattr(args, "spotting_target_mode", "hard_multiclass"),
                build_center_targets=use_center_regression,
                center_positive_threshold=center_positive_threshold,
            )
            dataset_Valid_metric = SoccerNetClipsTesting(
                path=args.SoccerNet_path,
                features=args.features,
                split=args.split_valid,
                version=args.version,
                framerate=args.framerate,
                window_size=args.window_size_spotting,
            )
        if args.test_only:
            dataset_Test  = SoccerNetClipsTesting(path=args.SoccerNet_path, features=args.features, split=args.split_test, version=args.version, framerate=args.framerate, window_size=args.window_size_spotting)

        if args.feature_dim is None:
            if not args.test_only:
                args.feature_dim = dataset_Train[0][1].shape[-1]
            else:
                args.feature_dim = dataset_Test[0][1].shape[-1]
            print("feature_dim found:", args.feature_dim)
        # create model
        dataset_for_classes = dataset_Test if args.test_only else dataset_Train
        spotting_num_classes = _resolve_spotting_num_classes(dataset_for_classes)
        model = Video2Spot(weights=args.load_weights, input_size=args.feature_dim,
                      num_classes=spotting_num_classes, window_size=args.window_size_spotting,
                      vlad_k=args.vlad_k,
                      framerate=args.framerate, pool=args.pool, freeze_encoder=args.freeze_encoder,
                      weights_encoder=args.weights_encoder, proj_size=getattr(args, "hidden_dim", 3584),
                      use_center_regression=use_center_regression)
        model.spotting_center_regression_weight = float(getattr(args, "spotting_center_regression_weight", 1.0))

    logging.info(model)
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    parameters_per_layer  = [p.numel() for p in model.parameters() if p.requires_grad]
    logging.info("Total number of parameters: " + str(total_params))

    bs = getattr(args, 'batch_size_spotting', None) or args.batch_size
    acc_steps = getattr(args, 'accumulation_steps_spotting', None) or getattr(args, 'accumulation_steps', 1)
    smoke_steps = getattr(args, 'smoke_steps_spotting', None)
    if smoke_steps is None:
        smoke_steps = getattr(args, 'smoke_steps', 0)
    spotting_valid_max_samples = int(getattr(args, "spotting_valid_max_samples", 0) or 0)
    if spotting_valid_max_samples < 0:
        logging.warning(
            "[spotting] spotting_valid_max_samples=%d is invalid. Resetting to 0 (full split).",
            spotting_valid_max_samples,
        )
        spotting_valid_max_samples = 0
    spotting_test_max_samples = int(getattr(args, "spotting_test_max_samples", 0) or 0)
    if spotting_test_max_samples < 0:
        logging.warning(
            "[spotting] spotting_test_max_samples=%d is invalid. Resetting to 0 (full split).",
            spotting_test_max_samples,
        )
        spotting_test_max_samples = 0
    spotting_epochs = getattr(args, "spotting_epochs", None) or args.max_epochs
    evaluation_frequency = getattr(args, "evaluation_frequency_spotting", None) or args.evaluation_frequency
    logging.info(
        f"Spotting phase effective batch_size={bs}, accumulation_steps={acc_steps}, "
        f"epochs={spotting_epochs}, smoke_steps={smoke_steps}, valid_max_samples={spotting_valid_max_samples}, "
        f"test_max_samples={spotting_test_max_samples}, "
        f"target_mode={getattr(args, 'spotting_target_mode', 'hard_multiclass')}, evaluation_frequency={evaluation_frequency}, "
        f"use_center_regression={use_center_regression}, center_positive_threshold={center_positive_threshold}, "
        f"center_regression_weight={getattr(args, 'spotting_center_regression_weight', 1.0)}"
    )

    # create dataloader
    if not args.test_only:
        _log_spotting_label_distribution("train", dataset_Train)
        _log_spotting_label_distribution("valid", dataset_Valid)

        train_loader = torch.utils.data.DataLoader(dataset_Train,
            batch_size=bs, shuffle=True,
            num_workers=args.max_num_worker, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(dataset_Valid,
            batch_size=bs, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True)

        if spotting_valid_max_samples > 0 and len(dataset_Valid_metric) > spotting_valid_max_samples:
            logging.info(
                "[spotting] limiting valid metric samples: %d -> %d",
                len(dataset_Valid_metric),
                spotting_valid_max_samples,
            )
            dataset_Valid_metric = torch.utils.data.Subset(dataset_Valid_metric, range(spotting_valid_max_samples))

        # Use game-level loader (batch_size=1) so validation metric matches official test-time spotting evaluation.
        val_metric_loader = torch.utils.data.DataLoader(
            dataset_Valid_metric,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )


    # training parameters
    if not args.test_only:
        criterion = _build_spotting_criterion(args)
        use_discriminative_ft = bool(getattr(args, "discriminative_ft_spotting", True))
        logging.info(
            f"Spotting phase LR base={args.LR}, discriminative_ft_spotting={use_discriminative_ft}, "
            f"spotting_num_classes={spotting_num_classes}"
        )
        if getattr(args, "spotting_target_mode", "hard_multiclass") == "soft_window_multiclass":
            logging.info(
                "[spotting] Soft-window targets are applied to both train/val loss splits; "
                "validation metric still uses hard labels through dataset_Valid_metric."
            )

        # start training
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
                    # Accelerate treats the presence of the optimizer key as "config defines optimizer".
                    # Remove it entirely so the code-defined AdamW can be used without conflict.
                    ds_config.pop('optimizer', None)
                    logging.info(
                        "[DeepSpeed] DS optimizer disabled for discriminative FT, "
                        "using code AdamW param groups."
                    )
                elif ds_config.get('optimizer') is not None:
                    params = ds_config['optimizer'].setdefault('params', {})
                    params['lr'] = float(args.LR)
                    params['betas'] = [0.9, 0.999]
                    params['eps'] = 1e-8
                    params['weight_decay'] = float(getattr(args, 'weight_decay', 0.01))
                    from accelerate.utils import DummyOptim
                    optimizer = DummyOptim(model.parameters())
                    scheduler = None
                    use_dummy_optimizer = True
                    logging.info(
                        f"[DeepSpeed] synced spotting config: micro_bs={bs}, acc_steps={acc_steps}, train_bs={ds_config['train_batch_size']}, lr={args.LR}"
                    )
                    logging.info("[DeepSpeed] optimizer is configured in JSON, using DummyOptim in code")

        if not use_dummy_optimizer:
            if use_discriminative_ft:
                param_groups = _build_spotting_param_groups(model, args, fallback_lr=args.LR)
            else:
                param_groups = model.parameters()
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=args.LR,
                weight_decay=getattr(args, "weight_decay", 0.01),
            )

            _tmax_spot = getattr(args, "lr_tmax_spotting", None) or spotting_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_tmax_spot)

        # 分布式模式: 用 accelerator 准备模型、优化器、DataLoader
        if accelerator is not None:
            accelerator.gradient_accumulation_steps = acc_steps
            logging.info(f"Accelerator gradient_accumulation_steps set to {acc_steps}")
            model, optimizer, train_loader, val_loader, val_metric_loader = accelerator.prepare(
                model, optimizer, train_loader, val_loader, val_metric_loader
            )
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

        trainer("spotting", train_loader, val_loader, val_metric_loader, 
                model, optimizer, scheduler, criterion,
                model_name=args.model_name,
                max_epochs=spotting_epochs, evaluation_frequency=evaluation_frequency,
                accumulation_steps=acc_steps,
                max_grad_norm=getattr(args, "max_grad_norm_spotting", None)
                if getattr(args, "max_grad_norm_spotting", None) is not None
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

    # For the best model only
    # ZeRO-3 时 model 可能是 DeepSpeedEngine，直接 load_state_dict 会因为 shard 而报错
    if accelerator is not None and getattr(accelerator.state, 'deepspeed_plugin', None) is not None:
        spotting_num_classes = _resolve_spotting_num_classes(dataset_for_classes)
        if use_dual:
            from dual_qformer import DualVideo2Spot
            model = DualVideo2Spot(
                num_classes=spotting_num_classes,
                video_input_dim=getattr(args, "video_input_dim", 1024),
                audio_input_dim=getattr(args, "audio_input_dim", 512),
                hidden_dim=getattr(args, "hidden_dim", 3584),
                dropout=getattr(args, "encoder_dropout", 0.1),
                weights=args.load_weights,
                weights_encoder=args.weights_encoder,
                freeze_encoder=args.freeze_encoder,
                use_center_regression=use_center_regression,
            )
            model.spotting_center_regression_weight = float(getattr(args, "spotting_center_regression_weight", 1.0))
        else:
            model = Video2Spot(
                weights=args.load_weights,
                input_size=args.feature_dim,
                num_classes=spotting_num_classes,
                window_size=args.window_size_spotting,
                vlad_k=args.vlad_k,
                framerate=args.framerate,
                pool=args.pool,
                freeze_encoder=args.freeze_encoder,
                weights_encoder=args.weights_encoder,
                proj_size=getattr(args, "hidden_dim", 3584),
                use_center_regression=use_center_regression,
            )
            model.spotting_center_regression_weight = float(getattr(args, "spotting_center_regression_weight", 1.0))

    checkpoint_path = _resolve_best_checkpoint_path(
        args.model_name,
        "spotting",
        use_metric_best=getattr(args, "load_best_metric_checkpoint", False),
    )
    checkpoint = _load_checkpoint_with_retry(
        checkpoint_path,
        accelerator=accelerator,
    )
    missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
    if missing or unexpected:
        logging.info(
            "[spotting] checkpoint load (strict=False) missing=%s unexpected=%s",
            missing,
            unexpected,
        )
    del checkpoint
    import gc
    gc.collect()
    device = accelerator.device if accelerator is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    # test on multiple splits [test/challenge]
    for split in args.split_test:
        if use_dual:
            from dataset_dual import SoccerNetClipsTestingDual
            dataset_Test = SoccerNetClipsTestingDual(
                vision_root=args.SoccerNet_path, audio_root=args.audio_root,
                features=args.features, split=[split],
                version=args.version, framerate=args.framerate,
                window_size=args.window_size_spotting,
            )
        else:
            dataset_Test  = SoccerNetClipsTesting(path=args.SoccerNet_path, features=args.features, split=[split], version=args.version, framerate=args.framerate, window_size=args.window_size_spotting)

        if spotting_test_max_samples > 0 and len(dataset_Test) > spotting_test_max_samples:
            logging.info(
                "[spotting] split=%s limiting test samples: %d -> %d",
                split,
                len(dataset_Test),
                spotting_test_max_samples,
            )
            dataset_Test = torch.utils.data.Subset(dataset_Test, range(spotting_test_max_samples))

        # Spotting test 每个样本是整场比赛，payload 很大；多 worker 在多卡下容易被 OOM killer 杀掉。
        test_loader = torch.utils.data.DataLoader(
            dataset_Test,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        results = test_spotting(
            test_loader,
            model=model,
            model_name=args.model_name,
            NMS_window=args.NMS_window,
            NMS_threshold=args.NMS_threshold,
            smoke_steps=smoke_steps,
        )
        if results is None:
            continue

        a_mAP_tight = results["a_mAP_tight"]
        a_mAP_loose = results["a_mAP_loose"]
        a_mAP_medium = results["a_mAP_medium"]

        logging.info("Best Performance at end of training ")
        logging.info("a_mAP tight: " +  str(a_mAP_tight))
        logging.info("a_mAP loose: " +  str(a_mAP_loose))
        logging.info("a_mAP_medium: " +  str(a_mAP_medium))
                
        log_dict = {f"{k}_{split}" : results[k] for k in ["a_mAP_tight", "a_mAP_loose", "a_mAP_medium"]}
        if getattr(args, 'accelerator', None) is not None:
            args.accelerator.log(log_dict)
        else:
            wandb.log(log_dict)

    return 

if __name__ == '__main__':


    parser = ArgumentParser(description='SoccerNet-Caption: Spotting training', formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--SoccerNet_path',   required=False, type=str,   default="/path/to/SoccerNet/",     help='Path for SoccerNet' )
    parser.add_argument('--features',   required=False, type=str,   default="ResNET_TF2.npy",     help='Video features' )
    parser.add_argument(
        '--max_epochs',
        required=False,
        type=int,
        default=1000,
        help='Legacy fallback for spotting. Prefer --spotting_epochs.',
    )
    parser.add_argument(
        '--spotting_epochs',
        required=False,
        type=int,
        default=None,
        help='Number of epochs for spotting phase (preferred name).',
    )
    parser.add_argument('--load_weights',   required=False, type=str,   default=None,     help='weights to load' )
    parser.add_argument('--model_name',   required=False, type=str,   default="NetVLAD++",     help='named of the model to save' )
    parser.add_argument('--test_only',   required=False, action='store_true',  help='Perform testing only' )

    parser.add_argument('--split_train', nargs='+', default=["train"], help='list of split for training')
    parser.add_argument('--split_valid', nargs='+', default=["valid"], help='list of split for validation')
    parser.add_argument('--split_test', nargs='+', default=["test", "challenge"], help='list of split for testing')
    parser.add_argument('--spotting_valid_max_samples', required=False, type=int, default=0,
                        help='If >0, limit spotting validation to first N samples per split (0 means full split)')
    parser.add_argument('--spotting_test_max_samples', required=False, type=int, default=0,
                        help='If >0, limit spotting test/eval to first N samples per split (0 means full split)')
    parser.add_argument('--load_best_metric_checkpoint', action=BooleanOptionalAction, default=False,
                        help='Load best-metric checkpoint for evaluation instead of the default best-loss checkpoint')

    parser.add_argument('--version', required=False, type=int,   default=2,     help='Version of the dataset' )
    parser.add_argument('--feature_dim', required=False, type=int,   default=None,     help='Number of input features' )
    parser.add_argument('--evaluation_frequency', required=False, type=int,   default=10,     help='Run metric validation every N epochs' )
    parser.add_argument('--framerate', required=False, type=int,   default=2,     help='Framerate of the input features' )
    parser.add_argument('--window_size', required=False, type=int,   default=15,     help='Size of the chunk (in seconds)' )
    parser.add_argument('--pool',       required=False, type=str,   default="NetVLAD++", help='How to pool' )
    parser.add_argument('--vlad_k',       required=False, type=int,   default=64, help='Size of the vocabulary for NetVLAD' )
    parser.add_argument('--NMS_window',       required=False, type=int,   default=30, help='NMS window in second' )
    parser.add_argument('--NMS_threshold',       required=False, type=float,   default=0.0, help='NMS threshold for positive results' )

    parser.add_argument('--batch_size', required=False, type=int,   default=256,     help='Batch size' )
    parser.add_argument('--LR',       required=False, type=float,   default=1e-03, help='Learning Rate' )
    parser.add_argument('--spotting_loss', required=False, type=str, default='ce', choices=['ce', 'focal'],
                        help='Loss for spotting phase')
    parser.add_argument('--focal_alpha', required=False, type=float, default=0.75,
                        help='Positive-class alpha for spotting focal loss')
    parser.add_argument('--focal_gamma', required=False, type=float, default=2.0,
                        help='Gamma for spotting focal loss')
    parser.add_argument('--LRe',       required=False, type=float,   default=1e-06, help='Learning Rate end' )
    parser.add_argument('--patience', required=False, type=int,   default=10,     help='Patience before reducing LR (ReduceLROnPlateau)' )

    parser.add_argument('--GPU',        required=False, type=int,   default=-1,     help='ID of the GPU to use' )
    parser.add_argument('--max_num_worker',   required=False, type=int,   default=4, help='number of worker to load data')
    parser.add_argument('--seed',   required=False, type=int,   default=0, help='seed for reproducibility')

    # parser.add_argument('--logging_dir',       required=False, type=str,   default="log", help='Where to log' )
    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')

    args = parser.parse_args()

    # for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

    os.makedirs(os.path.join("models", args.model_name), exist_ok=True)
    log_path = os.path.join("models", args.model_name,
                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))
    
    run = wandb.init(
    project="NetVLAD-spotting",
    name=args.model_name
    )

    wandb.config.update(args)
    
    logging.basicConfig(
        level=numeric_level,
        format=
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)


    start=time.time()
    logging.info('Starting main function')
    main(args)
    logging.info(f'Total Execution Time is {time.time()-start} seconds')
