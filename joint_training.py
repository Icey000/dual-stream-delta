import logging
import os
import time
import torch
import torch.nn as nn
import wandb

import captioning
from spotting import _build_spotting_criterion, _resolve_spotting_num_classes
from train import (
    _is_main_process,
    _phase_step_offset,
    _resolve_best_checkpoint_path,
    _to_spotting_multiclass_targets,
    validate_captioning,
    validate_spotting_official,
)


def _caption_checkpoint_path(args):
    candidates = [
        os.path.join("models", args.model_name, "caption", "best_caption_ce.pth.tar"),
        _resolve_best_checkpoint_path(
            args.model_name,
            "caption",
            use_metric_best=getattr(args, "load_best_metric_checkpoint", False),
        ),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[-1]


def _spotting_checkpoint_path(args):
    explicit = getattr(args, "joint_spotting_checkpoint_path", None)
    if explicit:
        return explicit
    return _resolve_best_checkpoint_path(
        args.model_name,
        "spotting",
        use_metric_best=getattr(args, "load_best_metric_checkpoint", False),
    )


def _safe_float_metric(metrics, *keys, default=float("-inf")):
    fallback_keys = ("mAP-sklearn", "mAP")
    for key in list(keys) + list(fallback_keys):
        value = metrics.get(key) if isinstance(metrics, dict) else None
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float(default)


def _joint_should_log_step(step, total_steps):
    if step < 3:
        return True
    if total_steps <= 0:
        return False
    interval = max(100, total_steps // 10)
    return ((step + 1) % interval == 0) or ((step + 1) == total_steps)


def _build_caption_datasets_and_loader(args):
    use_dual = getattr(args, "use_dual_stream", False)
    bs = getattr(args, "batch_size_caption", None) or args.batch_size

    if use_dual:
        from dataset_dual import SoccerNetCaptionsDual, CollateGPTDual

        logging.info("[joint][data] building dual caption datasets")
        train_dataset = SoccerNetCaptionsDual(
            vision_root=args.SoccerNet_path,
            audio_root=args.audio_root,
            features=args.features,
            split=args.split_train,
            version=args.version,
            framerate=args.framerate,
            window_size=args.window_size_caption,
            llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B"),
        )
        valid_dataset = SoccerNetCaptionsDual(
            vision_root=args.SoccerNet_path,
            audio_root=args.audio_root,
            features=args.features,
            split=args.split_valid,
            version=args.version,
            framerate=args.framerate,
            window_size=args.window_size_caption,
            llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B"),
        )
        collate_fn = CollateGPTDual(llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B"))
    else:
        from dataset import SoccerNetCaptions, CollateGPT

        logging.info("[joint][data] building single-stream caption datasets")
        train_dataset = SoccerNetCaptions(
            path=args.SoccerNet_path,
            features=args.features,
            split=args.split_train,
            version=args.version,
            framerate=args.framerate,
            window_size=args.window_size_caption,
            llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B"),
        )
        valid_dataset = SoccerNetCaptions(
            path=args.SoccerNet_path,
            features=args.features,
            split=args.split_valid,
            version=args.version,
            framerate=args.framerate,
            window_size=args.window_size_caption,
            llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B"),
        )
        collate_fn = CollateGPT(llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B"))

    valid_max_samples = int(getattr(args, "caption_valid_max_samples", 0) or 0)
    valid_metric_dataset = valid_dataset
    if valid_max_samples > 0 and len(valid_metric_dataset) > valid_max_samples:
        valid_metric_dataset = torch.utils.data.Subset(valid_metric_dataset, range(valid_max_samples))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=args.max_num_worker,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=args.max_num_worker,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    valid_metric_loader = torch.utils.data.DataLoader(
        valid_metric_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=args.max_num_worker,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    logging.info(
        "[joint][data] caption datasets ready: train=%d valid=%d valid_metric=%d",
        len(train_dataset),
        len(valid_dataset),
        len(valid_metric_dataset),
    )
    return train_dataset, valid_dataset, train_loader, valid_loader, valid_metric_loader


def _build_spotting_datasets_and_loader(args):
    use_dual = getattr(args, "use_dual_stream", False)
    bs = getattr(args, "batch_size_spotting", None) or args.batch_size

    if use_dual:
        from dataset_dual import SoccerNetClipsDual, SoccerNetClipsTestingDual

        logging.info("[joint][data] building dual spotting datasets")
        train_dataset = SoccerNetClipsDual(
            vision_root=args.SoccerNet_path,
            audio_root=args.audio_root,
            features=args.features,
            split=args.split_train,
            version=args.version,
            framerate=args.framerate,
            window_size=args.window_size_spotting,
            target_mode=getattr(args, "spotting_target_mode", "hard_multiclass"),
            soft_window_radius=getattr(args, "spotting_soft_window_radius", 2),
            soft_window_sigma=getattr(args, "spotting_soft_window_sigma", 1.0),
            build_center_targets=False,
        )
        valid_dataset = SoccerNetClipsDual(
            vision_root=args.SoccerNet_path,
            audio_root=args.audio_root,
            features=args.features,
            split=args.split_valid,
            version=args.version,
            framerate=args.framerate,
            window_size=args.window_size_spotting,
            target_mode=getattr(args, "spotting_target_mode", "hard_multiclass"),
            build_center_targets=False,
        )
        valid_metric_dataset = SoccerNetClipsTestingDual(
            vision_root=args.SoccerNet_path,
            audio_root=args.audio_root,
            features=args.features,
            split=args.split_valid,
            version=args.version,
            framerate=args.framerate,
            window_size=args.window_size_spotting,
        )
    else:
        from dataset import SoccerNetClips, SoccerNetClipsTesting

        logging.info("[joint][data] building single-stream spotting datasets")
        train_dataset = SoccerNetClips(
            path=args.SoccerNet_path,
            features=args.features,
            split=args.split_train,
            version=args.version,
            framerate=args.framerate,
            window_size=args.window_size_spotting,
            target_mode=getattr(args, "spotting_target_mode", "hard_multiclass"),
            soft_window_radius=getattr(args, "spotting_soft_window_radius", 2),
            soft_window_sigma=getattr(args, "spotting_soft_window_sigma", 1.0),
            build_center_targets=False,
        )
        valid_dataset = SoccerNetClips(
            path=args.SoccerNet_path,
            features=args.features,
            split=args.split_valid,
            version=args.version,
            framerate=args.framerate,
            window_size=args.window_size_spotting,
            target_mode=getattr(args, "spotting_target_mode", "hard_multiclass"),
            build_center_targets=False,
        )
        valid_metric_dataset = SoccerNetClipsTesting(
            path=args.SoccerNet_path,
            features=args.features,
            split=args.split_valid,
            version=args.version,
            framerate=args.framerate,
            window_size=args.window_size_spotting,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=args.max_num_worker,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=args.max_num_worker,
        pin_memory=True,
    )
    valid_metric_loader = torch.utils.data.DataLoader(
        valid_metric_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    logging.info(
        "[joint][data] spotting datasets ready: train=%d valid=%d valid_metric=%d",
        len(train_dataset),
        len(valid_dataset),
        len(valid_metric_dataset),
    )
    return train_dataset, valid_dataset, valid_metric_dataset, train_loader, valid_loader, valid_metric_loader


def _infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


class JointVideoCaptionSpot(nn.Module):
    def __init__(self, caption_model, num_spotting_classes):
        super().__init__()
        self.caption_model = caption_model
        hidden_dim = int(self.caption_model.llm.config.hidden_size)
        self.spot_norm = nn.LayerNorm(hidden_dim)
        self.spot_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, num_spotting_classes + 1),
        )
        for module in self.spot_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.constant_(module.bias, 0)

    def forward_caption(self, feats, captions, lengths):
        return self.caption_model(feats, captions, lengths)

    def forward_spotting(self, feats):
        tokens = self.caption_model.encode_projected_tokens(feats)
        pooled = self.spot_norm(tokens.mean(dim=1))
        return self.spot_head(pooled)

    def forward(self, feats, captions=None, lengths=None):
        if captions is None:
            return self.forward_spotting(feats)
        return self.forward_caption(feats, captions, lengths)

    def sample(self, feats, generation_config=None):
        return self.caption_model.sample(feats, generation_config=generation_config)

    def load_spotting_head_from_checkpoint(self, checkpoint_path):
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            logging.warning("[joint] spotting checkpoint missing, skipping head warm-start: %s", checkpoint_path)
            return
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        mapped = {}
        for key, value in state_dict.items():
            if key.startswith("head."):
                mapped[f"spot_head.{key[len('head.'):]}"] = value
            elif key.startswith("norm."):
                mapped[f"spot_norm.{key[len('norm.'):]}"] = value
        missing, unexpected = self.load_state_dict(mapped, strict=False)
        logging.info("[joint] spotting head warm-start loaded from %s (missing=%s unexpected=%s)", checkpoint_path, missing, unexpected)


class JointDualVideoCaptionSpot(nn.Module):
    def __init__(self, caption_model, num_spotting_classes):
        super().__init__()
        self.caption_model = caption_model
        hidden_dim = int(self.caption_model.llm.config.hidden_size)
        self.spot_norm = nn.LayerNorm(hidden_dim)
        self.spot_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, num_spotting_classes + 1),
        )
        for module in self.spot_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.constant_(module.bias, 0)

    def forward_caption(self, vfeats, afeats, captions, lengths):
        return self.caption_model(vfeats, afeats, captions, lengths)

    def forward_spotting(self, vfeats, afeats):
        tokens = self.caption_model.encode_projected_tokens(vfeats, afeats)
        pooled = self.spot_norm(tokens.mean(dim=1))
        return self.spot_head(pooled)

    def forward(self, vfeats, afeats, captions=None, lengths=None):
        if captions is None:
            return self.forward_spotting(vfeats, afeats)
        return self.forward_caption(vfeats, afeats, captions, lengths)

    def sample(self, vfeats, afeats, generation_config=None):
        return self.caption_model.sample(vfeats, afeats, generation_config=generation_config)

    def load_spotting_head_from_checkpoint(self, checkpoint_path):
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            logging.warning("[joint] spotting checkpoint missing, skipping head warm-start: %s", checkpoint_path)
            return
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        mapped = {}
        for key, value in state_dict.items():
            if key.startswith("head."):
                mapped[f"spot_head.{key[len('head.'):]}"] = value
            elif key.startswith("norm."):
                mapped[f"spot_norm.{key[len('norm.'):]}"] = value
        missing, unexpected = self.load_state_dict(mapped, strict=False)
        logging.info("[joint] spotting head warm-start loaded from %s (missing=%s unexpected=%s)", checkpoint_path, missing, unexpected)


def _build_joint_model(args, caption_train_dataset, spotting_train_dataset):
    use_dual = getattr(args, "use_dual_stream", False)
    generation_config = captioning.get_caption_generation_config_from_args(args)

    if use_dual:
        from dual_qformer import DualVideo2CaptionLLM

        caption_model = DualVideo2CaptionLLM(
            vocab_size=caption_train_dataset.vocab_size,
            video_input_dim=getattr(args, "video_input_dim", 1024),
            audio_input_dim=getattr(args, "audio_input_dim", 512),
            llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B"),
            lora_r=getattr(args, "lora_r", 8),
            lora_alpha=getattr(args, "lora_alpha", 16),
            lora_dropout=getattr(args, "lora_dropout", 0.05),
            weights=args.load_weights,
            weights_encoder=args.weights_encoder,
            freeze_encoder=bool(getattr(args, "freeze_encoder_joint", True)),
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
        captioning._load_optional_checkpoint(caption_model, _caption_checkpoint_path(args), source_name="joint_caption_init")
        logging.info("[joint][model] caption backbone warm-start complete")
        model = JointDualVideoCaptionSpot(
            caption_model=caption_model,
            num_spotting_classes=_resolve_spotting_num_classes(spotting_train_dataset),
        )
    else:
        from model_qwen import Video2CaptionQwen

        feature_dim = args.feature_dim
        if feature_dim is None:
            feature_dim = caption_train_dataset[0][0].shape[-1]
            args.feature_dim = feature_dim

        caption_model = Video2CaptionQwen(
            input_size=feature_dim,
            vlad_k=args.vlad_k,
            window_size=args.window_size_caption,
            framerate=args.framerate,
            pool=args.pool,
            llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B"),
            lora_r=getattr(args, "lora_r", 8),
            lora_alpha=getattr(args, "lora_alpha", 16),
            lora_dropout=getattr(args, "lora_dropout", 0.05),
            weights_encoder=args.weights_encoder,
            freeze_encoder=bool(getattr(args, "freeze_encoder_joint", True)),
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
        captioning._load_optional_checkpoint(caption_model, _caption_checkpoint_path(args), source_name="joint_caption_init")
        logging.info("[joint][model] caption backbone warm-start complete")
        model = JointVideoCaptionSpot(
            caption_model=caption_model,
            num_spotting_classes=_resolve_spotting_num_classes(spotting_train_dataset),
        )

    if bool(getattr(args, "joint_warm_start_spotting_head", True)):
        model.load_spotting_head_from_checkpoint(_spotting_checkpoint_path(args))
        logging.info("[joint][model] spotting head warm-start complete")
    return model


def _move_caption_batch_to_device(batch, device):
    data_tuple = batch[0]
    lengths = batch[1]

    if isinstance(data_tuple, (list, tuple)) and len(data_tuple) == 3:
        vfeats, afeats, tokens = data_tuple
        return (vfeats.to(device), afeats.to(device), tokens.to(device), lengths)

    feats, tokens = data_tuple
    return (feats.to(device), tokens.to(device), lengths)


def _move_spotting_batch_to_device(batch, device):
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        vfeats, afeats, labels = batch
        return (vfeats.to(device), afeats.to(device), labels.to(device))

    feats, labels = batch
    return (feats.to(device), labels.to(device))


def _should_step_optimizer(step_idx, accumulation_steps, effective_steps):
    if accumulation_steps <= 1:
        return True
    return ((step_idx + 1) % accumulation_steps == 0) or ((step_idx + 1) == effective_steps)


def _run_joint_epoch(model, caption_loader, spotting_loader, optimizer, criterion_spot, args, train, accelerator=None):
    model.train(train)
    device = accelerator.device if accelerator is not None else next(model.parameters()).device
    is_main = _is_main_process(accelerator)
    smoke_limit = int(getattr(args, "smoke_steps_joint", None) or getattr(args, "smoke_steps", 0) or 0)
    accumulation_steps = max(1, int(getattr(args, "accumulation_steps_joint", None) or 1))
    lambda_caption = float(getattr(args, "joint_lambda_caption", 1.0))
    max_grad_norm = getattr(args, "max_grad_norm_joint", None)
    if max_grad_norm is None:
        max_grad_norm = getattr(args, "max_grad_norm_caption", None)
    if max_grad_norm is None:
        max_grad_norm = getattr(args, "max_grad_norm", 0.5)
    max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None

    caption_cycle = _infinite_loader(caption_loader)
    spotting_cycle = _infinite_loader(spotting_loader)
    total_steps = max(len(caption_loader), len(spotting_loader))
    effective_steps = min(total_steps, smoke_limit) if smoke_limit > 0 else total_steps
    phase_name = "train" if train else "val"
    log_prefix = f"[joint][{phase_name}]"

    if train and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    caption_loss_total = 0.0
    spotting_loss_total = 0.0
    valid_steps = 0

    iterator = range(effective_steps)

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for step in iterator:
            if _joint_should_log_step(step, effective_steps):
                logging.info(
                    "%s step=%d/%d fetching batches",
                    log_prefix,
                    step + 1,
                    effective_steps,
                )
            step_start = time.time()
            caption_batch = next(caption_cycle)
            spotting_batch = next(spotting_cycle)
            if _joint_should_log_step(step, effective_steps):
                logging.info(
                    "%s step=%d/%d batches fetched in %.2fs",
                    log_prefix,
                    step + 1,
                    effective_steps,
                    time.time() - step_start,
                )

            cap_payload = _move_caption_batch_to_device(caption_batch, device)
            if len(cap_payload) == 4:
                vfeats, afeats, tokens, lengths = cap_payload
                cap_loss = model(vfeats, afeats, tokens, lengths)
            else:
                feats, tokens, lengths = cap_payload
                cap_loss = model(feats, tokens, lengths)

            spot_payload = _move_spotting_batch_to_device(spotting_batch, device)
            if len(spot_payload) == 3:
                vfeats, afeats, labels = spot_payload
                spot_logits = model(vfeats, afeats)
            else:
                feats, labels = spot_payload
                spot_logits = model(feats)

            multiclass_targets = _to_spotting_multiclass_targets(labels)
            if getattr(criterion_spot, "expects_prob_targets", False):
                spot_targets = multiclass_targets
            else:
                spot_targets = multiclass_targets.argmax(dim=1).long()
            spot_loss = criterion_spot(spot_logits, spot_targets)

            loss = spot_loss + lambda_caption * cap_loss

            if train and optimizer is not None:
                loss_to_backprop = loss / accumulation_steps
                if accelerator is not None:
                    accelerator.backward(loss_to_backprop)
                    if _should_step_optimizer(step, accumulation_steps, effective_steps):
                        if max_grad_norm is not None:
                            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                else:
                    loss_to_backprop.backward()
                    if _should_step_optimizer(step, accumulation_steps, effective_steps):
                        if max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
            if _joint_should_log_step(step, effective_steps):
                logging.info(
                    "%s step=%d/%d done: loss=%.5f cap=%.5f spot=%.5f elapsed=%.2fs",
                    log_prefix,
                    step + 1,
                    effective_steps,
                    float(loss.detach().item()),
                    float(cap_loss.detach().item()),
                    float(spot_loss.detach().item()),
                    time.time() - step_start,
                )

            total_loss += float(loss.detach().item())
            caption_loss_total += float(cap_loss.detach().item())
            spotting_loss_total += float(spot_loss.detach().item())
            valid_steps += 1

    if valid_steps == 0:
        return {"loss": float("nan"), "caption_loss": float("nan"), "spotting_loss": float("nan")}

    if is_main:
        logging.info(
            "[joint] %s epoch stats: total_loss=%.5f caption_loss=%.5f spotting_loss=%.5f steps=%d",
            "train" if train else "val",
            total_loss / valid_steps,
            caption_loss_total / valid_steps,
            spotting_loss_total / valid_steps,
            valid_steps,
        )
    return {
        "loss": total_loss / valid_steps,
        "caption_loss": caption_loss_total / valid_steps,
        "spotting_loss": spotting_loss_total / valid_steps,
    }


def _save_joint_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    model_name,
    best_loss,
    best_caption_score,
    best_spotting_score,
    is_best_loss=False,
    is_best_caption_metric=False,
    is_best_spotting_metric=False,
    accelerator=None,
):
    phase_dir = os.path.join("models", model_name, "joint")
    os.makedirs(phase_dir, exist_ok=True)
    state_dict = accelerator.get_state_dict(model) if accelerator is not None else model.state_dict()
    state = {
        "epoch": epoch,
        "state_dict": state_dict,
        "best_loss": best_loss,
        "best_eval_score": best_caption_score,
        "best_caption_score": best_caption_score,
        "best_spotting_score": best_spotting_score,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(state, os.path.join(phase_dir, "last_checkpoint.pth.tar"))
    if is_best_loss:
        torch.save(state, os.path.join(phase_dir, "model.pth.tar"))
        torch.save(state, os.path.join(phase_dir, "best_loss.pth.tar"))
    if is_best_caption_metric:
        torch.save(state, os.path.join(phase_dir, "best_metric.pth.tar"))
        torch.save(state, os.path.join(phase_dir, "best_joint.pth.tar"))
    if is_best_spotting_metric:
        torch.save(state, os.path.join(phase_dir, "best_spotting_metric.pth.tar"))


def main(args):
    logging.info("[joint] Starting Stage 4 joint training")
    logging.info(
        "[joint] config: epochs_joint=%s eval_freq=%s smoke_steps_joint=%s accumulation_steps_joint=%s",
        getattr(args, "epochs_joint", None),
        getattr(args, "evaluation_frequency_joint", None) or getattr(args, "evaluation_frequency_caption", None) or getattr(args, "evaluation_frequency", None),
        getattr(args, "smoke_steps_joint", None) or getattr(args, "smoke_steps", None),
        getattr(args, "accumulation_steps_joint", None),
    )

    generation_config = captioning.get_caption_generation_config_from_args(args)
    logging.info("[joint] building caption loaders")
    caption_train_dataset, _, caption_train_loader, caption_valid_loader, caption_valid_metric_loader = _build_caption_datasets_and_loader(args)
    logging.info("[joint] building spotting loaders")
    (
        spotting_train_dataset,
        spotting_valid_dataset,
        spotting_valid_metric_dataset,
        spotting_train_loader,
        spotting_valid_loader,
        spotting_valid_metric_loader,
    ) = _build_spotting_datasets_and_loader(args)
    logging.info("[joint] building joint model")
    model = _build_joint_model(args, caption_train_dataset, spotting_train_dataset)
    criterion_spot = _build_spotting_criterion(args)

    joint_lr = float(getattr(args, "lr_joint", None) or getattr(args, "LR_caption", None) or args.LR)
    use_discriminative_ft = bool(getattr(args, "discriminative_ft_caption", True))
    if use_discriminative_ft:
        param_groups = captioning._build_caption_param_groups(model, args, fallback_lr=joint_lr)
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=joint_lr,
        betas=(0.9, 0.999),
        eps=1e-4,
        weight_decay=getattr(args, "weight_decay", 0.01),
    )
    epochs_joint = int(getattr(args, "epochs_joint", 5))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_joint)

    accelerator = getattr(args, "accelerator", None)
    if accelerator is not None and getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_plugin = accelerator.state.deepspeed_plugin
        ds_config = getattr(ds_plugin, "deepspeed_config", None)
        if isinstance(ds_config, dict) and ds_config.get("optimizer") is not None:
            ds_config.pop("optimizer", None)
            logging.info("[joint] removed DeepSpeed JSON optimizer to use code-defined AdamW")

    if accelerator is not None:
        logging.info("[joint] preparing model and dataloaders with accelerator")
        model, optimizer, caption_train_loader, caption_valid_loader, caption_valid_metric_loader, spotting_train_loader, spotting_valid_loader = accelerator.prepare(
            model, optimizer, caption_train_loader, caption_valid_loader, caption_valid_metric_loader, spotting_train_loader, spotting_valid_loader
        )
        logging.info("[joint] accelerator.prepare complete")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logging.info("[joint] model moved to device=%s", device)

    evaluation_frequency = int(getattr(args, "evaluation_frequency_joint", None) or getattr(args, "evaluation_frequency_caption", None) or args.evaluation_frequency)
    is_main = _is_main_process(accelerator)
    best_loss = float("inf")
    best_cider = float("-inf")
    best_spotting_map = float("-inf")
    global_step = _phase_step_offset("joint")

    for epoch in range(epochs_joint):
        epoch_id = epoch + 1
        logging.info("[joint] epoch %d/%d train begin", epoch_id, epochs_joint)
        train_stats = _run_joint_epoch(model, caption_train_loader, spotting_train_loader, optimizer, criterion_spot, args, train=True, accelerator=accelerator)
        logging.info("[joint] epoch %d/%d val begin", epoch_id, epochs_joint)
        val_stats = _run_joint_epoch(model, caption_valid_loader, spotting_valid_loader, None, criterion_spot, args, train=False, accelerator=accelerator)

        loss_validation = float(val_stats["loss"])
        is_best_loss = loss_validation < best_loss
        if is_best_loss:
            best_loss = loss_validation

        eval_model = accelerator.unwrap_model(model) if accelerator is not None else model
        if evaluation_frequency and (epoch_id % evaluation_frequency == 0 or epoch_id == epochs_joint):
            logging.info("[joint] epoch %d/%d evaluation begin", epoch_id, epochs_joint)
            caption_metrics = validate_captioning(
                caption_valid_metric_loader,
                eval_model,
                args.model_name,
                smoke_steps=getattr(args, "smoke_steps_joint", None) or getattr(args, "smoke_steps", 0),
                generation_config=generation_config,
            )
            spotting_metrics = validate_spotting_official(
                spotting_valid_metric_loader,
                eval_model,
                args.model_name,
                smoke_steps=getattr(args, "smoke_steps_joint", None) or getattr(args, "smoke_steps", 0),
            )
            logging.info("[joint] epoch %d/%d evaluation complete", epoch_id, epochs_joint)
            cider = float(caption_metrics.get("CIDEr", float("-inf")))
            spotting_map = _safe_float_metric(spotting_metrics, "a_mAP_medium", "a_mAP")
            is_best_caption_metric = cider > best_cider
            if is_best_caption_metric:
                best_cider = cider
            is_best_spotting_metric = spotting_map > best_spotting_map
            if is_best_spotting_metric:
                best_spotting_map = spotting_map
        else:
            caption_metrics = {}
            spotting_metrics = {}
            is_best_caption_metric = False
            is_best_spotting_metric = False

        if accelerator is not None:
            accelerator.wait_for_everyone()
        if is_main:
            _save_joint_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch_id,
                args.model_name,
                best_loss,
                best_cider,
                best_spotting_map,
                is_best_loss=is_best_loss,
                is_best_caption_metric=is_best_caption_metric,
                is_best_spotting_metric=is_best_spotting_metric,
                accelerator=accelerator,
            )

            log_dict = {
                "joint_loss_train": train_stats["loss"],
                "joint_loss_val": val_stats["loss"],
                "joint_caption_loss_train": train_stats["caption_loss"],
                "joint_caption_loss_val": val_stats["caption_loss"],
                "joint_spotting_loss_train": train_stats["spotting_loss"],
                "joint_spotting_loss_val": val_stats["spotting_loss"],
                "joint_best_loss_so_far": best_loss,
                "joint_best_cider_so_far": best_cider,
                "joint_best_spotting_map_so_far": best_spotting_map,
                "epoch": epoch,
                "epoch_current": epoch_id,
                "epoch_total": epochs_joint,
            }
            log_dict.update({f"{k}_joint_val": v for k, v in caption_metrics.items()})
            log_dict.update({f"{k}_joint_val": v for k, v in spotting_metrics.items()})
            if accelerator is not None:
                accelerator.log(log_dict, step=global_step)
            else:
                wandb.log(log_dict, step=global_step)

        scheduler.step()
        global_step += 1

    logging.info("[joint] Finished Stage 4 joint training")
