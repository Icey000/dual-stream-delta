import logging
import os
import torch
import torch.nn.functional as F
import wandb

import captioning
from joint_training import _build_caption_datasets_and_loader
from train import _is_main_process, _phase_step_offset, validate_captioning


def _sanitize(text):
    if text is None:
        return ""
    text = str(text).replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split()).strip()
    if text.startswith(":"):
        text = text[1:].strip()
    return text


class CiderRewardScorer:
    def __init__(self):
        try:
            from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
            from pycocoevalcap.cider.cider import Cider
        except Exception as exc:
            raise RuntimeError(
                "SCST reward backend requires pycocoevalcap (PTBTokenizer + Cider). "
                "Please install it in the training environment before running --stage rl."
            ) from exc

        self.tokenizer = PTBTokenizer()
        self.cider = Cider()
        self._fallback_warned = False

    def score_batch(self, references, hypotheses):
        if len(references) != len(hypotheses):
            logging.warning(
                "[rl] reference/hypothesis size mismatch: %d vs %d; truncating to min length",
                len(references),
                len(hypotheses),
            )
        pairs = list(zip(references, hypotheses))
        if len(pairs) == 0:
            return []
        gts = {}
        res = {}
        for idx, (ref, hyp) in enumerate(pairs):
            ref_text = _sanitize(ref) or "<empty>"
            hyp_text = _sanitize(hyp) or "<empty>"
            gts[idx] = [{"caption": ref_text}]
            res[idx] = [{"caption": hyp_text}]
        try:
            gts = self.tokenizer.tokenize(gts)
            res = self.tokenizer.tokenize(res)
            _, scores = self.cider.compute_score(gts, res)
            return [float(x) for x in scores]
        except Exception as exc:
            if not self._fallback_warned:
                logging.warning("[rl] CIDEr scorer fallback to zero reward due to scorer failure: %s", exc)
                self._fallback_warned = True
            return [0.0 for _ in pairs]


def _caption_init_checkpoint_path(args):
    if str(getattr(args, "rl_init_stage", "caption")).strip().lower() == "joint":
        candidates = [
            os.path.join("models", args.model_name, "joint", "best_joint.pth.tar"),
            os.path.join("models", args.model_name, "joint", "best_metric.pth.tar"),
            os.path.join("models", args.model_name, "joint", "model.pth.tar"),
        ]
    else:
        candidates = [
            os.path.join("models", args.model_name, "caption", "best_caption_ce.pth.tar"),
            os.path.join("models", args.model_name, "caption", "best_metric.pth.tar"),
            os.path.join("models", args.model_name, "caption", "model.pth.tar"),
        ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def _build_caption_model(args, train_dataset):
    use_dual = getattr(args, "use_dual_stream", False)
    generation_config = captioning.get_caption_generation_config_from_args(args)

    if use_dual:
        from dual_qformer import DualVideo2CaptionLLM

        model = DualVideo2CaptionLLM(
            vocab_size=train_dataset.vocab_size,
            video_input_dim=getattr(args, "video_input_dim", 1024),
            audio_input_dim=getattr(args, "audio_input_dim", 512),
            llm_model_path=getattr(args, "llm_model_path", "Qwen/Qwen2.5-7B"),
            lora_r=getattr(args, "lora_r", 8),
            lora_alpha=getattr(args, "lora_alpha", 16),
            lora_dropout=getattr(args, "lora_dropout", 0.05),
            weights=args.load_weights,
            weights_encoder=args.weights_encoder,
            freeze_encoder=True,
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

        feature_dim = args.feature_dim
        if feature_dim is None:
            feature_dim = train_dataset[0][0].shape[-1]
            args.feature_dim = feature_dim

        model = Video2CaptionQwen(
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
            freeze_encoder=True,
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
    return model


def _load_initial_checkpoint(model, checkpoint_path, source_name):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[{source_name}] checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        if key.startswith("caption_model."):
            key = key[len("caption_model."):]
        normalized[key] = value
    captioning._load_state_dict_with_compact_mismatch(model, normalized, source_name=source_name)


def _configure_rl_trainable(model):
    trainable_names = []
    for name, param in model.named_parameters():
        param.requires_grad = False
        if "lora_" in name or name.startswith("proj.") or name.startswith("proj_norm.") or ".proj." in name or ".proj_norm." in name:
            param.requires_grad = True
            trainable_names.append(name)
    for param in model.encoder.parameters():
        param.requires_grad = False
    return trainable_names


def _should_step_optimizer(step_idx, accumulation_steps, effective_steps):
    if accumulation_steps <= 1:
        return True
    return ((step_idx + 1) % accumulation_steps == 0) or ((step_idx + 1) == effective_steps)


def _decode_generated_tokens(model, generated_token_ids):
    texts = model.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
    return [_sanitize(text) for text in texts]


def _extract_generated_token_ids(sequences, scores):
    num_steps = len(scores)
    if num_steps <= 0:
        return sequences[:, :0]
    if sequences.shape[1] >= num_steps:
        return sequences[:, -num_steps:]
    return sequences


def _compute_sequence_mean_logprob_with_grad(model, batch_inputs, token_ids):
    use_dual = len(batch_inputs) == 2
    if use_dual:
        prefix_embeds = model.build_prefix_embeds(batch_inputs[0], batch_inputs[1])
    else:
        prefix_embeds = model.build_prefix_embeds(batch_inputs[0])
    if token_ids.shape[1] == 0:
        return prefix_embeds.sum(dim=(1, 2)).float() * 0.0

    token_embeds = model.llm.get_input_embeddings()(token_ids).to(prefix_embeds.dtype)
    input_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

    prefix_mask = torch.ones(
        prefix_embeds.shape[:2],
        dtype=torch.long,
        device=prefix_embeds.device,
    )
    pad_token_id = model.tokenizer.pad_token_id
    if pad_token_id is None:
        token_mask = torch.ones_like(token_ids, dtype=torch.long, device=token_ids.device)
    else:
        token_mask = token_ids.ne(int(pad_token_id)).long()
    attention_mask = torch.cat([prefix_mask, token_mask], dim=1)

    outputs = model.llm(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        return_dict=True,
        use_cache=False,
    )
    logits = outputs.logits.float()

    prefix_len = prefix_embeds.shape[1]
    start_pos = max(prefix_len - 1, 0)
    end_pos = start_pos + token_ids.shape[1]
    token_logits = logits[:, start_pos:end_pos, :]
    token_logprobs = F.log_softmax(token_logits, dim=-1)
    chosen_logprobs = token_logprobs.gather(2, token_ids.unsqueeze(-1)).squeeze(-1)

    mask = torch.ones_like(chosen_logprobs, dtype=torch.float32)
    if pad_token_id is not None:
        mask = mask * token_ids.ne(int(pad_token_id)).float()
    eos_token_id = model.tokenizer.eos_token_id
    if eos_token_id is not None:
        eos_hits = token_ids.eq(int(eos_token_id))
        for row_idx in range(token_ids.shape[0]):
            eos_positions = torch.nonzero(eos_hits[row_idx], as_tuple=False)
            if eos_positions.numel() > 0:
                eos_pos = int(eos_positions[0].item())
                if eos_pos + 1 < token_ids.shape[1]:
                    mask[row_idx, eos_pos + 1 :] = 0.0
    token_count = mask.sum(dim=1).clamp_min(1.0)
    return (chosen_logprobs * mask).sum(dim=1) / token_count


def _ensure_scalar_loss(loss_tensor):
    """DeepSpeed backward expects a scalar tensor."""
    if not torch.is_tensor(loss_tensor):
        loss_tensor = torch.as_tensor(loss_tensor, dtype=torch.float32)
    if loss_tensor.ndim > 0:
        loss_tensor = loss_tensor.mean()
    return loss_tensor


def _sample_model_outputs(model, batch_inputs, sample_generation_config, greedy_generation_config):
    was_training = model.training
    model.eval()
    use_dual = len(batch_inputs) == 2
    with torch.no_grad():
        if use_dual:
            vfeats, afeats = batch_inputs
            sample_outputs = model.generate_sequences(
                vfeats,
                afeats,
                generation_config=sample_generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
            greedy_outputs = model.generate_sequences(
                vfeats,
                afeats,
                generation_config=greedy_generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
        else:
            feats = batch_inputs[0]
            sample_outputs = model.generate_sequences(
                feats,
                generation_config=sample_generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
            greedy_outputs = model.generate_sequences(
                feats,
                generation_config=greedy_generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
    if was_training:
        model.train()

    sampled_token_ids = _extract_generated_token_ids(sample_outputs.sequences, sample_outputs.scores)
    greedy_token_ids = _extract_generated_token_ids(greedy_outputs.sequences, greedy_outputs.scores)
    sampled_texts = _decode_generated_tokens(model, sampled_token_ids)
    greedy_texts = _decode_generated_tokens(model, greedy_token_ids)
    return sampled_texts, greedy_texts, sampled_token_ids


def _move_batch_to_device(batch, device):
    data_tuple = batch[0]
    if isinstance(data_tuple, (list, tuple)) and len(data_tuple) == 3:
        vfeats, afeats, _ = data_tuple
        return (vfeats.to(device), afeats.to(device))
    feats, _ = data_tuple
    return (feats.to(device),)


def _save_rl_checkpoint(model, optimizer, scheduler, epoch, model_name, best_cider, best_loss, is_best_metric=False, accelerator=None):
    phase_dir = os.path.join("models", model_name, "rl")
    os.makedirs(phase_dir, exist_ok=True)
    state_dict = accelerator.get_state_dict(model) if accelerator is not None else model.state_dict()
    state = {
        "epoch": epoch,
        "state_dict": state_dict,
        "best_eval_score": best_cider,
        "best_loss": best_loss,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(state, os.path.join(phase_dir, "last_checkpoint.pth.tar"))
    if is_best_metric:
        torch.save(state, os.path.join(phase_dir, "best_metric.pth.tar"))
        torch.save(state, os.path.join(phase_dir, "best_rl.pth.tar"))
        torch.save(state, os.path.join(phase_dir, "model.pth.tar"))


def main(args):
    logging.info("[rl] Starting Stage 5 SCST fine-tuning")
    reward_name = str(getattr(args, "rl_reward", "cider")).strip().lower()
    if reward_name != "cider":
        raise ValueError(f"Unsupported rl_reward='{reward_name}'. Only 'cider' is implemented in Stage 5 v1.")

    train_dataset, _, train_loader, _, valid_metric_loader = _build_caption_datasets_and_loader(args)
    model = _build_caption_model(args, train_dataset)
    _load_initial_checkpoint(model, _caption_init_checkpoint_path(args), source_name="rl_init")
    trainable_names = _configure_rl_trainable(model)
    if not trainable_names:
        raise RuntimeError("[rl] no trainable parameters found after applying the LoRA + proj/proj_norm filter")
    logging.info("[rl] trainable parameter groups are restricted to LoRA + proj/proj_norm")
    logging.info("[rl] trainable parameter sample: %s", trainable_names[:12])

    reward_scorer = CiderRewardScorer()
    rl_lr = float(getattr(args, "lr_rl", None) or getattr(args, "LR_caption", None) or args.LR)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=rl_lr,
        betas=(0.9, 0.999),
        eps=1e-4,
        weight_decay=getattr(args, "weight_decay", 0.01),
    )
    epochs_rl = int(getattr(args, "epochs_rl", 3))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs_rl))
    rl_weight = float(getattr(args, "rl_weight", 1.0))
    accelerator = getattr(args, "accelerator", None)

    if accelerator is not None and getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_plugin = accelerator.state.deepspeed_plugin
        ds_config = getattr(ds_plugin, "deepspeed_config", None)
        if isinstance(ds_config, dict) and ds_config.get("optimizer") is not None:
            ds_config.pop("optimizer", None)
            logging.info("[rl] removed DeepSpeed JSON optimizer to use code-defined AdamW")

    if accelerator is not None:
        model, optimizer, train_loader, valid_metric_loader = accelerator.prepare(model, optimizer, train_loader, valid_metric_loader)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    is_main = _is_main_process(accelerator)
    global_step = _phase_step_offset("rl")
    evaluation_frequency = int(getattr(args, "evaluation_frequency_rl", None) or getattr(args, "evaluation_frequency_caption", None) or args.evaluation_frequency)
    smoke_limit = int(getattr(args, "smoke_steps_rl", None) or getattr(args, "smoke_steps", 0) or 0)
    accumulation_steps = max(1, int(getattr(args, "accumulation_steps_rl", None) or 1))
    max_grad_norm = getattr(args, "max_grad_norm_rl", None)
    if max_grad_norm is None:
        max_grad_norm = getattr(args, "max_grad_norm_caption", None)
    if max_grad_norm is None:
        max_grad_norm = getattr(args, "max_grad_norm", 0.5)
    max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None
    eval_generation_config = captioning.get_rl_eval_generation_config_from_args(args)

    sample_generation_config = dict(eval_generation_config)
    sample_generation_config.update({
        "do_sample": True,
        "num_beams": 1,
        "temperature": float(getattr(args, "rl_sample_temperature", 0.7)),
        "top_p": float(getattr(args, "rl_sample_top_p", 0.9)),
        "max_new_tokens": int(getattr(args, "rl_sample_max_new_tokens", sample_generation_config.get("max_new_tokens", 30))),
    })
    greedy_generation_config = dict(sample_generation_config)
    greedy_generation_config.update({
        "do_sample": False,
        "num_beams": 1,
        "temperature": 1.0,
        "top_p": 1.0,
    })

    best_cider = float("-inf")
    best_loss = float("inf")
    effective_steps = min(len(train_loader), smoke_limit) if smoke_limit > 0 else len(train_loader)
    if effective_steps <= 0:
        raise RuntimeError("[rl] training dataloader is empty; cannot run SCST")

    for epoch in range(epochs_rl):
        epoch_id = epoch + 1
        model.train()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        loss_sum = 0.0
        reward_sum = 0.0
        baseline_sum = 0.0
        advantage_sum = 0.0
        valid_steps = 0

        for step_idx, batch in enumerate(train_loader):
            if smoke_limit > 0 and step_idx >= smoke_limit:
                break

            device = accelerator.device if accelerator is not None else next(model.parameters()).device
            batch_inputs = _move_batch_to_device(batch, device)
            references = batch[3]

            sampled_texts, greedy_texts, sampled_token_ids = _sample_model_outputs(
                model,
                batch_inputs,
                sample_generation_config=sample_generation_config,
                greedy_generation_config=greedy_generation_config,
            )
            mean_logprob = _compute_sequence_mean_logprob_with_grad(model, batch_inputs, sampled_token_ids)
            sampled_rewards = torch.tensor(reward_scorer.score_batch(references, sampled_texts), device=device, dtype=torch.float32)
            greedy_rewards = torch.tensor(reward_scorer.score_batch(references, greedy_texts), device=device, dtype=torch.float32)
            advantage = sampled_rewards - greedy_rewards
            loss = -(advantage.detach() * mean_logprob)
            loss = _ensure_scalar_loss(loss) * rl_weight

            loss_to_backprop = _ensure_scalar_loss(loss / accumulation_steps)
            if not (torch.is_tensor(loss_to_backprop) and loss_to_backprop.numel() == 1 and loss_to_backprop.grad_fn is not None):
                raise RuntimeError(
                    f"[rl] invalid backward loss: shape={tuple(loss_to_backprop.shape)} "
                    f"numel={loss_to_backprop.numel()} grad_fn={loss_to_backprop.grad_fn}"
                )
            if accelerator is not None:
                accelerator.backward(loss_to_backprop)
                if _should_step_optimizer(step_idx, accumulation_steps, effective_steps):
                    if max_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss_to_backprop.backward()
                if _should_step_optimizer(step_idx, accumulation_steps, effective_steps):
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            loss_sum += float(loss.detach().item())
            reward_sum += float(sampled_rewards.mean().item())
            baseline_sum += float(greedy_rewards.mean().item())
            advantage_sum += float(advantage.mean().item())
            valid_steps += 1

        if valid_steps == 0:
            raise RuntimeError("[rl] no valid RL steps executed; check dataset and smoke settings")

        train_loss = loss_sum / valid_steps
        best_loss = min(best_loss, train_loss)

        if evaluation_frequency and (epoch_id % evaluation_frequency == 0 or epoch_id == epochs_rl):
            eval_model = accelerator.unwrap_model(model) if accelerator is not None else model
            caption_metrics = validate_captioning(
                valid_metric_loader,
                eval_model,
                args.model_name,
                smoke_steps=getattr(args, "smoke_steps_rl", None) or getattr(args, "smoke_steps", 0),
                generation_config=eval_generation_config,
            )
            cider = float(caption_metrics.get("CIDEr", float("-inf")))
            is_best_metric = cider > best_cider
            if is_best_metric:
                best_cider = cider
        else:
            caption_metrics = {}
            is_best_metric = False

        if accelerator is not None:
            accelerator.wait_for_everyone()
        if is_main:
            _save_rl_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch_id,
                args.model_name,
                best_cider,
                best_loss,
                is_best_metric=is_best_metric,
                accelerator=accelerator,
            )
            log_dict = {
                "rl_loss_train": train_loss,
                "rl_reward_sample_mean": reward_sum / valid_steps,
                "rl_reward_baseline_mean": baseline_sum / valid_steps,
                "rl_advantage_mean": advantage_sum / valid_steps,
                "rl_best_cider_so_far": best_cider,
                "epoch": epoch,
                "epoch_current": epoch_id,
                "epoch_total": epochs_rl,
            }
            log_dict.update({f"{k}_rl_val": v for k, v in caption_metrics.items()})
            if accelerator is not None:
                accelerator.log(log_dict, step=global_step)
            else:
                wandb.log(log_dict, step=global_step)

        scheduler.step()
        global_step += 1

    logging.info("[rl] Finished Stage 5 SCST fine-tuning")
