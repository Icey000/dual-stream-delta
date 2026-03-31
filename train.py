import logging
import os
import zipfile
import sys
import json
import time
from tqdm import tqdm
import torch
import numpy as np

# Accelerate (仅在分布式训练时使用)
try:
    from accelerate import Accelerator
    _ACCELERATE_AVAILABLE = True
except ImportError:
    _ACCELERATE_AVAILABLE = False

from sklearn.metrics import average_precision_score
from SoccerNet.Evaluation.utils import AverageMeter, getMetaDataTask
import glob
from utils import evaluate as evaluate_spotting
from SoccerNet.Evaluation.DenseVideoCaptioning import evaluate as evaluate_dvc
from nlgeval import NLGEval
from torch.nn.utils.rnn import pack_padded_sequence

import wandb

caption_scorer = NLGEval(no_glove=True, no_skipthoughts=True)

_PHASE_STEP_OFFSETS = {
    "classifying": 0,
    "caption": 1_000_000,
    "spotting": 2_000_000,
    "joint": 3_000_000,
    "rl": 4_000_000,
}

_NAN_SKIP_PATIENCE = 8
_NAN_LOG_EVERY = 20


def _sanitize_caption_text(text, max_chars=220):
    """Normalize generated caption text before metric/cache usage."""
    if text is None:
        return ""
    text = str(text).replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split()).strip()
    if text.startswith(":"):
        text = text[1:].strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()
    return text


def _phase_step_offset(phase):
    return _PHASE_STEP_OFFSETS.get(phase, 0)


def _primary_eval_metric_name(phase):
    if phase == "caption":
        return "CIDEr"
    if phase == "joint":
        return "CIDEr"
    if phase == "rl":
        return "CIDEr"
    if phase == "spotting":
        return "a_mAP_medium"
    if phase == "classifying":
        return "accuracy"
    return None


def _primary_eval_metric_value(phase, metrics):
    metric_name = _primary_eval_metric_name(phase)
    if metric_name is None or not isinstance(metrics, dict):
        return metric_name, None

    metric_value = metrics.get(metric_name)
    if metric_value is None and metric_name == "a_mAP_medium":
        metric_value = metrics.get("a_mAP")
    if metric_value is None and metric_name == "mAP-sklearn":
        metric_value = metrics.get("mAP")

    if metric_value is None:
        return metric_name, None

    try:
        return metric_name, float(metric_value)
    except (TypeError, ValueError):
        return metric_name, None


def _unwrap_dataset(dataset):
    """
    Recursively unwrap torch.utils.data wrappers (e.g. Subset) to the base dataset.
    """
    current = dataset
    visited = set()
    while hasattr(current, "dataset"):
        current_id = id(current)
        if current_id in visited:
            break
        visited.add(current_id)
        current = current.dataset
    return current


def _get_dataset_attr(dataloader, attr_name, default=None):
    dataset = getattr(dataloader, "dataset", None)
    if dataset is not None and hasattr(dataset, attr_name):
        return getattr(dataset, attr_name)

    base_dataset = _unwrap_dataset(dataset)
    if base_dataset is not None and hasattr(base_dataset, attr_name):
        return getattr(base_dataset, attr_name)

    return default


def _resolve_best_checkpoint_path(model_name, phase, use_metric_best=False):
    phase_dir = os.path.join("models", model_name, phase)
    metric_path = os.path.join(phase_dir, "best_metric.pth.tar")
    loss_path = os.path.join(phase_dir, "model.pth.tar")
    alias_map = {
        "caption": os.path.join(phase_dir, "best_caption_ce.pth.tar"),
        "joint": os.path.join(phase_dir, "best_joint.pth.tar"),
        "rl": os.path.join(phase_dir, "best_rl.pth.tar"),
    }
    alias_path = alias_map.get(phase)
    if use_metric_best:
        return metric_path if os.path.exists(metric_path) else (alias_path if alias_path and os.path.exists(alias_path) else loss_path)
    if alias_path and os.path.exists(alias_path):
        return alias_path
    return loss_path


def _add_module_prefix_if_missing(state_dict):
    return {
        (k if k.startswith("module.") else f"module.{k}"): v
        for k, v in state_dict.items()
    }


def _strip_module_prefix_if_present(state_dict):
    return {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def _load_model_state_compatible(model, state_dict, source_name):
    """
    Load state_dict with key-prefix compatibility for plain nn.Module vs DeepSpeedEngine.
    """
    attempts = []
    seen = set()

    def _append_candidate(name, cand):
        keys = tuple(cand.keys())
        if keys in seen:
            return
        seen.add(keys)
        attempts.append((name, cand))

    _append_candidate("raw", state_dict)
    _append_candidate("add_module_prefix", _add_module_prefix_if_missing(state_dict))
    _append_candidate("strip_module_prefix", _strip_module_prefix_if_present(state_dict))

    errors = []
    for variant, candidate in attempts:
        try:
            model.load_state_dict(candidate)
            if variant != "raw":
                logging.info("[%s] loaded model state using variant=%s", source_name, variant)
            return
        except Exception as exc:
            errors.append((variant, exc))

    detail = " | ".join([f"{name}: {err}" for name, err in errors])
    raise RuntimeError(f"[{source_name}] all model state_dict load attempts failed -> {detail}")


def _is_main_process(accelerator=None):
    if accelerator is not None:
        return bool(
            getattr(accelerator, "is_local_main_process", False)
            or getattr(accelerator, "is_main_process", False)
        )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0

    return True


def _count_optimizer_param_groups(optimizer_state):
    if not isinstance(optimizer_state, dict):
        return "unknown"
    param_groups = optimizer_state.get("param_groups", None)
    if not isinstance(param_groups, list):
        return "unknown"
    return len(param_groups)


def _summarize_tensor(name, tensor):
    if tensor is None or not torch.is_tensor(tensor):
        return f"{name}=<none>"

    detached = tensor.detach()
    finite = bool(torch.isfinite(detached).all().item())
    absmax = float(detached.abs().max().item()) if detached.numel() > 0 else 0.0
    return f"{name}(shape={tuple(detached.shape)}, finite={finite}, absmax={absmax:.4g})"


def _find_nonfinite_parameters(model, limit=5):
    bad_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not torch.isfinite(param.detach()).all():
            bad_params.append(name)
            if len(bad_params) >= limit:
                break
    return bad_params


def _to_spotting_binary_targets(labels):
    multiclass_targets = _to_spotting_multiclass_targets(labels)
    if multiclass_targets.shape[1] < 2:
        raise ValueError(f"Unsupported spotting target shape: {tuple(multiclass_targets.shape)}")
    bg = multiclass_targets[:, 0]
    event = multiclass_targets[:, 1:].sum(dim=1)
    return torch.stack([bg, event], dim=1)


def _to_spotting_multiclass_targets(labels):
    if not torch.is_tensor(labels):
        raise ValueError(f"Unsupported spotting labels type: {type(labels)}")

    if labels.dim() == 1:
        num_classes = int(labels.max().item()) + 1 if labels.numel() > 0 else 2
        return torch.nn.functional.one_hot(labels.long(), num_classes=num_classes).float()

    if labels.dim() != 2:
        raise ValueError(f"Unsupported spotting label shape: {tuple(labels.shape)}")

    labels = labels.float()
    row_sums = labels.sum(dim=1)
    is_simplex = (
        torch.all(labels >= -1e-6)
        and torch.all(labels <= 1.0 + 1e-6)
        and torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4)
    )
    if is_simplex:
        return labels

    class_indices = labels.argmax(dim=1).long()
    return torch.nn.functional.one_hot(class_indices, num_classes=labels.shape[1]).float()


def _to_spotting_class_indices(labels):
    return _to_spotting_multiclass_targets(labels).argmax(dim=1).long()


def _to_spotting_binary_probabilities(logits):
    probs = _to_spotting_class_probabilities(logits)
    if probs.shape[1] == 2:
        return probs
    if probs.shape[1] > 2:
        return torch.stack([probs[:, 0], probs[:, 1:].sum(dim=1)], dim=1)
    raise ValueError(f"Unsupported spotting logit shape: {tuple(logits.shape)}")


def _to_spotting_class_probabilities(logits):
    if not torch.is_tensor(logits):
        raise ValueError(f"Unsupported spotting logits type: {type(logits)}")
    if logits.dim() != 2 or logits.shape[1] < 2:
        raise ValueError(f"Unsupported spotting logit shape: {tuple(logits.shape)}")

    logits = logits.float()
    row_sums = logits.sum(dim=1)
    if (
        torch.all(logits >= -1e-6)
        and torch.all(logits <= 1.0 + 1e-6)
        and torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4)
    ):
        return logits
    return torch.nn.functional.softmax(logits, dim=1)


def _extract_spotting_outputs(output):
    if isinstance(output, dict):
        return output.get("logits"), output.get("offsets")

    if isinstance(output, (list, tuple)):
        if len(output) == 2:
            return output[0], output[1]
        if len(output) == 1:
            return output[0], None

    return output, None


def _summarize_spotting_diagnostics(target_batches, output_batches, sample_limit=8):
    if not target_batches or not output_batches:
        return None

    targets = np.concatenate(target_batches, axis=0)
    outputs = np.concatenate(output_batches, axis=0)

    target_idx = targets.argmax(axis=1)
    pred_idx = outputs.argmax(axis=1)

    sample_count = min(sample_limit, len(target_idx))
    sample_pairs = []
    for i in range(sample_count):
        sample_pairs.append(
            f"(pred={int(pred_idx[i])}, target={int(target_idx[i])}, "
            f"bg={outputs[i, 0]:.4f}, event={outputs[i, 1]:.4f})"
        )

    return {
        "target_bg_count": int((target_idx == 0).sum()),
        "target_event_count": int((target_idx == 1).sum()),
        "pred_bg_count": int((pred_idx == 0).sum()),
        "pred_event_count": int((pred_idx == 1).sum()),
        "bg_prob_mean": float(outputs[:, 0].mean()),
        "event_prob_mean": float(outputs[:, 1].mean()),
        "bg_prob_max": float(outputs[:, 0].max()),
        "event_prob_max": float(outputs[:, 1].max()),
        "sample_pairs": sample_pairs,
    }

# ============================================================================
#  对比学习辅助函数 (Contrastive Learning Helpers)
# ============================================================================

# 全局懒加载文本编码器 (Lazy-loaded text encoder)
_text_encoder = None
_text_proj = None

def _get_text_encoder(hidden_dim, device):
    """
    懒加载文本编码器。优先使用 sentence-transformers (all-MiniLM-L6-v2, 384维 -> 投影到 hidden_dim)。
    若未安装，回退到简单的 TF-IDF 风格哈希编码 + 线性投影。
    """
    global _text_encoder, _text_proj
    if _text_encoder is not None:
        return _text_encoder, _text_proj

    try:
        from sentence_transformers import SentenceTransformer
        _text_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        _text_encoder.eval()
        for p in _text_encoder.parameters():
            p.requires_grad = False
        st_dim = _text_encoder.get_sentence_embedding_dimension()
        _text_proj = torch.nn.Linear(st_dim, hidden_dim).to(device)
        logging.info(f"Contrastive: using sentence-transformers (dim={st_dim} -> {hidden_dim})")
    except ImportError:
        _text_encoder = "hash"  # 标记使用哈希方式
        _text_proj = torch.nn.Linear(512, hidden_dim).to(device)
        logging.info(f"Contrastive: sentence-transformers not found, using hash encoding (512 -> {hidden_dim})")

    return _text_encoder, _text_proj


def _encode_text_batch(caption_texts, hidden_dim, device):
    """
    将一批文本编码为 [B, hidden_dim] 的特征向量。
    """
    encoder, proj = _get_text_encoder(hidden_dim, device)

    if encoder == "hash":
        # 哈希编码 fallback: 用字符级哈希得到固定维度表示
        B = len(caption_texts)
        hash_dim = 512
        text_vecs = torch.zeros(B, hash_dim, device=device)
        for i, text in enumerate(caption_texts):
            for ch in text:
                idx = hash(ch) % hash_dim
                text_vecs[i, idx] += 1.0
            # L2 归一化
            norm = text_vecs[i].norm() + 1e-8
            text_vecs[i] = text_vecs[i] / norm
        text_features = proj(text_vecs)
    else:
        # sentence-transformers 编码 (在 CPU 上计算，再 to device)
        with torch.no_grad():
            embeddings = encoder.encode(list(caption_texts), convert_to_tensor=True)
        text_features = proj(embeddings.to(device))

    return text_features


def _contrastive_loss(vis_features, text_features, temperature=0.07):
    """
    InfoNCE 对比学习 Loss (双向)。
    
    Args:
        vis_features:  [B, D] - 视频特征 (来自 Q-Former pooling)
        text_features: [B, D] - 文本特征 (来自文本编码器)
        temperature:   温度参数
    
    Returns:
        loss: scalar - 双向 InfoNCE loss 的平均值
    """
    # L2 归一化
    vis_norm = torch.nn.functional.normalize(vis_features, dim=-1)
    text_norm = torch.nn.functional.normalize(text_features, dim=-1)

    # 相似度矩阵 [B, B]
    logits = torch.matmul(vis_norm, text_norm.t()) / temperature

    # 对角线为正样本
    labels = torch.arange(logits.shape[0], device=logits.device)

    # 双向 InfoNCE: video->text + text->video
    loss_v2t = torch.nn.functional.cross_entropy(logits, labels)
    loss_t2v = torch.nn.functional.cross_entropy(logits.t(), labels)

    return (loss_v2t + loss_t2v) / 2.0


def train(phase,
          dataloader,
          model,
          criterion,
          optimizer,
          epoch,
          epoch_total=None,
          train=True,
          accumulation_steps=1,
          max_grad_norm=0.5,
          smoke_steps=0,
          accelerator=None,
          return_details=False):
    """
    One epoch train/val loop.

    Supports:
      - classifying / spotting: CrossEntropy on class indices
      - caption: model returns scalar loss (Qwen LoRA)
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cls_losses = AverageMeter()
    offset_losses = AverageMeter()

    if train:
        model.train()
    else:
        model.eval()

    device = accelerator.device if accelerator is not None else next(model.parameters()).device
    accumulation_steps = max(1, int(accumulation_steps))
    valid_batches = 0
    nan_batches = 0
    consecutive_nan_batches = 0
    nonfinite_reported = False
    is_main_process = _is_main_process(accelerator)
    rank = getattr(accelerator, "process_index", 0) if accelerator is not None else 0
    epoch_start = time.time()
    processed_samples = 0
    epoch_label = f"{epoch}/{epoch_total}" if epoch_total is not None else str(epoch)
    smoke_limit = int(smoke_steps) if smoke_steps is not None else 0
    if smoke_limit < 0:
        smoke_limit = 0
    clip_max_norm = None
    if max_grad_norm is not None:
        clip_max_norm = float(max_grad_norm)
        if clip_max_norm <= 0:
            clip_max_norm = None

    end = time.time()
    if train and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    spotting_diag_targets = []
    spotting_diag_outputs = []
    spotting_offset_positive_count = 0
    spotting_offset_total_count = 0

    trainable_params = []
    if train and optimizer is not None and clip_max_norm is not None:
        trainable_params = [p for p in model.parameters() if p.requires_grad]

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        with tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            disable=not is_main_process,
            dynamic_ncols=True,
        ) as t:
            for i, batch in t:
                if smoke_limit > 0 and i >= smoke_limit:
                    if is_main_process:
                        logging.info(
                            f"[{phase}] {'train' if train else 'val'} smoke_steps={smoke_limit} reached at epoch={epoch_label}, stopping early."
                        )
                    break
                data_time.update(time.time() - end)
                batch_summaries = []

                if phase == "caption":
                    data_tuple = batch[0]
                    lengths = batch[1]

                    if isinstance(data_tuple, (list, tuple)) and len(data_tuple) == 3:
                        vfeats, afeats, tokens = data_tuple
                        vfeats = vfeats.to(device)
                        afeats = afeats.to(device)
                        tokens = tokens.to(device)
                        batch_summaries = [
                            _summarize_tensor("video", vfeats),
                            _summarize_tensor("audio", afeats),
                            _summarize_tensor("tokens", tokens),
                        ]
                        processed_samples += int(tokens.shape[0])
                        loss = model(vfeats, afeats, tokens, lengths)
                    else:
                        feats, tokens = data_tuple
                        feats = feats.to(device)
                        tokens = tokens.to(device)
                        batch_summaries = [
                            _summarize_tensor("video", feats),
                            _summarize_tensor("tokens", tokens),
                        ]
                        processed_samples += int(tokens.shape[0])
                        loss = model(feats, tokens, lengths)

                else:
                    offset_targets = None
                    offset_masks = None
                    if phase == "spotting":
                        if isinstance(batch, (list, tuple)) and len(batch) == 5:
                            vfeats, afeats, labels, offset_targets, offset_masks = batch
                            vfeats = vfeats.to(device)
                            afeats = afeats.to(device)
                            labels = labels.to(device)
                            offset_targets = offset_targets.to(device).float()
                            offset_masks = offset_masks.to(device).float()
                            batch_summaries = [
                                _summarize_tensor("video", vfeats),
                                _summarize_tensor("audio", afeats),
                                _summarize_tensor("labels", labels),
                                _summarize_tensor("offset_targets", offset_targets),
                                _summarize_tensor("offset_masks", offset_masks),
                            ]
                            output = model(vfeats, afeats)
                        elif isinstance(batch, (list, tuple)) and len(batch) == 4:
                            feats, labels, offset_targets, offset_masks = batch
                            feats = feats.to(device)
                            labels = labels.to(device)
                            offset_targets = offset_targets.to(device).float()
                            offset_masks = offset_masks.to(device).float()
                            batch_summaries = [
                                _summarize_tensor("video", feats),
                                _summarize_tensor("labels", labels),
                                _summarize_tensor("offset_targets", offset_targets),
                                _summarize_tensor("offset_masks", offset_masks),
                            ]
                            output = model(feats)
                        elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                            vfeats, afeats, labels = batch
                            vfeats = vfeats.to(device)
                            afeats = afeats.to(device)
                            labels = labels.to(device)
                            batch_summaries = [
                                _summarize_tensor("video", vfeats),
                                _summarize_tensor("audio", afeats),
                                _summarize_tensor("labels", labels),
                            ]
                            output = model(vfeats, afeats)
                        else:
                            feats, labels = batch
                            feats = feats.to(device)
                            labels = labels.to(device)
                            batch_summaries = [
                                _summarize_tensor("video", feats),
                                _summarize_tensor("labels", labels),
                            ]
                            output = model(feats)
                    else:
                        if isinstance(batch, (list, tuple)) and len(batch) == 4:
                            vfeats, afeats, labels, _ = batch
                            vfeats = vfeats.to(device)
                            afeats = afeats.to(device)
                            labels = labels.to(device)
                            batch_summaries = [
                                _summarize_tensor("video", vfeats),
                                _summarize_tensor("audio", afeats),
                                _summarize_tensor("labels", labels),
                            ]
                            output = model(vfeats, afeats)
                        elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                            vfeats, afeats, labels = batch
                            vfeats = vfeats.to(device)
                            afeats = afeats.to(device)
                            labels = labels.to(device)
                            batch_summaries = [
                                _summarize_tensor("video", vfeats),
                                _summarize_tensor("audio", afeats),
                                _summarize_tensor("labels", labels),
                            ]
                            output = model(vfeats, afeats)
                        else:
                            feats, labels = batch
                            feats = feats.to(device)
                            labels = labels.to(device)
                            batch_summaries = [
                                _summarize_tensor("video", feats),
                                _summarize_tensor("labels", labels),
                            ]
                            output = model(feats)

                    if phase == "spotting":
                        logits, predicted_offsets = _extract_spotting_outputs(output)
                        multiclass_targets = _to_spotting_multiclass_targets(labels)
                        if getattr(criterion, "expects_prob_targets", False):
                            labels = multiclass_targets
                        else:
                            labels = multiclass_targets.argmax(dim=1).long()
                        cls_loss = criterion(logits, labels)
                        loss = cls_loss
                        cls_losses.update(cls_loss.item(), 1)

                        if predicted_offsets is not None and offset_targets is not None and offset_masks is not None:
                            offset_loss_weight = float(
                                getattr(
                                    model,
                                    "spotting_center_regression_weight",
                                    getattr(getattr(model, "module", None), "spotting_center_regression_weight", 1.0),
                                )
                            )
                            predicted_offsets = predicted_offsets.view(-1).float()
                            offset_targets = offset_targets.view(-1).float()
                            offset_masks = offset_masks.view(-1).float()
                            positive_mask = offset_masks > 0
                            positive_count = int(positive_mask.sum().item())
                            spotting_offset_positive_count += positive_count
                            spotting_offset_total_count += int(offset_masks.numel())
                            if positive_count > 0:
                                per_example_offset_loss = torch.nn.functional.smooth_l1_loss(
                                    predicted_offsets,
                                    offset_targets,
                                    reduction="none",
                                )
                                # 只对正样本位置计算 offset，并按正样本数做 epoch 级平均。
                                offset_loss = (
                                    per_example_offset_loss * offset_masks
                                ).sum() / offset_masks.sum().clamp_min(1.0)
                                loss = loss + offset_loss_weight * offset_loss
                                offset_losses.update(offset_loss.item(), positive_count)
                            else:
                                offset_losses.update(0.0, 0)
                        if not train:
                            spotting_diag_targets.append(
                                _to_spotting_binary_targets(multiclass_targets).detach().cpu().numpy()
                            )
                            spotting_diag_outputs.append(
                                _to_spotting_binary_probabilities(logits.detach()).cpu().numpy()
                            )
                    else:
                        if labels.dim() > 1:
                            labels = labels.argmax(dim=1)
                        labels = labels.long()
                        loss = criterion(output, labels)
                    processed_samples += int(labels.shape[0])

                if not torch.isfinite(loss.detach()):
                    nan_batches += 1
                    consecutive_nan_batches += 1

                    if not nonfinite_reported:
                        bad_params = _find_nonfinite_parameters(model)
                        logging.error(
                            f"First non-finite loss detected: phase={phase}, epoch={epoch_label}, batch={i}, rank={rank}, "
                            f"loss={loss.detach().float().cpu().item()}, inputs=[{'; '.join(batch_summaries)}], "
                            f"nonfinite_params={bad_params if bad_params else 'none'}"
                        )
                        nonfinite_reported = True
                    elif nan_batches % _NAN_LOG_EVERY == 0 or consecutive_nan_batches == _NAN_SKIP_PATIENCE:
                        logging.error(
                            f"Non-finite loss persists: phase={phase}, epoch={epoch_label}, total_nan_batches={nan_batches}, "
                            f"consecutive_nan_batches={consecutive_nan_batches}, latest_batch={i}, rank={rank}"
                        )

                    if consecutive_nan_batches >= _NAN_SKIP_PATIENCE:
                        raise RuntimeError(
                            f"Aborting because loss stayed NaN/Inf for {consecutive_nan_batches} consecutive batches "
                            f"(phase={phase}, epoch={epoch_label}, latest_batch={i}, rank={rank}). "
                            f"This usually means the model/optimizer state is corrupted rather than a single bad batch."
                        )
                    continue
                elif consecutive_nan_batches > 0:
                    logging.warning(
                        f"Recovered from {consecutive_nan_batches} consecutive non-finite batches "
                        f"at phase={phase}, epoch={epoch_label}, batch={i}, rank={rank}"
                    )
                    consecutive_nan_batches = 0

                if train and optimizer is not None:
                    loss_to_backprop = loss / accumulation_steps
                    if accelerator is not None:
                        accelerator.backward(loss_to_backprop)
                        if accelerator.sync_gradients:
                            if trainable_params:
                                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=clip_max_norm)
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                    else:
                        loss_to_backprop.backward()
                        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                            if trainable_params:
                                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=clip_max_norm)
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)

                losses.update(loss.item(), 1)
                valid_batches += 1
                batch_time.update(time.time() - end)
                end = time.time()

                desc = f'{"Train" if train else "Val"} ({phase}) [epoch {epoch_label}]: '
                desc += f'Time {batch_time.avg:.3f}s (it:{batch_time.val:.3f}s) '
                desc += f'Data {data_time.avg:.3f}s (it:{data_time.val:.3f}s) '
                desc += f'Loss {losses.avg:.5f}'
                if is_main_process:
                    t.set_description(desc)

    epoch_seconds = max(time.time() - epoch_start, 1e-9)
    if valid_batches == 0:
        if return_details:
            return {
                "loss": float("nan"),
                "iter_time_avg": float("nan"),
                "data_time_avg": float("nan"),
                "samples_per_sec": 0.0,
                "valid_batches": 0,
                "processed_samples": int(processed_samples),
                "epoch_seconds": epoch_seconds,
                "spotting_diag": None,
            }
        return float("nan")

    if return_details:
        spotting_diag = None
        if phase == "spotting" and not train:
            spotting_diag = _summarize_spotting_diagnostics(spotting_diag_targets, spotting_diag_outputs)
        return {
            "loss": float(losses.avg),
            "cls_loss": float(cls_losses.avg) if getattr(cls_losses, "count", 0) > 0 else None,
            "offset_loss": float(offset_losses.avg) if getattr(offset_losses, "count", 0) > 0 else None,
            "iter_time_avg": float(batch_time.avg),
            "data_time_avg": float(data_time.avg),
            "samples_per_sec": float(processed_samples / epoch_seconds),
            "valid_batches": int(valid_batches),
            "processed_samples": int(processed_samples),
            "epoch_seconds": float(epoch_seconds),
            "spotting_diag": spotting_diag,
            "spotting_offset_positive_count": int(spotting_offset_positive_count),
            "spotting_offset_total_count": int(spotting_offset_total_count),
        }

    return losses.avg


def trainer(phase, train_loader,
            val_loader,
            val_metric_loader,
            model,
            optimizer,
            scheduler,
            criterion,
            model_name,
            max_epochs=1000,
            evaluation_frequency=20,
            accumulation_steps=1,
            max_grad_norm=0.5,
            smoke_steps=0,
            continue_training=False,
            accelerator=None,
            use_metric_best_checkpoint=False):

    logging.info("start training")
    logging.info(
        f"[{phase}] grad_clip_max_norm="
        f"{max_grad_norm if max_grad_norm is not None else 'disabled'}"
    )
    if smoke_steps and int(smoke_steps) > 0:
        logging.info(f"[{phase}] smoke mode enabled: first {int(smoke_steps)} batches per epoch")

    # ----- Accelerate 辅助函数 -----
    # is_main: 主进程 (rank 0) 才做保存 / wandb 日志
    is_main = (accelerator is None) or accelerator.is_main_process

    best_loss = 9e99
    best_eval_score = float("-inf")
    start_epoch = 0

    os.makedirs(os.path.join("models", model_name, phase), exist_ok=True)
    best_model_path = os.path.join("models", model_name, phase, "model.pth.tar")
    best_loss_path = os.path.join("models", model_name, phase, "best_loss.pth.tar")
    best_metric_path = os.path.join("models", model_name, phase, "best_metric.pth.tar")
    last_checkpoint_path = os.path.join("models", model_name, phase, "last_checkpoint.pth.tar")
    phase_alias_best = {
        "caption": "best_caption_ce.pth.tar",
    }.get(phase)

    # ---- 完美断点加载逻辑 ----
    if continue_training:
        resume_candidates = [
            ("last checkpoint", last_checkpoint_path),
            ("best checkpoint fallback", best_model_path),
            ("best loss fallback", best_loss_path),
        ]
        if use_metric_best_checkpoint:
            resume_candidates.insert(1, ("best metric fallback", best_metric_path))
        resumed = False
        for label, ckpt_path in resume_candidates:
            if not os.path.exists(ckpt_path):
                continue
            logging.info(f"[resume] Loading {label} from {ckpt_path}...")
            try:
                checkpoint = torch.load(ckpt_path, map_location="cpu")
                _load_model_state_compatible(model, checkpoint["state_dict"], source_name=f"resume:{phase}")

                start_epoch = int(checkpoint.get("epoch", 0))
                best_loss = checkpoint.get("best_loss", 9e99)
                best_eval_score = float(checkpoint.get("best_eval_score", float("-inf")))

                optimizer_loaded = False
                optimizer_state = checkpoint.get("optimizer")
                if optimizer_state is not None:
                    try:
                        optimizer.load_state_dict(optimizer_state)
                        optimizer_loaded = True
                    except Exception as oe:
                        ckpt_groups = _count_optimizer_param_groups(optimizer_state)
                        cur_groups = len(getattr(optimizer, "param_groups", []) or [])
                        logging.warning(
                            f"[resume] Optimizer state restore failed for {phase}: {type(oe).__name__}: {oe}. "
                            f"checkpoint param_groups={ckpt_groups}, current param_groups={cur_groups}. "
                            "Continuing with a freshly initialized optimizer; model weights and epoch were restored, "
                            "but Adam moments / momentum state were reset."
                        )
                else:
                    optimizer_loaded = True
                    logging.info(
                        f"[resume] No optimizer state found for {phase}; using freshly initialized optimizer."
                    )

                scheduler_state = checkpoint.get("scheduler")
                if scheduler is not None and scheduler_state is not None:
                    try:
                        scheduler.load_state_dict(scheduler_state)
                        if optimizer_loaded:
                            logging.info(f"[resume] Scheduler state restored for {phase}.")
                        else:
                            logging.warning(
                                f"[resume] Scheduler state restored for {phase} even though optimizer restore failed. "
                                "The LR schedule should continue from the checkpointed epoch, but optimizer moments were reset."
                            )
                    except Exception as se:
                        logging.warning(
                            f"[resume] Scheduler state restore failed for {phase}: {type(se).__name__}: {se}. "
                            "Continuing with the newly constructed scheduler."
                        )

                logging.info(
                    f"[resume] Resuming {phase} from checkpoint epoch {start_epoch}; "
                    f"next epoch will be {start_epoch + 1}/{max_epochs} "
                    f"(source={os.path.basename(ckpt_path)})"
                )
                resumed = True
                del checkpoint
                import gc
                gc.collect()
                break
            except Exception as e:
                if phase == "spotting":
                    logging.warning(
                        "[resume] Skipping %s for spotting because it is incompatible with the current binary spotting head: %s",
                        label,
                        e,
                    )
                else:
                    logging.error(f"[resume] Failed to load {label}: {e}")

        if continue_training and not resumed:
            logging.warning(f"[resume] No valid checkpoint could be restored for phase={phase}; starting from epoch 0.")

    if start_epoch >= max_epochs:
        logging.info(f"Phase {phase} has already completed {start_epoch} / {max_epochs} epochs. Skipping training.")
        return

    # ---- wandb step offset: 续训时从正确的 global_step 开始 ----
    # 每个 epoch 的 step 数 = dataloader 长度 (近似；用于保证续训曲线连续)
    if smoke_steps and int(smoke_steps) > 0:
        steps_per_epoch = min(len(train_loader), int(smoke_steps))
    else:
        steps_per_epoch = len(train_loader)
    global_step = _phase_step_offset(phase) + start_epoch * steps_per_epoch
    logging.info(f"[{phase}] training epochs will run from {start_epoch + 1}/{max_epochs} to {max_epochs}/{max_epochs}")

    for epoch in range(start_epoch, max_epochs):
        current_epoch = epoch + 1
        epoch_label = f"{current_epoch}/{max_epochs}"
        # train for one epoch
        train_stats = train(phase, train_loader, model, criterion,
                            optimizer, current_epoch, max_epochs, train=True,
                            accumulation_steps=accumulation_steps,
                            max_grad_norm=max_grad_norm,
                            smoke_steps=smoke_steps,
                            accelerator=accelerator,
                            return_details=True)
        loss_training = train_stats["loss"]

        # evaluate on validation set
        val_stats = train(phase, val_loader, model, criterion, optimizer, current_epoch, max_epochs, train=False,
                          max_grad_norm=max_grad_norm,
                          smoke_steps=smoke_steps,
                          accelerator=accelerator,
                          return_details=True)
        loss_validation = val_stats["loss"]

        if phase == "caption" and is_main:
            logging.info(
                "[caption][epoch %s] train(iter=%.3fs,data=%.3fs,samples/s=%.2f,batches=%d,samples=%d) "
                "val(iter=%.3fs,data=%.3fs,samples/s=%.2f,batches=%d,samples=%d)",
                epoch_label,
                train_stats["iter_time_avg"],
                train_stats["data_time_avg"],
                train_stats["samples_per_sec"],
                train_stats["valid_batches"],
                train_stats["processed_samples"],
                val_stats["iter_time_avg"],
                val_stats["data_time_avg"],
                val_stats["samples_per_sec"],
                val_stats["valid_batches"],
                val_stats["processed_samples"],
            )
        elif phase == "spotting" and is_main:
            spotting_diag = val_stats.get("spotting_diag")
            train_offset_total = max(1, int(train_stats.get("spotting_offset_total_count") or 0))
            val_offset_total = max(1, int(val_stats.get("spotting_offset_total_count") or 0))
            logging.info(
                "[spotting][epoch %s] train(loss=%.5f, cls=%.5f, offset=%s, pos=%d/%d) "
                "val(loss=%.5f, cls=%.5f, offset=%s, pos=%d/%d)",
                epoch_label,
                loss_training,
                train_stats.get("cls_loss") if train_stats.get("cls_loss") is not None else float("nan"),
                "n/a" if train_stats.get("offset_loss") is None else f"{train_stats.get('offset_loss'):.5f}",
                int(train_stats.get("spotting_offset_positive_count") or 0),
                train_offset_total,
                loss_validation,
                val_stats.get("cls_loss") if val_stats.get("cls_loss") is not None else float("nan"),
                "n/a" if val_stats.get("offset_loss") is None else f"{val_stats.get('offset_loss'):.5f}",
                int(val_stats.get("spotting_offset_positive_count") or 0),
                val_offset_total,
            )
            if spotting_diag is not None:
                logging.info(
                    "[spotting][epoch %s] val_diag target_bg=%d target_event=%d pred_bg=%d pred_event=%d "
                    "bg_prob_mean=%.4f event_prob_mean=%.4f samples=%s",
                    epoch_label,
                    spotting_diag["target_bg_count"],
                    spotting_diag["target_event_count"],
                    spotting_diag["pred_bg_count"],
                    spotting_diag["pred_event_count"],
                    spotting_diag["bg_prob_mean"],
                    spotting_diag["event_prob_mean"],
                    spotting_diag["sample_pairs"],
                )

        # ---- 模型保存：多卡模式下只有主进程保存 ----
        state_dict = None
        if accelerator is not None:
            accelerator.wait_for_everyone()
            # ZeRO-3 下必须通过 accelerate 收集完整参数
            state_dict = accelerator.get_state_dict(model)
        else:
            # 单卡/非 accelerate 模式
            state_dict = model.state_dict()

        if is_main:
            state = {
                'epoch': current_epoch,
                'state_dict': state_dict,
                'best_loss': best_loss,
                'best_eval_score': best_eval_score,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
            }
            os.makedirs(os.path.join("models", model_name), exist_ok=True)

            # 保存每个 epoch 的 last_checkpoint，防止中途断电
            torch.save(state, last_checkpoint_path)
            logging.info(f"[{phase}] saved last checkpoint at epoch {epoch_label}")

        # remember best prec@1 and save checkpoint
        is_better = loss_validation < best_loss
        best_loss = min(loss_validation, best_loss)

        # Always keep a dedicated best-loss checkpoint for safety / resume fallback.
        if is_better and is_main:
            torch.save(state, best_loss_path)
            torch.save(state, best_model_path)
            if phase_alias_best is not None:
                torch.save(state, os.path.join("models", model_name, phase, phase_alias_best))

        global_step += steps_per_epoch

        # Test the model on the validation set
        if evaluation_frequency and current_epoch % evaluation_frequency == 0:
            if phase == "caption":
                test = validate_captioning
            elif phase == "spotting":
                test = validate_spotting_official
            elif phase == "classifying":
                test = validate_classifying
            # 多卡推理时需要 unwrap 模型
            eval_model = accelerator.unwrap_model(model) if accelerator is not None else model
            performance_validation = test(
                val_metric_loader,
                eval_model,
                model_name,
                smoke_steps=smoke_steps)

            logging.info("Validation performance at epoch " +
                         epoch_label + " -> " + str(performance_validation))

            primary_metric_name, primary_metric_value = _primary_eval_metric_value(
                phase, performance_validation
            )
            if primary_metric_name is not None and primary_metric_value is not None:
                if primary_metric_value > best_eval_score:
                    best_eval_score = primary_metric_value
                    if is_main:
                        torch.save(state, best_metric_path)

            log_dict = {**{
                f"loss_train_{phase}": loss_training,
                f"loss_val_{phase}": loss_validation,
                "epoch" : epoch,
                "epoch_current": current_epoch,
                "epoch_total": max_epochs,
                "best_loss_so_far": best_loss,
                }, **{f"{k}_val" : v for k, v in performance_validation.items()}}
            if phase == "caption":
                log_dict.update({
                    "caption_train_iter_time_s": train_stats["iter_time_avg"],
                    "caption_train_data_time_s": train_stats["data_time_avg"],
                    "caption_train_samples_per_sec": train_stats["samples_per_sec"],
                    "caption_val_iter_time_s": val_stats["iter_time_avg"],
                    "caption_val_data_time_s": val_stats["data_time_avg"],
                    "caption_val_samples_per_sec": val_stats["samples_per_sec"],
                })
            if phase == "spotting":
                log_dict.update({
                    "spotting_loss_cls_train": train_stats.get("cls_loss"),
                    "spotting_loss_cls_val": val_stats.get("cls_loss"),
                    "spotting_loss_offset_train": train_stats.get("offset_loss"),
                    "spotting_loss_offset_val": val_stats.get("offset_loss"),
                    "spotting_offset_positive_count_train": train_stats.get("spotting_offset_positive_count"),
                    "spotting_offset_positive_count_val": val_stats.get("spotting_offset_positive_count"),
                })
            if primary_metric_name is not None and primary_metric_value is not None:
                log_dict[f"{primary_metric_name}_best_so_far"] = best_eval_score
            
            if accelerator is not None:
                accelerator.log(log_dict, step=global_step)
            else:
                if is_main:
                    wandb.log(log_dict, step=global_step)

            if is_main:
                if 'state' in dir():
                    torch.save(state, os.path.join("models", model_name, phase, f"model_{current_epoch}.pth.tar"))
        else:
            log_dict = {
                f"loss_train_{phase}": loss_training,
                f"loss_val_{phase}": loss_validation,
                "epoch" : epoch,
                "epoch_current": current_epoch,
                "epoch_total": max_epochs,
                "best_loss_so_far": best_loss,
            }
            if phase == "caption":
                log_dict.update({
                    "caption_train_iter_time_s": train_stats["iter_time_avg"],
                    "caption_train_data_time_s": train_stats["data_time_avg"],
                    "caption_train_samples_per_sec": train_stats["samples_per_sec"],
                    "caption_val_iter_time_s": val_stats["iter_time_avg"],
                    "caption_val_data_time_s": val_stats["data_time_avg"],
                    "caption_val_samples_per_sec": val_stats["samples_per_sec"],
                })
            if phase == "spotting":
                log_dict.update({
                    "spotting_loss_cls_train": train_stats.get("cls_loss"),
                    "spotting_loss_cls_val": val_stats.get("cls_loss"),
                    "spotting_loss_offset_train": train_stats.get("offset_loss"),
                    "spotting_loss_offset_val": val_stats.get("offset_loss"),
                    "spotting_offset_positive_count_train": train_stats.get("spotting_offset_positive_count"),
                    "spotting_offset_positive_count_val": val_stats.get("spotting_offset_positive_count"),
                })
            if accelerator is not None:
                accelerator.log(log_dict, step=global_step)
            else:
                if is_main:
                    wandb.log(log_dict, step=global_step)

        # Reduce LR on Plateau after patience reached
        # prevLR = optimizer.param_groups[0]['lr']
        # scheduler.step(loss_validation)
        # currLR = optimizer.param_groups[0]['lr']
        # if (currLR is not prevLR and scheduler.num_bad_epochs == 0):
        #     logging.info("Plateau Reached!")

        # if (prevLR < 2 * scheduler.eps and
        #         scheduler.num_bad_epochs >= scheduler.patience):
        #     logging.info(
        #         "Plateau Reached and no more reduction -> Exiting Loop")
        #     break
        if scheduler is not None:
            scheduler.step()
        model.train()

def validate_spotting(dataloader, model, model_name, smoke_steps=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()
    device = next(model.parameters()).device

    end = time.time()
    all_labels = []
    all_outputs = []
    show_progress = _is_main_process()
    smoke_limit = int(smoke_steps) if smoke_steps is not None else 0
    if smoke_limit < 0:
        smoke_limit = 0
    with tqdm(enumerate(dataloader), total=len(dataloader), disable=not show_progress, dynamic_ncols=True) as t:
        for i, batch in t:
            if smoke_limit > 0 and i >= smoke_limit:
                if show_progress:
                    logging.info(f"[spotting] validate smoke_steps={smoke_limit} reached, stopping early.")
                break
            # measure data loading time
            data_time.update(time.time() - end)

            if len(batch) == 5:
                vfeats, afeats, labels, _, _ = batch
                vfeats = vfeats.to(device)
                afeats = afeats.to(device)
                output = model(vfeats, afeats)
            elif len(batch) == 4:
                feats, labels, _, _ = batch
                feats = feats.to(device)
                output = model(feats)
            elif len(batch) == 3:
                # 双流: (vfeats, afeats, labels)
                vfeats, afeats, labels = batch
                vfeats = vfeats.to(device)
                afeats = afeats.to(device)
                output = model(vfeats, afeats)
            else:
                feats, labels = batch
                feats = feats.to(device)
                output = model(feats)
            
            multiclass_targets = _to_spotting_multiclass_targets(labels)
            all_labels.append(multiclass_targets.detach().cpu().numpy())

            logits, _ = _extract_spotting_outputs(output)
            output = _to_spotting_class_probabilities(logits)
            all_outputs.append(output.cpu().detach().numpy())

            batch_time.update(time.time() - end)
            end = time.time()

            desc = f'Test (spot.): '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            if show_progress:
                t.set_description(desc)

    labels_np = np.concatenate(all_labels)
    outputs_np = np.concatenate(all_outputs)
    num_classes = int(_get_dataset_attr(dataloader, "num_classes", 0) or 0)
    if num_classes <= 0:
        # probabilities include background at channel 0
        num_classes = max(0, int(outputs_np.shape[1]) - 1)

    AP = []
    for i in range(1, num_classes + 1):
        AP.append(average_precision_score(labels_np[:, i], outputs_np[:, i]))

    mAP = np.mean(AP)

    diag = _summarize_spotting_diagnostics(
        [
            _to_spotting_binary_targets(torch.from_numpy(x)).numpy()
            for x in all_labels
        ],
        [
            _to_spotting_binary_probabilities(torch.from_numpy(x)).numpy()
            for x in all_outputs
        ],
    )
    if diag is not None and show_progress:
        logging.info(
            "[spotting][metric] target_bg=%d target_event=%d pred_bg=%d pred_event=%d "
            "bg_prob_mean=%.4f event_prob_mean=%.4f samples=%s",
            diag["target_bg_count"],
            diag["target_event_count"],
            diag["pred_bg_count"],
            diag["pred_event_count"],
            diag["bg_prob_mean"],
            diag["event_prob_mean"],
            diag["sample_pairs"],
        )

    return {"mAP-sklearn" : mAP}


def validate_spotting_official(dataloader, model, model_name, smoke_steps=0):
    """Official SoccerNet spotting validation (same metric family as test)."""
    results = test_spotting(
        dataloader,
        model=model,
        model_name=model_name,
        smoke_steps=smoke_steps,
    )
    if results is None:
        return {}
    return results

def validate_classifying(dataloader, model, model_name, smoke_steps=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()
    device = next(model.parameters()).device

    end = time.time()
    correct_predictions = 0.0
    total_predictions = 0.0
    smoke_limit = int(smoke_steps) if smoke_steps is not None else 0
    if smoke_limit < 0:
        smoke_limit = 0
    with torch.no_grad():
        show_progress = _is_main_process()
        with tqdm(enumerate(dataloader), total=len(dataloader), disable=not show_progress, dynamic_ncols=True) as t:
            for i, batch in t:
                if smoke_limit > 0 and i >= smoke_limit:
                    if show_progress:
                        logging.info(f"[classifying] validate smoke_steps={smoke_limit} reached, stopping early.")
                    break
                # measure data loading time
                data_time.update(time.time() - end)

                if len(batch) == 4:
                    vfeats, afeats, labels, _ = batch
                    vfeats = vfeats.to(device)
                    afeats = afeats.to(device)
                    labels = labels.to(device)
                    output = model(vfeats, afeats)
                elif len(batch) == 3:
                    # 双流: (vfeats, afeats, labels)
                    vfeats, afeats, labels = batch
                    vfeats = vfeats.to(device)
                    afeats = afeats.to(device)
                    labels = labels.to(device)
                    output = model(vfeats, afeats)
                else:
                    feats, labels = batch
                    feats = feats.to(device)
                    labels = labels.to(device)
                    output = model(feats)

                _, predicted = torch.max(output.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted.cpu().detach() == labels.cpu()).sum().item()


                batch_time.update(time.time() - end)
                end = time.time()

                desc = f'Test (cls): '
                desc += f'Time {batch_time.avg:.3f}s '
                desc += f'(it:{batch_time.val:.3f}s) '
                desc += f'Data:{data_time.avg:.3f}s '
                desc += f'(it:{data_time.val:.3f}s) '
                if show_progress:
                    t.set_description(desc)

    return {"accuracy" : correct_predictions/total_predictions}

def test_spotting(dataloader, model, model_name, save_predictions=True, NMS_window=30, NMS_threshold=0.5, smoke_steps=0):
    dataset_split = _get_dataset_attr(dataloader, "split", [])
    split = "_".join(dataset_split)
    output_folder = f"outputs/{split}"
    output_results = os.path.join("models", model_name, output_folder)
    

    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()
    device = next(model.parameters()).device

    dataset_version = _get_dataset_attr(dataloader, "version")
    _, _, _, inv_dict = getMetaDataTask("caption", "SoccerNet", dataset_version)

    end = time.time()
    show_progress = _is_main_process()
    smoke_limit = int(smoke_steps) if smoke_steps is not None else 0
    if smoke_limit < 0:
        smoke_limit = 0
    write_raw_predictions = False

    def _forward_half_predictions(feat_half, audio_half=None):
        class_scores = []
        offset_scores = []
        for b in range(int(np.ceil(len(feat_half) / BS))):
            start_frame = BS * b
            end_frame = BS * (b + 1) if BS * (b + 1) < len(feat_half) else len(feat_half)
            feat = feat_half[start_frame:end_frame].to(device)
            if audio_half is not None:
                afeat = audio_half[start_frame:end_frame].to(device)
                output = model(feat, afeat)
            else:
                output = model(feat)

            logits, offsets = _extract_spotting_outputs(output)
            probs = torch.nn.functional.softmax(logits, dim=1)
            class_scores.append(probs[:, 1:].cpu().detach().numpy())
            if offsets is not None:
                offset_scores.append(offsets.view(-1).cpu().detach().numpy())

        class_scores = np.concatenate(class_scores, axis=0)
        offset_scores = np.concatenate(offset_scores, axis=0) if offset_scores else None
        return class_scores, offset_scores

    def _build_prediction_entry(frame_index, confidence, class_index, half_index, framerate):
        seconds = int((frame_index // framerate) % 60)
        minutes = int((frame_index // framerate) // 60)
        return {
            "gameTime": f"{half_index} - {int(minutes):02d}:{int(seconds):02d}",
            "label": inv_dict[class_index],
            "position": str(int((frame_index / framerate) * 1000)),
            "half": str(half_index),
            "confidence": str(confidence),
        }

    with tqdm(enumerate(dataloader), total=len(dataloader), disable=not show_progress, dynamic_ncols=True) as t:
        for i, batch_data in t:
            if smoke_limit > 0 and i >= smoke_limit:
                if show_progress:
                    logging.info(f"[spotting] test smoke_steps={smoke_limit} reached, stopping early.")
                break
            data_time.update(time.time() - end)

            # 兼容双流 (7元素) 和单流 (5元素) 的整场比赛级 batch
            if len(batch_data) == 7:
                # 双流: (game_ID, vfeat_h1, vfeat_h2, afeat_h1, afeat_h2, label_h1, label_h2)
                game_ID, feat_half1, feat_half2, audio_half1, audio_half2, label_half1, label_half2 = batch_data
                is_dual = True
            elif len(batch_data) == 5:
                # 单流: (game_ID, feat_h1, feat_h2, label_h1, label_h2)
                game_ID, feat_half1, feat_half2, label_half1, label_half2 = batch_data
                is_dual = False
            else:
                raise ValueError(
                    f"test_spotting expects a full-game spotting dataset batch with 5 or 7 elements, "
                    f"but received {len(batch_data)}. "
                    "If you are validating clip-level spotting data, call validate_spotting() "
                    "or pass SoccerNetClipsTesting/SoccerNetClipsTestingDual instead."
                )

            # Batch size of 1
            game_ID = game_ID[0]
            feat_half1 = feat_half1.squeeze(0)
            label_half1 = label_half1.float().squeeze(0)
            feat_half2 = feat_half2.squeeze(0)
            label_half2 = label_half2.float().squeeze(0)
            if is_dual:
                audio_half1 = audio_half1.squeeze(0)
                audio_half2 = audio_half2.squeeze(0)

            BS = 256
            if is_dual:
                timestamp_long_half_1, offset_half_1 = _forward_half_predictions(feat_half1, audio_half1)
                timestamp_long_half_2, offset_half_2 = _forward_half_predictions(feat_half2, audio_half2)
            else:
                timestamp_long_half_1, offset_half_1 = _forward_half_predictions(feat_half1)
                timestamp_long_half_2, offset_half_2 = _forward_half_predictions(feat_half2)
            if offset_half_1 is not None or offset_half_2 is not None:
                write_raw_predictions = True

            batch_time.update(time.time() - end)
            end = time.time()

            desc = f'Test (spot.): '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            if show_progress:
                t.set_description(desc)



            def get_spot_from_NMS(Input, window=60, thresh=0.0):

                detections_tmp = np.copy(Input)
                indexes = []
                MaxValues = []
                while(np.max(detections_tmp) >= thresh):

                    # Get the max remaining index and value
                    max_value = np.max(detections_tmp)
                    max_index = np.argmax(detections_tmp)
                    MaxValues.append(max_value)
                    indexes.append(max_index)
                    # detections_NMS[max_index,i] = max_value

                    nms_from = int(np.maximum(-(window/2)+max_index,0))
                    nms_to = int(np.minimum(max_index+int(window/2), len(detections_tmp)))
                    detections_tmp[nms_from:nms_to] = -1

                return np.transpose([indexes, MaxValues])

            framerate = _get_dataset_attr(dataloader, "framerate")
            dataset_num_classes = int(_get_dataset_attr(dataloader, "num_classes", 0) or 0)
            if dataset_num_classes <= 0:
                dataset_num_classes = int(timestamp_long_half_1.shape[1])
            window_size_frame = float(_get_dataset_attr(dataloader, "window_size_frame", 1.0) or 1.0)
            get_spot = get_spot_from_NMS

            json_data = dict()
            json_data["UrlLocal"] = game_ID
            json_data["predictions"] = list()
            json_data_raw = None
            if write_raw_predictions:
                json_data_raw = dict()
                json_data_raw["UrlLocal"] = game_ID
                json_data_raw["predictions"] = list()

            for half, (timestamp, offsets) in enumerate(
                [
                    (timestamp_long_half_1, offset_half_1),
                    (timestamp_long_half_2, offset_half_2),
                ]
            ):
                for l in range(dataset_num_classes):
                    spots = get_spot(
                        timestamp[:, l], window=NMS_window*framerate, thresh=NMS_threshold) # l = 0 which is out[:, 1:][:, 0]
                    for spot in spots:
                        frame_index = int(spot[0])
                        confidence = float(spot[1])
                        corrected_frame_index = frame_index
                        if offsets is not None:
                            corrected_frame_index = int(np.round(frame_index + offsets[frame_index] * window_size_frame))
                            corrected_frame_index = int(np.clip(corrected_frame_index, 0, timestamp.shape[0] - 1))

                        json_data["predictions"].append(
                            _build_prediction_entry(corrected_frame_index, confidence, l, half + 1, framerate)
                        )
                        if json_data_raw is not None:
                            json_data_raw["predictions"].append(
                                _build_prediction_entry(frame_index, confidence, l, half + 1, framerate)
                            )
            
            json_data["predictions"] = sorted(json_data["predictions"], key=lambda x: (int(x["half"]), int(x["position"])))
            if save_predictions:
                os.makedirs(os.path.join("models", model_name, output_folder, game_ID), exist_ok=True)
                with open(os.path.join("models", model_name, output_folder, game_ID, "results_spotting.json"), 'w') as output_file:
                    json.dump(json_data, output_file, indent=4)
                if json_data_raw is not None:
                    json_data_raw["predictions"] = sorted(
                        json_data_raw["predictions"],
                        key=lambda x: (int(x["half"]), int(x["position"])),
                    )
                    with open(os.path.join("models", model_name, output_folder, game_ID, "results_spotting_raw.json"), 'w') as output_file:
                        json.dump(json_data_raw, output_file, indent=4)

    if split == "challenge": 
        print("Visit eval.ai to evalaute performances on Challenge set")
        return None
    
    dataset_path = _get_dataset_attr(dataloader, "path")
    dataset_split = _get_dataset_attr(dataloader, "split")
    dataset_version = _get_dataset_attr(dataloader, "version")
    framerate = _get_dataset_attr(dataloader, "framerate")

    tight = evaluate_spotting(SoccerNet_path=dataset_path, 
                Predictions_path=output_results,
                split=dataset_split,
                prediction_file="results_spotting.json", 
                version=dataset_version, 
                framerate=framerate, metric="tight")
    
    loose = evaluate_spotting(SoccerNet_path=dataset_path, 
                Predictions_path=output_results,
                split=dataset_split,
                prediction_file="results_spotting.json", 
                version=dataset_version, 
                framerate=framerate, metric="loose")
    
    medium = evaluate_spotting(SoccerNet_path=dataset_path, 
                Predictions_path=output_results,
                split=dataset_split,
                prediction_file="results_spotting.json", 
                version=dataset_version, 
                framerate=framerate, metric="medium")

    tight = {f"{k}_tight" : v for k, v in tight.items() if v!= None}
    loose = {f"{k}_loose" : v for k, v in loose.items() if v!= None}
    medium = {f"{k}_medium" : v for k, v in medium.items() if v!= None}

    results = {**tight, **loose, **medium}

    if write_raw_predictions:
        raw_tight = evaluate_spotting(SoccerNet_path=dataset_path,
                    Predictions_path=output_results,
                    split=dataset_split,
                    prediction_file="results_spotting_raw.json",
                    version=dataset_version,
                    framerate=framerate, metric="tight")

        raw_loose = evaluate_spotting(SoccerNet_path=dataset_path,
                    Predictions_path=output_results,
                    split=dataset_split,
                    prediction_file="results_spotting_raw.json",
                    version=dataset_version,
                    framerate=framerate, metric="loose")

        raw_medium = evaluate_spotting(SoccerNet_path=dataset_path,
                    Predictions_path=output_results,
                    split=dataset_split,
                    prediction_file="results_spotting_raw.json",
                    version=dataset_version,
                    framerate=framerate, metric="medium")

        raw_results = {
            **{f"raw_{k}_tight": v for k, v in raw_tight.items() if v is not None},
            **{f"raw_{k}_loose": v for k, v in raw_loose.items() if v is not None},
            **{f"raw_{k}_medium": v for k, v in raw_medium.items() if v is not None},
        }
        results.update(raw_results)

    return results

@torch.no_grad()
def validate_captioning(dataloader, model, model_name, smoke_steps=0, generation_config=None, return_examples=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()
    device = next(model.parameters()).device

    end = time.time()
    all_labels = []
    all_outputs = []
    examples = []
    
    show_progress = _is_main_process()
    smoke_limit = int(smoke_steps) if smoke_steps is not None else 0
    if smoke_limit < 0:
        smoke_limit = 0
    with tqdm(dataloader, disable=not show_progress, dynamic_ncols=True) as t:
        for i, batch in enumerate(t):
            if smoke_limit > 0 and i >= smoke_limit:
                if show_progress:
                    logging.info(f"[caption] validate smoke_steps={smoke_limit} reached, stopping early.")
                break
            # measure data loading time
            data_time.update(time.time() - end)

            # 兼容双流和单流 batch 格式
            data_tuple = batch[0]
            lengths = batch[1]
            mask = batch[2]
            caption_or = batch[3]
            cap_id = batch[4]

            if isinstance(data_tuple, (list, tuple)) and len(data_tuple) == 3:
                # 双流: (vfeats, afeats, caption)
                vfeats, afeats, caption = data_tuple
                vfeats = vfeats.to(device)
                afeats = afeats.to(device)
                output = [
                    model.sample(vfeats[idx], afeats[idx], generation_config=generation_config)
                    for idx in range(vfeats.shape[0])
                ]
            else:
                # 单流: (feats, caption)
                feats, caption = data_tuple
                feats = feats.to(device)
                output = [
                    model.sample(feats[idx], generation_config=generation_config)
                    for idx in range(feats.shape[0])
                ]
            
            output = [_sanitize_caption_text(x) for x in output]
            all_outputs.extend(output)
            all_labels.extend([_sanitize_caption_text(x) for x in caption_or])
            if return_examples and len(examples) < int(return_examples):
                for reference, prediction in zip(caption_or, output):
                    if len(examples) >= int(return_examples):
                        break
                    examples.append({
                        "reference": _sanitize_caption_text(reference),
                        "prediction": prediction,
                    })
            batch_time.update(time.time() - end)
            end = time.time()

            desc = f'Test (cap): '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            if show_progress:
                t.set_description(desc)

    all_labels = [_sanitize_caption_text(x) for x in all_labels]
    all_outputs = [_sanitize_caption_text(x) for x in all_outputs]
    scores = caption_scorer.compute_metrics(ref_list=[all_labels,], hyp_list=all_outputs)
    if return_examples:
        return scores, examples
    return scores

@torch.no_grad()
def test_captioning(dataloader, model, model_name, output_filename = "results_dense_captioning.json", input_filename="results_spotting.json", smoke_steps=0, generation_config=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()
    device = next(model.parameters()).device

    end = time.time()
    all_outputs = []
    all_index = []

    dataset_split = _get_dataset_attr(dataloader, "split", [])
    split = "_".join(dataset_split)
    output_folder = f"outputs/{split}"
    output_results = os.path.join("models", model_name, f"results_dense_captioning_{split}.zip")

    show_progress = _is_main_process()
    smoke_limit = int(smoke_steps) if smoke_steps is not None else 0
    if smoke_limit < 0:
        smoke_limit = 0
    with tqdm(dataloader, disable=not show_progress, dynamic_ncols=True) as t:
        for i, batch in enumerate(t):
            if smoke_limit > 0 and i >= smoke_limit:
                if show_progress:
                    logging.info(f"[caption] test smoke_steps={smoke_limit} reached, stopping early.")
                break
            # measure data loading time
            data_time.update(time.time() - end)

            # 兼容双流和单流: 双流返回 (vfeats, afeats, game_id, cap_id), 单流返回 (feats, game_id, cap_id)
            if len(batch) == 4:
                # 双流 PredictionCaptionsDual
                vfeats, afeats, game_id, cap_id = batch
                vfeats = vfeats.to(device)
                afeats = afeats.to(device)
                output = [
                    model.sample(vfeats[idx], afeats[idx], generation_config=generation_config)
                    for idx in range(vfeats.shape[0])
                ]
            else:
                # 单流 PredictionCaptions
                feats, game_id, cap_id = batch
                feats = feats.to(device)
                output = [
                    model.sample(feats[idx], generation_config=generation_config)
                    for idx in range(feats.shape[0])
                ]
            
            output = [_sanitize_caption_text(x) for x in output]
            all_outputs.extend(output)
            all_index.extend([(i.item(), j.item()) for i, j in zip(game_id, cap_id)])

            batch_time.update(time.time() - end)
            end = time.time()

            desc = f'Test (dense_caption): '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            if show_progress:
                t.set_description(desc)
    
    #store output
    captions = dict(zip(all_index, all_outputs))
    skipped_games = 0
    list_games = _get_dataset_attr(dataloader, "listGames", [])
    for game_id, game in enumerate(list_games):
        path = os.path.join("models", model_name, output_folder, game, input_filename)
        with open(path, 'r') as pred_file:
            preds = json.load(pred_file)
        # Skip games with no spotting predictions to avoid UnboundLocalError
        # in SoccerNet evaluator (evaluate_detection iterates over empty list)
        if len(preds["predictions"]) == 0:
            logging.warning(f"Game '{game}' has 0 spotting predictions - inserting a dummy prediction to avoid evaluation crash.")
            skipped_games += 1
            # Insert a harmless dummy prediction to bypass the SoccerNet library bug
            preds["predictions"].append({
                "gameTime": "1 - 00:00",
                "label": "Corner",
                "position": "0",
                "half": "1",
                "confidence": "0.0",
                "comment": ""
            })
            
        for caption_id, annotation in enumerate(preds["predictions"]):
            annotation["comment"] = captions.get((game_id, caption_id), "")
        with open(os.path.join("models", model_name, output_folder, game, output_filename), 'w') as output_file:
            json.dump(preds, output_file, indent=4)
    if skipped_games > 0:
        logging.warning(f"Skipped {skipped_games} games with empty predictions during captioning output.")
    
    def zipResults(zip_path, target_dir, filename="results_spotting.json"):
            rootlen = len(target_dir) + 1
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipobj:
                for base, dirs, files in os.walk(target_dir):
                    for file in files:
                        if file == filename:
                            fn = os.path.join(base, file)
                            # 统一使用 / 作为 zip 内部路径分隔符，避免 Windows 平台问题
                            arcname = fn[rootlen:].replace(os.sep, '/')
                            zipobj.write(fn, arcname)
    
    zipResults(zip_path = output_results,
            target_dir = os.path.join("models", model_name, output_folder),
            filename=output_filename)

    if split == "challenge": 
        print("Visit eval.ai to evalaute performances on Challenge set")
        return None
    
    dataset_path = _get_dataset_attr(dataloader, "path")
    dataset_split = _get_dataset_attr(dataloader, "split")
    dataset_version = _get_dataset_attr(dataloader, "version")
    tight = evaluate_dvc(SoccerNet_path=dataset_path, Predictions_path=output_results, split=dataset_split, version=dataset_version, prediction_file=output_filename, window_size=5, include_SODA=False)
    loose = evaluate_dvc(SoccerNet_path=dataset_path, Predictions_path=output_results, split=dataset_split, version=dataset_version, prediction_file=output_filename, window_size=30, include_SODA=False)

    results = {**{f"{k}_tight" : v for k, v in tight.items()}, **{f"{k}_loose" : v for k, v in loose.items()}}

    return results
