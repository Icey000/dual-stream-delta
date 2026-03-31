import json
import logging
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, BooleanOptionalAction
from datetime import datetime
from itertools import product

import torch
import wandb

import captioning
from dataset import CollateGPT, SoccerNetCaptions
from train import _resolve_best_checkpoint_path, validate_captioning


def _build_validation_loader(args):
    use_dual = getattr(args, "use_dual_stream", False)

    if use_dual:
        from dataset_dual import CollateGPTDual, SoccerNetCaptionsDual

        dataset = SoccerNetCaptionsDual(
            vision_root=args.SoccerNet_path,
            audio_root=args.audio_root,
            features=args.features,
            split=args.split_valid,
            version=args.version,
            framerate=args.framerate,
            window_size=args.window_size_caption,
            llm_model_path=args.llm_model_path,
        )
        collate_fn = CollateGPTDual(llm_model_path=args.llm_model_path)
    else:
        dataset = SoccerNetCaptions(
            path=args.SoccerNet_path,
            features=args.features,
            split=args.split_valid,
            version=args.version,
            framerate=args.framerate,
            window_size=args.window_size_caption,
            llm_model_path=args.llm_model_path,
        )
        collate_fn = CollateGPT(llm_model_path=args.llm_model_path)

    max_samples = int(getattr(args, "caption_valid_max_samples", 0) or 0)
    if max_samples > 0 and len(dataset) > max_samples:
        logging.info("[decoding_eval] limiting validation samples: %d -> %d", len(dataset), max_samples)
        dataset = torch.utils.data.Subset(dataset, range(max_samples))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=getattr(args, "batch_size_caption", None) or args.batch_size,
        shuffle=False,
        num_workers=args.max_num_worker,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return dataset, loader


def _resolve_dataset_for_model(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        return dataset.dataset
    return dataset


def _build_model(args, dataset):
    generation_config = captioning.get_caption_generation_config_from_args(args)
    use_dual = getattr(args, "use_dual_stream", False)

    if use_dual:
        from dual_qformer import DualVideo2CaptionLLM

        model = DualVideo2CaptionLLM(
            vocab_size=dataset.vocab_size,
            video_input_dim=getattr(args, "video_input_dim", 1024),
            audio_input_dim=getattr(args, "audio_input_dim", 512),
            hidden_dim=getattr(args, "hidden_dim", 3584),
            llm_model_path=args.llm_model_path,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
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

        feature_dim = args.feature_dim
        if feature_dim is None:
            sample = dataset[0][0]
            feature_dim = sample.shape[-1]

        model = Video2CaptionQwen(
            input_size=feature_dim,
            vlad_k=args.vlad_k,
            window_size=args.window_size_caption,
            framerate=args.framerate,
            pool=args.pool,
            llm_model_path=args.llm_model_path,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
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

    return model


def run(args):
    if getattr(args, "use_dual_stream", False) and not getattr(args, "audio_root", None):
        raise ValueError("--audio_root is required when --use_dual_stream is enabled")

    dataset, loader = _build_validation_loader(args)
    model_dataset = _resolve_dataset_for_model(dataset)
    model = _build_model(args, model_dataset)

    checkpoint_path = getattr(args, "checkpoint_path", None) or _resolve_best_checkpoint_path(
        args.model_name,
        "caption",
        use_metric_best=getattr(args, "load_best_metric_checkpoint", False),
    )
    if not os.path.exists(checkpoint_path):
        fallback_path = os.path.join("models", args.model_name, "caption", "best_caption_ce.pth.tar")
        if os.path.exists(fallback_path):
            checkpoint_path = fallback_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found for decoding_eval: {checkpoint_path}")

    wandb_run = None
    wandb_project = getattr(args, "wandb_project", None) or os.environ.get("WANDB_PROJECT")
    if wandb_project:
        wandb_mode = getattr(args, "wandb_mode", None) or os.environ.get("WANDB_MODE") or "online"
        wandb_name = getattr(args, "wandb_name", None)
        wandb_group = getattr(args, "wandb_group", None)
        wandb_tags = ["decode_sweep", "compare_ckpts"]
        if getattr(args, "checkpoint_path", None):
            wandb_tags.append(os.path.basename(checkpoint_path))
        try:
            wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_name,
                group=wandb_group,
                mode=wandb_mode,
                tags=wandb_tags,
            )
            wandb_run.config.update(
                {
                    "model_name": args.model_name,
                    "checkpoint_path": checkpoint_path,
                    "split_valid": list(args.split_valid),
                    "caption_valid_max_samples": int(getattr(args, "caption_valid_max_samples", 0) or 0),
                    "batch_size_caption": int(getattr(args, "batch_size_caption", None) or getattr(args, "batch_size", 0)),
                    "decode_num_examples": int(getattr(args, "decode_num_examples", 5)),
                    "sweep_max_new_tokens": list(getattr(args, "sweep_max_new_tokens", [])),
                    "sweep_no_repeat_ngram_size": list(getattr(args, "sweep_no_repeat_ngram_size", [])),
                    "sweep_num_beams": list(getattr(args, "sweep_num_beams", [])),
                    "sweep_temperature": list(getattr(args, "sweep_temperature", [])),
                },
                allow_val_change=True,
            )
        except Exception as exc:
            logging.warning("[decoding_eval] wandb init failed, continuing without W&B logging: %s", exc)
            wandb_run = None

    logging.info("[decoding_eval] loading checkpoint: %s", checkpoint_path)
    captioning._load_optional_checkpoint(model, checkpoint_path, source_name="decoding_eval")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    output_dir = os.path.join("models", args.model_name, "caption", "decoding_sweeps")
    os.makedirs(output_dir, exist_ok=True)

    results = []
    best_entry = None
    total_configs = (
        len(args.sweep_max_new_tokens)
        * len(args.sweep_no_repeat_ngram_size)
        * len(args.sweep_num_beams)
        * len(args.sweep_temperature)
    )
    logging.info("[decoding_eval] evaluating %d decoding configurations", total_configs)

    base_generation_config = captioning.get_caption_generation_config_from_args(args)

    for idx, values in enumerate(
        product(
            args.sweep_max_new_tokens,
            args.sweep_no_repeat_ngram_size,
            args.sweep_num_beams,
            args.sweep_temperature,
        ),
        start=1,
    ):
        max_new_tokens, no_repeat_ngram_size, num_beams, temperature = values
        generation_config = dict(base_generation_config)
        generation_config.update({
            "max_new_tokens": max_new_tokens,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "num_beams": num_beams,
            "do_sample": True,
            "temperature": temperature,
        })
        logging.info("[decoding_eval] (%d/%d) config=%s", idx, total_configs, generation_config)
        scores, examples = validate_captioning(
            loader,
            model,
            args.model_name,
            smoke_steps=getattr(args, "smoke_steps_caption", None) or getattr(args, "smoke_steps", 0),
            generation_config=generation_config,
            return_examples=getattr(args, "decode_num_examples", 5),
        )
        entry = {
            "config": generation_config,
            "metrics": scores,
            "examples": examples,
        }
        results.append(entry)

        cider = float(scores.get("CIDEr", float("-inf")))
        if best_entry is None or cider > float(best_entry["metrics"].get("CIDEr", float("-inf"))):
            best_entry = entry

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = os.path.join(output_dir, f"decode_sweep_{timestamp}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "checkpoint_path": checkpoint_path,
                "best_by_cider": best_entry,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    best_config_path = os.path.join(output_dir, "best_decode_config.json")
    if best_entry is not None:
        with open(best_config_path, "w", encoding="utf-8") as f:
            json.dump(best_entry["config"], f, indent=2, ensure_ascii=False)
        logging.info("[decoding_eval] best CIDEr=%.6f config=%s", best_entry["metrics"]["CIDEr"], best_entry["config"])

    if wandb_run is not None and best_entry is not None:
        metric_prefix = {
            f"best/{k}": v
            for k, v in best_entry["metrics"].items()
            if isinstance(v, (int, float))
        }
        metric_prefix["best/checkpoint_path"] = checkpoint_path
        metric_prefix["best/summary_path"] = summary_path
        wandb_run.log(metric_prefix)

        examples = best_entry.get("examples") or []
        if examples:
            table = wandb.Table(columns=["idx", "reference", "prediction"])
            for idx, example in enumerate(examples):
                table.add_data(idx, example.get("reference", ""), example.get("prediction", ""))
            wandb_run.log({"best/examples": table})

        wandb_run.summary["best_cider"] = float(best_entry["metrics"].get("CIDEr", float("nan")))
        wandb_run.summary["checkpoint_path"] = checkpoint_path
        wandb_run.summary["summary_path"] = summary_path
        wandb_run.finish()

    logging.info("[decoding_eval] saved results to %s", summary_path)
    logging.info("[decoding_eval] saved best config to %s", best_config_path)


def build_arg_parser():
    parser = ArgumentParser(
        description="Official caption decoding sweep on the validation split",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--SoccerNet_path", type=str, required=True)
    parser.add_argument("--audio_root", type=str, default=None)
    parser.add_argument("--features", type=str, default="baidu_soccer_embeddings.npy")
    parser.add_argument("--model_name", type=str, default="Dual-QFormer-Qwen")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--load_weights", type=str, default=None)
    parser.add_argument("--weights_encoder", type=str, default=None)
    parser.add_argument("--llm_model_path", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--split_valid", nargs="+", default=["valid"])
    parser.add_argument("--version", type=int, default=2)
    parser.add_argument("--framerate", type=int, default=1)
    parser.add_argument("--pool", type=str, default="QFormer")
    parser.add_argument("--window_size_caption", type=int, default=45)
    parser.add_argument("--feature_dim", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batch_size_caption", type=int, default=8)
    parser.add_argument("--max_num_worker", type=int, default=4)
    parser.add_argument("--caption_valid_max_samples", type=int, default=50)
    parser.add_argument("--use_dual_stream", action="store_true")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--video_input_dim", type=int, default=8576)
    parser.add_argument("--audio_input_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=3584)
    parser.add_argument("--encoder_dropout", type=float, default=0.3)
    parser.add_argument("--vlad_k", type=int, default=64)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--load_best_metric_checkpoint", action=BooleanOptionalAction, default=False)
    parser.add_argument("--caption_generation_config_json", type=str, default=None)
    parser.add_argument("--caption_max_new_tokens", type=int, default=48)
    parser.add_argument("--caption_no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--caption_num_beams", type=int, default=1)
    parser.add_argument("--caption_length_penalty", type=float, default=0.9)
    parser.add_argument("--caption_do_sample", action=BooleanOptionalAction, default=False)
    parser.add_argument("--caption_temperature", type=float, default=1.0)
    parser.add_argument("--caption_top_p", type=float, default=1.0)
    parser.add_argument("--caption_repetition_penalty", type=float, default=1.15)
    parser.add_argument("--sweep_max_new_tokens", nargs="+", type=int, default=[20, 30, 40])
    parser.add_argument("--sweep_no_repeat_ngram_size", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--sweep_num_beams", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--sweep_temperature", nargs="+", type=float, default=[0.6, 0.7, 0.8])
    parser.add_argument("--decode_num_examples", type=int, default=5)
    parser.add_argument("--smoke_steps", type=int, default=0)
    parser.add_argument("--smoke_steps_caption", type=int, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None)
    parser.add_argument("--loglevel", type=str, default="INFO")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    numeric_level = getattr(logging, args.loglevel.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="%(asctime)s [%(levelname)s] %(message)s")
    run(args)


if __name__ == "__main__":
    main()
