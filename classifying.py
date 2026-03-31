import os
import logging
from datetime import datetime
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch

from dataset import SoccerNetClassification
from model import Video2Classifcation
from train import trainer

import wandb

def main(args):

    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(arg.rjust(15) + " : " + str(getattr(args, arg)))

    use_dual = getattr(args, "use_dual_stream", False)

    if use_dual:
        # ==================== 双流模式 ====================
        from dataset_dual import SoccerNetClassificationDual
        from dual_qformer import DualVideo2Classification

        audio_root = args.audio_root
        assert audio_root is not None, "Dual-stream classifying requires --audio_root"

        if not args.test_only:
            dataset_Train = SoccerNetClassificationDual(
                vision_root=args.SoccerNet_path, audio_root=audio_root,
                features=args.features, split=args.split_train,
                version=args.version, framerate=args.framerate,
                window_size=args.window_size_caption,
            )
            dataset_Valid = SoccerNetClassificationDual(
                vision_root=args.SoccerNet_path, audio_root=audio_root,
                features=args.features, split=args.split_valid,
                version=args.version, framerate=args.framerate,
                window_size=args.window_size_caption,
            )
            dataset_Valid_metric = dataset_Valid
        if args.test_only:
            dataset_Test = SoccerNetClassificationDual(
                vision_root=args.SoccerNet_path, audio_root=audio_root,
                features=args.features, split=args.split_test,
                version=args.version, framerate=args.framerate,
                window_size=args.window_size_caption,
            )

        if args.feature_dim is None:
            if not args.test_only and len(dataset_Train) > 0:
                args.feature_dim = dataset_Train[0][0].shape[-1]
            elif args.test_only and len(dataset_Test) > 0:
                args.feature_dim = dataset_Test[0][0].shape[-1]
            print("feature_dim found:", args.feature_dim)

        class_labels = dataset_Test.class_labels if args.test_only else dataset_Train.class_labels
        model = DualVideo2Classification(
            num_classes=len(class_labels),
            video_input_dim=getattr(args, "video_input_dim", 1024),
            audio_input_dim=getattr(args, "audio_input_dim", 512),
            hidden_dim=getattr(args, "hidden_dim", 3584),
            dropout=getattr(args, "encoder_dropout", 0.1),
            weights=args.load_weights,
            weights_encoder=args.weights_encoder,
            freeze_encoder=args.freeze_encoder,
        )

    else:
        # ==================== 原有单流模式 ====================
        if not args.test_only:
            dataset_Train = SoccerNetClassification(path=args.SoccerNet_path, features=args.features, split=args.split_train, version=args.version, framerate=args.framerate, window_size=args.window_size_caption)
            dataset_Valid = SoccerNetClassification(path=args.SoccerNet_path, features=args.features, split=args.split_valid, version=args.version, framerate=args.framerate, window_size=args.window_size_caption)
            dataset_Valid_metric  = dataset_Valid
        
        if args.test_only:
            dataset_Test  = SoccerNetClassification(path=args.SoccerNet_path, features=args.features, split=args.split_test, version=args.version, framerate=args.framerate, window_size=args.window_size_caption)

        if args.feature_dim is None:
            if not args.test_only:
                args.feature_dim = dataset_Train[0][0].shape[-1]
            else:
                args.feature_dim = dataset_Test[0][0].shape[-1]
            print("feature_dim found:", args.feature_dim)
        
        class_labels = dataset_Test.class_labels if args.test_only else dataset_Train.class_labels
        model = Video2Classifcation(num_classes=len(class_labels), weights=args.load_weights, input_size=args.feature_dim,
                      window_size=args.window_size_spotting, 
                      vlad_k=args.vlad_k,
                      framerate=args.framerate, pool=args.pool, freeze_encoder=args.freeze_encoder, weights_encoder=args.weights_encoder,
                      proj_size=getattr(args, "hidden_dim", 3584))

    logging.info(model)
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    parameters_per_layer  = [p.numel() for p in model.parameters() if p.requires_grad]
    logging.info("Total number of parameters: " + str(total_params))

    bs = getattr(args, 'batch_size_classify', None) or args.batch_size
    acc_steps = getattr(args, 'accumulation_steps_classify', None) or getattr(args, 'accumulation_steps', 1)
    smoke_steps = getattr(args, 'smoke_steps_classify', None)
    if smoke_steps is None:
        smoke_steps = getattr(args, 'smoke_steps', 0)
    evaluation_frequency = getattr(args, "evaluation_frequency_classify", None) or args.evaluation_frequency
    logging.info(
        f"Classifying phase effective batch_size={bs}, accumulation_steps={acc_steps}, "
        f"epochs={getattr(args, 'epochs_classify', 10)}, smoke_steps={smoke_steps}, "
        f"evaluation_frequency={evaluation_frequency}"
    )

    # create dataloader
    if not args.test_only:
        train_loader = torch.utils.data.DataLoader(dataset_Train,
            batch_size=bs, shuffle=True,
            num_workers=args.max_num_worker, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(dataset_Valid,
            batch_size=bs, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True)

        val_metric_loader = torch.utils.data.DataLoader(dataset_Valid_metric,
            batch_size=bs, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True)


    # training parameters
    if not args.test_only:
        criterion = torch.nn.CrossEntropyLoss()

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
                if ds_config.get('optimizer') is not None:
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
                        f"[DeepSpeed] synced classifying config: micro_bs={bs}, acc_steps={acc_steps}, train_bs={ds_config['train_batch_size']}, lr={args.LR}"
                    )
                    logging.info("[DeepSpeed] optimizer is configured in JSON, using DummyOptim in code")

        if not use_dummy_optimizer:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.LR,
                weight_decay=getattr(args, "weight_decay", 0.01),
            )

            _epochs_classify = getattr(args, "epochs_classify", 10)
            _tmax_classify   = getattr(args, "lr_tmax_classify", None) or _epochs_classify
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_tmax_classify)
        else:
            _epochs_classify = getattr(args, "epochs_classify", 10)

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

        trainer("classifying", train_loader, val_loader, val_metric_loader, 
                model, optimizer, scheduler, criterion,
                model_name=args.model_name,
                max_epochs=_epochs_classify, evaluation_frequency=evaluation_frequency,
                accumulation_steps=acc_steps,
                max_grad_norm=getattr(args, "max_grad_norm_classify", None)
                if getattr(args, "max_grad_norm_classify", None) is not None
                else getattr(args, "max_grad_norm", 0.5),
                smoke_steps=smoke_steps,
                continue_training=getattr(args, 'continue_training', False),
                accelerator=accelerator)

        del train_loader
        del val_loader
        del val_metric_loader
        del dataset_Train
        del dataset_Valid

    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return 
