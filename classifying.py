import os
import logging
from datetime import datetime
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch

from dataset import SoccerNetClips, SoccerNetClipsTesting, SoccerNetClassification
from model import Video2Spot, Video2Classifcation
from train import trainer, test_spotting
from loss import NLLLoss

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
        feature_size = {"gpt2": 768, "gpt2-medium": 1024, "gpt2-large": 1280, "gpt2-xl": 1600}

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
            dataset_Valid_metric = SoccerNetClassificationDual(
                vision_root=args.SoccerNet_path, audio_root=audio_root,
                features=args.features, split=args.split_valid,
                version=args.version, framerate=args.framerate,
                window_size=args.window_size_caption,
            )
        dataset_Test = SoccerNetClassificationDual(
            vision_root=args.SoccerNet_path, audio_root=audio_root,
            features=args.features, split=args.split_test,
            version=args.version, framerate=args.framerate,
            window_size=args.window_size_caption,
        )

        if args.feature_dim is None and len(dataset_Test) > 0:
            args.feature_dim = dataset_Test[0][0].shape[-1]
            print("feature_dim found:", args.feature_dim)

        model = DualVideo2Classification(
            num_classes=len(dataset_Test.class_labels),
            video_input_dim=getattr(args, "video_input_dim", 1024),
            audio_input_dim=getattr(args, "audio_input_dim", 512),
            hidden_dim=feature_size[args.gpt_type],
            dropout=getattr(args, "encoder_dropout", 0.1),
            weights=args.load_weights,
            weights_encoder=args.weights_encoder,
            freeze_encoder=args.freeze_encoder,
        ).cuda()

    else:
        # ==================== 原有单流模式 ====================
        if not args.test_only:
            dataset_Train = SoccerNetClassification(path=args.SoccerNet_path, features=args.features, split=args.split_train, version=args.version, framerate=args.framerate, window_size=args.window_size_caption)
            dataset_Valid = SoccerNetClassification(path=args.SoccerNet_path, features=args.features, split=args.split_valid, version=args.version, framerate=args.framerate, window_size=args.window_size_caption)
            dataset_Valid_metric  = SoccerNetClassification(path=args.SoccerNet_path, features=args.features, split=args.split_valid, version=args.version, framerate=args.framerate, window_size=args.window_size_caption)
        dataset_Test  = SoccerNetClassification(path=args.SoccerNet_path, features=args.features, split=args.split_test, version=args.version, framerate=args.framerate, window_size=args.window_size_caption)

        if args.feature_dim is None:
            args.feature_dim = dataset_Test[0][0].shape[-1]
            print("feature_dim found:", args.feature_dim)
        feature_size = {"gpt2": 768, "gpt2-medium": 1024, "gpt2-large": 1280, "gpt2-xl": 1600}
        model = Video2Classifcation(num_classes=len(dataset_Test.class_labels), weights=args.load_weights, input_size=args.feature_dim,
                      window_size=args.window_size_spotting, 
                      vlad_k=args.vlad_k,
                      framerate=args.framerate, pool=args.pool, freeze_encoder=args.freeze_encoder, weights_encoder=args.weights_encoder,
                      proj_size=feature_size[args.gpt_type]).cuda()

    logging.info(model)
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    parameters_per_layer  = [p.numel() for p in model.parameters() if p.requires_grad]
    logging.info("Total number of parameters: " + str(total_params))

    # create dataloader
    if not args.test_only:
        train_loader = torch.utils.data.DataLoader(dataset_Train,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.max_num_worker, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(dataset_Valid,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True)

        val_metric_loader = torch.utils.data.DataLoader(dataset_Valid_metric,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True)


    # training parameters
    if not args.test_only:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.LR,
            weight_decay=getattr(args, "weight_decay", 0.01),
        )

        _epochs_classify = getattr(args, "epochs_classify", 10)
        _tmax_classify   = getattr(args, "lr_tmax_classify", None) or _epochs_classify
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_tmax_classify)

        # start training
        trainer("classifying", train_loader, val_loader, val_metric_loader, 
                model, optimizer, scheduler, criterion,
                model_name=args.model_name,
                max_epochs=_epochs_classify, evaluation_frequency=args.evaluation_frequency)

    return 