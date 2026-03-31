"""
model.py - Core single-stream model components

Provides:
  - VideoEncoder:        Video feature pooling (QFormer / TRANS only)
  - Video2Classifcation: Event classification model (encoder + FC head)
  - Video2Spot:          Action spotting model (encoder + classification head)

For captioning, see model_qwen.py (single-stream Qwen) or dual_qformer.py (dual-stream).
"""

import __future__

import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from pooling import TransformerVideoPooling, QFormerVideoPooling


class VideoEncoder(nn.Module):
    """
    Temporal video feature encoder.

    Supported pooling modes:
      - "QFormer": Learnable query tokens + Transformer Decoder cross-attention
      - "TRANS":   Dual-stage Transformer Encoder + QFormer

    Input:  [B, T, D_input]
    Output: [B, num_tokens, hidden_dim]  (for QFormer/TRANS)
    """

    def __init__(self, input_size=512, vlad_k=64, window_size=15, framerate=2,
                 pool="QFormer", dropout=0.1, proj_size=768):
        super(VideoEncoder, self).__init__()

        self.window_size_frame = window_size * framerate
        self.input_size = input_size
        self.framerate = framerate
        self.pool = pool

        # Project input features to target dimension if needed
        if not self.input_size == proj_size:
            self.feature_extractor = nn.Linear(self.input_size, proj_size)
            input_size = proj_size
            self.input_size = proj_size

        if self.pool == "TRANS":
            self.hidden_size = input_size
            self.pool_layer = TransformerVideoPooling(input_size, self.hidden_size)

        elif self.pool == "QFormer":
            self.hidden_size = input_size
            self.pool_layer = QFormerVideoPooling(input_size, self.hidden_size, dropout=dropout)

        else:
            raise ValueError(f"Unsupported pooling mode: '{self.pool}'. Use 'QFormer' or 'TRANS'.")

    def forward(self, inputs):
        # input_shape: (batch, frames, dim_features)
        BS, FR, IC = inputs.shape
        if hasattr(self, 'feature_extractor'):
            inputs = self.feature_extractor(inputs)

        # Temporal pooling
        inputs_pooled = self.pool_layer(inputs)
        return inputs_pooled


class Video2Classifcation(nn.Module):
    """
    Single-stream event classification model.

    Architecture: VideoEncoder -> mean pool -> FC -> class logits
    """

    def __init__(self, num_classes, weights=None, input_size=512, vlad_k=64,
                 window_size=15, framerate=2, pool="QFormer",
                 weights_encoder=None, freeze_encoder=False, proj_size=768):
        super(Video2Classifcation, self).__init__()
        self.encoder = VideoEncoder(input_size, vlad_k, window_size, framerate, pool, proj_size=proj_size)
        self.load_weights(weights=weights)
        self.load_encoder(weights_encoder=weights_encoder, freeze_encoder=freeze_encoder)
        self.num_classes = num_classes
        self.fc = nn.Linear(self.encoder.hidden_size, num_classes)
        self.pool = pool

    def load_weights(self, weights=None):
        if weights is not None:
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(weights, checkpoint['epoch']))

    def load_encoder(self, weights_encoder=None, freeze_encoder=False):
        if weights_encoder is not None:
            print("=> loading encoder '{}'".format(weights_encoder))
            checkpoint = torch.load(weights_encoder, map_location=torch.device('cpu'))
            self.load_state_dict({k: v for k, v in checkpoint['state_dict'].items() if "encoder." in k}, strict=False)
            print("=> loaded encoder '{}' (epoch {})".format(weights_encoder, checkpoint['epoch']))

            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False

    def forward(self, features):
        features = self.encoder(features)

        if self.pool == "TRANS":
            features = features[:, 0]

        if features.dim() == 3:
            features = features.mean(dim=1)

        output = self.fc(features)
        return output


class Video2Spot(nn.Module):
    """
    Single-stream action spotting model.

    Architecture: VideoEncoder -> mean pool -> LayerNorm -> MLP head -> class logits
    """

    def __init__(self, weights=None, input_size=512, num_classes=17, vlad_k=64,
                 window_size=15, framerate=2, pool="QFormer",
                 weights_encoder=None, freeze_encoder=False, proj_size=768,
                 use_center_regression=False):
        super(Video2Spot, self).__init__()
        self.encoder = VideoEncoder(input_size, vlad_k, window_size, framerate, pool, proj_size=proj_size)
        self.use_center_regression = bool(use_center_regression)
        self.norm = nn.LayerNorm(self.encoder.hidden_size)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.hidden_size, 32),
            nn.GELU(),
            nn.Linear(32, num_classes + 1),
        )
        if self.use_center_regression:
            self.offset_head = nn.Sequential(
                nn.Linear(self.encoder.hidden_size, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Tanh(),
            )
        self.sigm = nn.Sigmoid()
        self.load_weights(weights=weights)
        self.load_encoder(weights_encoder=weights_encoder, freeze_encoder=freeze_encoder)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def load_weights(self, weights=None):
        if weights is not None:
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights, map_location="cpu")
            missing, unexpected = self.load_state_dict(checkpoint['state_dict'], strict=False)
            if missing or unexpected:
                print("=> load_weights non-strict: missing={}, unexpected={}".format(missing, unexpected))
            print("=> loaded checkpoint '{}' (epoch {})".format(weights, checkpoint['epoch']))

    def load_encoder(self, weights_encoder=None, freeze_encoder=False):
        if weights_encoder is not None:
            print("=> loading encoder '{}'".format(weights_encoder))
            checkpoint = torch.load(weights_encoder, map_location=torch.device('cpu'))
            self.load_state_dict({k: v for k, v in checkpoint['state_dict'].items() if "encoder." in k}, strict=False)
            print("=> loaded encoder '{}' (epoch {})".format(weights_encoder, checkpoint['epoch']))

            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False

    def forward(self, inputs):
        # input_shape: (batch, frames, dim_features)
        inputs_pooled = self.encoder(inputs)  # B x num_tokens x D

        # avg pool in sequence dimension
        if inputs_pooled.dim() == 3:
            inputs_pooled = inputs_pooled.mean(dim=1)
        inputs_pooled = self.norm(inputs_pooled)

        # Classification head
        logits = self.head(inputs_pooled)
        if not self.use_center_regression:
            return logits

        offsets = self.offset_head(inputs_pooled).squeeze(-1)
        return {"logits": logits, "offsets": offsets}
