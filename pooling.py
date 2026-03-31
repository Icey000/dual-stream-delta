"""
pooling.py - Video Temporal Pooling Modules

Provides two pooling strategies for compressing variable-length video features
into fixed-size representations:

  1. QFormerVideoPooling  - Learnable query tokens + Transformer Decoder (primary)
  2. TransformerVideoPooling - Dual-stage Transformer Encoder + QFormer (secondary)

Both output shape: [B, num_tokens, hidden_dim]
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence inputs."""

    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class QFormerVideoPooling(nn.Module):
    """
    Q-Former style pooling: uses learnable query tokens to cross-attend
    to the input sequence via a Transformer Decoder.

    Input:  [B, T, D]  (variable-length video features)
    Output: [B, num_tokens, D]  (fixed-size representation)
    """

    def __init__(self, input_dim, output_dim, num_heads=8, num_layers=4,
                 dropout=0.0, num_tokens=8):
        super(QFormerVideoPooling, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tranformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=input_dim, nhead=num_heads, dropout=dropout,
                batch_first=True, activation="gelu"
            ),
            num_layers=num_layers,
        )
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, input_dim))
        self.pos_encoder = PositionalEncoding(input_dim, max_len=10000)

    def forward(self, x):
        # x [BS, T, D]
        BS, T, D = x.size()
        x = self.pos_encoder(x)
        queries = self.query_tokens.expand(BS, -1, -1)
        queries = self.pos_encoder(queries)
        x = self.tranformer_decoder(queries, x)  # [BS, num_tokens, D]
        return x


class TransformerVideoPooling(nn.Module):
    """
    Dual-stage Transformer Encoder pooling followed by Q-Former compression.

    Stage 1: Transpose to [B, D, T] -> TransformerEncoder over T dimension
    Stage 2: Transpose back -> TransformerEncoder over D dimension
    Stage 3: QFormer compression to fixed tokens

    Input:  [B, T, D]
    Output: [B, num_tokens, D]
    """

    def __init__(self, input_dim, output_dim, num_heads=6, num_layers=2, dropout=0.1):
        super(TransformerVideoPooling, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pos_encoder = PositionalEncoding(30, max_len=1000)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=30, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_layers,
        )
        self.transformer_2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_layers,
        )
        self.qformer = QFormerVideoPooling(input_dim, output_dim, num_heads, num_layers, dropout, num_tokens=8)

    def forward(self, x):
        # x [BS, T, D]
        x = x.transpose(1, 2)
        BS, T, D = x.size()
        x = self.pos_encoder(x)   # [BS, D, T]
        x = self.transformer(x)   # [BS, D, T]
        x = self.pos_encoder(x)
        x = x.transpose(1, 2)
        x = self.transformer_2(x)
        x = self.qformer(x)
        return x
