"""
dual_qformer.py - 双流晚期融合 (Dual-Stream Late Fusion) 核心模块

架构概览:
    Video [B, T_vid, D_vid] -> video_qformer -> [B, N_v, D_vid] -> video_proj -> [B, N_v, D_hidden]
                                                                                          |
                                                                                    cat(dim=1)
                                                                                          |
    Audio [B, T_aud, D_aud] -> audio_qformer -> [B, N_a, D_aud] -> audio_proj -> [B, N_a, D_hidden]
                                                                                          |
                                                                                 [B, N_v+N_a, D_hidden]

    Captioning 任务: 拼接特征 -> GPT Decoder
    Spotting 任务:   拼接特征 -> mean pool -> Linear 分类头

作者注: 本模块完全解耦自原有 model.py, 通过 import 复用 netvlad.py 中的 QFormerVideoPooling。
       对外暴露干净的 forward 接口，可在 captioning.py / spotting.py 中灵活切换使用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from netvlad import QFormerVideoPooling


# ============================================================================
#  1. 双流编码器 (Dual-Stream Encoder)
# ============================================================================

class DualStreamEncoder(nn.Module):
    """
    双流晚期融合编码器。

    将视觉特征和音频特征分别通过独立的 Q-Former 压缩为固定数量的 Token，
    再各自通过线性投影层映射到统一的隐藏维度 (hidden_dim)，
    最后在 **序列长度维度 (dim=1)** 上拼接。

    参数:
        video_input_dim (int):  视觉特征的原始维度，例如百度特征 = 1024
        audio_input_dim (int):  音频特征的原始维度，例如 CLAP = 512
        hidden_dim (int):       统一的输出隐藏维度，默认 768 (与 GPT-2 对齐)
        video_tokens (int):     视觉 Q-Former 输出的 Token 数量，默认 8
        audio_tokens (int):     音频 Q-Former 输出的 Token 数量，默认 8
        num_heads (int):        Q-Former 内 Transformer 的注意力头数
        num_layers (int):       Q-Former 内 Transformer Decoder 的层数
        dropout (float):        Dropout 概率

    输入:
        video_feats: [B, T_vid, video_input_dim]  - 预提取的视觉 .npy 张量
        audio_feats: [B, T_aud, audio_input_dim]  - 预提取的音频 .npy 张量

    输出:
        fused: [B, video_tokens + audio_tokens, hidden_dim]
               默认形状 = [B, 16, 768]
    """

    def __init__(
        self,
        video_input_dim=1024,
        audio_input_dim=512,
        hidden_dim=768,
        video_tokens=8,
        audio_tokens=8,
        num_heads=8,
        num_layers=4,
        dropout=0.0,
    ):
        super(DualStreamEncoder, self).__init__()

        # ---------- 保存维度信息，供下游模块读取 ----------
        self.video_input_dim = video_input_dim
        self.audio_input_dim = audio_input_dim
        self.hidden_dim = hidden_dim
        self.video_tokens = video_tokens
        self.audio_tokens = audio_tokens
        self.total_tokens = video_tokens + audio_tokens   # 默认 16
        self.hidden_size = hidden_dim                     # 兼容原有 VideoEncoder 的属性命名

        # ---------- 视觉流: 先降维投影，再进 Q-Former ----------
        # 关键修复: 先用 Linear 把原始大维度 (如 1024, 8576) 压到 hidden_dim，
        # 让 Q-Former 内部 Transformer 的 d_model = hidden_dim (如 768)，
        # 而不是用原始大维度作 d_model，否则参数量会平方级爆炸！
        # (与原版 VideoEncoder 里 if not input_size==proj_size: Linear 的做法完全一致)
        self.video_pre_proj = nn.Linear(video_input_dim, hidden_dim)
        self.video_qformer = QFormerVideoPooling(
            input_dim=hidden_dim,        # Q-Former 内部维度统一为 hidden_dim
            output_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_tokens=video_tokens,
        )
        # 出 Q-Former 后不再需要额外映射，已经是 hidden_dim 了

        # ---------- 音频流: 先降维投影，再进 Q-Former ----------
        # 同理，audio_input_dim (如 512) 也先压到 hidden_dim，
        # 保证音频的 Q-Former 内部维度与视觉流一致。
        self.audio_pre_proj = nn.Linear(audio_input_dim, hidden_dim)
        self.audio_qformer = QFormerVideoPooling(
            input_dim=hidden_dim,        # Q-Former 内部维度统一为 hidden_dim
            output_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_tokens=audio_tokens,
        )

        # ---------- 融合后的 LayerNorm ----------
        self.fusion_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _find_compatible_heads(dim, desired_heads):
        """
        找到一个 <= desired_heads 且能整除 dim 的注意力头数。
        """
        for h in range(desired_heads, 0, -1):
            if dim % h == 0:
                return h
        return 1

    def forward(self, video_feats, audio_feats):
        """
        前向传播: 先降维 -> Q-Former 压缩 -> 晚期融合拼接。

        Args:
            video_feats: [B, T_vid, video_input_dim]   (任意输入维度)
            audio_feats: [B, T_aud, audio_input_dim]   (任意输入维度)

        Returns:
            fused: [B, total_tokens, hidden_dim]  (默认 [B, 16, 768])
        """
        # ---- 视觉流: 先降维，再 Q-Former ----
        v = self.video_pre_proj(video_feats)   # [B, T_vid, hidden_dim]  <-- 降维到 768
        v_tokens = self.video_qformer(v)        # [B, video_tokens, hidden_dim]

        # ---- 音频流: 先降维，再 Q-Former ----
        a = self.audio_pre_proj(audio_feats)   # [B, T_aud, hidden_dim]  <-- 降维到 768
        a_tokens = self.audio_qformer(a)        # [B, audio_tokens, hidden_dim]

        # ---- 晚期融合: 在序列长度维度拼接 ----
        fused = torch.cat([v_tokens, a_tokens], dim=1)  # [B, total_tokens, hidden_dim]
        fused = self.fusion_norm(fused)

        return fused


# ============================================================================
#  2. Captioning 任务模型 (DualVideo2Caption)
# ============================================================================

class DualVideo2Caption(nn.Module):
    """
    双流 + GPT-2 的 Dense Video Captioning 模型。

    架构:
        (video_feats, audio_feats) -> DualStreamEncoder -> [B, 16, 768]
                                                               |
                                                      Linear 投影 (可选)
                                                               |
                                                         GPT-2 Decoder

    参数:
        vocab_size (int):            词表大小 (GPT-2 tokenizer)
        video_input_dim (int):       视觉特征维度
        audio_input_dim (int):       音频特征维度
        hidden_dim (int):            统一隐藏维度 (应等于 GPT 的 n_embd)
        video_tokens / audio_tokens: Q-Former 产出的 Token 数
        gpt_path (str):              GPT-2 预训练权重路径
        gpt_type (str):              GPT-2 模型类型 ("gpt2", "gpt2-medium", ...)
        weights (str):               可选的整体权重加载路径
        weights_encoder (str):       可选的编码器权重加载路径
        freeze_encoder (bool):       是否冻结编码器参数
        top_k (int):                 采样时的 top-k 值
    """

    def __init__(
        self,
        vocab_size,
        video_input_dim=1024,
        audio_input_dim=512,
        hidden_dim=768,
        video_tokens=8,
        audio_tokens=8,
        num_heads=8,
        num_layers=4,
        dropout=0.0,          # 备用，保持向下兼容性
        gpt_dropout=0.1,       # GPT Decoder各层dropout；原先硬编砍为0.0！
        encoder_dropout=0.1,   # Q-Former编码器dropout
        gpt_path="gpt2",
        gpt_type="gpt2",
        weights=None,
        weights_encoder=None,
        freeze_encoder=False,
        top_k=5,
        **kwargs,
    ):
        super(DualVideo2Caption, self).__init__()

        # ---- GPT Decoder (先创建，以获取 n_embd) ----
        from gpt import GPT
        self.decoder = GPT.from_pretrained(gpt_type, dict(dropout=gpt_dropout), path=gpt_path)
        actual_hidden = self.decoder.config.n_embd  # GPT-2: 768, medium: 1024, ...

        # ---- 双流编码器 ----
        self.encoder = DualStreamEncoder(
            video_input_dim=video_input_dim,
            audio_input_dim=audio_input_dim,
            hidden_dim=actual_hidden,    # 确保与 GPT 维度对齐
            video_tokens=video_tokens,
            audio_tokens=audio_tokens,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=encoder_dropout,
        )

        # ---- 特征投影 (编码器输出 -> GPT 输入) ----
        self.proj = nn.Linear(actual_hidden, actual_hidden)

        # ---- 其他属性 ----
        self.top_k = top_k
        self.vocab_size = vocab_size

        # ---- 加载权重 ----
        self.load_weights(weights=weights)
        self.load_encoder(weights_encoder=weights_encoder, freeze_encoder=freeze_encoder)

    # -------------------- 权重加载 --------------------

    def load_weights(self, weights=None):
        """加载整体模型权重"""
        if weights is not None:
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights, map_location="cpu")
            self.load_state_dict(checkpoint["state_dict"])
            print("=> loaded checkpoint '{}' (epoch {})".format(weights, checkpoint['epoch']))

    def load_encoder(self, weights_encoder=None, freeze_encoder=False):
        """加载编码器权重 (兼容从单流 VideoEncoder 迁移)"""
        if weights_encoder is not None:
            print("=> loading encoder '{}'".format(weights_encoder))
            checkpoint = torch.load(weights_encoder, map_location="cpu")
            # 只加载 encoder 开头的 key，忽略不匹配的
            encoder_state = {
                k: v for k, v in checkpoint["state_dict"].items()
                if k.startswith("encoder.")
            }
            self.load_state_dict(encoder_state, strict=False)
            print("=> loaded encoder '{}' (epoch {})".format(weights_encoder, checkpoint['epoch']))

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    # -------------------- 前向传播 --------------------

    def forward(self, video_feats, audio_feats, captions, lengths):
        """
        训练时的前向传播。

        Args:
            video_feats: [B, T_vid, D_vid]
            audio_feats: [B, T_aud, D_aud]
            captions:    [B, max_caption_len]  - token IDs
            lengths:     [B]                   - 每条 caption 的实际长度

        Returns:
            decoder_output: packed 的 logits
        """
        # 双流编码
        fused = self.encoder(video_feats, audio_feats)   # [B, 16, hidden]
        fused = self.proj(fused)                          # [B, 16, hidden]

        # GPT Decoder
        decoder_output, _ = self.decoder(captions, lengths=lengths, features=fused)
        return decoder_output

    def sample(self, video_feats, audio_feats, max_seq_length=100):
        """
        推理时的自回归采样。

        Args:
            video_feats: [T_vid, D_vid]  - 单条样本（无 batch 维度）
            audio_feats: [T_aud, D_aud]  - 单条样本（无 batch 维度）
            max_seq_length: 最大生成长度

        Returns:
            text: [1, seq_len]  - 生成的 token ID 序列
        """
        import tiktoken

        # 添加 batch 维度
        fused = self.encoder(
            video_feats.unsqueeze(0),
            audio_feats.unsqueeze(0),
        )  # [1, 16, hidden]
        fused = self.proj(fused)

        # 构造 BOS token (使用 ":" 作为起始，与原 model_gpt.py 一致)
        enc = tiktoken.get_encoding("gpt2")
        eot = enc.eot_token
        start_text = ":"
        start_ids = enc.encode(start_text)
        x = torch.tensor(start_ids, dtype=torch.long, device=fused.device).unsqueeze(0)

        # 自回归生成
        text = self.decoder.generate(x, max_seq_length, top_k=self.top_k, features=fused, eos_token=eot)
        text = text[:, 1:]  # 去掉起始 token
        return text


# ============================================================================
#  3. Spotting 任务模型 (DualVideo2Spot)
# ============================================================================

class DualVideo2Spot(nn.Module):
    """
    双流 + 分类头的 Action Spotting 模型。

    架构:
        (video_feats, audio_feats) -> DualStreamEncoder -> [B, 16, 768]
                                                               |
                                                          mean(dim=1)
                                                               |
                                                          LayerNorm
                                                               |
                                                      Linear -> ReLU -> Linear
                                                               |
                                                    [B, num_classes + 1]

    参数:
        num_classes (int):           动作类别数
        video_input_dim (int):       视觉特征维度
        audio_input_dim (int):       音频特征维度
        hidden_dim (int):            统一隐藏维度
        video_tokens / audio_tokens: Q-Former 产出的 Token 数
        weights (str):               可选的整体权重加载路径
        weights_encoder (str):       可选的编码器权重加载路径
        freeze_encoder (bool):       是否冻结编码器参数
    """

    def __init__(
        self,
        num_classes=17,
        video_input_dim=1024,
        audio_input_dim=512,
        hidden_dim=768,
        video_tokens=8,
        audio_tokens=8,
        num_heads=8,
        num_layers=4,
        dropout=0.0,
        weights=None,
        weights_encoder=None,
        freeze_encoder=False,
        **kwargs,
    ):
        super(DualVideo2Spot, self).__init__()

        # ---- 双流编码器 ----
        self.encoder = DualStreamEncoder(
            video_input_dim=video_input_dim,
            audio_input_dim=audio_input_dim,
            hidden_dim=hidden_dim,
            video_tokens=video_tokens,
            audio_tokens=audio_tokens,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        # ---- 分类头: 融合特征 -> 动作类别概率 ----
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes + 1),
        )
        self.sigm = nn.Sigmoid()

        # ---- 加载权重 ----
        self.load_weights(weights=weights)
        self.load_encoder(weights_encoder=weights_encoder, freeze_encoder=freeze_encoder)

        # ---- 初始化分类头权重 ----
        self.init_weights()

    def init_weights(self):
        """使用 Kaiming 初始化分类头的线性层"""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def load_weights(self, weights=None):
        """加载整体模型权重"""
        if weights is not None:
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights, map_location="cpu")
            self.load_state_dict(checkpoint["state_dict"])
            print("=> loaded checkpoint '{}' (epoch {})".format(weights, checkpoint['epoch']))

    def load_encoder(self, weights_encoder=None, freeze_encoder=False):
        """加载编码器权重"""
        if weights_encoder is not None:
            print("=> loading encoder '{}'".format(weights_encoder))
            checkpoint = torch.load(weights_encoder, map_location="cpu")
            encoder_state = {
                k: v for k, v in checkpoint["state_dict"].items()
                if k.startswith("encoder.")
            }
            self.load_state_dict(encoder_state, strict=False)
            print("=> loaded encoder '{}' (epoch {})".format(weights_encoder, checkpoint['epoch']))

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, video_feats, audio_feats):
        """
        前向传播。

        Args:
            video_feats: [B, T_vid, D_vid]
            audio_feats: [B, T_aud, D_aud]

        Returns:
            output: [B, num_classes + 1]  - 各类别的得分
        """
        # 双流编码
        fused = self.encoder(video_feats, audio_feats)  # [B, 16, hidden]

        # 在 Token 维度做 mean pooling
        fused = fused.mean(dim=1)   # [B, hidden]
        fused = self.norm(fused)

        # 分类头
        output = self.head(fused)   # [B, num_classes + 1]
        return output


# ============================================================================
#  4. Classifying 任务模型 (DualVideo2Classification)
# ============================================================================

class DualVideo2Classification(nn.Module):
    """
    双流 + 分类头的事件分类模型 (用于预训练编码器)。

    与 DualVideo2Spot 类似，但分类头更简单，
    直接输出 num_classes 个类别的概率。
    """

    def __init__(
        self,
        num_classes,
        video_input_dim=1024,
        audio_input_dim=512,
        hidden_dim=768,
        video_tokens=8,
        audio_tokens=8,
        num_heads=8,
        num_layers=4,
        dropout=0.0,
        weights=None,
        weights_encoder=None,
        freeze_encoder=False,
        **kwargs,
    ):
        super(DualVideo2Classification, self).__init__()

        self.encoder = DualStreamEncoder(
            video_input_dim=video_input_dim,
            audio_input_dim=audio_input_dim,
            hidden_dim=hidden_dim,
            video_tokens=video_tokens,
            audio_tokens=audio_tokens,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.num_classes = num_classes
        self.fc = nn.Linear(hidden_dim, num_classes)

        self.load_weights(weights=weights)
        self.load_encoder(weights_encoder=weights_encoder, freeze_encoder=freeze_encoder)

    def load_weights(self, weights=None):
        if weights is not None:
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights, map_location="cpu")
            self.load_state_dict(checkpoint["state_dict"])
            print("=> loaded checkpoint '{}' (epoch {})".format(weights, checkpoint['epoch']))

    def load_encoder(self, weights_encoder=None, freeze_encoder=False):
        if weights_encoder is not None:
            print("=> loading encoder '{}'".format(weights_encoder))
            checkpoint = torch.load(weights_encoder, map_location="cpu")
            encoder_state = {
                k: v for k, v in checkpoint["state_dict"].items()
                if k.startswith("encoder.")
            }
            self.load_state_dict(encoder_state, strict=False)
            print("=> loaded encoder '{}' (epoch {})".format(weights_encoder, checkpoint['epoch']))
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, video_feats, audio_feats):
        """
        Args:
            video_feats: [B, T_vid, D_vid]
            audio_feats: [B, T_aud, D_aud]
        Returns:
            output: [B, num_classes]
        """
        fused = self.encoder(video_feats, audio_feats)  # [B, 16, hidden]
        fused = fused.mean(dim=1)  # [B, hidden]
        output = self.fc(fused)    # [B, num_classes]
        return output


# ============================================================================
#  5. Smoke Test (独立运行验证)
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Dual-Stream Late Fusion - Smoke Test")
    print("=" * 60)

    BS = 4          # batch size
    T_VID = 60      # 视觉时间步 (例如 30秒 * 2fps)
    T_AUD = 30      # 音频时间步
    D_VID = 1024    # 百度视觉特征维度
    D_AUD = 512     # CLAP 音频特征维度
    HIDDEN = 768    # 统一隐藏维度

    # ---- 1. 测试 DualStreamEncoder ----
    print("\n[1] Testing DualStreamEncoder...")
    encoder = DualStreamEncoder(
        video_input_dim=D_VID,
        audio_input_dim=D_AUD,
        hidden_dim=HIDDEN,
        video_tokens=8,
        audio_tokens=8,
    )
    v_in = torch.randn(BS, T_VID, D_VID)
    a_in = torch.randn(BS, T_AUD, D_AUD)
    fused = encoder(v_in, a_in)
    print("  video input:   ", v_in.shape)     # [4, 60, 1024]
    print("  audio input:   ", a_in.shape)     # [4, 30, 512]
    print("  fused output:  ", fused.shape)    # [4, 16, 768]
    assert fused.shape == (BS, 16, HIDDEN), "Shape mismatch!"
    print("  [PASS] DualStreamEncoder output shape is correct!")

    # ---- 2. 测试 DualVideo2Spot ----
    print("\n[2] Testing DualVideo2Spot...")
    num_classes = 17
    spot_model = DualVideo2Spot(
        num_classes=num_classes,
        video_input_dim=D_VID,
        audio_input_dim=D_AUD,
        hidden_dim=HIDDEN,
    )
    spot_out = spot_model(v_in, a_in)
    print("  spot output:   ", spot_out.shape)  # [4, 18]
    assert spot_out.shape == (BS, num_classes + 1), "Shape mismatch!"
    print("  [PASS] DualVideo2Spot output shape is correct!")

    # ---- 3. 测试 DualVideo2Classification ----
    print("\n[3] Testing DualVideo2Classification...")
    cls_model = DualVideo2Classification(
        num_classes=7,
        video_input_dim=D_VID,
        audio_input_dim=D_AUD,
        hidden_dim=HIDDEN,
    )
    cls_out = cls_model(v_in, a_in)
    print("  cls output:    ", cls_out.shape)   # [4, 7]
    assert cls_out.shape == (BS, 7), "Shape mismatch!"
    print("  [PASS] DualVideo2Classification output shape is correct!")

    # ---- 4. 参数统计 ----
    print("\n[4] Parameter counts:")
    for name, m in [("DualStreamEncoder", encoder), ("DualVideo2Spot", spot_model), ("DualVideo2Classification", cls_model)]:
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print("  {} : total={:,}  trainable={:,}".format(name, total, trainable))

    print("\n" + "=" * 60)
    print("  All smoke tests passed!")
    print("=" * 60)
