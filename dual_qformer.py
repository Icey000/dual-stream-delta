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

    Captioning 任务: 拼接特征 -> Qwen2.5 (LoRA) Decoder
    Spotting 任务:   拼接特征 -> mean pool -> Linear 分类头
    Classifying 任务: 拼接特征 -> mean pool -> FC 分类

本模块通过 import 复用 pooling.py 中的 QFormerVideoPooling。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pooling import QFormerVideoPooling


# ============================================================================
#  1. 双流编码器 (Dual-Stream Encoder)
# ============================================================================

class DualStreamEncoder(nn.Module):
    """
    双流晚期融合编码器。

    将视觉特征和音频特征分别通过独立的 Q-Former 压缩为固定数量的 Token，
    再各自通过线性投影层映射到统一的隐藏维度 (hidden_dim)，
    最后在序列长度维度 (dim=1) 上拼接。

    参数:
        video_input_dim (int):  视觉特征的原始维度
        audio_input_dim (int):  音频特征的原始维度
        hidden_dim (int):       统一的输出隐藏维度
        video_tokens (int):     视觉 Q-Former 输出的 Token 数量
        audio_tokens (int):     音频 Q-Former 输出的 Token 数量
        num_heads (int):        Q-Former 内 Transformer 的注意力头数
        num_layers (int):       Q-Former 内 Transformer Decoder 的层数
        dropout (float):        Dropout 概率

    输入:
        video_feats: [B, T_vid, video_input_dim]
        audio_feats: [B, T_aud, audio_input_dim]

    输出:
        fused: [B, video_tokens + audio_tokens, hidden_dim]
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

        self.video_input_dim = video_input_dim
        self.audio_input_dim = audio_input_dim
        self.hidden_dim = hidden_dim
        self.video_tokens = video_tokens
        self.audio_tokens = audio_tokens
        self.total_tokens = video_tokens + audio_tokens
        self.hidden_size = hidden_dim  # 兼容原有属性命名

        # 视觉流: 先降维投影，再进 Q-Former
        self.video_pre_proj = nn.Linear(video_input_dim, hidden_dim)
        self.video_qformer = QFormerVideoPooling(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_tokens=video_tokens,
        )

        # 音频流: 先降维投影，再进 Q-Former
        self.audio_pre_proj = nn.Linear(audio_input_dim, hidden_dim)
        self.audio_qformer = QFormerVideoPooling(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_tokens=audio_tokens,
        )

        # 融合后的 LayerNorm
        self.fusion_norm = nn.LayerNorm(hidden_dim)

    def forward(self, video_feats, audio_feats):
        """
        前向传播: 先降维 -> Q-Former 压缩 -> 晚期融合拼接。

        Args:
            video_feats: [B, T_vid, video_input_dim]
            audio_feats: [B, T_aud, audio_input_dim]

        Returns:
            fused: [B, total_tokens, hidden_dim]
        """
        # 统一输入 dtype/device，避免 fp32 特征喂给 fp16 权重时触发
        # mat1/mat2 dtype mismatch。
        video_feats = video_feats.to(
            device=self.video_pre_proj.weight.device,
            dtype=self.video_pre_proj.weight.dtype,
        )
        audio_feats = audio_feats.to(
            device=self.audio_pre_proj.weight.device,
            dtype=self.audio_pre_proj.weight.dtype,
        )

        v = self.video_pre_proj(video_feats)
        v_tokens = self.video_qformer(v)

        a = self.audio_pre_proj(audio_feats)
        a_tokens = self.audio_qformer(a)

        fused = torch.cat([v_tokens, a_tokens], dim=1)
        fused = self.fusion_norm(fused)
        return fused


# ============================================================================
#  2. Spotting 任务模型 (DualVideo2Spot)
# ============================================================================

class DualVideo2Spot(nn.Module):
    """
    双流 + 分类头的 Action Spotting 模型。

    架构:
        (video_feats, audio_feats) -> DualStreamEncoder -> [B, 16, hidden]
                                                               |
                                                          mean(dim=1) -> LayerNorm -> MLP head
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
        use_center_regression=False,
        **kwargs,
    ):
        super(DualVideo2Spot, self).__init__()
        self.use_center_regression = bool(use_center_regression)

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

        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, num_classes + 1),
        )
        if self.use_center_regression:
            self.offset_head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Tanh(),
            )
        self.sigm = nn.Sigmoid()

        self.load_weights(weights=weights)
        self.load_encoder(weights_encoder=weights_encoder, freeze_encoder=freeze_encoder)
        self.init_weights()

    def init_weights(self):
        modules = [self.head]
        if self.use_center_regression:
            modules.append(self.offset_head)
        for module in modules:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    nn.init.constant_(m.bias, 0)

    def load_weights(self, weights=None):
        if weights is not None:
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights, map_location="cpu")
            missing, unexpected = self.load_state_dict(checkpoint["state_dict"], strict=False)
            if missing or unexpected:
                print("=> load_weights non-strict: missing={}, unexpected={}".format(missing, unexpected))
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
        fused = self.encoder(video_feats, audio_feats)
        fused = fused.mean(dim=1)
        fused = self.norm(fused)
        logits = self.head(fused)
        if not self.use_center_regression:
            return logits

        offsets = self.offset_head(fused).squeeze(-1)
        return {"logits": logits, "offsets": offsets}


# ============================================================================
#  3. Classifying 任务模型 (DualVideo2Classification)
# ============================================================================

class DualVideo2Classification(nn.Module):
    """
    双流 + 分类头的事件分类模型 (用于预训练编码器)。
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

    def forward(self, video_feats, audio_feats, return_features=False):
        fused = self.encoder(video_feats, audio_feats)
        fused = fused.mean(dim=1)
        output = self.fc(fused)
        if return_features:
            return output, fused
        return output


# ============================================================================
#  4. 双流 Qwen2.5 + LoRA Captioning 模型 (DualVideo2CaptionLLM)
# ============================================================================

class DualVideo2CaptionLLM(nn.Module):
    """
    双流 + Qwen2.5 LoRA 的 Dense Video Captioning 模型。

    架构:
        (video_feats, audio_feats) -> DualStreamEncoder -> [B, 16, hidden]
                                                               |
                                                         MLP (GELU)
                                                               |
                                                     Qwen2.5 (LoRA) -> Caption
    """

    def __init__(
        self,
        vocab_size=None,
        video_input_dim=1024,
        audio_input_dim=512,
        hidden_dim=768,
        video_tokens=8,
        audio_tokens=8,
        num_heads=8,
        num_layers=4,
        dropout=0.0,
        encoder_dropout=0.1,
        llm_model_path="Qwen/Qwen2.5-7B",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        weights=None,
        weights_encoder=None,
        freeze_encoder=False,
        top_k=5,
        max_new_tokens=48,
        no_repeat_ngram_size=3,
        num_beams=1,
        length_penalty=0.9,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.15,
        **kwargs,
    ):
        super(DualVideo2CaptionLLM, self).__init__()

        import logging
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import get_peft_model, LoraConfig, TaskType

        # 1. 加载 Qwen Tokenizer + Model
        logging.info(f"Loading LLM: {llm_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        base_model = AutoModelForCausalLM.from_pretrained(
            llm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        # 2. LoRA
        # 注意: 不将 gate_proj/up_proj/down_proj 列入 target_modules
        # 这些 FFN 层在训练时需要保存巨大的激活值 (batch × seq_len × 14336)
        # 只在 attention 投影层加 LoRA，参数量更少, 显存占用更低
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=max(lora_dropout, 0.1),  # FP16 稳定性: dropout 下限 0.1
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",
            ],
        )
        self.llm = get_peft_model(base_model, peft_config)
        self.llm.print_trainable_parameters()

        # 2b. Gradient Checkpointing: 以重新计算换显存，可节省约 30-40% 激活内存
        # 注意: 需要在 get_peft_model 之后调用
        self.llm.enable_input_require_grads()  # PEFT 要求
        # ZeRO-3 + checkpoint 在新 PyTorch 下可能触发 metadata 校验冲突，强制使用 reentrant 规避
        self.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True}
        )

        actual_hidden = self.llm.config.hidden_size

        # 3. 双流编码器
        self.encoder = DualStreamEncoder(
            video_input_dim=video_input_dim,
            audio_input_dim=audio_input_dim,
            hidden_dim=actual_hidden,
            video_tokens=video_tokens,
            audio_tokens=audio_tokens,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=encoder_dropout,
        )

        # 4. 双层 MLP 投影 (GELU)
        # 中间维度用 1x (而非 2x/4x) 以减少显存和数值范围
        proj_hidden = actual_hidden
        self.proj = nn.Sequential(
            nn.Linear(actual_hidden, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, actual_hidden),
        )
        # proj 之后的 LayerNorm: 把投影输出归一化，防止 fp16 溢出
        self.proj_norm = nn.LayerNorm(actual_hidden)

        # ====== FIX FP16 NaN ======
        # 使用极小的初始化 std，防止首个 forward pass 就数值爆炸
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.top_k = top_k
        self.vocab_size = len(self.tokenizer)
        self.generation_config = {
            "max_new_tokens": int(max_new_tokens),
            "no_repeat_ngram_size": int(no_repeat_ngram_size),
            "num_beams": int(num_beams),
            "length_penalty": float(length_penalty),
            "do_sample": bool(do_sample),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "repetition_penalty": float(repetition_penalty),
            "top_k": int(top_k),
        }

        # 5. 加载权重
        if weights is not None:
            logging.info(f"=> loading checkpoint '{weights}'")
            ckpt = torch.load(weights, map_location="cpu")
            self.load_state_dict(ckpt["state_dict"], strict=False)

        if weights_encoder is not None:
            logging.info(f"=> loading encoder '{weights_encoder}'")
            ckpt = torch.load(weights_encoder, map_location="cpu")
            enc_state = {k: v for k, v in ckpt["state_dict"].items() if k.startswith("encoder.")}
            self.load_state_dict(enc_state, strict=False)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logging.info("Dual encoder frozen.")

    def encode_projected_tokens(self, video_feats, audio_feats):
        if video_feats.dim() == 2:
            video_feats = video_feats.unsqueeze(0)
        if audio_feats.dim() == 2:
            audio_feats = audio_feats.unsqueeze(0)

        fused = self.encoder(video_feats, audio_feats)
        proj_weight = self.proj[0].weight
        fused = fused.to(device=proj_weight.device, dtype=proj_weight.dtype)
        fused = self.proj(fused)
        fused = self.proj_norm(fused)
        fused = fused.clamp(-10.0, 10.0)
        return fused

    def build_prefix_embeds(self, video_feats, audio_feats):
        fused = self.encode_projected_tokens(video_feats, audio_feats).to(self.llm.dtype)

        bos_ids = self.tokenizer.encode(":", add_special_tokens=False)
        bos_tensor = torch.tensor([bos_ids], dtype=torch.long, device=fused.device)
        bos_embeds = self.llm.get_input_embeddings()(bos_tensor).to(fused.dtype)
        if fused.shape[0] > 1:
            bos_embeds = bos_embeds.expand(fused.shape[0], -1, -1)

        return torch.cat([fused, bos_embeds], dim=1)

    def generate_sequences(self, video_feats, audio_feats, max_seq_length=None, generation_config=None, **extra_kwargs):
        prefix_embeds = self.build_prefix_embeds(video_feats, audio_feats)
        prefix_attention_mask = torch.ones(
            prefix_embeds.shape[:2],
            dtype=torch.long,
            device=prefix_embeds.device,
        )
        generation_kwargs = self._build_generation_kwargs(
            max_seq_length=max_seq_length,
            generation_config=generation_config,
        )
        generation_kwargs.update(extra_kwargs)
        return self.llm.generate(
            inputs_embeds=prefix_embeds,
            attention_mask=prefix_attention_mask,
            **generation_kwargs,
        )

    def forward(self, video_feats, audio_feats, captions, lengths):
        """训练时的前向传播。返回 scalar loss。"""
        fused = self.encode_projected_tokens(video_feats, audio_feats).to(self.llm.dtype)

        token_embeds = self.llm.get_input_embeddings()(captions).to(fused.dtype)
        input_embeds = torch.cat([fused, token_embeds], dim=1)

        v_mask = torch.ones(fused.shape[0], fused.shape[1], dtype=torch.long, device=fused.device)
        text_mask = (captions != self.tokenizer.pad_token_id).long()
        attention_mask = torch.cat([v_mask, text_mask], dim=1)

        labels = captions.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        v_labels = torch.full(
            (fused.shape[0], fused.shape[1]), -100,
            dtype=labels.dtype, device=labels.device,
        )
        labels = torch.cat([v_labels, labels], dim=1)

        # 整个 batch 都没有有效 caption token 时，避免 ignore_index CE 产生 NaN。
        if not torch.any(labels != -100):
            return fused.sum() * 0.0

        outputs = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss

    def _build_generation_kwargs(self, max_seq_length=None, generation_config=None):
        config = dict(self.generation_config)
        if generation_config:
            for key, value in generation_config.items():
                if value is not None:
                    config[key] = value

        max_new_tokens = int(max_seq_length) if max_seq_length is not None else int(config.get("max_new_tokens", 48))
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": bool(config.get("do_sample", False)),
            "num_beams": max(1, int(config.get("num_beams", 1))),
            "length_penalty": float(config.get("length_penalty", 1.0)),
            "repetition_penalty": float(config.get("repetition_penalty", 1.0)),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        no_repeat_ngram_size = int(config.get("no_repeat_ngram_size", 0))
        if no_repeat_ngram_size > 0:
            generation_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

        top_k = int(config.get("top_k", self.top_k))
        if top_k > 0:
            generation_kwargs["top_k"] = top_k

        if generation_kwargs["do_sample"]:
            generation_kwargs["temperature"] = max(float(config.get("temperature", 1.0)), 1e-5)
            top_p = float(config.get("top_p", 1.0))
            if 0.0 < top_p < 1.0:
                generation_kwargs["top_p"] = top_p

        if generation_kwargs["num_beams"] > 1:
            generation_kwargs["early_stopping"] = True

        return generation_kwargs

    def sample(self, video_feats, audio_feats, max_seq_length=None, generation_config=None):
        """推理时的自回归采样。返回生成的文本字符串。"""
        self.eval()
        with torch.no_grad():
            outputs = self.generate_sequences(
                video_feats.unsqueeze(0),
                audio_feats.unsqueeze(0),
                max_seq_length=max_seq_length,
                generation_config=generation_config,
            )

            generated_text = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

            # 清理可能出现的起始符和多余空白，进一步减少脏输出
            generated_text = generated_text.strip()
            if generated_text.startswith(":"):
                generated_text = generated_text[1:].strip()
            if "\n" in generated_text:
                generated_text = generated_text.split("\n", 1)[0].strip()

        return generated_text

    def get_trainable_parameters(self):
        """返回可训练参数列表 (LoRA + MLP + 未冻结的 encoder)"""
        return [p for p in self.parameters() if p.requires_grad]
