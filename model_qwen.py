"""
model_qwen.py - Qwen2.5 + LoRA 大语言模型 Dense Video Captioning

单流版本: VideoEncoder -> MLP -> Qwen (LoRA)
需要安装: pip install transformers peft accelerate

架构概览:
    Video [B, T, D] -> VideoEncoder -> Q-Former -> [B, N, hidden]
                                                        |
                                                  MLP (GELU)
                                                        |
                                               Qwen2.5 (LoRA) -> Caption

作者注: 本模块替代 model_gpt.py + gpt.py，使用 HuggingFace transformers + peft 实现。
"""

import logging
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


class Video2CaptionQwen(nn.Module):
    """
    单流视觉特征 + Qwen2.5 LoRA Dense Video Captioning 模型。

    参数:
        input_size (int):       视觉特征的原始维度
        vlad_k (int):           聚类数 (传给 VideoEncoder, 未使用)
        window_size (int):      窗口大小 (秒)
        framerate (int):        帧率
        pool (str):             池化方式 ("QFormer", "TRANS")
        llm_model_path (str):   HuggingFace 模型路径或本地路径
        lora_r (int):           LoRA 秩
        lora_alpha (int):       LoRA alpha
        lora_dropout (float):   LoRA dropout
        weights_encoder (str):  编码器预训练权重路径
        freeze_encoder (bool):  是否冻结编码器
        top_k (int):            采样 top-k
    """

    def __init__(
        self,
        input_size,
        vlad_k=64,
        window_size=15,
        framerate=2,
        pool="QFormer",
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
        super(Video2CaptionQwen, self).__init__()

        # ---- 1. 加载 Qwen 模型和 Tokenizer ----
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

        # ---- 2. 应用 LoRA ----
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        self.llm = get_peft_model(base_model, peft_config)
        self.llm.print_trainable_parameters()

        self.hidden_dim = self.llm.config.hidden_size  # Qwen2.5-7B: 3584

        # ---- 3. 视频编码器 ----
        from model import VideoEncoder
        self.encoder = VideoEncoder(
            input_size, vlad_k, window_size, framerate, pool,
            proj_size=self.hidden_dim,
        )

        # 加载权重
        if weights is not None:
            logging.info(f"=> loading full checkpoint '{weights}'")
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
            logging.info("Encoder frozen.")

        # ---- 4. 双层 MLP 投影 (GELU) ----
        proj_hidden = self.hidden_dim * 4
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_dim, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, self.hidden_dim),
        )

        # ====== FIX FP16 NaN ======
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
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

    def encode_projected_tokens(self, features):
        if features.dim() == 2:
            features = features.unsqueeze(0)

        v = self.encoder(features)
        proj_weight = self.proj[0].weight
        v = v.to(device=proj_weight.device, dtype=proj_weight.dtype)
        v = self.proj(v)
        return v

    def build_prefix_embeds(self, features):
        v = self.encode_projected_tokens(features).to(self.llm.dtype)

        bos_ids = self.tokenizer.encode(":", add_special_tokens=False)
        bos_tensor = torch.tensor([bos_ids], dtype=torch.long, device=v.device)
        bos_embeds = self.llm.get_input_embeddings()(bos_tensor).to(v.dtype)
        if v.shape[0] > 1:
            bos_embeds = bos_embeds.expand(v.shape[0], -1, -1)

        return torch.cat([v, bos_embeds], dim=1)

    def generate_sequences(self, features, max_seq_length=None, generation_config=None, **extra_kwargs):
        prefix_embeds = self.build_prefix_embeds(features)
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

    # ----------------------------------------------------------------
    #  forward: 训练时的前向传播
    # ----------------------------------------------------------------
    def forward(self, features, captions, lengths):
        """
        训练时的前向传播。

        Args:
            features:  [B, T, D_vid]        - 视频特征
            captions:  [B, max_caption_len]  - token IDs (已 tokenize + padding)
            lengths:   [B]                   - 每条 caption 的实际长度

        Returns:
            loss: scalar - Causal LM loss (由 HuggingFace 内部计算)
        """
        # 1. 视频编码 + 投影
        v = self.encode_projected_tokens(features).to(self.llm.dtype)

        # 2. 获取 caption 的 token embeddings
        token_embeds = self.llm.get_input_embeddings()(captions)  # [B, L, hidden_dim]
        token_embeds = token_embeds.to(v.dtype)

        # 3. 拼接: [video_tokens | caption_tokens]
        input_embeds = torch.cat([v, token_embeds], dim=1)  # [B, N+L, hidden_dim]

        # 4. 构造 attention mask
        v_mask = torch.ones(v.shape[0], v.shape[1], dtype=torch.long, device=v.device)
        text_mask = (captions != self.tokenizer.pad_token_id).long()
        attention_mask = torch.cat([v_mask, text_mask], dim=1)  # [B, N+L]

        # 5. 构造 labels: 只在 caption 部分计算 loss，视频 token 部分置 -100
        labels = captions.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        v_labels = torch.full(
            (v.shape[0], v.shape[1]), -100,
            dtype=labels.dtype, device=labels.device,
        )
        labels = torch.cat([v_labels, labels], dim=1)  # [B, N+L]

        # 整个 batch 都是 pad 时，HF 的 ignore_index CE 可能返回 NaN。
        if not torch.any(labels != -100):
            return v.sum() * 0.0

        # 6. LLM forward (HuggingFace 自动计算 cross-entropy loss)
        outputs = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        return outputs.loss

    # ----------------------------------------------------------------
    #  sample: 推理时的自回归采样
    # ----------------------------------------------------------------
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

    def sample(self, features, max_seq_length=None, generation_config=None):
        """
        推理时的自回归采样。

        Args:
            features: [T, D_vid]  - 单条样本（无 batch 维度）
            max_seq_length: 最大生成长度

        Returns:
            text: str - 生成的文本
        """
        self.eval()
        with torch.no_grad():
            outputs = self.generate_sequences(
                features.unsqueeze(0),
                max_seq_length=max_seq_length,
                generation_config=generation_config,
            )

            # 解码
            generated_text = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
            generated_text = generated_text.strip()
            if generated_text.startswith(":"):
                generated_text = generated_text[1:].strip()
            if "\n" in generated_text:
                generated_text = generated_text.split("\n", 1)[0].strip()

        return generated_text

    # ----------------------------------------------------------------
    #  get_trainable_parameters: 返回可训练参数
    # ----------------------------------------------------------------
    def get_trainable_parameters(self):
        """返回可训练参数列表 (LoRA + MLP + 未冻结的 encoder)"""
        return [p for p in self.parameters() if p.requires_grad]


# ============================================================================
#  Smoke Test
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  model_qwen.py - Architecture Info (no actual model load)")
    print("=" * 60)
    print()
    print("  This module provides Video2CaptionQwen:")
    print("    - Qwen2.5-7B + LoRA for Dense Video Captioning")
    print("    - VideoEncoder -> MLP(GELU) -> Qwen(LoRA)")
    print()
    print("  To test with actual model weights, run:")
    print("    python model_qwen.py --llm_model_path Qwen/Qwen2.5-7B")
    print()
    print("  Required packages: transformers, peft, accelerate")
    print("=" * 60)
