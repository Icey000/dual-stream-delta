"""
inspect_model_flow.py
=====================
诊断脚本：打印整个双流模型结构，并用假数据追踪每一层的张量维度变化。

使用方法（不需要真实数据和GPU也能跑）:
    python inspect_model_flow.py

可选参数（命令行）:
    --video_input_dim   视觉特征原始维度  (default: 8576)
    --audio_input_dim   音频特征原始维度  (default: 512)
    --hidden_dim        encoder内部维度   (default: 768  ← classifying阶段)
                        captioning阶段会自动用 Qwen hidden_size (3584)
    --video_tokens      视觉Q-Former输出token数 (default: 8)
    --audio_tokens      音频Q-Former输出token数 (default: 8)
    --batch_size        (default: 2)
    --T_vid             视频序列长度 (default: 30)
    --T_aud             音频序列长度 (default: 30)
    --no_llm            跳过加载Qwen LLM，仅分析Encoder部分 (default: False)
    --log_file          输出到文件，如 model_flow.txt (default: 仅打印到终端)



python inspect_model_flow.py --no_llm --log_file model_flow.txt


python inspect_model_flow.py --log_file model_flow.txt


python inspect_model_flow.py --no_llm --log_file model_flow.txt --video_input_dim 8576 --audio_input_dim 512 --hidden_dim 768 --video_tokens 8 --audio_tokens 8 --batch_size 2 --T_vid 30 --T_aud 30


输入: Video [B, T, 8576]  +  Audio [B, T, 512]
         ↓                         ↓
   video_pre_proj             audio_pre_proj       ← 普通线性层，降/升维
   Linear(8576 → H)           Linear(512 → H)
         ↓                         ↓
   video_qformer             audio_qformer          ← Q-Former（核心）
   [B, T, H] → [B, 8, H]     [B, T, H] → [B, 8, H]  T帧 → 固定8个token
         ↓_________________________↓
              cat(dim=1) + LayerNorm
              [B, 16, H]                            ← 双流晚期融合
                    ↓
        ┌───────────┼──────────────┐
        ↓           ↓              ↓
   Classifying   Spotting      Captioning
   mean→Linear  mean→MLP      MLP→Qwen2.5-7B(LoRA)




"""



import argparse
import sys
import io
import textwrap
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────
# 彩色终端输出 helper
# ─────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
MAGENTA= "\033[95m"
BLUE   = "\033[94m"

def h1(msg):  print(f"\n{BOLD}{CYAN}{'='*70}{RESET}\n{BOLD}{CYAN}  {msg}{RESET}\n{BOLD}{CYAN}{'='*70}{RESET}")
def h2(msg):  print(f"\n{BOLD}{BLUE}  ── {msg}{RESET}")
def h3(msg):  print(f"\n{BOLD}{MAGENTA}    ▶ {msg}{RESET}")
def ok(msg):  print(f"      {GREEN}✓ {msg}{RESET}")
def info(msg):print(f"      {YELLOW}→ {msg}{RESET}")
def err(msg): print(f"      {RED}✗ {msg}{RESET}")
def sep():    print(f"  {BOLD}{'─'*66}{RESET}")

# ─────────────────────────────────────────────────────────────
# 张量 shape 追踪 hook
# ─────────────────────────────────────────────────────────────
_shape_log = []   # [(module_name, input_shapes, output_shape)]

def make_hook(name):
    def hook(module, inp, out):
        in_shapes  = [tuple(t.shape) if isinstance(t, torch.Tensor) else type(t).__name__
                      for t in inp]
        out_shape  = tuple(out.shape) if isinstance(out, torch.Tensor) else type(out).__name__
        _shape_log.append((name, in_shapes, out_shape))
    return hook

def register_hooks(model, prefix=""):
    """为模型所有子模块注册前向hook（只记录叶子层或关键层）。"""
    handles = []
    for name, module in model.named_modules():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(module, (nn.Linear, nn.LayerNorm,
                                nn.TransformerDecoder, nn.TransformerEncoder,
                                nn.MultiheadAttention)):
            handles.append(module.register_forward_hook(make_hook(full_name)))
    return handles

def remove_hooks(handles):
    for h in handles:
        h.remove()

# ─────────────────────────────────────────────────────────────
# 打印参数表
# ─────────────────────────────────────────────────────────────
def print_param_table(model, model_name="Model"):
    h2(f"参数统计: {model_name}")
    total = 0
    trainable = 0
    rows = []
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
        rows.append((name, tuple(param.shape), n, param.requires_grad))

    # 只打印最重要的几层
    print(f"    {'参数名':<55} {'Shape':<25} {'数量':>10}  {'可训练'}")
    print(f"    {'─'*55} {'─'*25} {'─'*10}  {'─'*4}")
    for (nm, sh, n, req) in rows[:60]:   # 最多60行
        flag = GREEN+"✓"+RESET if req else RED+"✗"+RESET
        print(f"    {nm:<55} {str(sh):<25} {n:>10,}  {flag}")
    if len(rows) > 60:
        print(f"    ... (共 {len(rows)} 个参数组，只显示前60个)")
    print()
    ok(f"总参数:      {total:>15,}")
    ok(f"可训练参数:  {trainable:>15,}")
    ok(f"冻结参数:    {total-trainable:>15,}")

# ─────────────────────────────────────────────────────────────
# 打印 shape flow 日志
# ─────────────────────────────────────────────────────────────
def print_shape_log(log, title="张量维度流"):
    h2(title)
    print(f"    {'层名':<55} {'输入shape':<35} {'输出shape'}")
    print(f"    {'─'*55} {'─'*35} {'─'*25}")
    for (name, in_sh, out_sh) in log:
        in_str = str(in_sh[0]) if len(in_sh)==1 else str(in_sh)
        print(f"    {name:<55} {in_str:<35} {str(out_sh)}")

# ─────────────────────────────────────────────────────────────
# 架构文字说明
# ─────────────────────────────────────────────────────────────
ARCH_EXPLAIN = """
┌─────────────────────────────────────────────────────────────────────────────┐
│               双流模型架构总览 (Dual-Stream Architecture)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────┐                      │
│  │         输入 (Input Features)                    │                      │
│  │   Video: [B, T_vid, video_input_dim=8576]        │                      │
│  │   Audio: [B, T_aud, audio_input_dim=512]         │                      │
│  └──────────────┬───────────────────┬───────────────┘                      │
│                 │                   │                                       │
│       ┌─────────▼──────┐   ┌────────▼────────┐                             │
│       │  video_pre_proj│   │  audio_pre_proj │   ← nn.Linear               │
│       │ Linear(8576,H) │   │  Linear(512, H) │     H = hidden_dim          │
│       └─────────┬──────┘   └────────┬────────┘                             │
│                 │                   │                                       │
│    [B,T_vid,H] ─┘           └─[B,T_aud,H]                                  │
│                                                                             │
│       ┌──────────────┐      ┌───────────────┐                              │
│       │ video_qformer│      │ audio_qformer │  ← QFormerVideoPooling       │
│       │ (TransDecoder│      │ (TransDecoder │    可学习Query tokens        │
│       │ + learnable Q│      │ + learnable Q │    cross-attend输入序列      │
│       │  Tokens)     │      │  Tokens)      │                              │
│       └──────┬───────┘      └───────┬───────┘                              │
│              │                      │                                       │
│   [B,N_v,H] ─┘                      └─ [B,N_a,H]                           │
│         N_v=video_tokens=8               N_a=audio_tokens=8                 │
│                                                                             │
│       ┌──────────────────────────────────┐                                 │
│       │      cat(dim=1) + LayerNorm      │  晚期融合 Late Fusion            │
│       │    [B, N_v+N_a, H] = [B,16, H]  │                                 │
│       └──────────────────┬───────────────┘                                 │
│                          │                                                  │
│          ┌───────────────┼──────────────────┐                              │
│          │               │                  │                              │
│   ┌──────▼──────┐ ┌──────▼──────┐ ┌────────▼──────┐                       │
│   │ Classifying │ │  Spotting   │ │  Captioning   │  ← 三个下游任务        │
│   │     Task    │ │    Task     │ │     Task      │                        │
│   │             │ │             │ │               │                        │
│   │  mean(dim=1)│ │  mean(dim=1)│ │  MLP(GELU)    │                       │
│   │    [B,H]    │ │    [B,H]    │ │  [B,16,H]     │                       │
│   │             │ │             │ │               │                        │
│   │  Linear(H,C)│ │ MLP head    │ │  Qwen2.5-7B   │                       │
│   │  [B, n_cls] │ │ [B, n_cls] │ │  (LoRA)       │                       │
│   └─────────────┘ └─────────────┘ │               │                       │
│                                   │  hidden_size= │                        │
│                                   │    3584       │                        │
│                                   └───────────────┘                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  ⚠️  关键问题：H (hidden_dim) 在两个阶段不同！                               │
│                                                                             │
│   Phase 1 Classifying: H = 768  (gpt_type="gpt2" → feature_size["gpt2"])  │
│   Phase 2 Captioning:  H = 3584 (自动从 Qwen2.5-7B.config.hidden_size 读)  │
│                                                                             │
│   → video_pre_proj.weight 在checkpoint=[768,8576], 当前模型=[3584,8576]     │
│   → 这就是 RuntimeError: size mismatch 的根本原因！                         │
└─────────────────────────────────────────────────────────────────────────────┘
"""

# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="模型结构与维度流诊断工具")
    parser.add_argument("--video_input_dim", type=int, default=8576)
    parser.add_argument("--audio_input_dim", type=int, default=512)
    parser.add_argument("--hidden_dim",      type=int, default=768,
                        help="Classifying阶段的hidden_dim(默认768); "
                             "Captioning阶段Qwen会自动覆盖为3584")
    parser.add_argument("--video_tokens",    type=int, default=8)
    parser.add_argument("--audio_tokens",    type=int, default=8)
    parser.add_argument("--batch_size",      type=int, default=2)
    parser.add_argument("--T_vid",           type=int, default=30)
    parser.add_argument("--T_aud",           type=int, default=30)
    parser.add_argument("--no_llm",          action="store_true",
                        help="跳过Qwen LLM加载，只分析Encoder")
    parser.add_argument("--log_file",        type=str, default=None,
                        help="同时写入到文件（如 model_flow.txt）")
    args = parser.parse_args()

    # ── 可选：同时写文件 ──────────────────────────────────────
    if args.log_file:
        tee = open(args.log_file, "w", encoding="utf-8")
        class Tee:
            def __init__(self, *streams): self.streams = streams
            def write(self, s):
                for st in self.streams: st.write(s)
            def flush(self):
                for st in self.streams: st.flush()
        sys.stdout = Tee(sys.__stdout__, tee)

    # ═══════════════════════════════════════════════════════════
    h1("架构总览说明")
    print(ARCH_EXPLAIN)

    # ═══════════════════════════════════════════════════════════
    h1("Phase 1: DualVideo2Classification (Classifying 阶段)")
    # ────────────────────────────────────────────────────────────
    # 这个阶段用 gpt_type="gpt2" → hidden_dim=768
    CLASSIFY_HIDDEN = 768
    info(f"hidden_dim = {CLASSIFY_HIDDEN}  (来自 gpt_type='gpt2' → feature_size['gpt2'])")
    info(f"video_input_dim = {args.video_input_dim}")
    info(f"audio_input_dim = {args.audio_input_dim}")
    info(f"video_tokens = {args.video_tokens}, audio_tokens = {args.audio_tokens}")

    # 导入类
    sys.path.insert(0, ".")
    try:
        from dual_qformer import DualVideo2Classification, DualStreamEncoder
        from pooling import QFormerVideoPooling
    except ImportError as e:
        err(f"导入失败: {e}")
        err("请确保在项目根目录下运行此脚本")
        sys.exit(1)

    num_classes = 17
    classify_model = DualVideo2Classification(
        num_classes=num_classes,
        video_input_dim=args.video_input_dim,
        audio_input_dim=args.audio_input_dim,
        hidden_dim=CLASSIFY_HIDDEN,
        video_tokens=args.video_tokens,
        audio_tokens=args.audio_tokens,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
    )
    classify_model.eval()

    h2("模型结构打印 (DualVideo2Classification)")
    print(classify_model)

    print_param_table(classify_model, "DualVideo2Classification")

    # ── 假数据前向传播 ──────────────────────────────────────
    h2(f"假数据前向传播追踪 (B={args.batch_size}, T_vid={args.T_vid}, T_aud={args.T_aud})")
    vid = torch.randn(args.batch_size, args.T_vid, args.video_input_dim)
    aud = torch.randn(args.batch_size, args.T_aud, args.audio_input_dim)
    info(f"输入 video_feats: {tuple(vid.shape)}")
    info(f"输入 audio_feats: {tuple(aud.shape)}")

    _shape_log.clear()
    handles = register_hooks(classify_model)
    with torch.no_grad():
        out = classify_model(vid, aud)
    remove_hooks(handles)

    print_shape_log(_shape_log, f"DualVideo2Classification 维度流 (hidden={CLASSIFY_HIDDEN})")
    sep()
    ok(f"最终分类输出: {tuple(out.shape)}  (B={args.batch_size}, num_classes={num_classes})")

    # ── 关键维度手动标注 ────────────────────────────────────
    h2("关键维度节点 (手动标注)")
    steps_classify = [
        ("video_feats 输入",    f"[{args.batch_size}, {args.T_vid}, {args.video_input_dim}]",
                                 "原始视觉特征"),
        ("video_pre_proj 输出", f"[{args.batch_size}, {args.T_vid}, {CLASSIFY_HIDDEN}]",
                                 "Linear(8576→768) 降维投影"),
        ("video_qformer 输出",  f"[{args.batch_size}, {args.video_tokens}, {CLASSIFY_HIDDEN}]",
                                 "Q-Former压缩：T_vid → N_v token"),
        ("audio_feats 输入",    f"[{args.batch_size}, {args.T_aud}, {args.audio_input_dim}]",
                                 "原始音频特征"),
        ("audio_pre_proj 输出", f"[{args.batch_size}, {args.T_aud}, {CLASSIFY_HIDDEN}]",
                                 "Linear(512→768) 投影"),
        ("audio_qformer 输出",  f"[{args.batch_size}, {args.audio_tokens}, {CLASSIFY_HIDDEN}]",
                                 "Q-Former压缩：T_aud → N_a token"),
        ("cat + LayerNorm",     f"[{args.batch_size}, {args.video_tokens+args.audio_tokens}, {CLASSIFY_HIDDEN}]",
                                 "融合: [B,8,768] + [B,8,768] → [B,16,768]"),
        ("mean(dim=1)",         f"[{args.batch_size}, {CLASSIFY_HIDDEN}]",
                                 "时间池化"),
        ("fc 输出",             f"[{args.batch_size}, {num_classes}]",
                                 "分类头 Linear(768→17)"),
    ]
    print(f"\n    {'节点':<35} {'张量Shape':<40} {'说明'}")
    print(f"    {'─'*35} {'─'*40} {'─'*30}")
    for step, shape, note in steps_classify:
        print(f"    {step:<35} {shape:<40} {note}")

    # ═══════════════════════════════════════════════════════════
    h1("Phase 2: DualVideo2CaptionLLM (Captioning 阶段)")
    # ────────────────────────────────────────────────────────────
    # 这个阶段的 hidden_dim 来自 Qwen2.5-7B.config.hidden_size = 3584
    CAPTION_HIDDEN = 3584  # Qwen2.5-7B hidden size
    info(f"hidden_dim = {CAPTION_HIDDEN}  ← 自动从 Qwen2.5-7B.config.hidden_size 读取")
    info(f"  (代码: actual_hidden = self.llm.config.hidden_size)")
    info(f"video_input_dim = {args.video_input_dim}")
    info(f"audio_input_dim = {args.audio_input_dim}")
    info(f"video_tokens = {args.video_tokens}, audio_tokens = {args.audio_tokens}")

    if args.no_llm:
        h2("--no_llm 模式：跳过Qwen加载，模拟CaptionLLM的Encoder部分")
        caption_encoder = DualStreamEncoder(
            video_input_dim=args.video_input_dim,
            audio_input_dim=args.audio_input_dim,
            hidden_dim=CAPTION_HIDDEN,
            video_tokens=args.video_tokens,
            audio_tokens=args.audio_tokens,
            num_heads=8,
            num_layers=4,
            dropout=0.1,
        )
        caption_encoder.eval()

        h2("模型结构打印 (DualStreamEncoder with Qwen hidden=3584)")
        print(caption_encoder)

        print_param_table(caption_encoder, "DualStreamEncoder (hidden=3584)")

        _shape_log.clear()
        handles = register_hooks(caption_encoder)
        with torch.no_grad():
            fused = caption_encoder(vid, aud)
        remove_hooks(handles)
        print_shape_log(_shape_log, f"DualStreamEncoder 维度流 (hidden={CAPTION_HIDDEN})")
        sep()
        ok(f"Encoder 输出 fused: {tuple(fused.shape)}")

        # MLP proj 模拟
        h2("MLP Projection 层 (GELU, 双层)")
        proj_hidden = CAPTION_HIDDEN * 4
        proj = nn.Sequential(
            nn.Linear(CAPTION_HIDDEN, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, CAPTION_HIDDEN),
        )
        proj.eval()
        with torch.no_grad():
            proj_out = proj(fused)
        ok(f"proj 输入: {tuple(fused.shape)}")
        ok(f"proj 中间: [{args.batch_size}, {args.video_tokens+args.audio_tokens}, {proj_hidden}]  (×4 expand)")
        ok(f"proj 输出: {tuple(proj_out.shape)}  → 这就是喂给Qwen的visual tokens")

    else:
        h2("尝试加载 DualVideo2CaptionLLM (需要 Qwen 模型文件)")
        try:
            from dual_qformer import DualVideo2CaptionLLM
            caption_model = DualVideo2CaptionLLM(
                video_input_dim=args.video_input_dim,
                audio_input_dim=args.audio_input_dim,
                llm_model_path="Qwen/Qwen2.5-7B",
                lora_r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                weights=None,
                weights_encoder=None,
                freeze_encoder=False,
            )
            caption_model.eval()

            h2("模型结构打印 (DualVideo2CaptionLLM)")
            print(caption_model)
            print_param_table(caption_model, "DualVideo2CaptionLLM")

            _shape_log.clear()
            handles = register_hooks(caption_model.encoder)
            with torch.no_grad():
                fused = caption_model.encoder(vid, aud)
            remove_hooks(handles)
            print_shape_log(_shape_log, f"Encoder 维度流 (hidden={CAPTION_HIDDEN})")
            sep()
            ok(f"Encoder 输出: {tuple(fused.shape)}")

            with torch.no_grad():
                proj_out = caption_model.proj(fused.float())
            ok(f"MLP proj 输出: {tuple(proj_out.shape)}  → 喂入Qwen作为visual prefix tokens")

        except Exception as e:
            err(f"加载 Qwen 失败（可能模型文件不在本地）: {e}")
            info("请使用 --no_llm 跳过LLM加载，只分析Encoder部分")

    # ── 关键维度节点（Captioning） ──────────────────────────
    h2("关键维度节点 (手动标注, Captioning阶段)")
    N = args.video_tokens + args.audio_tokens
    steps_caption = [
        ("video_feats 输入",     f"[B, T_vid, 8576]",       "原始视觉特征"),
        ("video_pre_proj 输出",  f"[B, T_vid, 3584]",       "Linear(8576→3584) ← 不同于Phase1!"),
        ("video_qformer 输出",   f"[B, {args.video_tokens}, 3584]",  "Q-Former压缩"),
        ("audio_feats 输入",     f"[B, T_aud, 512]",        "原始音频特征"),
        ("audio_pre_proj 输出",  f"[B, T_aud, 3584]",       "Linear(512→3584)"),
        ("audio_qformer 输出",   f"[B, {args.audio_tokens}, 3584]",  "Q-Former压缩"),
        ("cat + LayerNorm",      f"[B, {N}, 3584]",          f"融合: [B,8,3584]+[B,8,3584]→[B,{N},3584]"),
        ("MLP proj 中间",        f"[B, {N}, {3584*4}]",      "×4 expand (GELU)"),
        ("MLP proj 输出",        f"[B, {N}, 3584]",          "visual prefix tokens for Qwen"),
        ("Qwen2.5-7B 输入",      f"[B, T_text+{N}, 3584]",  "visual tokens拼接文本embeddings"),
        ("Qwen2.5-7B 输出",      f"[B, T_text+{N}, vocab]", "最终Caption预测"),
    ]
    print(f"\n    {'节点':<37} {'张量Shape':<30} {'说明'}")
    print(f"    {'─'*37} {'─'*30} {'─'*35}")
    for step, shape, note in steps_caption:
        print(f"    {step:<37} {shape:<30} {note}")

    # ═══════════════════════════════════════════════════════════
    h1("⚠️  维度不匹配：根本原因分析")

    print(f"""
  {BOLD}问题根源:{RESET}
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  classifying.py 第67行:                                         │
  │    hidden_dim = feature_size[args.gpt_type]                     │
  │                            ↓                                    │
  │    feature_size = {{"gpt2": 768, ...}}  → hidden_dim = {YELLOW}768{RESET}        │
  │                                                                  │
  │  dual_qformer.py 第365行 (DualVideo2CaptionLLM):                │
  │    actual_hidden = self.llm.config.hidden_size                  │
  │                            ↓                                    │
  │    Qwen2.5-7B.hidden_size = {RED}3584{RESET}  → hidden_dim = {RED}3584{RESET}        │
  │                                                                  │
  │  所以:                                                           │
  │    checkpoint 保存的是: video_pre_proj.weight [{YELLOW}768{RESET}, 8576]      │
  │    当前模型期望的是:    video_pre_proj.weight [{RED}3584{RESET}, 8576]     │
  │                                                                  │
  │  {BOLD}修复方案:{RESET}                                                      │
  │    在 classifying.py 第67行，将 hidden_dim 改为 {GREEN}3584{RESET}:            │
  │                                                                  │
  │    {GREEN}hidden_dim = 3584  # 与 Qwen2.5-7B hidden_size 一致{RESET}          │
  │                                                                  │
  │    然后重新训练 classifying 阶段，保存的 encoder 权重维度        │
  │    就能正确匹配 captioning 阶段了。                              │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
""")

    if args.log_file:
        sys.stdout = sys.__stdout__
        tee.close()
        print(f"\n{GREEN}已保存到: {args.log_file}{RESET}")

    h1("诊断完成 ✓")


if __name__ == "__main__":
    main()
