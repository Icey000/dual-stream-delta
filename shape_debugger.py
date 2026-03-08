#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
shape_debugger.py  ——  逐层打印每个子模块的 输入/输出 shape

使用方法（在服务器上的项目根目录执行）:
    python shape_debugger.py --task classify   # 双流 Classification
    python shape_debugger.py --task spotting   # 双流 Spotting
    python shape_debugger.py --task caption    # 双流 Caption + GPT-2
    python shape_debugger.py --task single     # 单流 VideoEncoder

可选参数:
    --batch-size  N   一次喂进去多少个样本 (默认 2)
    --seq-len     N   时间步长度 (默认 30, 对应 15秒×2fps 的窗口)

输出说明:
    每行格式:  [模块名]  输入shape -> 输出shape
    箭头左边是进入该层之前的张量 shape，右边是该层处理后的 shape。
"""

import argparse
import sys
import os
from typing import List
import torch
import torch.nn as nn

# ─── Tee: 同时向终端 & 文件输出 ────────────────────────────────────────────────
class _Tee:
    """把 stdout 的输出同时写到文件和终端。"""
    def __init__(self, filepath: str):
        self._terminal = sys.stdout
        self._file = open(filepath, "w", encoding="utf-8")
        sys.stdout = self

    def write(self, msg):
        self._terminal.write(msg)
        self._file.write(msg)

    def flush(self):
        self._terminal.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._terminal
        self._file.close()

# ─── 导入项目模型 ───────────────────────────────────────────────────────────────
from dual_qformer import (
    DualVideo2Caption,
    DualVideo2Spot,
    DualVideo2Classification,
    DualStreamEncoder,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助：把 shape 转成好看的字符串
# ═══════════════════════════════════════════════════════════════════════════════
def _shape(t) -> str:
    """把张量（或张量列表/元组）转成易读的 shape 字符串。"""
    if isinstance(t, torch.Tensor):
        return str(tuple(t.shape))
    if isinstance(t, (list, tuple)):
        parts = [_shape(x) for x in t if isinstance(x, torch.Tensor)]
        return "(" + ", ".join(parts) + ")"
    return f"<{type(t).__name__}>"


# ═══════════════════════════════════════════════════════════════════════════════
# 核心：给每个子模块注册 hook，打印 输入→输出 shape
# ═══════════════════════════════════════════════════════════════════════════════
def register_all_hooks(model: nn.Module, skip_types=(nn.ModuleList, nn.Sequential)):
    """
    遍历 model 的所有具名子模块，给它们每一个注册 forward hook。
    每次 forward 时自动打印：  [层名]  输入shape -> 输出shape

    skip_types: 这些"容器"模块本身不包含计算，跳过以避免重复打印。
    """
    handles = []

    for name, module in model.named_modules():
        if name == "":           # 跳过模型本身（最顶层）
            continue
        if isinstance(module, skip_types):
            continue

        # 这里用 default argument capture 防止 Python 闭包坑
        def make_hook(n):
            def hook_fn(mod, inp, out):
                # 处理输入：取第一个张量作为代表
                if isinstance(inp, (list, tuple)):
                    inp_str = "(" + ", ".join(_shape(x) for x in inp) + ")"
                else:
                    inp_str = _shape(inp)
                out_str = _shape(out)
                print(f"  {n:55s}  {inp_str}  ->  {out_str}")
            return hook_fn

        h = module.register_forward_hook(make_hook(name))
        handles.append(h)

    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ═══════════════════════════════════════════════════════════════════════════════
# 生成 dummy 输入
# ═══════════════════════════════════════════════════════════════════════════════
def make_dummy(batch_size=2, v_dim=1024, a_dim=512, seq_len=30):
    """
    返回 video_feats [B, T, D_v] 和 audio_feats [B, T, D_a]。
    seq_len 不能超过 netvlad.py 中 PositionalEncoding 的 max_len（已改成 10000）。
    """
    v = torch.randn(batch_size, seq_len, v_dim)
    a = torch.randn(batch_size, seq_len, a_dim)
    return v, a


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="逐层 Shape Debugger for SoccerNet-DVC")
    parser.add_argument("--task", choices=["caption", "spotting", "classify", "single"],
                        default="classify",
                        help="单独运行某个任务")
    parser.add_argument("--all", action="store_true",
                        help="一次性运行 classify / spotting / caption 三个任务，"
                             "分别保存到 shape_debug_classify.txt / shape_debug_spotting.txt / shape_debug_caption.txt")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len",    type=int, default=30,
                        help="时间步长度 (默认30 = 15秒×2fps窗口)")
    args = parser.parse_args()

    tasks = ["classify", "spotting", "caption"] if args.all else [args.task]
    for task in tasks:
        _run_task(task, args.batch_size, args.seq_len)


def _run_task(task: str, batch_size: int, seq_len: int):
    """执行单个任务的 shape debug，并把输出保存到 shape_debug_{task}.txt。"""
    log_path = f"shape_debug_{task}.txt"
    tee = _Tee(log_path)

    vfeats, afeats = make_dummy(batch_size=batch_size, seq_len=seq_len)

    print(f"\n{'='*70}")
    print(f"  任务: {task}   batch_size={batch_size}   seq_len={seq_len}")
    print(f"  输入视频特征:  {tuple(vfeats.shape)}  (B, T, D_video=1024)")
    print(f"  输入音频特征:  {tuple(afeats.shape)}  (B, T, D_audio=512)")
    print(f"{'='*70}")
    print(f"\n  {'层名':55s}  输入 shape  ->  输出 shape")
    print(f"  {'-'*100}")

    # ── 构建 Encoder ──────────────────────────────────────────────────────────
    encoder = DualStreamEncoder(
        video_input_dim=vfeats.shape[-1],   # 1024
        audio_input_dim=afeats.shape[-1],   # 512
        hidden_dim=768,
        video_tokens=8,
        audio_tokens=8,
        num_heads=8,
        num_layers=4,
        dropout=0.0,
    )

    # ── 根据任务构建完整模型 ──────────────────────────────────────────────────
    if task == "classify":
        model = DualVideo2Classification(
            encoder=encoder,
            hidden_dim=768,
            num_classes=10,
        )
        forward_args = (vfeats, afeats)

    elif task == "spotting":
        model = DualVideo2Spot(
            encoder=encoder,
            hidden_dim=768,
            num_classes=18,         # SoccerNet 17个动作 + 1背景
        )
        forward_args = (vfeats, afeats)

    elif task == "caption":
        model = DualVideo2Caption(
            encoder=encoder,
            hidden_dim=768,
            vocab_size=50257,       # GPT-2 vocab size
        )
        # Caption 需要额外的 caption tokens 输入，这里用全零占位
        dummy_caption = torch.zeros(batch_size, 20, dtype=torch.long)
        dummy_lengths = torch.full((batch_size,), 20, dtype=torch.long)
        forward_args = (vfeats, afeats, dummy_caption, dummy_lengths)

    elif task == "single":
        # 单流：只给视频，不需要 audio
        from model import VideoEncoder, Video2Spot
        single_enc = VideoEncoder(
            input_size=vfeats.shape[-1],
            num_classes=18,
            pool="QFormer",
            proj_size=768,
        )
        model = single_enc
        forward_args = (vfeats,)

    else:
        raise ValueError(task)

    model.eval()

    # ── 注册所有子模块的 hook ─────────────────────────────────────────────────
    handles = register_all_hooks(model)

    # ── 前向传播（会自动触发所有 hook）──────────────────────────────────────
    with torch.no_grad():
        output = model(*forward_args)

    # ── 清除 hook，打印最终输出 ───────────────────────────────────────────────
    remove_hooks(handles)

    print(f"\n{'='*70}")
    print(f"  最终输出 shape: {_shape(output)}")
    print(f"{'='*70}\n")

    # ── 额外：打印整个模型的层次结构（不含参数）────────────────────────────
    print("\n模型层次结构（named_modules）：")
    for name, mod in model.named_modules():
        if name:
            depth = name.count(".")
            indent = "  " + "  " * depth
            print(f"{indent}{name:60s}  [{mod.__class__.__name__}]")

    # ── 关闭 Tee，恢复 stdout，告知保存路径 ──────────────────────────────────
    tee.close()
    print(f"\n[已保存] {log_path}")


if __name__ == "__main__":
    main()
