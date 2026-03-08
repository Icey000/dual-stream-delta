"""
=====================================
# 1. 先查缺失（有没有 audio_clap.npy 根本就没生成的）
python check_and_download_missing.py

# 2. 检查已有 WAV 的实际质量，清理掉损坏的
python 3_check_wav_integrity.py
# → 生成 broken_wav_games.txt（坏 WAV）
# → 生成 need_720p_games.txt（MKV 头骗人、需进一步调查的）

# 3. 对 broken_wav_games.txt → 重提音频（MKV 好的直接重跑 ffmpeg）
python 1_download_and_extract.py --from_txt broken_wav_games.txt

# 4. 对还是提不出来的（原始音轨本身就断）→ 直接生成全 0 dummy
python 4_generate_dummy_audio.py --from_txt need_720p_games.txt

# 5. 对有了正常 WAV 的 → 提 CLAP 特征
python 2_extract_clap_features.py --from_txt broken_wav_games.txt

======================================================
4_generate_dummy_audio.py
=========================
对 WAV 音频损坏（MKV 原始音轨就缺失/截断）的比赛，
生成一个全 0 的"静音" audio_clap.npy，形状与视觉特征对齐。

为什么这样做：
  - TEST 需要所有比赛都出预测，不能跳过（否则 evaluate_dvc 找不到文件会崩溃）
  - 如果 audio_clap.npy 不存在，PredictionCaptionsDual 会跳过这场比赛
  - 有了全 0 的 NPY，dataset_dual 可以正常加载，模型能用视觉特征做预测
  - 全 0 音频 → Q-Former 把音频 token 压到全 0 → 模型事实上只靠视觉输出预测
  - 这比完全跳过（0预测）至少还能拿一点视觉维度的分

用法:
  # 对需要处理的游戏列表生成 dummy 音频（通常是 need_720p_games.txt 或 broken_wav_games.txt）
  python 4_generate_dummy_audio.py --from_txt need_720p_games.txt

  # 也可以不加 --from_txt，自动扫描所有缺少 audio_clap.npy 的场次
  python 4_generate_dummy_audio.py
"""

import os
import argparse
import numpy as np

# ================= 配置区 =================
VISION_ROOT  = os.environ.get("SOCCERNET_VISION", "./data/caption-2024/")
AUDIO_ROOT   = os.environ.get("SOCCERNET_AUDIO", "./data/SoccerNet/")
VISION_FEAT  = "baidu_soccer_embeddings.npy"
AUDIO_DIM    = 512    # LAION-CLAP 输出维度，与 args.audio_input_dim 一致
SPLITS       = ["train", "valid", "test"]
# ==========================================

parser = argparse.ArgumentParser()
parser.add_argument("--from_txt",    type=str, default=None,
                    help="只对此 txt 文件中的比赛生成 dummy 音频")
parser.add_argument("--vision_root", type=str, default=VISION_ROOT)
parser.add_argument("--audio_root",  type=str, default=AUDIO_ROOT)
parser.add_argument("--audio_dim",   type=int, default=AUDIO_DIM,
                    help="CLAP 特征维度（默认 512）")
parser.add_argument("--splits",      nargs="+", default=SPLITS)
parser.add_argument("--overwrite",   action="store_true",
                    help="如果 audio_clap.npy 已存在，是否覆盖（默认跳过）")
args = parser.parse_args()


# ══════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════

def fmt_size(path):
    if not os.path.isfile(path):
        return "不存在"
    return f"{os.path.getsize(path) / (1024*1024):.2f} MB"


def get_vision_timesteps(vision_path):
    """
    读取视觉特征 NPY 文件，返回时序长度 T。
    视觉特征形状通常是 [T, D_vision]。
    """
    try:
        feat = np.load(vision_path)
        return feat.shape[0]
    except Exception as e:
        return None


# ══════════════════════════════════════════════════════
#  确定要处理的比赛列表
# ══════════════════════════════════════════════════════

vision_root = args.vision_root.rstrip("/")
audio_root  = args.audio_root.rstrip("/")

print(f"\n{'='*65}")
print(f"  STEP 4: 为音轨损坏的比赛生成 dummy audio_clap.npy")
print(f"  视觉根目录 : {vision_root}")
print(f"  音频根目录 : {audio_root}")
print(f"  CLAP 维度  : {args.audio_dim}")
print(f"  覆盖已有   : {'是' if args.overwrite else '否（跳过）'}")
print(f"{'='*65}\n")

if args.from_txt:
    if not os.path.isfile(args.from_txt):
        print(f"❌ 找不到文件: {args.from_txt}")
        exit(1)
    with open(args.from_txt) as f:
        target_games = [line.strip() for line in f if line.strip()]
    print(f"📋 从 {args.from_txt} 读入 {len(target_games)} 场比赛\n")
else:
    # 自动扫描：找所有缺 audio_clap.npy 的场次
    print("📋 自动扫描模式：在音频目录里找所有缺 audio_clap.npy 的场次...\n")
    from SoccerNet.Downloader import getListGames
    all_from_sn = []
    for split in args.splits:
        all_from_sn.extend(getListGames(split, task="caption"))
    target_games = []
    for game in all_from_sn:
        a1 = os.path.join(audio_root, game, "1_audio_clap.npy")
        a2 = os.path.join(audio_root, game, "2_audio_clap.npy")
        if not os.path.isfile(a1) or not os.path.isfile(a2):
            target_games.append(game)
    print(f"  发现 {len(target_games)} 场缺少音频特征的比赛\n")


# ══════════════════════════════════════════════════════
#  逐场生成 dummy NPY
# ══════════════════════════════════════════════════════

ok_count      = 0   # 成功生成
skip_count    = 0   # 已存在跳过
fail_count    = 0   # 失败
fail_list     = []

print(f"{'─'*65}")

for i, game in enumerate(target_games):
    print(f"\n[{i+1}/{len(target_games)}] {game}")

    for half in [1, 2]:
        audio_dir  = os.path.join(audio_root, game)
        npy_path   = os.path.join(audio_dir, f"{half}_audio_clap.npy")
        vision_path = os.path.join(vision_root, game, f"{half}_{args.vision_root.split('/')[-2] if False else 'baidu_soccer_embeddings'}.npy")
        # 更可靠的视觉路径拼接
        vision_path = os.path.join(vision_root, game, f"{half}_baidu_soccer_embeddings.npy")

        print(f"  half={half}:")

        # 已存在？
        if os.path.isfile(npy_path) and not args.overwrite:
            print(f"    ⏭  audio_clap.npy 已存在 [{fmt_size(npy_path)}]，跳过（加 --overwrite 强制覆盖）")
            skip_count += 1
            continue

        # 获取视觉特征时序长度（让 dummy 长度相近）
        T = get_vision_timesteps(vision_path)
        if T is None:
            # 视觉特征也读不到？用保守默认值（45分钟 × 0.5fps = 1350）
            T = 1350
            print(f"    ⚠️  视觉特征读取失败，使用默认时序长度 T={T}")
            print(f"       视觉路径不存在: {vision_path}")
        else:
            print(f"    📐 视觉特征时序长度 T={T}（与之对齐）")

        # 生成全 0 dummy NPY
        os.makedirs(audio_dir, exist_ok=True)
        dummy = np.zeros((T, args.audio_dim), dtype=np.float32)

        try:
            np.save(npy_path, dummy)
            saved_size = fmt_size(npy_path)
            print(f"    ✅ 已生成 dummy NPY → 形状: {dummy.shape}  [{saved_size}]")
            print(f"       （全 0 静音特征，双流 Q-Former 会仅靠视觉特征输出预测）")
            ok_count += 1
        except Exception as e:
            print(f"    ❌ 写入失败: {e}")
            fail_count += 1
            fail_list.append((game, half, str(e)))


# ══════════════════════════════════════════════════════
#  汇总
# ══════════════════════════════════════════════════════

print(f"\n{'='*65}")
print(f"  最终汇总")
print(f"{'='*65}")
print(f"  处理比赛场次        : {len(target_games)} 场")
print(f"  ✅ 成功生成 dummy    : {ok_count} 个半场")
print(f"  ⏭  跳过 (NPY已存在) : {skip_count} 个半场")
print(f"  ❌ 生成失败          : {fail_count} 个半场")

if fail_list:
    print(f"\n  失败详情：")
    for game, half, reason in fail_list:
        print(f"    · {game}  half={half}  原因: {reason}")

if fail_count == 0 and ok_count > 0:
    print(f"""
  ✅ 全部 dummy 音频已生成！

  接下来可以直接跑 TEST：
  → python main.py ... --use_dual_stream  （你的训练/测试命令）

  这些場次在推理时会用全 0 音频，模型靠视觉特征做预测，
  不会跳过也不会崩溃，比 0 分强。
""")
elif ok_count == 0 and skip_count > 0:
    print(f"\n  所有 NPY 均已存在，无需生成。")

print(f"{'='*65}\n")
