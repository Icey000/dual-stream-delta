import os
import numpy as np
import librosa
import torch
import laion_clap
import argparse
import time
from SoccerNet.Downloader import getListGames

# ================= 配置区 =================
ROOT_DIR   = os.environ.get("SOCCERNET_ROOT", "./data/SoccerNet")
SPLITS     = ["train", "valid", "test"]
WINDOW_SEC = 2.0    # 每 2 秒切一刀提取一个特征 token
BATCH_SIZE = 32     # 显卡每次吞吐的音频块数
# ==========================================

parser = argparse.ArgumentParser()
parser.add_argument("--from_txt", type=str, default=None,
                    help="只对此 txt 文件内的比赛提取 CLAP 特征，"
                         "通常是 missing_audio_games.txt 或 broken_wav_games.txt")
args = parser.parse_args()


# ══════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════

def fmt_size(path):
    if not os.path.isfile(path):
        return "不存在"
    mb = os.path.getsize(path) / (1024 * 1024)
    return f"{mb:.2f} MB"


def extract_clap_from_wav(model, wav_path, npy_path):
    """
    从 WAV 文件提取 CLAP 特征并保存为 NPY。
    返回 True/False。
    """
    # 已存在跳过
    if os.path.isfile(npy_path):
        npy_size = fmt_size(npy_path)
        print(f"      ⏭  NPY 已存在 [{npy_size}]，跳过")
        return "skip"

    # WAV 不存在
    if not os.path.isfile(wav_path):
        print(f"      ❌ WAV 不存在: {os.path.basename(wav_path)}")
        print(f"         请先运行 1_download_and_extract.py！")
        return "missing"

    wav_size = fmt_size(wav_path)
    print(f"      📂 加载 WAV [{wav_size}]...")
    t0 = time.time()

    try:
        y, sr = librosa.load(wav_path, sr=48000)
    except Exception as e:
        print(f"      ❌ librosa 加载失败: {e}")
        return "fail"

    load_time = time.time() - t0
    duration_min = len(y) / sr / 60
    chunk_size   = int(WINDOW_SEC * sr)
    total_chunks = len(y) // chunk_size

    print(f"      ⏱  加载耗时: {load_time:.1f}s  |  时长: {duration_min:.1f}分钟  |  "
          f"共 {total_chunks} 个 {WINDOW_SEC:.0f}s 切片")
    print(f"      🔊 开始 CLAP 特征提取 (batch_size={BATCH_SIZE})...")

    all_embeddings = []
    t1 = time.time()

    try:
        for i in range(0, total_chunks, BATCH_SIZE):
            batch_y = []
            for j in range(i, min(i + BATCH_SIZE, total_chunks)):
                start = j * chunk_size
                end   = start + chunk_size
                batch_y.append(y[start:end])

            with torch.no_grad():
                embeds = model.get_audio_embedding_from_data(x=batch_y, use_tensor=False)
                all_embeddings.append(embeds)

            # 进度打印（每 10 个 batch 报一次）
            batches_done = (i // BATCH_SIZE) + 1
            total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE
            if batches_done % 10 == 0 or batches_done == total_batches:
                pct = batches_done / total_batches * 100
                print(f"      ... batch {batches_done}/{total_batches} ({pct:.0f}%)")

    except Exception as e:
        print(f"      ❌ CLAP 提取过程出错: {e}")
        return "fail"

    if all_embeddings:
        features = np.vstack(all_embeddings)
        np.save(npy_path, features)
        extract_time = time.time() - t1
        npy_size = fmt_size(npy_path)
        print(f"      ✅ NPY 已保存 → 形状: {features.shape}  [{npy_size}]  提取耗时: {extract_time:.1f}s")
        return "ok"
    else:
        print(f"      ❌ all_embeddings 为空，NPY 未保存")
        return "fail"


# ══════════════════════════════════════════════════════
#  主逻辑
# ══════════════════════════════════════════════════════

print(f"\n{'='*65}")
print(f"  STEP 2: 提取 LAION-CLAP 音频特征")
print(f"  ROOT_DIR   : {ROOT_DIR}")
print(f"  WINDOW_SEC : {WINDOW_SEC}s  |  BATCH_SIZE: {BATCH_SIZE}")
print(f"{'='*65}\n")

print(">>> 加载 LAION-CLAP 模型...")
t_load = time.time()
global_clap_model = laion_clap.CLAP_Module(enable_fusion=False)
global_clap_model.load_ckpt()
print(f"    ✅ 模型加载完成，耗时 {time.time()-t_load:.1f}s\n")

# ── 决定要处理的比赛列表 ─────────────────────────────
if args.from_txt:
    if not os.path.isfile(args.from_txt):
        print(f"❌ 找不到文件: {args.from_txt}")
        exit(1)
    with open(args.from_txt) as f:
        all_target_games = [line.strip() for line in f if line.strip()]
    print(f"📋 从 {args.from_txt} 读入 {len(all_target_games)} 场比赛\n")
    games_with_split = [(g, "(from txt)") for g in all_target_games]
else:
    games_with_split = []
    for split in SPLITS:
        for g in getListGames(split, task="caption"):
            games_with_split.append((g, split))

# ── 汇总统计 ─────────────────────────────────────────
ok_count      = 0
skip_count    = 0
fail_count    = 0
missing_count = 0
fail_list     = []   # [(game, half, reason)]

script_start = time.time()
total = len(games_with_split)

print(f"共 {total} 场比赛，开始逐一处理...\n")
print(f"{'─'*65}")

for i, (game, split_label) in enumerate(games_with_split):
    print(f"\n[{i+1}/{total}] {game}  [{split_label}]")
    game_path = os.path.join(ROOT_DIR, game)

    for half in [1, 2]:
        wav_path = os.path.join(game_path, f"{half}_audio.wav")
        npy_path = os.path.join(game_path, f"{half}_audio_clap.npy")

        print(f"    half={half}:")
        result = extract_clap_from_wav(global_clap_model, wav_path, npy_path)

        if result == "ok":
            ok_count += 1
        elif result == "skip":
            skip_count += 1
        elif result == "missing":
            missing_count += 1
            fail_list.append((game, half, "WAV 不存在"))
        elif result == "fail":
            fail_count += 1
            fail_list.append((game, half, "提取过程报错"))


# ══════════════════════════════════════════════════════
#  最终汇总
# ══════════════════════════════════════════════════════

total_elapsed = time.time() - script_start
print(f"\n{'='*65}")
print(f"  最终汇总报告  (总耗时: {total_elapsed/60:.1f} 分钟)")
print(f"{'='*65}")
print(f"  处理比赛场次        : {total} 场")
print(f"  ─── CLAP 提取结果（半场粒度） ──────────────────")
print(f"  ✅ 提取成功         : {ok_count} 个半场")
print(f"  ⏭  跳过(NPY已存在)  : {skip_count} 个半场")
print(f"  ❌ WAV 不存在       : {missing_count} 个半场")
print(f"  ❌ 提取过程失败     : {fail_count} 个半场")

if fail_list:
    print(f"\n  失败详情:")
    for game, half, reason in fail_list:
        print(f"    · {game}  half={half}  原因: {reason}")

total_fail = fail_count + missing_count
if total_fail == 0:
    print(f"\n  🎉 全部成功！没有任何失败！")
else:
    print(f"\n  ⚠️  共 {total_fail} 个半场失败，请检查上方详情")
    if missing_count > 0:
        print(f"     → 有 WAV 不存在的场次，请先运行 1_download_and_extract.py")

print(f"{'='*65}\n")