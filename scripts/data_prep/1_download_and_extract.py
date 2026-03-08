import os
import subprocess
import argparse
import time
from SoccerNet.Downloader import SoccerNetDownloader
import SoccerNet.Downloader
import SoccerNet.utils

# ================= 配置区 =================
ROOT_DIR = os.environ.get("SOCCERNET_ROOT", "./data/SoccerNet")
PASSWORD = "s0cc3rn3t"
SPLITS = ["train", "valid", "test"]
# ==========================================

parser = argparse.ArgumentParser()
parser.add_argument("--from_txt", type=str, default=None,
                    help="只处理此 txt 文件中列出的比赛（每行一个相对路径），"
                         "通常是 missing_audio_games.txt 或 broken_wav_games.txt")
parser.add_argument("--quality", type=str, default="720p",
                    choices=["224p", "720p"],
                    help="下载画质（默认 720p）")
args = parser.parse_args()

# 根据画质决定文件名和任务名
MKV_FILENAME = f"{{half}}_{args.quality}.mkv"   # 占位符，实际用时填 half
DL_TASK      = "caption-2024"                  # SoccerNet 任务名

print(f"画质模式: {args.quality}  (MKV 文件名: {{half}}_{args.quality}.mkv)\n")


# ══════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════

def fmt_size(path):
    """返回文件大小字符串，文件不存在返回 '不存在'"""
    if not os.path.isfile(path):
        return "不存在"
    mb = os.path.getsize(path) / (1024 * 1024)
    return f"{mb:.1f} MB"


def extract_audio_from_mkv(mkv_path, wav_path):
    """
    调用 FFmpeg 提取音频。
    返回 True/False，并打印 ffmpeg 的错误信息（不再 DEVNULL 静默吞错误）。
    """
    print(f"    [ffmpeg] {os.path.basename(mkv_path)} → {os.path.basename(wav_path)}")
    t0 = time.time()
    cmd = [
        'ffmpeg', '-y', '-i', mkv_path,
        '-vn', '-acodec', 'pcm_s16le', '-ar', '48000', '-ac', '1',
        wav_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    elapsed = time.time() - t0

    if result.returncode != 0:
        err_msg = result.stderr.decode('utf-8', errors='replace').strip()
        # 只打印最后几行（ffmpeg 错误通常在末尾）
        err_tail = "\n".join(err_msg.splitlines()[-5:]) if err_msg else "(无错误信息)"
        print(f"    ❌ ffmpeg 失败 (耗时 {elapsed:.1f}s):")
        print(f"       {err_tail}")
        return False

    wav_size = fmt_size(wav_path)
    print(f"    ✅ 提取成功！WAV 大小: {wav_size}，耗时: {elapsed:.1f}s")
    return True


# ══════════════════════════════════════════════════════
#  主逻辑
# ══════════════════════════════════════════════════════

print(f"\n{'='*65}")
print(f"  STEP 1+2: 下载 {args.quality} MKV + 提取 WAV 音频")
print(f"  ROOT_DIR: {ROOT_DIR}")
print(f"{'='*65}\n")

my_downloader = SoccerNetDownloader(LocalDirectory=ROOT_DIR)
my_downloader.password = PASSWORD
original_getListGames = SoccerNet.utils.getListGames

# ── 决定要处理的比赛列表 ──────────────────────────────
if args.from_txt:
    if not os.path.isfile(args.from_txt):
        print(f"❌ 找不到文件: {args.from_txt}")
        exit(1)
    with open(args.from_txt) as f:
        target_games = [line.strip() for line in f if line.strip()]
    print(f"📋 从 {args.from_txt} 读入 {len(target_games)} 场比赛\n")
    split_games_map = {split: [] for split in SPLITS}
    for g in target_games:
        placed = False
        for split in SPLITS:
            if g in original_getListGames(split, task="caption"):
                split_games_map[split].append(g)
                placed = True
                break
        if not placed:
            split_games_map["test"].append(g)
else:
    split_games_map = None

# ── 汇总统计 ──────────────────────────────────────────
total_games_processed = 0
dl_ok = 0;   dl_skip = 0;  dl_fail = 0
wav_ok = 0;  wav_skip = 0; wav_fail = 0
wav_fail_list = []    # [(game, half, reason)]
dl_fail_list  = []    # [game]

script_start = time.time()

for split in SPLITS:
    # 拿到这个 split 要处理的比赛列表
    if split_games_map is not None:
        all_games = split_games_map[split]
        if not all_games:
            print(f"[{split}] 无需处理的场次，跳过\n")
            continue
    else:
        all_games = original_getListGames(split, task="caption")

    print(f"\n{'─'*65}")
    print(f"  [{split}] 共 {len(all_games)} 场比赛")
    print(f"{'─'*65}")

    # ── STEP 1：扫描哪些场次需要下载 MKV ─────────────
    print(f"\n>>> 扫描硬盘状态...")
    games_to_download = []
    for game in all_games:
        game_path = os.path.join(ROOT_DIR, game)
        need_dl = False
        for half in [1, 2]:
            mkv_path = os.path.join(game_path, f"{half}_{args.quality}.mkv")
            wav_path = os.path.join(game_path, f"{half}_audio.wav")
            mkv_exists = os.path.isfile(mkv_path)
            wav_exists = os.path.isfile(wav_path)
            mkv_mb     = os.path.getsize(mkv_path) / (1024*1024) if mkv_exists else 0

            if wav_exists:
                print(f"  ✅ WAV已有  {game} half={half}  [{fmt_size(wav_path)}]，跳过下载")
            elif mkv_exists:
                print(f"  ⏭  MKV已有  {game} half={half}  [{mkv_mb:.0f}MB]({args.quality})，直接提 WAV，跳过下载")
            else:
                print(f"  ⬇  需下载   {game} half={half}  MKV({args.quality}) 和 WAV 均不存在")
                need_dl = True

        if need_dl:
            games_to_download.append(game)

    # ── STEP 2：只下载缺 MKV 的场次 ──────────────────
    if games_to_download:
        print(f"\n>>> 需要下载 MKV 的场次: {len(games_to_download)} 场")
        fake_getListGames = lambda s, task="": games_to_download
        SoccerNet.utils.getListGames = fake_getListGames
        if hasattr(SoccerNet.Downloader, 'getListGames'):
            SoccerNet.Downloader.getListGames = fake_getListGames
        try:
            my_downloader.downloadGames(
                files=[f"1_{args.quality}.mkv", f"2_{args.quality}.mkv"],
                split=[split],
                task=DL_TASK,
                verbose=True,
            )
        except Exception as e:
            print(f"  ❌ 下载过程中发生错误: {e}")
            dl_fail_list.append(f"{split}: {e}")
            dl_fail += 1
        finally:
            SoccerNet.utils.getListGames = original_getListGames
            if hasattr(SoccerNet.Downloader, 'getListGames'):
                SoccerNet.Downloader.getListGames = original_getListGames
        dl_ok += len(games_to_download)
    else:
        print(f"\n>>> ✅ 所有 MKV 或 WAV 均已就位，跳过下载")
        dl_skip += len(all_games)

    # ── STEP 3：逐场逐半场提取 WAV ────────────────────
    print(f"\n>>> 开始 ffmpeg 提取音频...")
    for i, game in enumerate(all_games):
        game_path = os.path.join(ROOT_DIR, game)
        os.makedirs(game_path, exist_ok=True)
        total_games_processed += 1

        print(f"\n  [{i+1}/{len(all_games)}] {game}")

        for half in [1, 2]:
            mkv_path = os.path.join(game_path, f"{half}_{args.quality}.mkv")
            wav_path = os.path.join(game_path, f"{half}_audio.wav")

            # A: WAV 已有，跳过
            if os.path.isfile(wav_path):
                wav_size = fmt_size(wav_path)
                print(f"    ⏭  half={half}: WAV 已存在 [{wav_size}]，跳过")
                wav_skip += 1
                continue

            # B: MKV 存在，提取音频
            if os.path.isfile(mkv_path):
                mkv_size = fmt_size(mkv_path)
                print(f"    ▶  half={half}: MKV [{mkv_size}] → 提取 WAV...")
                success = extract_audio_from_mkv(mkv_path, wav_path)
                if success:
                    wav_ok += 1
                else:
                    wav_fail += 1
                    wav_fail_list.append((game, half, "ffmpeg 返回非零退出码"))
            else:
                # C: WAV 没有，MKV 也没有（下载可能失败了）
                print(f"    ❌ half={half}: MKV 和 WAV 均不存在！下载可能失败了，请检查网络")
                wav_fail += 1
                wav_fail_list.append((game, half, "MKV 不存在，无法提取 WAV"))


# ══════════════════════════════════════════════════════
#  最终汇总
# ══════════════════════════════════════════════════════

total_elapsed = time.time() - script_start
print(f"\n{'='*65}")
print(f"  最终汇总报告  (总耗时: {total_elapsed/60:.1f} 分钟)")
print(f"{'='*65}")
print(f"  处理比赛场次      : {total_games_processed} 场")
print(f"  ─── MKV 下载 ───────────────────────────────────")
print(f"  下载成功 (场)     : {dl_ok}")
print(f"  跳过下载 (场)     : {dl_skip}")
print(f"  下载失败 (场)     : {dl_fail}")
print(f"  ─── WAV 提取 ───────────────────────────────────")
print(f"  提取成功 (半场)   : {wav_ok}")
print(f"  跳过已有 (半场)   : {wav_skip}")
print(f"  提取失败 (半场)   : {wav_fail}")

if wav_fail_list:
    print(f"\n  ❌ 提取失败的半场详情:")
    for game, half, reason in wav_fail_list:
        print(f"    · {game}  half={half}  原因: {reason}")

if dl_fail_list:
    print(f"\n  ❌ 下载失败:")
    for item in dl_fail_list:
        print(f"    · {item}")

if wav_fail == 0 and dl_fail == 0:
    print(f"\n  🎉 全部成功！没有任何失败！")
else:
    print(f"\n  ⚠️  有 {wav_fail + dl_fail} 个操作失败，请检查上方详情")

print(f"{'='*65}\n")