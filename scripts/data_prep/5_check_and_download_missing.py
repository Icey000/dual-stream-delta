"""
check_and_download_missing.py
=============================
用法:
  # 只查漏 (不下载)
  python check_and_download_missing.py

  # 查漏 + 下载缺失比赛的 224p MKV
  python check_and_download_missing.py --download --password YOUR_PWD
"""

import os
import argparse
import time

# ──────────────────────────────────────────────
# 路径配置（与 run_train.sh 保持一致）
# ──────────────────────────────────────────────
VISION_ROOT = os.environ.get("SOCCERNET_VISION", "./data/caption-2024/")
AUDIO_ROOT  = os.environ.get("SOCCERNET_AUDIO", "./data/SoccerNet/")
VISION_FEAT = "baidu_soccer_embeddings.npy"
LABEL_FILE  = "Labels-caption.json"
OUTPUT_TXT  = "missing_audio_games.txt"

parser = argparse.ArgumentParser()
parser.add_argument("--vision_root", type=str, default=VISION_ROOT)
parser.add_argument("--audio_root",  type=str, default=AUDIO_ROOT)
parser.add_argument("--features",    type=str, default=VISION_FEAT)
parser.add_argument("--download",    action="store_true", help="下载缺失比赛的 224p MKV")
parser.add_argument("--password",    type=str, default=None, help="SoccerNet 下载密码")
args = parser.parse_args()


# ══════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════

def fmt_size(path):
    """返回文件大小字符串，不存在则返回 '—'"""
    if not os.path.isfile(path):
        return "—"
    mb = os.path.getsize(path) / (1024 * 1024)
    return f"{mb:.1f} MB"


# ══════════════════════════════════════════════════════════════
#  STEP 1: 扫描，找出缺失音频特征的比赛
#  逻辑：以 caption-2024 目录为"真实基准"直接 walk，
#        不依赖 getListGames，避免网络请求和版本差异
# ══════════════════════════════════════════════════════════════

script_start = time.time()
vision_root = args.vision_root.rstrip("/")
audio_root  = args.audio_root.rstrip("/")

print(f"\n{'='*65}")
print(f"  STEP 1: 扫描并比对视觉 / 音频目录")
print(f"  视觉基准 : {vision_root}")
print(f"  音频检查 : {audio_root}")
print(f"{'='*65}\n")

complete_vision_games = []     # 视觉完备的比赛（基准）
incomplete_vision_games = []   # 视觉本身有问题的（正常不该出现）
missing_audio_games = []       # 音频缺失的比赛 [(game_rel, [missing_halves])]
ok_audio_games = []            # 音频也完备的比赛

games_scanned = 0

for league in sorted(os.listdir(vision_root)):
    league_dir = os.path.join(vision_root, league)
    if not os.path.isdir(league_dir):
        continue

    for season in sorted(os.listdir(league_dir)):
        season_dir = os.path.join(league_dir, season)
        if not os.path.isdir(season_dir):
            continue

        for game in sorted(os.listdir(season_dir)):
            game_dir = os.path.join(season_dir, game)
            if not os.path.isdir(game_dir):
                continue

            games_scanned += 1
            # 统一用 / 作路径分隔符（服务器是 Linux）
            game_rel = f"{league}/{season}/{game}"

            # ── 检查视觉特征 ──────────────────────────────
            v1 = os.path.join(vision_root, game_rel, f"1_{args.features}")
            v2 = os.path.join(vision_root, game_rel, f"2_{args.features}")
            lb = os.path.join(vision_root, game_rel, LABEL_FILE)

            missing_vision = []
            if not os.path.isfile(v1): missing_vision.append(f"1_{args.features}")
            if not os.path.isfile(v2): missing_vision.append(f"2_{args.features}")
            if not os.path.isfile(lb): missing_vision.append(LABEL_FILE)

            if missing_vision:
                incomplete_vision_games.append((game_rel, missing_vision))
                print(f"  [WARN 视觉缺失] {game_rel}")
                for f_name in missing_vision:
                    print(f"       缺: {f_name}")
                continue

            complete_vision_games.append(game_rel)

            # 视觉完备 → 打印视觉文件大小
            v1_sz = fmt_size(v1)
            v2_sz = fmt_size(v2)
            lb_sz = fmt_size(lb)

            # ── 检查音频特征 ──────────────────────────────
            a1 = os.path.join(audio_root, game_rel, "1_audio_clap.npy")
            a2 = os.path.join(audio_root, game_rel, "2_audio_clap.npy")

            a1_exists = os.path.isfile(a1)
            a2_exists = os.path.isfile(a2)
            a1_sz = fmt_size(a1)
            a2_sz = fmt_size(a2)

            missing_halves = []
            if not a1_exists: missing_halves.append(1)
            if not a2_exists: missing_halves.append(2)

            if missing_halves:
                missing_audio_games.append((game_rel, missing_halves))
                status = "❌ 音频缺失"
            else:
                ok_audio_games.append(game_rel)
                status = "✅ 完备"

            # 详细输出每场比赛的状态
            print(f"  {status}  {game_rel}")
            print(f"    视觉: half1={v1_sz}  half2={v2_sz}  label={lb_sz}")
            if missing_halves:
                print(f"    音频: half1={'✅ '+a1_sz if a1_exists else '❌ 缺失'}  "
                      f"half2={'✅ '+a2_sz if a2_exists else '❌ 缺失'}")
            else:
                print(f"    音频: half1=✅ {a1_sz}  half2=✅ {a2_sz}")

scan_elapsed = time.time() - script_start

# ── 汇总报告 ──────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  STEP 1 扫描结果  (耗时 {scan_elapsed:.1f}s)")
print(f"{'='*65}")
print(f"  扫描比赛目录数          : {games_scanned}")
print(f"  视觉完备 (基准)          : {len(complete_vision_games)} 场")
print(f"  视觉本身有问题           : {len(incomplete_vision_games)} 场 {'← 注意！' if incomplete_vision_games else ''}")
print(f"  ─────────────────────────────────────────────")
print(f"  ✅ 音视频均完备          : {len(ok_audio_games)} 场")
print(f"  ❌ 缺音频特征            : {len(missing_audio_games)} 场")

if missing_audio_games:
    print(f"\n  缺音频特征的比赛列表：")
    for game, halves in missing_audio_games:
        print(f"    · [缺 half {halves}]  {game}")

if incomplete_vision_games:
    print(f"\n  ⚠️  视觉数据有缺失的比赛（正常不该出现）：")
    for game, files in incomplete_vision_games:
        print(f"    · {game}  缺: {files}")

# ── 写 txt ───────────────────────────────────────────────
missing_games_list = [g for g, _ in missing_audio_games]
with open(OUTPUT_TXT, "w") as f:
    f.write("\n".join(missing_games_list))
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"\n  已写入: {os.path.join(script_dir, OUTPUT_TXT)}  ({len(missing_games_list)} 场)")
print(f"{'='*65}\n")


# ══════════════════════════════════════════════════════════════
#  STEP 2: 下载缺失比赛的 224p MKV
# ══════════════════════════════════════════════════════════════

if not args.download:
    print("(仅查漏模式。要下载 224p MKV，加参数: --download --password YOUR_PWD)")
    exit(0)

if not args.password:
    print("⚠️  指定了 --download 但没有 --password，无法下载。")
    print("    请用: python check_and_download_missing.py --download --password YOUR_PWD")
    exit(1)

if not missing_games_list:
    print("✅ 没有缺失的音频特征，无需下载任何 MKV！")
    exit(0)

print(f"\n{'='*65}")
print(f"  STEP 2: 下载 {len(missing_games_list)} 场比赛的 224p MKV")
print(f"  下载目标目录: {audio_root}")
print(f"{'='*65}\n")

try:
    from SoccerNet.Downloader import SoccerNetDownloader
except ImportError:
    print("❌ 未找到 SoccerNet 库，请先 pip install SoccerNet")
    exit(1)

downloader = SoccerNetDownloader(LocalDirectory=audio_root)
downloader.password = args.password

dl_ok   = 0
dl_skip = 0
dl_fail = 0
dl_fail_list = []

dl_start = time.time()

for idx, (game_rel, missing_halves) in enumerate(missing_audio_games):
    print(f"\n[{idx+1}/{len(missing_audio_games)}] {game_rel}")
    print(f"    需要补充 half: {missing_halves}")
    game_dir_local = os.path.join(audio_root, game_rel)
    os.makedirs(game_dir_local, exist_ok=True)

    for half in missing_halves:
        mkv_filename    = f"{half}_720p.mkv"
        local_file_path = os.path.join(game_dir_local, mkv_filename)
        remote_path     = f"{game_rel}/{mkv_filename}"

        # 已存在就跳过
        if os.path.isfile(local_file_path):
            existing_sz = fmt_size(local_file_path)
            print(f"    ⏭  half={half}: MKV 已存在 [{existing_sz}]，跳过")
            dl_skip += 1
            continue

        print(f"    ⬇  half={half}: 开始下载 {mkv_filename} ...")
        t_dl = time.time()
        try:
            downloader.downloadFile(local_file_path, remote_path)
            elapsed = time.time() - t_dl
            final_sz = fmt_size(local_file_path)
            print(f"    ✅ half={half}: 下载完成 [{final_sz}]，耗时 {elapsed:.1f}s")
            dl_ok += 1
        except Exception as e:
            elapsed = time.time() - t_dl
            print(f"    ❌ half={half}: 下载失败 (耗时 {elapsed:.1f}s)  原因: {e}")
            dl_fail += 1
            dl_fail_list.append((game_rel, half, str(e)))

# ── 下载汇总 ─────────────────────────────────────────────
dl_elapsed = time.time() - dl_start
print(f"\n{'='*65}")
print(f"  STEP 2 下载汇总  (总耗时 {dl_elapsed/60:.1f} 分钟)")
print(f"{'='*65}")
print(f"  ✅ 下载成功  : {dl_ok} 个文件")
print(f"  ⏭  跳过已有  : {dl_skip} 个文件")
print(f"  ❌ 下载失败  : {dl_fail} 个文件")

if dl_fail_list:
    print(f"\n  失败详情：")
    for game, half, reason in dl_fail_list:
        print(f"    · {game}  half={half}")
        print(f"      原因: {reason}")

if dl_fail == 0:
    print(f"\n  🎉 全部文件下载成功！")
else:
    print(f"\n  ⚠️  有 {dl_fail} 个文件下载失败，请检查网络或密码后重试")

print(f"\n  下一步：")
print(f"  → python 1_download_and_extract.py --from_txt {OUTPUT_TXT}")
print(f"    (ffmpeg 从 MKV 提取 WAV)")
print(f"  → python 2_extract_clap_features.py --from_txt {OUTPUT_TXT}")
print(f"    (CLAP 提取 audio_clap.npy 特征)")
print(f"{'='*65}\n")
