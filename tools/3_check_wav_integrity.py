"""
======================================
用法：

  # 先空跑，只看报告，不删任何东西
  python 3_check_wav_integrity.py --dry_run

  # 正式运行：发现坏文件就清理（WAV一定删，MKV先检查再决定）
  python 3_check_wav_integrity.py
======================================

3_check_wav_integrity.py
========================
检测所有 WAV 文件是否正常，并智能清理：
  - WAV 时长 < 阈值 → 坏，必须删
  - WAV 读取失败    → 坏，必须删
  - NPY 如果存在    → 是用坏 WAV 生成的，一起删
  - MKV 如果存在    → 先检查 MKV 是否正常：
      * MKV 太小（< 50MB）或时长异常 → MKV 也坏，删掉重下
      * MKV 大小正常                 → MKV 没问题！保留，下次可以直接从 MKV 重提音频

最终生成 broken_wav_games.txt，方便后续重跑。
"""

import os
import argparse
import wave
import contextlib
import subprocess

# ================= 配置区 =================
ROOT_DIR = os.getenv("SOCCERNET_AUDIO_ROOT", "/path/to/SoccerNet-audio")
SPLITS   = ["train", "valid", "test"]

MIN_DURATION_SEC   = 20 * 60   # WAV 最短时长阈值：20 分钟（正常半场 ≥ 40 分钟）
MIN_MKV_SIZE_MB    = 50        # MKV 最小合理大小：50 MB（正常半场几百 MB）
# ==========================================

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir",     type=str,   default=ROOT_DIR)
parser.add_argument("--splits",       nargs="+",  default=SPLITS)
parser.add_argument("--min_duration", type=float, default=MIN_DURATION_SEC,
                    help="低于此秒数的 WAV 视为损坏（默认 1200s = 20分钟）")
parser.add_argument("--min_mkv_mb",   type=float, default=MIN_MKV_SIZE_MB,
                    help="MKV 低于此大小（MB）视为损坏需删除（默认 50MB）")
parser.add_argument("--dry_run",      action="store_true",
                    help="只打印报告，不真正删除任何文件")
parser.add_argument("--quality",       type=str, default="720p",
                    choices=["224p", "720p"],
                    help="MKV 画质（默认 720p）")
args = parser.parse_args()

from SoccerNet.Downloader import getListGames


# ══════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════

def get_wav_duration(wav_path):
    """读 WAV 头部获取时长（秒），失败返回 None。用标准库，极快，不加载音频数据。"""
    try:
        with contextlib.closing(wave.open(wav_path, 'r')) as f:
            frames     = f.getnframes()
            frame_rate = f.getframerate()
            if frame_rate == 0:
                return None
            return frames / float(frame_rate)
    except Exception:
        return None


def get_mkv_size_mb(mkv_path):
    """返回 MKV 文件大小（MB），文件不存在返回 None。"""
    if not os.path.isfile(mkv_path):
        return None
    return os.path.getsize(mkv_path) / (1024 * 1024)


def get_mkv_audio_info(mkv_path):
    """
    用 ffprobe 探测 MKV 文件里的音轨情况。
    返回 dict:
        {
          'has_audio'  : bool,      # 是否有音频流
          'duration'   : float|None,# 音轨时长（秒），None 表示无法读取
          'codec'      : str,       # 音频编解码器名称
          'channels'   : int,       # 声道数
          'sample_rate': int,       # 采样率
        }
    返回 None 表示 ffprobe 调用失败（工具不存在或文件损坏）。
    """
    if not os.path.isfile(mkv_path):
        return None
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-select_streams', 'a:0',          # 只看第一条音频流
        '-show_entries',
        'stream=codec_name,channels,sample_rate,duration',
        '-of', 'default=noprint_wrappers=1',
        mkv_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()
        if not output:                     # 无音频流
            return {'has_audio': False, 'duration': None,
                    'codec': '', 'channels': 0, 'sample_rate': 0}
        info = {}
        for line in output.splitlines():
            if '=' in line:
                k, v = line.split('=', 1)
                info[k.strip()] = v.strip()
        duration_raw = info.get('duration', 'N/A')
        try:
            duration = float(duration_raw)
        except (ValueError, TypeError):
            duration = None
        return {
            'has_audio'  : True,
            'duration'   : duration,
            'codec'      : info.get('codec_name', 'unknown'),
            'channels'   : int(info.get('channels', 0)),
            'sample_rate': int(info.get('sample_rate', 0)),
        }
    except FileNotFoundError:
        # ffprobe 不在 PATH 里
        return None
    except Exception:
        return None


def safe_delete(path, label, dry_run, deleted_list, failed_list):
    """
    尝试删除文件，结果追加到 deleted_list 或 failed_list。
    """
    if not os.path.isfile(path):
        return  # 根本不存在，无需处理
    if dry_run:
        print(f"       [DRY RUN] 将删除 {label}: {os.path.basename(path)}")
        return
    try:
        os.remove(path)
        deleted_list.append(path)
        print(f"       ✂  已删除 {label}: {os.path.basename(path)}")
    except Exception as e:
        failed_list.append((path, str(e)))
        print(f"       ❌ 删除失败 {label}: {os.path.basename(path)}  原因: {e}")


# ══════════════════════════════════════════════════════
#  主逻辑
# ══════════════════════════════════════════════════════

print(f"\n{'='*65}")
print(f"  WAV 完整性检查 {'[DRY RUN]' if args.dry_run else '[清理模式]'}")
print(f"  根目录        : {args.root_dir}")
print(f"  Split         : {args.splits}")
print(f"  WAV 最短时长  : {args.min_duration/60:.0f} 分钟 ({args.min_duration:.0f} 秒)")
print(f"  MKV 最小大小  : {args.min_mkv_mb:.0f} MB")
print(f"{'='*65}\n")

# -- 统计变量 --
ok_count    = 0
bad_count   = 0
missing_wav = 0
broken_games = set()

deleted_files = []   # 成功删掉的文件路径
failed_files  = []   # 删除失败的 (path, reason)

# 详细记录：每场坏的情况
bad_details = []     # list of (game, half, reason, wav_dur, mkv_mb, mkv_kept)

all_games = []
for split in args.splits:
    all_games.extend(getListGames(split, task="caption"))

print(f"共 {len(all_games)} 场比赛，开始逐一检查...\n")
print(f"{'─'*65}")

for game in all_games:
    game_path = os.path.join(args.root_dir, game)

    for half in [1, 2]:
        wav_path = os.path.join(game_path, f"{half}_audio.wav")
        npy_path = os.path.join(game_path, f"{half}_audio_clap.npy")
        mkv_path = os.path.join(game_path, f"{half}_{args.quality}.mkv")

        # ── 情况 1：WAV 根本不存在 ──────────────────────────
        if not os.path.isfile(wav_path):
            missing_wav += 1
            # 不属于"坏文件"，属于"根本没提取"，check_and_download_missing 已处理
            continue

        # ── 读取 WAV 信息 ──────────────────────────────────—
        wav_size_mb = os.path.getsize(wav_path) / (1024 * 1024)
        duration    = get_wav_duration(wav_path)

        # ── 情况 2：WAV 读取失败（文件损坏/非标准 PCM）─────
        if duration is None:
            reason = "WAV 读取失败（文件损坏或非标准 PCM）"
            print(f"\n  ❌ [损坏] {game}  half={half}")
            print(f"     WAV 大小: {wav_size_mb:.1f} MB  |  无法读取时长")
        # ── 情况 3：WAV 时长太短（ffmpeg 中途挂掉的典型症状）
        elif duration < args.min_duration:
            reason = f"WAV 时长过短 ({duration:.1f}s = {duration/60:.1f}分钟，阈值 {args.min_duration/60:.0f}分钟)"
            print(f"\n  ⚠️  [时长异常] {game}  half={half}")
            print(f"     WAV 大小: {wav_size_mb:.1f} MB  |  时长: {duration:.1f}s ({duration/60:.1f}分钟)  ← 异常！")
        else:
            # ── 正常 ──────────────────────────────────────—
            ok_count += 1
            continue

        # ── 到这里说明 WAV 确认有问题，开始清理 ──────────—
        bad_count += 1
        broken_games.add(game)

        # 1. 删坏 WAV（必须删）
        safe_delete(wav_path, "WAV", args.dry_run, deleted_files, failed_files)

        # 2. 删 NPY（基于坏 WAV 生成的特征也是坏的，必须删）
        if os.path.isfile(npy_path):
            npy_size_mb = os.path.getsize(npy_path) / (1024 * 1024)
            print(f"     NPY 大小: {npy_size_mb:.1f} MB  → 由坏 WAV 生成，删除")
            safe_delete(npy_path, "NPY", args.dry_run, deleted_files, failed_files)
        else:
            print(f"     NPY: 不存在（尚未提取特征）")

        # 3. 检查 MKV：先看大小，再决定删不删
        mkv_mb         = get_mkv_size_mb(mkv_path)
        mkv_no_audio   = False   # flag：MKV 里根本没有音频流
        mkv_audio_short= False   # flag：MKV 音轨本身过短

        if mkv_mb is None:
            print(f"     MKV: 不存在（需要重新下载）")
            mkv_kept = False
        elif mkv_mb < args.min_mkv_mb:
            print(f"     MKV 大小: {mkv_mb:.1f} MB  → 太小，MKV 本身也坏了，删除重下")
            safe_delete(mkv_path, "MKV", args.dry_run, deleted_files, failed_files)
            mkv_kept = False
        else:
            print(f"     MKV 大小: {mkv_mb:.1f} MB  → 大小正常，MKV 完好！保留，可直接重提音频")

            # ── 进一步诊断：MKV 里的音轨本身情况 ──
            audio_info = get_mkv_audio_info(mkv_path)
            if audio_info is None:
                print(f"       ⚠️  ffprobe 不可用或调用失败，无法诊断 MKV 音轨")
            elif not audio_info['has_audio']:
                print(f"       🔇 MKV 里根本没有音频流！（原始录像可能就无音轨）")
                print(f"          → 建议尝试下载 720p 版本，确认高清版有无音轨")
                mkv_no_audio = True
            else:
                dur_str = f"{audio_info['duration']:.1f}s" if audio_info['duration'] else "未知时长"
                dur_min = audio_info['duration'] / 60 if audio_info['duration'] else 0
                print(f"       🎵 MKV 音轨: codec={audio_info['codec']}  "
                      f"channels={audio_info['channels']}  "
                      f"rate={audio_info['sample_rate']}Hz  "
                      f"duration={dur_str} ({dur_min:.1f}分钟)")
                if audio_info['duration'] and audio_info['duration'] < args.min_duration:
                    print(f"          ⚠️  MKV 音轨本身也很短！WAV 2秒是因为 MKV 音轨就只有这么长")
                    print(f"          → 原始视频音轨被截断，建议尝试下载 720p 版本重试")
                    mkv_audio_short = True
                else:
                    print(f"          ✅ MKV 音轨时长正常，只是 ffmpeg 提取时出了问题")
            mkv_kept = True

        bad_details.append({
            "game"          : game,
            "half"          : half,
            "reason"        : reason,
            "wav_size_mb"   : wav_size_mb,
            "mkv_mb"        : mkv_mb if mkv_mb else 0,
            "mkv_kept"      : mkv_kept,
            "mkv_no_audio"  : mkv_no_audio,
            "mkv_audio_short": mkv_audio_short,
        })

print(f"\n{'─'*65}")


# ══════════════════════════════════════════════════════
#  汇总报告
# ══════════════════════════════════════════════════════

print(f"\n{'='*65}")
print(f"  最终汇总报告")
print(f"{'='*65}")
print(f"  总比赛场次              : {len(all_games)} 场")
print(f"  WAV 正常               : {ok_count} 个半场")
print(f"  WAV 异常（坏文件）      : {bad_count} 个半场（涉及 {len(broken_games)} 场比赛）")
print(f"  WAV 完全缺失           : {missing_wav} 个半场（用 check_and_download_missing.py 补）")
print(f"{'─'*65}")

if bad_details:
    print(f"\n  【坏文件详情】")
    for d in bad_details:
        mkv_status = f"MKV={d['mkv_mb']:.0f}MB ({'保留✅' if d['mkv_kept'] else '已删❌'})"
        extra = ""
        if d.get('mkv_no_audio'):    extra = "  ⚠️  MKV 无音轨 → 建议试 720p"
        if d.get('mkv_audio_short'): extra = "  ⚠️  MKV 音轨过短 → 建议试 720p"
        print(f"  · {d['game']}  half={d['half']}")
        print(f"    原因: {d['reason']}")
        print(f"    {mkv_status}{extra}")

print(f"\n{'─'*65}")
if not args.dry_run:
    print(f"  【删除操作结果】")
    print(f"  成功删除文件数 : {len(deleted_files)} 个")
    print(f"  删除失败文件数 : {len(failed_files)} 个")
    if failed_files:
        print(f"\n  ⚠️  以下文件删除失败：")
        for path, reason in failed_files:
            print(f"    ✗ {path}")
            print(f"      原因: {reason}")
    # MKV 保留的场次：只需重提音频，不需重下
    need_re_extract  = [d for d in bad_details if d["mkv_kept"] and
                        not d.get('mkv_no_audio') and not d.get('mkv_audio_short')]
    need_re_download = [d for d in bad_details if not d["mkv_kept"]]
    need_720p        = [d for d in bad_details if d.get('mkv_no_audio') or d.get('mkv_audio_short')]

    print(f"\n{'─'*65}")
    print(f"  【下一步行动】")
    if need_re_download:
        games_to_dl = list({d["game"] for d in need_re_download})
        print(f"  ❌ 需重新下载 MKV ({args.quality}) 并提取: {len(games_to_dl)} 场")
        print(f"     → python 1_download_and_extract.py --from_txt broken_wav_games.txt")
    if need_re_extract:
        games_re = list({d["game"] for d in need_re_extract})
        print(f"\n  ♻️  MKV 完好、只需重提 WAV: {len(games_re)} 场")
        print(f"     → python 1_download_and_extract.py --from_txt broken_wav_games.txt")
        print(f"       (检测到 MKV 已存在则跳过下载，直接 ffmpeg 提取 WAV)")
    if need_720p:
        games_720p = list({d["game"] for d in need_720p})
        print(f"\n  🔄 MKV 音轨本身损坏/缺失，需尝试 720p: {len(games_720p)} 场")
        print(f"     → python 1_download_and_extract.py --from_txt need_720p_games.txt --quality 720p")
    print(f"\n  最后统一跑 CLAP 提特征:")
    print(f"     → python 2_extract_clap_features.py --from_txt broken_wav_games.txt")
else:
    print(f"  [DRY RUN] 没有实际删除任何文件")
    print(f"  确认无误后，去掉 --dry_run 参数重新运行以执行删除")

# ── 写 txt ────────────────────────────────────────────────
if broken_games:
    out_txt = "broken_wav_games.txt"
    with open(out_txt, "w") as f:
        f.write("\n".join(sorted(broken_games)))
    print(f"\n  📄 已写入: {out_txt}  ({len(broken_games)} 场)")

# 额外：需要试 720p 的场次
if not args.dry_run:
    need_720p_list = list({d["game"] for d in bad_details
                           if d.get('mkv_no_audio') or d.get('mkv_audio_short')})
    if need_720p_list:
        out_720p = "need_720p_games.txt"
        with open(out_720p, "w") as f:
            f.write("\n".join(sorted(need_720p_list)))
        print(f"  📄 已写入: {out_720p}  ({len(need_720p_list)} 场，这些需要尝试下载 720p)")

if not broken_games:
    print(f"\n  ✅ 所有 WAV 文件均正常！无需任何修复。")

print(f"{'='*65}\n")
