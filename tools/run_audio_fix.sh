#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  run_audio_fix.sh
#  一键查漏补缺流水线 (720p 优先)
#
#  执行顺序：
#  ╔═══════════════════════════════════════════╗
#  ║  [0] 5_: 扫描，找出完全缺音频NPY的比赛   ║
#  ╚═════════════════════╦═════════════════════╝
#                        ↓
#  ╔═══════════════════ 第一轮 ═════════════════╗
#  ║  [1] 1_: 下载 720p MKV + ffmpeg 提 WAV   ║
#  ║  [2] 3_: 检查 WAV 时长是否正常            ║
#  ║  [3] 2_: 提取 CLAP 特征 (audio_clap.npy) ║
#  ╚═════════════════════╦═════════════════════╝
#                        ↓
#  ╔═══════════════════ 第二轮 ═════════════════╗
#  ║  [4] 3_: 再次检查WAV：720p还是只有几秒？  ║
#  ║        → 是：原始视频根本没音轨，无解      ║
#  ║  [5] 4_: 对无解的 → 生成全0 dummy NPY     ║
#  ╚═════════════════════╦═════════════════════╝
#                        ↓
#           [6] 最终验证：确认所有比赛有 NPY
#
#  用法：bash run_audio_fix.sh
# ═══════════════════════════════════════════════════════════

set -e   # 遇到错误停止

# ──────────────────────────────────────────────────────────
#  Python 路径与工作目录配置
#  公开版默认使用当前目录；需要时可由调用者覆盖环境变量
# ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_APOLLO="${PYTHON_APOLLO:-python}"
PYTHON_CLAP="${PYTHON_CLAP:-python}"
WORKDIR="${WORKDIR:-$SCRIPT_DIR}"

cd "$WORKDIR"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  一键查漏补缺流水线 (720p 优先)"
echo "  Apollo    Python: $PYTHON_APOLLO"
echo "  CLAP      Python: $PYTHON_CLAP"
echo "  工作目录: $WORKDIR"
echo "════════════════════════════════════════════════════════"


# ──────────────────────────────────────────────────────────
#  STEP 0: 用 5_ 扫描，找出完全缺音频NPY的比赛
#  输出 missing_audio_games.txt
#  使用 Apollo 环境（有 SoccerNet 库）
# ──────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  [0] 扫描缺失音频特征的比赛 (5_)        ║"
echo "╚══════════════════════════════════════════╝"

$PYTHON_APOLLO 5_check_and_download_missing.py

# 检查是否有缺失场次
if [ ! -f "missing_audio_games.txt" ] || [ ! -s "missing_audio_games.txt" ]; then
    echo ""
    echo "✅ 所有比赛音频特征均已存在，无需修复！"
    exit 0
fi

missing_count=$(wc -l < missing_audio_games.txt)
echo ""
echo ">>> 发现 ${missing_count} 场缺失音频特征，开始修复..."


# ──────────────────────────────────────────────────────────
#  第一轮：下载 720p MKV + 提取 WAV + 检查 + 提 CLAP 特征
# ──────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  第一轮：用 720p 修复缺失音频            ║"
echo "╚══════════════════════════════════════════╝"

# [1] 下载 720p MKV + ffmpeg 提取 WAV（Apollo 环境有 SoccerNet 下载器）
echo ""
echo ">>> [1/3] 下载 720p MKV 并提取 WAV..."
$PYTHON_APOLLO 1_download_and_extract.py \
    --from_txt missing_audio_games.txt \
    --quality  720p

# [2] 第一次 WAV 完整性检查（Apollo 环境，wave 是标准库无依赖要求）
echo ""
echo ">>> [2/3] 第一次检查提取出的 WAV 完整性..."
$PYTHON_APOLLO 3_check_wav_integrity.py --quality 720p

# [3] 提取 CLAP 特征（soccernet-DVC 环境有 laion_clap / librosa）
echo ""
echo ">>> [3/3] 提取 CLAP 音频特征..."
$PYTHON_CLAP 2_extract_clap_features.py --from_txt missing_audio_games.txt


# ──────────────────────────────────────────────────────────
#  第二轮：再次检查，用 dummy 对无法修复的兜底
# ──────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  第二轮：检查720p是否也救不了，再兜底    ║"
echo "╚══════════════════════════════════════════╝"

# [4] 再次检查 WAV（720p 也没救则删掉，记录到 broken_wav_games.txt）
echo ""
echo ">>> [4/5] 第二次检查 WAV 完整性..."
$PYTHON_APOLLO 3_check_wav_integrity.py --quality 720p

# [5] 对 broken_wav_games.txt 里剩余无救的 → 生成全 0 dummy NPY
if [ -f "broken_wav_games.txt" ] && [ -s "broken_wav_games.txt" ]; then
    broken_count=$(wc -l < broken_wav_games.txt)
    echo ""
    echo ">>> [5/5] 720p 也救不了 ${broken_count} 场（原始视频无音轨），生成 dummy 静音 NPY..."
    $PYTHON_APOLLO 4_generate_dummy_audio.py \
        --from_txt broken_wav_games.txt \
        --overwrite
else
    echo ""
    echo ">>> [5/5] ✅ 所有 WAV 均已正常，无需 dummy 兜底！"
fi


# ──────────────────────────────────────────────────────────
#  最终验证：确认所有比赛都有 audio_clap.npy
# ──────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  [6] 最终验证                            ║"
echo "╚══════════════════════════════════════════╝"
echo ""
# 用 4_ 的自动扫描模式，不加 --from_txt，自动找还缺 NPY 的场次
$PYTHON_APOLLO 4_generate_dummy_audio.py

echo ""
echo "════════════════════════════════════════════════════════"
echo "  流水线完成！"
echo ""
echo "  现在可以直接回到根目录跑训练/测试："
echo "  → cd .. && sbatch run_train.sh"
echo "════════════════════════════════════════════════════════"
echo ""
