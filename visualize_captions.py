"""
visualize_captions.py  —  [Optional] Caption Visualization Tool

Overlays model-generated captions onto a SoccerNet match video clip.
Only predicted captions are shown (no ground truth labels).

Requirements:
  pip install opencv-python pillow SoccerNet

Usage:
  python visualize_captions.py \
      --pred_json   /path/to/results_dense_captioning.json \
      --video_dir   /path/to/match_folder \
      --half        1 \
      --start_min   30 \
      --end_min     35 \
      --output      caption_demo.mp4

  # To download a match video first (requires SoccerNet password):
  python visualize_captions.py --download_only \
      --match_id  "england_epl/2016-2017/2017-05-06 - 17-00 Leicester 3 - 0 Watford" \
      --video_dir /path/to/save \
      --soccernet_password s0cc3rn3t
"""

import argparse
import json
import os
import sys
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Caption display tuning
# ---------------------------------------------------------------------------
CAPTION_BAR_HEIGHT = 160      # px of black bar added below the video frame
FONT_SIZE_TIME     = 16       # px — timestamp label
FONT_SIZE_CAPTION  = 20       # px — caption text (auto-scales down if needed)
MIN_FONT_SIZE      = 12       # px — never go below this
CAPTION_DURATION_MS = 7000    # ms — how long each caption stays on screen
CONFIDENCE_THRESHOLD = 0.05   # skip predictions below this confidence
TEXT_COLOR_CAPTION  = (255, 220, 50)   # warm yellow — stands out on black bar
TEXT_COLOR_TIME     = (180, 180, 180)  # light grey  — subtle timestamp
DIVIDER_COLOR       = (80, 80, 80)     # dim grey divider line

# How to align the caption timestamp relative to its position field:
# pos is ms to event; caption shows OFFSET_BEFORE_MS before the event
# and stays for CAPTION_DURATION_MS total.
OFFSET_BEFORE_MS   = 5000    # start showing caption 5 s before the event


# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------
def _load_font(size: int) -> ImageFont.ImageFont:
    """Try to load a proportional font; fall back to default."""
    candidates = [
        "arial.ttf",
        "ArialUnicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


def _measure_text_width(draw: ImageDraw.Draw, text: str, font) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_predictions(pred_json_path: str, target_half: int) -> list:
    """
    Load model predictions from a results_dense_captioning.json file.

    Supported schemas:
      - {"predictions": [...]}
      - [...]   (bare list)

    Each event dict returned has keys: start_ms, end_ms, text, conf
    """
    if not os.path.exists(pred_json_path):
        print(f"[WARN] Prediction JSON not found: {pred_json_path}")
        return []

    with open(pred_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        items = raw.get("predictions", raw.get("annotations", []))
    elif isinstance(raw, list):
        items = raw
    else:
        print("[WARN] Unrecognised JSON schema.")
        return []

    events = []
    for item in items:
        half_str = str(item.get("half", "1"))
        if half_str != str(target_half):
            continue

        conf = float(item.get("confidence", 1.0))
        if conf < CONFIDENCE_THRESHOLD:
            continue

        pos_ms = int(item.get("position", 0))
        text   = str(item.get("comment", item.get("caption", ""))).strip()
        if not text:
            continue

        events.append({
            "start_ms": max(0, pos_ms - OFFSET_BEFORE_MS),
            "end_ms":   pos_ms - OFFSET_BEFORE_MS + CAPTION_DURATION_MS,
            "text":     text,
            "conf":     conf,
            "pos_ms":   pos_ms,
        })

    print(f"[INFO] Loaded {len(events)} captions for half {target_half} "
          f"(confidence > {CONFIDENCE_THRESHOLD})")
    return events


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------
def draw_caption_frame(
    frame_bgr: np.ndarray,
    current_ms: int,
    active_events: list,
    target_half: int,
) -> np.ndarray:
    """
    Extend the frame downward with a black caption bar and render
    the active caption (highest-confidence one if multiple overlap).
    """
    h, w = frame_bgr.shape[:2]
    bar_h = CAPTION_BAR_HEIGHT

    # Build canvas: original frame + black bar
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[:h, :w] = frame_bgr

    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    font_time    = _load_font(FONT_SIZE_TIME)
    font_caption = _load_font(FONT_SIZE_CAPTION)

    # --- Timestamp (top-left of bar) ---
    secs     = current_ms // 1000
    time_str = f"  Half {target_half}  {secs // 60:02d}:{secs % 60:02d}"
    draw.text((8, h + 8), time_str, font=font_time, fill=TEXT_COLOR_TIME)

    # --- Thin divider line ---
    draw.line([(0, h + 30), (w, h + 30)], fill=DIVIDER_COLOR, width=1)

    # --- Caption text ---
    if active_events:
        best = max(active_events, key=lambda e: e["conf"])
        caption_text = best["text"]

        # Determine how many chars fit per line given video width
        # Approximate: measure one character at chosen font size
        avg_char_w = max(1, _measure_text_width(draw, "W", font_caption))
        chars_per_line = max(10, int(w * 0.92 / avg_char_w))

        lines = textwrap.wrap(caption_text, width=chars_per_line)

        # If text won't fit in the bar, shrink font
        font_to_use = font_caption
        line_h = FONT_SIZE_CAPTION + 4
        while len(lines) * line_h > (bar_h - 44) and FONT_SIZE_CAPTION > MIN_FONT_SIZE:
            smaller_size = max(MIN_FONT_SIZE, (FONT_SIZE_CAPTION + MIN_FONT_SIZE) // 2)
            font_to_use  = _load_font(smaller_size)
            avg_char_w   = max(1, _measure_text_width(draw, "W", font_to_use))
            chars_per_line = max(10, int(w * 0.92 / avg_char_w))
            lines  = textwrap.wrap(caption_text, width=chars_per_line)
            line_h = smaller_size + 4
            break   # one shrink attempt is enough; avoid infinite loop

        # Draw centered lines
        y = h + 38
        for line in lines:
            lw = _measure_text_width(draw, line, font_to_use)
            x  = (w - lw) // 2
            draw.text((x, y), line, font=font_to_use, fill=TEXT_COLOR_CAPTION)
            y += line_h

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------
def find_video_file(video_dir: str, half: int) -> str | None:
    """Locate the half video file (tries several naming conventions)."""
    candidates = [
        f"{half}.mkv",
        f"{half}_224p.mkv",
        f"{half}_720p.mkv",
        f"{half}.mp4",
        f"{half}_224p.mp4",
    ]
    for name in candidates:
        path = os.path.join(video_dir, name)
        if os.path.exists(path):
            return path
    return None


def render_video(
    pred_json:  str,
    video_dir:  str,
    half:       int,
    start_min:  float,
    end_min:    float,
    output:     str,
):
    video_path = find_video_file(video_dir, half)
    if not video_path:
        sys.exit(
            f"[ERROR] No video found for half={half} in: {video_dir}\n"
            "  Expected files like: 1.mkv / 1_224p.mkv / 1_720p.mkv\n"
            "  Run with --download_only first to fetch the video."
        )

    print(f"[INFO] Video: {video_path}")

    events = load_predictions(pred_json, target_half=half)
    if not events:
        print("[WARN] No captions to render. Check your JSON path and half number.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_min * 60 * fps)
    end_frame   = int(end_min   * 60 * fps)
    total_frames = max(1, end_frame - start_frame)

    out_height = height + CAPTION_BAR_HEIGHT
    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    writer     = cv2.VideoWriter(output, fourcc, fps, (width, out_height))

    print(f"[INFO] Rendering {start_min:.1f}–{end_min:.1f} min "
          f"→ {output}  ({width}×{out_height} @ {fps:.1f} fps)")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    curr_frame = start_frame

    while curr_frame < end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        curr_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        active  = [e for e in events if e["start_ms"] <= curr_ms <= e["end_ms"]]

        out_frame = draw_caption_frame(frame, curr_ms, active, half)
        writer.write(out_frame)

        done = (curr_frame - start_frame + 1) / total_frames * 100
        if (curr_frame - start_frame) % 100 == 0:
            print(f"\r  Progress: {done:5.1f}%", end="", flush=True)

        curr_frame += 1

    cap.release()
    writer.release()
    print(f"\n[OK] Saved: {output}")


# ---------------------------------------------------------------------------
# Optional video download
# ---------------------------------------------------------------------------
def download_video(match_id: str, save_dir: str, password: str):
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError:
        sys.exit("[ERROR] SoccerNet package not found. Run: pip install SoccerNet")

    print(f"[INFO] Downloading match: {match_id}")
    dl = SoccerNetDownloader(LocalDirectory=save_dir)
    dl.password = password
    try:
        dl.downloadGame(match_id, files=["1_224p.mkv", "2_224p.mkv"])
        print("[OK] Download complete.")
    except Exception as e:
        sys.exit(f"[ERROR] Download failed: {e}\n"
                 "  Check: password, network connection, and exact match_id path.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise dual-stream Qwen captions overlaid on a match video clip."
    )

    # ---- Core rendering args ----
    p.add_argument("--pred_json",  default="",
                   help="Path to results_dense_captioning.json produced by DVC inference.")
    p.add_argument("--video_dir",  default="",
                   help="Folder containing the half video files (e.g. 1_224p.mkv).")
    p.add_argument("--half",       type=int, default=1, choices=[1, 2],
                   help="Which half to render (1 or 2). Default: 1.")
    p.add_argument("--start_min",  type=float, default=0.0,
                   help="Clip start time in minutes. Default: 0.")
    p.add_argument("--end_min",    type=float, default=5.0,
                   help="Clip end time in minutes. Default: 5.")
    p.add_argument("--output",     default="caption_demo.mp4",
                   help="Output video file path. Default: caption_demo.mp4.")

    # ---- Download args ----
    p.add_argument("--download_only", action="store_true",
                   help="Download match video only; skip rendering.")
    p.add_argument("--match_id",  default="",
                   help="SoccerNet match path, e.g. 'england_epl/2016-2017/...'.")
    p.add_argument("--soccernet_password", default="s0cc3rn3t",
                   help="SoccerNet download password. Default: s0cc3rn3t.")

    return p.parse_args()


def main():
    args = parse_args()

    if args.download_only:
        if not args.match_id or not args.video_dir:
            sys.exit("[ERROR] --download_only requires --match_id and --video_dir.")
        download_video(args.match_id, args.video_dir, args.soccernet_password)
        return

    # ---- Validate inputs ----
    if not args.pred_json:
        sys.exit("[ERROR] --pred_json is required.")
    if not args.video_dir:
        sys.exit("[ERROR] --video_dir is required.")
    if args.start_min >= args.end_min:
        sys.exit("[ERROR] --start_min must be less than --end_min.")

    render_video(
        pred_json  = args.pred_json,
        video_dir  = args.video_dir,
        half       = args.half,
        start_min  = args.start_min,
        end_min    = args.end_min,
        output     = args.output,
    )


if __name__ == "__main__":
    main()
