#!/bin/bash
# Cut a single clip from already-rendered frames to a chosen frame range.
# No CARLA — operates purely on <dir>/clip_viz/frame_*.jpg.
#
# Usage:
#   DIR=outputs/eval_viz/clips/route04_stopsign START=300 LEN=200 \
#     OUT=outputs/eval_viz/clips/stopsign_cut.mp4 \
#     bash rl_finetuning/eval_viz/stitch_clip.sh
#
#   DIR can point at the clip folder or its clip_viz subfolder.
#   START = first frame index (default 0). LEN = number of frames (default: to end).
#   Scrub the jpgs to find the action frames; filenames are zero-padded indices.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../.."

DIR="${DIR:?set DIR (clip folder or its clip_viz subfolder)}"
[[ -d "$DIR/clip_viz" ]] && DIR="$DIR/clip_viz"
START="${START:-0}"
FPS="${FPS:-20}"
OUT="${OUT:-$DIR/../cut.mp4}"

TOTAL=$(ls "$DIR"/frame_*.jpg 2>/dev/null | wc -l)
[[ "$TOTAL" -gt 0 ]] || { echo "[error] no frames in $DIR"; exit 1; }
LEN="${LEN:-$((TOTAL - START))}"
echo "== $DIR has $TOTAL frames; cutting [$START, $((START + LEN))) at ${FPS}fps =="

ffmpeg -y -framerate "$FPS" -start_number "$START" -i "$DIR/frame_%06d.jpg" \
  -frames:v "$LEN" -c:v libx264 -pix_fmt yuv420p -crf 18 "$OUT"
echo "Done -> $OUT"
