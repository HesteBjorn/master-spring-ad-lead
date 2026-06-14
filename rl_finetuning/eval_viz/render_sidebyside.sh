#!/bin/bash
# Render a side-by-side clip on ONE route: frozen TFv6 (left) vs TD3 fine-tuned
# (right). Both runs use the same TD3 checkpoint (its frozen backbone IS the base
# TFv6); the left run zeroes the residual (CLIP_RESIDUAL_OFF=1) so it drives as
# pure TFv6. Same route + traffic-manager-seed=0 => identical scene, fair compare.
#
# Usage:
#   ROUTE_ID=17 bash rl_finetuning/eval_viz/render_sidebyside.sh
#   ROUTE_ID=17 NAME=intro bash rl_finetuning/eval_viz/render_sidebyside.sh
#
# Produces outputs/eval_viz/clips/<NAME>_sidebyside.mp4 (plus the two halves).

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."

ROUTE_ID="${ROUTE_ID:?set ROUTE_ID (e.g. 17)}"
NAME="${NAME:-route${ROUTE_ID}_sxs}"
OUT_ROOT="${OUT_ROOT:-outputs/eval_viz/clips}"
FPS="${FPS:-20}"
LEFT="$OUT_ROOT/${NAME}_tfv6"
RIGHT="$OUT_ROOT/${NAME}_td3"
OUT="$OUT_ROOT/${NAME}_sidebyside.mp4"

# 1. Left: pure TFv6 (residual zeroed). 2. Right: TD3 residual on.
echo "== rendering LEFT (TFv6, residual off) =="
CLIP_RESIDUAL_OFF=1 FPS="$FPS" ROUTE_ID="$ROUTE_ID" NAME="${NAME}_tfv6" \
  bash rl_finetuning/eval_viz/render_clip.sh
echo "== rendering RIGHT (TD3 fine-tuned) =="
FPS="$FPS" ROUTE_ID="$ROUTE_ID" NAME="${NAME}_td3" \
  bash rl_finetuning/eval_viz/render_clip.sh

LMP4="$LEFT/${NAME}_tfv6.mp4"
RMP4="$RIGHT/${NAME}_td3.mp4"
[[ -f "$LMP4" && -f "$RMP4" ]] || { echo "[error] missing a half ($LMP4 / $RMP4)"; exit 1; }

# 3. Pad both to equal length (freeze last frame), title each side, hstack.
dur() { ffprobe -v error -show_entries format=duration -of csv=p=0 "$1"; }
DL=$(dur "$LMP4"); DR=$(dur "$RMP4")
MAXD=$(python -c "print(max($DL,$DR))")

FONT=$(fc-match -f '%{file}' bold 2>/dev/null || true)
[[ -f "$FONT" ]] || FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
if [[ -f "$FONT" ]]; then
  LTXT="drawtext=fontfile=$FONT:text='TFv6':x=(w-tw)/2:y=12:fontsize=40:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8,"
  RTXT="drawtext=fontfile=$FONT:text='TD3 fine-tuned':x=(w-tw)/2:y=12:fontsize=40:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8,"
else
  echo "[warn] no font found; stitching without per-side titles"
  LTXT=""; RTXT=""
fi

ffmpeg -y -i "$LMP4" -i "$RMP4" -filter_complex \
  "[0:v]${LTXT}tpad=stop_mode=clone:stop_duration=30[l];\
   [1:v]${RTXT}tpad=stop_mode=clone:stop_duration=30[r];\
   [l][r]hstack=inputs=2[v]" \
  -map "[v]" -t "$MAXD" -c:v libx264 -pix_fmt yuv420p -crf 18 -r "$FPS" "$OUT"

echo "Done -> $OUT"
