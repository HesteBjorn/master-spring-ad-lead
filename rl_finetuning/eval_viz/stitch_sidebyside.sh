#!/bin/bash
# Stitch a side-by-side clip from already-rendered frames, with an INDEPENDENT
# start frame per side so you can align the action even when it happens at
# different frame numbers in the two runs. No CARLA — operates on saved frames.
#
# Usage (after render_sidebyside.sh produced <NAME>_tfv6 and <NAME>_td3):
#   NAME=intro LEFT_START=140 RIGHT_START=95 LEN=220 \
#     bash rl_finetuning/eval_viz/stitch_sidebyside.sh
#
# Or point at arbitrary clip folders directly:
#   LEFT_DIR=outputs/eval_viz/clips/a RIGHT_DIR=outputs/eval_viz/clips/b \
#     LEFT_START=10 RIGHT_START=0 LEN=200 OUT=out.mp4 \
#     bash rl_finetuning/eval_viz/stitch_sidebyside.sh
#
# LEN defaults to the most frames available from both chosen starts. Each side is
# trimmed to exactly LEN frames so the panels stay the same length.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../.."

OUT_ROOT="${OUT_ROOT:-outputs/eval_viz/clips}"
NAME="${NAME:-}"
LEFT_DIR="${LEFT_DIR:-$OUT_ROOT/${NAME}_tfv6}"
RIGHT_DIR="${RIGHT_DIR:-$OUT_ROOT/${NAME}_td3}"
[[ -d "$LEFT_DIR/clip_viz" ]] && LEFT_DIR="$LEFT_DIR/clip_viz"
[[ -d "$RIGHT_DIR/clip_viz" ]] && RIGHT_DIR="$RIGHT_DIR/clip_viz"
LEFT_START="${LEFT_START:-0}"
RIGHT_START="${RIGHT_START:-0}"
FPS="${FPS:-20}"
LTITLE="${LTITLE:-TFv6}"
RTITLE="${RTITLE:-TD3 fine-tuned}"
OUT="${OUT:-$OUT_ROOT/${NAME:-sxs}_sidebyside_cut.mp4}"

nframes() { ls "$1"/frame_*.jpg 2>/dev/null | wc -l; }
NL=$(nframes "$LEFT_DIR"); NR=$(nframes "$RIGHT_DIR")
[[ "$NL" -gt 0 && "$NR" -gt 0 ]] || { echo "[error] frames missing ($LEFT_DIR=$NL $RIGHT_DIR=$NR)"; exit 1; }
AVL=$((NL - LEFT_START)); AVR=$((NR - RIGHT_START))
LEN="${LEN:-$(( AVL < AVR ? AVL : AVR ))}"
[[ "$LEN" -gt 0 ]] || { echo "[error] LEN<=0 (check START values)"; exit 1; }
echo "== left $LEFT_DIR [$LEFT_START..+$LEN]  right $RIGHT_DIR [$RIGHT_START..+$LEN]  @${FPS}fps =="

FONT=$(fc-match -f '%{file}' bold 2>/dev/null || true)
[[ -f "$FONT" ]] || FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
if [[ -f "$FONT" ]]; then
  L="drawtext=fontfile=$FONT:text='$LTITLE':x=(w-tw)/2:y=12:fontsize=40:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8,"
  R="drawtext=fontfile=$FONT:text='$RTITLE':x=(w-tw)/2:y=12:fontsize=40:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8,"
else
  echo "[warn] no font; stitching without per-side titles"; L=""; R=""
fi

ffmpeg -y \
  -framerate "$FPS" -start_number "$LEFT_START" -i "$LEFT_DIR/frame_%06d.jpg" \
  -framerate "$FPS" -start_number "$RIGHT_START" -i "$RIGHT_DIR/frame_%06d.jpg" \
  -filter_complex \
  "[0:v]trim=end_frame=$LEN,setpts=PTS-STARTPTS,${L}null[l];\
   [1:v]trim=end_frame=$LEN,setpts=PTS-STARTPTS,${R}null[r];\
   [l][r]hstack=inputs=2[v]" \
  -map "[v]" -frames:v "$LEN" -c:v libx264 -pix_fmt yuv420p -crf 18 -r "$FPS" "$OUT"
echo "Done -> $OUT"
