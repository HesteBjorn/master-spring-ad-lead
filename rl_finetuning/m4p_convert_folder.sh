#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $(basename "$0") <debug_viz_folder>" >&2
  exit 1
fi

INPUT_DIR="${1%/}"
FPS=10
CRF=18
PRESET="medium"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Error: folder does not exist: $INPUT_DIR" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Error: ffmpeg is not available in PATH." >&2
  exit 1
fi

mapfile -d '' IMAGE_FILES < <(
  find "$INPUT_DIR" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) -print0 | sort -z
)

if [[ ${#IMAGE_FILES[@]} -eq 0 ]]; then
  echo "Error: no .jpg/.jpeg images found in: $INPUT_DIR" >&2
  exit 1
fi

RUN_DIR="$(dirname "$INPUT_DIR")"
RUN_BASENAME="$(basename "$RUN_DIR")"
if [[ "$RUN_BASENAME" =~ ^run_([0-9]{8}_[0-9]{6})$ ]]; then
  OUTPUT_NAME="debug_run_${BASH_REMATCH[1]}.mp4"
else
  OUTPUT_NAME="debug_${RUN_BASENAME}.mp4"
fi
OUTPUT_PATH="${RUN_DIR}/${OUTPUT_NAME}"

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

index=0
for image_path in "${IMAGE_FILES[@]}"; do
  printf -v seq_name "frame_%06d.jpg" "$index"
  abs_image_path="$(realpath "$image_path")"
  ln -s "$abs_image_path" "$TMP_DIR/$seq_name"
  ((index += 1))
done

ffmpeg -hide_banner -loglevel error -y \
  -framerate "$FPS" \
  -start_number 0 \
  -i "$TMP_DIR/frame_%06d.jpg" \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2:in_range=pc:out_range=tv,format=yuv420p" \
  -c:v libx264 \
  -preset "$PRESET" \
  -crf "$CRF" \
  -movflags +faststart \
  "$OUTPUT_PATH"

if [[ ! -s "$OUTPUT_PATH" ]]; then
  echo "Error: output video missing or empty: $OUTPUT_PATH" >&2
  exit 1
fi

rm -rf "$INPUT_DIR"

echo "Wrote: $OUTPUT_PATH"
echo "Frames: ${#IMAGE_FILES[@]} | FPS: $FPS"
echo "Deleted source folder: $INPUT_DIR"
