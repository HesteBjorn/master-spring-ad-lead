#!/bin/bash
# Render a clean demonstration clip for one chosen route.
#
# Re-runs deterministic closed-loop inference on a SINGLE staged route with the
# clip visualizer on (CLIP_VIZ=1), writing camera + BEV-route + speed-bar frames
# to <out>/clip_viz, then stitches them into an mp4 with ffmpeg.
#
# Usage:
#   ROUTE_ID=11 bash rl_finetuning/eval_viz/render_clip.sh
#   ROUTE_ID=4 NAME=stopsign bash rl_finetuning/eval_viz/render_clip.sh
#
# Pick ROUTE_ID from outputs/eval_viz/outcomes_Town13.tsv (the route column).
# Defaults target the TD3 residual checkpoint used in the scan.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."

ROUTE_ID="${ROUTE_ID:?set ROUTE_ID (e.g. 11) — see outcomes_Town13.tsv}"
NAME="${NAME:-route${ROUTE_ID}}"
AGENT="${AGENT:-rl_finetuning/inference/td3_sensor_agent.py}"
CONFIG="${CONFIG:-outputs/eval_viz/checkpoints/td3_final}"
STAGED="${STAGED:-outputs/eval_viz/routes_Town13/route_Town13_00.xml}"
OUT_ROOT="${OUT_ROOT:-outputs/eval_viz/clips}"
# Clip viz writes one frame per control step, and the leaderboard ticks the sim
# at carla_fps=20 Hz (action_repeat=1), so 20 fps plays the clip back in real time
# — matching the framework's own debug video_fps (20/produce_frame_frequency).
FPS="${FPS:-20}"
# KEEP_WEATHER=1 preserves each route's packaged <weathers> (e.g. the adverse
# bench2drive benchmark weather). Default (0) forces the fixed ClearNoon used for
# the training-route demonstration clips (those routes ship no weather).
KEEP_WEATHER="${KEEP_WEATHER:-0}"

export CARLA_ROOT="${CARLA_ROOT:-3rd_party/CARLA_0915}"
OUT_DIR="$OUT_ROOT/$NAME"
# route.xml lives OUTSIDE OUT_DIR: eval_bench2drive.sh with RESUME=0 wipes
# EVALUATION_OUTPUT_DIR before running, which would delete a route file inside it.
ROUTE_XML="$OUT_ROOT/${NAME}_route.xml"
mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# 1. Extract the single chosen route into its own xml.
# ---------------------------------------------------------------------------
python - "$STAGED" "$ROUTE_ID" "$ROUTE_XML" "$KEEP_WEATHER" <<'PY'
import sys, xml.etree.ElementTree as ET
src, rid, dst = sys.argv[1], sys.argv[2], sys.argv[3]
keep_weather = len(sys.argv) > 4 and sys.argv[4] == "1"
tree = ET.parse(src); root = tree.getroot()
keep = [r for r in root.findall("route") if r.get("id") == rid]
if not keep:
    raise SystemExit(f"route id={rid} not found in {src}")
route_el = keep[0]
for r in list(root.findall("route")):
    if r is not route_el:
        root.remove(r)

if keep_weather:
    # Keep the route's packaged <weathers> untouched (benchmark weather).
    note = "packaged weather"
else:
    # Force clear-noon weather to match the fixed training weather. The training
    # routes carry no <weather>, so the leaderboard would otherwise apply its dim
    # partly-cloudy fallback (sun_altitude=70, cloudiness=50). Values mirror
    # lead/common/weathers.py "ClearNoon".
    for w in route_el.findall("weathers"):
        route_el.remove(w)
    weathers = ET.Element("weathers")
    ET.SubElement(weathers, "weather", {
        "route_percentage": "0", "cloudiness": "5.0", "precipitation": "0.0",
        "precipitation_deposits": "0.0", "wind_intensity": "10.0",
        "sun_azimuth_angle": "-1.0", "sun_altitude_angle": "90.0",
        "fog_density": "2.0", "fog_distance": "0.75", "fog_falloff": "0.1",
        "wetness": "0.0", "scattering_intensity": "1.0",
        "mie_scattering_scale": "0.03", "rayleigh_scattering_scale": "0.0331",
        "dust_storm": "0.0"})
    route_el.insert(0, weathers)
    note = "clear-noon"

tree.write(dst, encoding="utf-8", xml_declaration=True)
print(f"[render] staged route {rid} ({note}) -> {dst}")
PY

# ---------------------------------------------------------------------------
# 2. Fresh CARLA, run the one route with clip viz on (frames, not outcome-only).
# ---------------------------------------------------------------------------
trap 'bash scripts/clean_carla.sh' EXIT
bash scripts/clean_carla.sh
bash scripts/start_carla.sh
sleep 12

CHECKPOINT_DIR="$CONFIG" AGENT="$AGENT" ROUTES="$ROUTE_XML" \
  EVALUATION_OUTPUT_DIR="$OUT_DIR" \
  NO_SAVE=0 CLIP_VIZ=1 REPETITIONS=1 RESUME=0 \
  bash scripts/eval_bench2drive.sh

bash scripts/clean_carla.sh
trap - EXIT

# ---------------------------------------------------------------------------
# 3. Stitch frames -> mp4.
# ---------------------------------------------------------------------------
FRAMES="$OUT_DIR/clip_viz"
N=$(ls "$FRAMES"/frame_*.jpg 2>/dev/null | wc -l)
echo "== $N frames in $FRAMES =="
if [[ "$N" -gt 0 ]]; then
  ffmpeg -y -framerate "$FPS" -i "$FRAMES/frame_%06d.jpg" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 "$OUT_DIR/$NAME.mp4"
  echo "Done -> $OUT_DIR/$NAME.mp4"
else
  echo "[error] no frames were written — check that CLIP_VIZ wired in and the route ran." >&2
  exit 1
fi
