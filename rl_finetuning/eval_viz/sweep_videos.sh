#!/bin/bash
# Full demo-video sweep. For each town (Town12, then Town13) runs the frozen TFv6
# (residual zeroed) and the TD3 fine-tuned policy over every route WITH video, and
# a live monitor that — as each route finishes — deletes route-deviation footage
# and keeps/catalogs everything else (success / isolated infraction / timeout /
# collision combos). After both passes per town it builds the side-by-side
# comparisons (TD3 wins; both win but TFv6 slower) and the overview document.
#
# Bench2drive-style: one CARLA session per (town, model) pass, routes run
# sequentially, checkpoint + video processed per route as they complete.
#
# Run:
#   conda activate lead
#   export CARLA_ROOT=.../3rd_party/CARLA_0915
#   bash rl_finetuning/eval_viz/sweep_videos.sh
#
# Override TOWNS / ROUTE_SRC / SWEEP_ROOT via the environment.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."

TOWNS="${TOWNS:-12 13}"
ROUTE_SRC="${ROUTE_SRC:-data/rl_finetuning_training_data/NonSignalizedJunctionLeftTurnEnterFlow}"
SWEEP_ROOT="${SWEEP_ROOT:-outputs/eval_viz/sweep}"
CONFIG="${CONFIG:-outputs/eval_viz/checkpoints/td3_final}"
AGENT="${AGENT:-rl_finetuning/inference/td3_sensor_agent.py}"
OVERVIEW="${OVERVIEW:-outputs/eval_viz/videos_to_report.md}"
FPS="${FPS:-20}"
# Cap the route game-time timeout so stuck/hesitating routes end promptly and
# their clips stay bounded (the agent applies this via route_timeout_patch).
export ROUTE_TIMEOUT_S="${ROUTE_TIMEOUT_S:-100}"
export CARLA_ROOT="${CARLA_ROOT:-3rd_party/CARLA_0915}"

[[ -e "$CONFIG" ]] || { echo "[error] checkpoint dir not found: $CONFIG (run extract_checkpoints.sh)"; exit 1; }

run_pass() {  # town_num pass_name extra_env
  local town="$1" pass="$2" extra="$3"
  local out="$SWEEP_ROOT/town${town}/${pass}"
  local routes="$SWEEP_ROOT/town${town}/routes_combined.xml"
  local done="$out/.run_done"

  rm -rf "$out"; mkdir -p "$out/clip_viz"
  rm -f "$done"

  echo "================================================================"
  echo "== town ${town}  pass=${pass}  -> ${out}"
  echo "================================================================"

  # Live monitor (background): processes each route as the checkpoint grows.
  python rl_finetuning/eval_viz/monitor_routes.py \
    --checkpoint "$out/checkpoint_endpoint.json" --clip-root "$out/clip_viz" \
    --model "$pass" --town "$town" --out-jsonl "$out/kept.jsonl" \
    --done-file "$done" --fps "$FPS" &
  local mon=$!

  # Fresh CARLA for this pass.
  trap 'bash scripts/clean_carla.sh; kill '"$mon"' 2>/dev/null || true' EXIT
  bash scripts/clean_carla.sh
  bash scripts/start_carla.sh
  sleep 12

  # RESUME=1 so eval_bench2drive does NOT wipe the dir we share with the monitor.
  # CLIP_PER_ROUTE_DIR makes the agent write clip_viz/route_<NNNN> per route.
  env $extra \
    CHECKPOINT_DIR="$CONFIG" AGENT="$AGENT" ROUTES="$routes" \
    EVALUATION_OUTPUT_DIR="$out" \
    NO_SAVE=0 CLIP_VIZ=1 CLIP_PER_ROUTE_DIR=1 RESUME=1 REPETITIONS=1 \
    bash scripts/eval_bench2drive.sh || echo "[warn] eval_bench2drive returned non-zero for town${town}/${pass}"

  touch "$done"
  wait "$mon" || true
  bash scripts/clean_carla.sh
  trap - EXIT
  sleep 3
}

for town in $TOWNS; do
  STAGE_DIR="$SWEEP_ROOT/town${town}/staged"
  COMBINED="$SWEEP_ROOT/town${town}/routes_combined.xml"
  mkdir -p "$STAGE_DIR"

  echo "== staging Town${town} routes from $ROUTE_SRC =="
  # MAX_ROUTES>0 trims each staged file (handy for a quick smoke of the pipeline).
  python rl_finetuning/eval_viz/stage_routes.py \
    --src "$ROUTE_SRC" --town "Town${town}" --out "$STAGE_DIR" \
    ${MAX_ROUTES:+--max-routes "$MAX_ROUTES"}

  # Merge all staged route files into one combined file with sequential ids
  # (ids stay unique even if several source files overlap). LIMIT_ROUTES>0 caps
  # the TOTAL number of routes in the combined file — unlike MAX_ROUTES (which
  # trims per source file and so can't bound single-route lead files). Handy for
  # a quick dry run over the whole route set. ROUTE_IDS="0 2 5 ..." selects only
  # those source positions and KEEPS each route's positional id (so a follow-up
  # pass aligns with an earlier full run's per-id results); it takes precedence
  # over LIMIT_ROUTES.
  python - "$STAGE_DIR" "$COMBINED" "${LIMIT_ROUTES:-0}" "${ROUTE_IDS:-}" <<'PY'
import sys, glob, os, xml.etree.ElementTree as ET
stage_dir, dst = sys.argv[1], sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
ids_env = sys.argv[4] if len(sys.argv) > 4 else ""
selected = {int(x) for x in ids_env.split()} if ids_env.strip() else None
root = ET.Element("routes"); pos = 0; kept = 0
for f in sorted(glob.glob(os.path.join(stage_dir, "*.xml"))):
    if selected is None and limit and kept >= limit:
        break
    for route in ET.parse(f).getroot().findall("route"):
        if selected is not None:
            if pos in selected:
                route.set("id", str(pos)); root.append(route); kept += 1
        else:
            if limit and kept >= limit:
                break
            route.set("id", str(pos)); root.append(route); kept += 1
        pos += 1
ET.ElementTree(root).write(dst, encoding="utf-8", xml_declaration=True)
if selected is not None:
    print(f"[sweep] combined {kept} route(s) -> {dst} (ROUTE_IDS={sorted(selected)})")
else:
    print(f"[sweep] combined {kept} route(s) -> {dst}" + (f" (capped at LIMIT_ROUTES={limit})" if limit else ""))
PY

  # PASSES selects which model passes to run (default both). Use PASSES=td3 to
  # run only the fine-tuned model (skips the residual-off TFv6 baseline pass);
  # PASSES=tfv6 for only the base. build_overview tolerates a missing pass.
  for pass in ${PASSES:-tfv6 td3}; do
    case "$pass" in
      tfv6) run_pass "$town" "tfv6" "CLIP_RESIDUAL_OFF=1" ;;
      td3)  run_pass "$town" "td3" "" ;;
      *)    echo "[warn] unknown pass '$pass' (expected: tfv6 td3)" ;;
    esac
  done

  echo "== building overview after Town${town} =="
  python rl_finetuning/eval_viz/build_overview.py \
    --sweep-root "$SWEEP_ROOT" --overview "$OVERVIEW" --fps "$FPS"
done

echo ""
echo "Sweep complete. Catalogue: $OVERVIEW"
echo "Comparisons in: $SWEEP_ROOT/comparisons/   Kept clips under: $SWEEP_ROOT/town*/{tfv6,td3}/clip_viz/route_*/"
