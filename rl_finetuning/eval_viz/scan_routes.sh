#!/bin/bash
# Scan a route set with one or more models and record per-route outcomes.
#
# Finds which Town13 left-turn routes show each behaviour we want to film
# (stop-sign violation, gap-taking success, impatient collision, TFv6-vs-TD3
# divergence) by running deterministic closed-loop inference and reading the
# leaderboard checkpoint JSON. No frames are written here (NO_SAVE=1); pick the
# routes from the aggregated table, then re-run only those with viz on.
#
# This wraps the proven local harness scripts/eval_bench2drive.sh (same one the
# normal bench2drive eval uses), once per model, over a staged Town13 route file.
# Run it exactly like your other local evals: conda env active, CARLA_ROOT set.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROUTE_SRC="${ROUTE_SRC:-data/rl_finetuning_training_data/NonSignalizedJunctionLeftTurnEnterFlow}"
TOWN="${TOWN:-Town13}"
MAX_ROUTES="${MAX_ROUTES:-0}"          # >0 trims the file for a cheap first probe
OUT_ROOT="${OUT_ROOT:-outputs/eval_viz}"
STAGE_DIR="${STAGE_DIR:-$OUT_ROOT/routes_${TOWN}}"

export CARLA_ROOT="${CARLA_ROOT:-3rd_party/CARLA_0915}"
export REPETITIONS="${REPETITIONS:-1}"  # deterministic inference: one rep is enough
export NO_SAVE="${NO_SAVE:-1}"          # scan: outcomes only, no debug frames
export RESUME="${RESUME:-1}"            # resume: keep done routes, continue where it stopped

records_in() {  # echo number of leaderboard records in a checkpoint json (0 if absent)
  local j="$1"
  [[ -f "$j" ]] || { echo 0; return; }
  python -c "import json,sys;
try: print(len(json.load(open('$j'))['_checkpoint']['records']))
except Exception: print(0)" 2>/dev/null
}

# name|agent|config  — base first (reference), policy last (matches aggregator).
# Override from the environment with MODELS_SPEC (entries separated by ';').
if [[ -n "${MODELS_SPEC:-}" ]]; then
  IFS=';' read -r -a MODELS <<< "$MODELS_SPEC"
else
  MODELS=(
    "base|lead/inference/sensor_agent.py|outputs/checkpoints/tfv6_resnet34"
    "td3|rl_finetuning/inference/td3_sensor_agent.py|outputs/eval_viz/checkpoints/td3_final"
  )
fi

# ---------------------------------------------------------------------------
# 1. Stage routes (gunzip + town filter + dedup + optional trim) into one .xml
# ---------------------------------------------------------------------------
echo "== staging routes =="
python rl_finetuning/eval_viz/stage_routes.py \
  --src "$ROUTE_SRC" --town "$TOWN" --out "$STAGE_DIR" \
  ${MAX_ROUTES:+--max-routes "$MAX_ROUTES"}
ROUTES_XML="$(ls "$STAGE_DIR"/*.xml | head -n1)"
N_ROUTES="$(grep -c "<route " "$ROUTES_XML")"
echo "== routes: $ROUTES_XML ($N_ROUTES routes) =="

# ---------------------------------------------------------------------------
# 2. Run each model over the staged routes (scan mode: no frames, 1 rep)
# ---------------------------------------------------------------------------
for entry in "${MODELS[@]}"; do
  IFS='|' read -r NAME AGENT CONFIG <<< "$entry"
  echo ""
  echo "===================================================================="
  echo "== model: $NAME   agent=$AGENT   config=$CONFIG"
  echo "===================================================================="
  if [[ ! -e "$CONFIG" ]]; then
    echo "[error] config not found: $CONFIG (run extract_checkpoints.sh first)" >&2
    exit 1
  fi

  # Skip a model that is already complete (avoids booting CARLA just to no-op).
  DONE="$(records_in "$OUT_ROOT/$NAME/checkpoint_endpoint.json")"
  if [[ "$RESUME" == "1" && "$DONE" -ge "$N_ROUTES" ]]; then
    echo "== $NAME already complete ($DONE/$N_ROUTES) — skipping =="
    continue
  fi
  [[ "$DONE" -gt 0 ]] && echo "== $NAME resuming from $DONE/$N_ROUTES =="

  # Fresh CARLA per model on the fixed port, mirroring eval_bench2drive_multi_sequential.sh.
  trap 'bash scripts/clean_carla.sh' EXIT
  bash scripts/clean_carla.sh
  bash scripts/start_carla.sh
  sleep 12
  CHECKPOINT_DIR="$CONFIG" AGENT="$AGENT" ROUTES="$ROUTES_XML" \
    EVALUATION_OUTPUT_DIR="$OUT_ROOT/$NAME" \
    bash scripts/eval_bench2drive.sh
  bash scripts/clean_carla.sh
  sleep 5
  trap - EXIT
done

# ---------------------------------------------------------------------------
# 3. Aggregate into the selection table
# ---------------------------------------------------------------------------
echo ""
echo "== aggregating outcomes =="
AGG_ARGS=()
for entry in "${MODELS[@]}"; do
  IFS='|' read -r NAME _ _ <<< "$entry"
  AGG_ARGS+=(--model "$NAME=$OUT_ROOT/$NAME")
done
python rl_finetuning/eval_viz/aggregate_outcomes.py "${AGG_ARGS[@]}" \
  --out "$OUT_ROOT/outcomes_${TOWN}.tsv"

echo ""
echo "Done. Table: $OUT_ROOT/outcomes_${TOWN}.tsv"
