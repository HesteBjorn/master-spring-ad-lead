#!/bin/bash
# Build inference checkpoint dirs for the route scan / video runs.
#
# The TD3 sensor agent needs an extracted checkpoint dir (config.json +
# model*.pth + td3_policy.pth + rl_config.json), not the raw training .pth.
# This wraps scripts/extract_trained_tfv6_model_from_policy.py for the two
# baseline checkpoints we use:
#   td3_final     converged baseline (1.5M steps)  -> clips #4 #5 #6
#   td3_stopsign  stop-sign spike (~75k steps)     -> clip #3
#
# td3_final is the IDUN converged baseline (TD3_idun_baseline), the checkpoint the
# rest of the thesis reports. Its run config.json points at a /cluster/... base
# path, so we pass --base-checkpoint-dir explicitly. IDUN only saved the final
# checkpoint, so the stop-sign spike (a transient ~75k phenomenon) comes from the
# local baseline run T1213, which used the same speed-only residual config.
#
# The base TFv6 model (outputs/checkpoints/tfv6_resnet34) is already a usable
# inference dir for lead/inference/sensor_agent.py and needs no extraction.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."

# lead.common.common_utils imports agents.navigation.* from CARLA's PythonAPI;
# without it extraction fails with ModuleNotFoundError: No module named 'agents'.
: "${CARLA_ROOT:=3rd_party/CARLA_0915}"
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI:${PYTHONPATH:-}"

FINAL_RUN="outputs/rl_logs/TD3_idun_baseline"
SPIKE_RUN="outputs/rl_logs/TFV6_TD3_LOCAL_RESIDUAL_8_baseline_T1213"
BASE="outputs/checkpoints/tfv6_resnet34"
OUT="outputs/eval_viz/checkpoints"
mkdir -p "$OUT"

python scripts/extract_trained_tfv6_model_from_policy.py \
  "$FINAL_RUN/model_latest_000001500000.pth" \
  --base-checkpoint-dir "$BASE" \
  --output-dir "$OUT/td3_final" --force

python scripts/extract_trained_tfv6_model_from_policy.py \
  "$SPIKE_RUN/model_latest_000000075000.pth" \
  --base-checkpoint-dir "$BASE" \
  --output-dir "$OUT/td3_stopsign" --force

echo "Done. Extracted to $OUT/{td3_final,td3_stopsign}"
