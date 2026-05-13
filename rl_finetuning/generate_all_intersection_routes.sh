#!/usr/bin/env bash
# Generate data/rl_finetuning_training_data/all_intersection from route XMLs.
#
# Prerequisites:
#   conda activate lead_carla_fork
#   export CARLA_ROOT=...
#   unset PYTHONPATH
#   CARLA server running (default localhost:2000)
#
# Run from repo root:
#   bash rl_finetuning/generate_all_intersection_routes.sh [--source-dataset lead|50x] [--train-towns "5 12 13 15"] [--interleaved-2env]
#
# --source-dataset  Route source to convert. "lead" uses data/data_routes/lead.
#                   "50x" uses data/data_routes/50x38_Town12 and
#                   data/data_routes/50x36_Town13 with the official CARLA
#                   intersection scenarios.
# --train-towns      Optional town list to convert. Accepts numbers or CARLA
#                    town names, separated by spaces or commas. If omitted,
#                    all towns present in the selected source dataset are used.
# --interleaved-2env  Split each town file into _00 (even routes) and _01 (odd
#                     routes) so two parallel envs on the same town together
#                     cover all routes without overlap.

set -euo pipefail

INTERLEAVED_2ENV=0
SOURCE_DATASET="lead"
TRAIN_TOWNS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --interleaved-2env) INTERLEAVED_2ENV=1 ;;
        --source-dataset)
            shift
            if [[ $# -eq 0 || "$1" == --* ]]; then
                echo "--source-dataset requires one value: lead or 50x"
                exit 1
            fi
            SOURCE_DATASET="$1"
            ;;
        --train-towns)
            shift
            if [[ $# -eq 0 || "$1" == --* ]]; then
                echo "--train-towns requires at least one town"
                exit 1
            fi
            while [[ $# -gt 0 && "$1" != --* ]]; do
                town_arg="${1//,/ }"
                read -r -a town_parts <<< "${town_arg}"
                for town in "${town_parts[@]}"; do
                    if [[ -n "${town}" ]]; then
                        TRAIN_TOWNS+=("${town}")
                    fi
                done
                shift
            done
            continue
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

REPO_ROOT="$(pwd)"
TMP_BASE="${REPO_ROOT}/data/rl_finetuning_training_data/_tmp_all_intersection"
CARLA_HOST="${CARLA_HOST:-127.0.0.1}"
CARLA_PORT="${CARLA_PORT:-2000}"

if [[ "${INTERLEAVED_2ENV}" -eq 1 ]]; then
    DEST="${REPO_ROOT}/data/rl_finetuning_training_data/all_intersection_Interleaved2env"
else
    DEST="${REPO_ROOT}/data/rl_finetuning_training_data/all_intersection"
fi

case "${SOURCE_DATASET}" in
    lead)
        SOURCE_ROOTS=("${REPO_ROOT}/data/data_routes/lead")
        SCENARIOS=(
            NonSignalizedJunctionLeftTurn
            NonSignalizedJunctionLeftTurnEnterFlow
            CrossJunctionDefectTrafficLight
            SignalizedJunctionLeftTurn
            SignalizedJunctionLeftTurnEnterFlow
            SignalizedJunctionRightTurn
        )
        DEFAULT_TRAIN_TOWNS_HINT="1 2 3 4 5 7 10 12 13 15"
        ;;
    50x)
        SOURCE_ROOTS=(
            "${REPO_ROOT}/data/data_routes/50x38_Town12"
            "${REPO_ROOT}/data/data_routes/50x36_Town13"
        )
        SCENARIOS=(
            SignalizedJunctionLeftTurn
            NonSignalizedJunctionLeftTurn
            SignalizedJunctionRightTurn
            NonSignalizedJunctionRightTurn
            OppositeVehicleTakingPriority
            OppositeVehicleRunningRedLight
            CrossingBicycleFlow
        )
        DEFAULT_TRAIN_TOWNS_HINT="12 13"
        ;;
    *)
        echo "Unknown --source-dataset: ${SOURCE_DATASET}. Expected lead or 50x."
        exit 1
        ;;
esac

TOWN_ARGS=()
if [[ "${#TRAIN_TOWNS[@]}" -gt 0 ]]; then
    TOWN_ARGS=(--towns "${TRAIN_TOWNS[@]}")
fi

rm -rf "${TMP_BASE}" "${DEST}"

# ---------------------------------------------------------------------------
# Step 1: convert each scenario type into its own temp subdir
# ---------------------------------------------------------------------------
echo "=== Step 1: Converting each scenario type ==="
for scenario in "${SCENARIOS[@]}"; do
    echo "--- ${scenario} ---"
    found_source=0
    for source_root in "${SOURCE_ROOTS[@]}"; do
        source_dir="${source_root}/${scenario}"
        if [[ ! -d "${source_dir}" ]]; then
            continue
        fi
        found_source=1
        python rl_finetuning/convert_lead_routes_to_carl.py \
            --source_dir "${source_dir}" \
            --dest_dir   "${TMP_BASE}/${scenario}" \
            --carla_host "${CARLA_HOST}" \
            --carla_port "${CARLA_PORT}" \
            "${TOWN_ARGS[@]}"
    done
    if [[ "${found_source}" -eq 0 ]]; then
        echo "No source folder found for ${scenario}; skipped"
    fi
done

# ---------------------------------------------------------------------------
# Step 2: round-robin merge per town, then optionally split for 2-env mode
# ---------------------------------------------------------------------------
echo "=== Step 2: Round-robin merge by town ==="
python - "${TMP_BASE}" "${DEST}" "${INTERLEAVED_2ENV}" "${#TRAIN_TOWNS[@]}" "${TRAIN_TOWNS[@]}" "${SCENARIOS[@]}" <<'PYEOF'
import gzip
import pathlib
import sys
import xml.etree.ElementTree as ET

tmp = pathlib.Path(sys.argv[1])
dest = pathlib.Path(sys.argv[2])
interleaved_2env = sys.argv[3] == "1"
town_count = int(sys.argv[4])
requested_towns = set(sys.argv[5:5 + town_count])
scenarios = sys.argv[5 + town_count:]
dest.mkdir(parents=True, exist_ok=True)

def normalize_town_name(town):
    town = town.strip()
    if town.lower().startswith("town"):
        suffix = town[4:]
        if suffix.lower() == "10hd":
            return "Town10HD"
        number = int(suffix)
    else:
        number = int(town)
    if number == 10:
        return "Town10HD"
    return f"Town{number:02d}"

requested_towns = {normalize_town_name(town) for town in requested_towns}

towns = set()
for s in scenarios:
    for f in (tmp / s).glob("*.xml.gz"):
        # route_Town12_00.xml.gz -> Town12
        parts = f.name.replace(".xml.gz", "").split("_")
        town = "_".join(parts[1:-1])
        if not requested_towns or town in requested_towns:
            towns.add(town)

for town in sorted(towns):
    routes_by_type = []
    for s in scenarios:
        gz_files = sorted((tmp / s).glob(f"*{town}*.xml.gz"))
        if not gz_files:
            continue
        type_routes = []
        for gz in gz_files:
            with gzip.open(gz, "rb") as fh:
                type_routes.extend(ET.parse(fh).getroot().findall("route"))
        routes_by_type.append((s, type_routes))

    if not routes_by_type:
        continue

    # Round-robin interleave across scenario types
    merged = []
    iters = [iter(rs) for _, rs in routes_by_type]
    active = list(range(len(iters)))
    route_id = 0
    while active:
        next_active = []
        for idx in active:
            try:
                r = next(iters[idx])
                r.set("id", str(route_id))
                merged.append(r)
                route_id += 1
                next_active.append(idx)
            except StopIteration:
                pass
        active = next_active

    counts = ", ".join(f"{s}: {len(rs)}" for s, rs in routes_by_type)

    def write_gz(path, routes):
        root = ET.Element("routes")
        for r in routes:
            root.append(r)
        ET.indent(root, space="  ")
        xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        with gzip.open(path, "wb") as fh:
            fh.write(xml_bytes)

    if interleaved_2env:
        even = [r for r in merged if int(r.get("id")) % 2 == 0]
        odd  = [r for r in merged if int(r.get("id")) % 2 == 1]
        write_gz(dest / f"route_{town}_00.xml.gz", even)
        write_gz(dest / f"route_{town}_01.xml.gz", odd)
        print(f"  route_{town}_00/01.xml.gz: {len(even)}/{len(odd)} routes ({counts})")
    else:
        write_gz(dest / f"route_{town}_00.xml.gz", merged)
        print(f"  route_{town}_00.xml.gz: {len(merged)} routes ({counts})")

PYEOF

echo "=== Done. Output: ${DEST} ==="
ls -lh "${DEST}"
echo ""
if [[ "${#TRAIN_TOWNS[@]}" -gt 0 ]]; then
    TRAIN_TOWNS_HINT="${TRAIN_TOWNS[*]}"
else
    TRAIN_TOWNS_HINT="${DEFAULT_TRAIN_TOWNS_HINT}"
fi

if [[ "${INTERLEAVED_2ENV}" -eq 1 ]]; then
    echo "Use in your .env file:"
    echo "  ROUTES_FOLDER=data/rl_finetuning_training_data/all_intersection_Interleaved2env"
    echo "  TRAIN_TOWNS=\"${TRAIN_TOWNS_HINT}\""
else
    echo "Use in your .env file:"
    echo "  ROUTES_FOLDER=data/rl_finetuning_training_data/all_intersection"
    echo "  TRAIN_TOWNS=\"${TRAIN_TOWNS_HINT}\""
fi
