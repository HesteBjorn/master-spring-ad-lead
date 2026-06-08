#!/usr/bin/env bash
# Generate RL-compatible route bundles from route XMLs.
#
# Prerequisites:
#   conda activate lead_carla_fork
#   export CARLA_ROOT=...
#   unset PYTHONPATH
#   CARLA server running (default localhost:2000)
#
# Run from repo root:
#   bash rl_finetuning/generate_all_intersection_routes.sh [--source-dataset lead|50x|all4lt|allscenarios_b2d] [--train-towns "5 12 13 15"] [--interleaved-2env|--mixed-2env]
#
# --source-dataset  Route source to convert. "lead" uses data/data_routes/lead.
#                   "50x" uses data/data_routes/50x38_Town12 and
#                   data/data_routes/50x36_Town13 with the official CARLA
#                   intersection scenarios.
#                   "all4lt" outputs only the four bench2drive left-turn scenario
#                   types into data/rl_finetuning_training_data/all_4_leftturn_Scenarios:
#                   base Signalized/NonSignalized LeftTurn from the 50x Town12/Town13
#                   sets, plus the EnterFlow variants from lead/ (labels preserved).
#                   "allscenarios_b2d" uses all 50x route folders, adds
#                   Bench2Drive training YieldToEmergencyVehicle routes, and
#                   adds ported enter-flow/T-junction sources where available.
# --train-towns      Optional town list to convert. Accepts numbers or CARLA
#                    town names, separated by spaces or commas. If omitted,
#                    all towns present in the selected source dataset are used.
# --interleaved-2env  Split each town file into _00 (even routes) and _01 (odd
#                     routes) so two parallel envs on the same town together
#                     cover all routes without overlap.
# --mixed-2env        Write route_mixed_00.xml.gz and route_mixed_01.xml.gz,
#                     each containing all selected towns/scenarios interleaved.

set -euo pipefail

INTERLEAVED_2ENV=0
MIXED_2ENV=0
SOURCE_DATASET="lead"
TRAIN_TOWNS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --interleaved-2env) INTERLEAVED_2ENV=1 ;;
        --mixed-2env) MIXED_2ENV=1 ;;
        --source-dataset)
            shift
            if [[ $# -eq 0 || "$1" == --* ]]; then
                echo "--source-dataset requires one value: lead, 50x, or allscenarios_b2d"
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
TMP_BASE="${REPO_ROOT}/data/rl_finetuning_training_data/_tmp_${SOURCE_DATASET}"
CARLA_HOST="${CARLA_HOST:-127.0.0.1}"
CARLA_PORT="${CARLA_PORT:-2000}"

if [[ "${SOURCE_DATASET}" == "allscenarios_b2d" && "${INTERLEAVED_2ENV}" -eq 0 ]]; then
    MIXED_2ENV=1
fi

if [[ "${SOURCE_DATASET}" == "all4lt" ]]; then
    DEST="${REPO_ROOT}/data/rl_finetuning_training_data/all_4_leftturn_Scenarios"
elif [[ "${MIXED_2ENV}" -eq 1 ]]; then
    DEST="${REPO_ROOT}/data/rl_finetuning_training_data/all_scenarios_b2d_train"
elif [[ "${INTERLEAVED_2ENV}" -eq 1 ]]; then
    if [[ "${SOURCE_DATASET}" == "allscenarios_b2d" ]]; then
        DEST="${REPO_ROOT}/data/rl_finetuning_training_data/all_scenarios_b2d_train_Interleaved2env"
    else
        DEST="${REPO_ROOT}/data/rl_finetuning_training_data/all_intersection_Interleaved2env"
    fi
else
    if [[ "${SOURCE_DATASET}" == "allscenarios_b2d" ]]; then
        DEST="${REPO_ROOT}/data/rl_finetuning_training_data/all_scenarios_b2d_train"
    else
        DEST="${REPO_ROOT}/data/rl_finetuning_training_data/all_intersection"
    fi
fi

case "${SOURCE_DATASET}" in
    lead)
        SOURCE_ROOTS=("${REPO_ROOT}/data/data_routes/lead")
        B2D_SOURCE_FILE=""
        LEAD_EXTRA_ROOT=""
        LEADERBOARD_EXTRA_ROOT=""
        SCENARIOS=(
            NonSignalizedJunctionLeftTurn
            NonSignalizedJunctionLeftTurnEnterFlow
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
        B2D_SOURCE_FILE=""
        LEAD_EXTRA_ROOT=""
        LEADERBOARD_EXTRA_ROOT=""
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
    allscenarios_b2d)
        SOURCE_ROOTS=(
            "${REPO_ROOT}/data/data_routes/50x38_Town12"
            "${REPO_ROOT}/data/data_routes/50x36_Town13"
        )
        B2D_SOURCE_FILE="${REPO_ROOT}/3rd_party/Bench2Drive/leaderboard/data/routes_training.xml"
        LEAD_EXTRA_ROOT="${REPO_ROOT}/data/data_routes/lead"
        LEADERBOARD_EXTRA_ROOT="${REPO_ROOT}/data/data_routes/leaderboard1"
        SCENARIOS=(
            Accident
            AccidentTwoWays
            BlockedIntersection
            ConstructionObstacle
            ConstructionObstacleTwoWays
            ControlLoss
            CrossingBicycleFlow
            DynamicObjectCrossing
            EnterActorFlow
            EnterActorFlowV2
            HardBreakRoute
            HazardAtSideLane
            HazardAtSideLaneTwoWays
            HighwayCutIn
            HighwayExit
            InterurbanActorFlow
            InterurbanAdvancedActorFlow
            InvadingTurn
            MergerIntoSlowTraffic
            MergerIntoSlowTrafficV2
            NonSignalizedJunctionLeftTurn
            NonSignalizedJunctionRightTurn
            OppositeVehicleRunningRedLight
            OppositeVehicleTakingPriority
            ParkedObstacle
            ParkedObstacleTwoWays
            ParkingCrossingPedestrian
            ParkingCutIn
            ParkingExit
            PedestrianCrossing
            PriorityAtJunction
            SignalizedJunctionLeftTurn
            SignalizedJunctionRightTurn
            StaticCutIn
            VehicleOpensDoorTwoWays
            VehicleTurningRoute
            VehicleTurningRoutePedestrian
            YieldToEmergencyVehicle
            NonSignalizedJunctionLeftTurnEnterFlow
            SignalizedJunctionLeftTurnEnterFlow
            T_Junction
        )
        DEFAULT_TRAIN_TOWNS_HINT="12 13"
        ;;
    all4lt)
        # The four bench2drive left-turn scenario types only. Base (non-EnterFlow)
        # types come from the 50x Town12/Town13 sets, which carry the Town12/Town13
        # SignalizedJunctionLeftTurn routes that lead/ lacks. The EnterFlow variants
        # come from lead/ with their type labels preserved. The two source families
        # never share a (scenario, town) pair, so no per-town output files collide.
        SOURCE_ROOTS=(
            "${REPO_ROOT}/data/data_routes/50x38_Town12"
            "${REPO_ROOT}/data/data_routes/50x36_Town13"
        )
        B2D_SOURCE_FILE=""
        LEAD_EXTRA_ROOT="${REPO_ROOT}/data/data_routes/lead"
        LEADERBOARD_EXTRA_ROOT=""
        SCENARIOS=(
            NonSignalizedJunctionLeftTurn
            SignalizedJunctionLeftTurn
            NonSignalizedJunctionLeftTurnEnterFlow
            SignalizedJunctionLeftTurnEnterFlow
        )
        DEFAULT_TRAIN_TOWNS_HINT="12 13"
        ;;
    *)
        echo "Unknown --source-dataset: ${SOURCE_DATASET}. Expected lead, 50x, all4lt, or allscenarios_b2d."
        exit 1
        ;;
esac

TOWN_ARGS=()
if [[ "${#TRAIN_TOWNS[@]}" -gt 0 ]]; then
    TOWN_ARGS=(--towns "${TRAIN_TOWNS[@]}")
fi

rm -rf "${TMP_BASE}" "${DEST}"

python - "${REPO_ROOT}" "${SCENARIOS[@]}" <<'PYEOF'
import ast
import pathlib
import sys

repo_root = pathlib.Path(sys.argv[1])
requested = set(sys.argv[2:])
scenario_root = repo_root / "3rd_party/CaRL/CARLA/custom_leaderboard/scenario_runner/srunner/scenarios"
available = set()
for path in scenario_root.glob("*.py"):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            available.add(node.name)

missing = sorted(requested - available)
if missing:
    print("ERROR: CaRL custom scenario runner is missing scenario classes:")
    for scenario in missing:
        print(f"  {scenario}")
    raise SystemExit(1)
PYEOF

# ---------------------------------------------------------------------------
# Step 1: convert each scenario type into its own temp subdir
# ---------------------------------------------------------------------------
echo "=== Step 1: Converting each scenario type ==="
for scenario in "${SCENARIOS[@]}"; do
    echo "--- ${scenario} ---"
    found_source=0

    if [[ "${scenario}" == "YieldToEmergencyVehicle" && -n "${B2D_SOURCE_FILE}" && -f "${B2D_SOURCE_FILE}" ]]; then
        found_source=1
        python rl_finetuning/convert_lead_routes_to_carl.py \
            --source_file "${B2D_SOURCE_FILE}" \
            --dest_dir   "${TMP_BASE}/${scenario}" \
            --include_scenario_types "${scenario}" \
            --preserve_scenario_types \
            --carla_host "${CARLA_HOST}" \
            --carla_port "${CARLA_PORT}" \
            "${TOWN_ARGS[@]}"
    fi

    if [[ "${scenario}" == "NonSignalizedJunctionLeftTurnEnterFlow" || "${scenario}" == "SignalizedJunctionLeftTurnEnterFlow" ]]; then
        if [[ -n "${LEAD_EXTRA_ROOT}" && -d "${LEAD_EXTRA_ROOT}/${scenario}" ]]; then
            found_source=1
            python rl_finetuning/convert_lead_routes_to_carl.py \
                --source_dir "${LEAD_EXTRA_ROOT}/${scenario}" \
                --dest_dir   "${TMP_BASE}/${scenario}" \
                --preserve_scenario_types \
                --carla_host "${CARLA_HOST}" \
                --carla_port "${CARLA_PORT}" \
                "${TOWN_ARGS[@]}"
        fi
        if [[ "${found_source}" -eq 0 ]]; then
            echo "No source folder found for ${scenario}; skipped"
        fi
        continue
    fi

    if [[ "${scenario}" == "T_Junction" ]]; then
        if [[ -n "${LEADERBOARD_EXTRA_ROOT}" && -d "${LEADERBOARD_EXTRA_ROOT}/${scenario}" ]]; then
            found_source=1
            python rl_finetuning/convert_lead_routes_to_carl.py \
                --source_dir "${LEADERBOARD_EXTRA_ROOT}/${scenario}" \
                --dest_dir   "${TMP_BASE}/${scenario}" \
                --preserve_scenario_types \
                --carla_host "${CARLA_HOST}" \
                --carla_port "${CARLA_PORT}" \
                "${TOWN_ARGS[@]}"
        fi
        if [[ "${found_source}" -eq 0 ]]; then
            echo "No source folder found for ${scenario}; skipped"
        fi
        continue
    fi

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
echo "=== Step 2: Round-robin merge ==="
python - "${TMP_BASE}" "${DEST}" "${INTERLEAVED_2ENV}" "${MIXED_2ENV}" "${REPO_ROOT}" "${#TRAIN_TOWNS[@]}" "${TRAIN_TOWNS[@]}" "${SCENARIOS[@]}" <<'PYEOF'
import collections
import gzip
import hashlib
import json
import pathlib
import sys
import xml.etree.ElementTree as ET

tmp = pathlib.Path(sys.argv[1])
dest = pathlib.Path(sys.argv[2])
interleaved_2env = sys.argv[3] == "1"
mixed_2env = sys.argv[4] == "1"
repo_root = pathlib.Path(sys.argv[5])
town_count = int(sys.argv[6])
requested_towns = set(sys.argv[7:7 + town_count])
scenarios = sys.argv[7 + town_count:]
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

def route_town_from_file(path):
    parts = path.name.replace(".xml.gz", "").split("_")
    return "_".join(parts[1:-1])

def route_scenario_types(route):
    scenarios_elem = route.find("scenarios")
    if scenarios_elem is None:
        return ["<missing>"]
    types = [scenario.get("type", "<missing>") for scenario in scenarios_elem.findall("scenario")]
    return types or ["<missing>"]

def route_signature(route):
    waypoints = []
    waypoints_elem = route.find("waypoints")
    if waypoints_elem is not None:
        for position in waypoints_elem.findall("position"):
            waypoints.append(tuple(position.get(key, "") for key in ("x", "y", "z")))
    payload = {
        "town": route.get("town", ""),
        "types": route_scenario_types(route),
        "waypoints": waypoints,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

def write_gz(path, routes):
    root = ET.Element("routes")
    for route_id, route in enumerate(routes):
        route.set("id", str(route_id))
        root.append(route)
    ET.indent(root, space="  ")
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    with gzip.open(path, "wb") as fh:
        fh.write(xml_bytes)

def summarize(output_files):
    manifest = {
        "files": {},
        "overall": {
            "routes": 0,
            "by_scenario": collections.Counter(),
            "by_town": collections.Counter(),
            "unique_by_scenario": collections.defaultdict(set),
        },
    }
    for path in output_files:
        with gzip.open(path, "rb") as fh:
            routes = ET.parse(fh).getroot().findall("route")
        file_summary = {
            "routes": len(routes),
            "by_scenario": collections.Counter(),
            "by_town": collections.Counter(),
            "unique_by_scenario": collections.defaultdict(set),
        }
        for route in routes:
            town = route.get("town", "")
            sig = route_signature(route)
            manifest["overall"]["routes"] += 1
            manifest["overall"]["by_town"][town] += 1
            file_summary["by_town"][town] += 1
            for scenario_type in route_scenario_types(route):
                manifest["overall"]["by_scenario"][scenario_type] += 1
                manifest["overall"]["unique_by_scenario"][scenario_type].add(sig)
                file_summary["by_scenario"][scenario_type] += 1
                file_summary["unique_by_scenario"][scenario_type].add(sig)
        manifest["files"][path.name] = file_summary

    def plain(summary):
        return {
            "routes": summary["routes"],
            "by_town": dict(sorted(summary["by_town"].items())),
            "by_scenario": dict(sorted(summary["by_scenario"].items())),
            "unique_by_scenario": {
                key: len(value)
                for key, value in sorted(summary["unique_by_scenario"].items())
            },
        }

    plain_manifest = {
        "overall": plain(manifest["overall"]),
        "files": {
            name: plain(summary)
            for name, summary in sorted(manifest["files"].items())
        },
    }
    (dest / "manifest.json").write_text(json.dumps(plain_manifest, indent=2) + "\n")
    lines = [
        f"routes: {plain_manifest['overall']['routes']}",
        "",
        "scenario\tinstances\tunique",
    ]
    for scenario_type, count in plain_manifest["overall"]["by_scenario"].items():
        unique = plain_manifest["overall"]["unique_by_scenario"].get(scenario_type, 0)
        lines.append(f"{scenario_type}\t{count}\t{unique}")
    (dest / "manifest.tsv").write_text("\n".join(lines) + "\n")
    write_bench2drive_gap_report(plain_manifest)

def write_bench2drive_gap_report(plain_manifest):
    b2d_ref = repo_root / "3rd_party/Bench2Drive/leaderboard/data/bench2drive220.xml"
    if not b2d_ref.exists():
        return
    b2d_counts = collections.Counter()
    b2d_towns = collections.defaultdict(set)
    root = ET.parse(b2d_ref).getroot()
    for route in root.findall("route"):
        town = route.get("town", "")
        for scenario in route.findall("./scenarios/scenario"):
            scenario_type = scenario.get("type", "<missing>")
            b2d_counts[scenario_type] += 1
            b2d_towns[scenario_type].add(town)

    train_counts = plain_manifest["overall"]["by_scenario"]
    unique_counts = plain_manifest["overall"]["unique_by_scenario"]
    rows = ["scenario\tb2d220_routes\ttrain_instances\ttrain_unique\tb2d_towns\tstatus"]
    for scenario_type, b2d_count in sorted(b2d_counts.items()):
        train_count = train_counts.get(scenario_type, 0)
        unique_count = unique_counts.get(scenario_type, 0)
        if train_count == 0:
            status = "missing"
        elif unique_count < b2d_count:
            status = "weak_unique_coverage"
        else:
            status = "covered"
        rows.append(
            "\t".join(
                [
                    scenario_type,
                    str(b2d_count),
                    str(train_count),
                    str(unique_count),
                    ",".join(sorted(b2d_towns[scenario_type])),
                    status,
                ]
            )
        )
    (dest / "bench2drive_gap_report.tsv").write_text("\n".join(rows) + "\n")

def collect_routes_for_town(town):
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
    return routes_by_type

def round_robin(routes_by_type):
    merged = []
    iters = [iter(rs) for _, rs in routes_by_type]
    active = list(range(len(iters)))
    while active:
        next_active = []
        for idx in active:
            try:
                r = next(iters[idx])
                merged.append(r)
                next_active.append(idx)
            except StopIteration:
                pass
        active = next_active
    return merged

towns = set()
for s in scenarios:
    for f in (tmp / s).glob("*.xml.gz"):
        town = route_town_from_file(f)
        if not requested_towns or town in requested_towns:
            towns.add(town)

output_files = []

if mixed_2env:
    routes_by_type = []
    for s in scenarios:
        type_routes = []
        for gz in sorted((tmp / s).glob("*.xml.gz")):
            town = route_town_from_file(gz)
            if requested_towns and town not in requested_towns:
                continue
            with gzip.open(gz, "rb") as fh:
                type_routes.extend(ET.parse(fh).getroot().findall("route"))
        if type_routes:
            routes_by_type.append((s, type_routes))

    merged = round_robin(routes_by_type)
    split_a = [route for idx, route in enumerate(merged) if idx % 2 == 0]
    split_b = [route for idx, route in enumerate(merged) if idx % 2 == 1]
    path_a = dest / "route_mixed_00.xml.gz"
    path_b = dest / "route_mixed_01.xml.gz"
    write_gz(path_a, split_a)
    write_gz(path_b, split_b)
    output_files.extend([path_a, path_b])
    counts = ", ".join(f"{s}: {len(rs)}" for s, rs in routes_by_type)
    print(f"  route_mixed_00/01.xml.gz: {len(split_a)}/{len(split_b)} routes ({counts})")
else:
    for town in sorted(towns):
        routes_by_type = collect_routes_for_town(town)
        if not routes_by_type:
            continue

        merged = round_robin(routes_by_type)
        counts = ", ".join(f"{s}: {len(rs)}" for s, rs in routes_by_type)

        if interleaved_2env:
            even = [r for idx, r in enumerate(merged) if idx % 2 == 0]
            odd  = [r for idx, r in enumerate(merged) if idx % 2 == 1]
            path_even = dest / f"route_{town}_00.xml.gz"
            path_odd = dest / f"route_{town}_01.xml.gz"
            write_gz(path_even, even)
            write_gz(path_odd, odd)
            output_files.extend([path_even, path_odd])
            print(f"  route_{town}_00/01.xml.gz: {len(even)}/{len(odd)} routes ({counts})")
        else:
            path = dest / f"route_{town}_00.xml.gz"
            write_gz(path, merged)
            output_files.append(path)
            print(f"  route_{town}_00.xml.gz: {len(merged)} routes ({counts})")

if not output_files:
    raise SystemExit("No output route files were generated")

summarize(output_files)
PYEOF

echo "=== Done. Output: ${DEST} ==="
ls -lh "${DEST}"
echo ""
if [[ "${#TRAIN_TOWNS[@]}" -gt 0 ]]; then
    TRAIN_TOWNS_HINT="${TRAIN_TOWNS[*]}"
else
    TRAIN_TOWNS_HINT="${DEFAULT_TRAIN_TOWNS_HINT}"
fi

if [[ "${MIXED_2ENV}" -eq 1 ]]; then
    echo "Use in your .env file:"
    echo "  ROUTES_FOLDER=data/rl_finetuning_training_data/all_scenarios_b2d_train"
    echo "  ROUTE_ASSIGNMENT_MODE=mixed"
    echo "  TRAIN_TOWNS=\"${TRAIN_TOWNS_HINT}\""
elif [[ "${INTERLEAVED_2ENV}" -eq 1 ]]; then
    echo "Use in your .env file:"
    if [[ "${SOURCE_DATASET}" == "allscenarios_b2d" ]]; then
        echo "  ROUTES_FOLDER=data/rl_finetuning_training_data/all_scenarios_b2d_train_Interleaved2env"
    else
        echo "  ROUTES_FOLDER=data/rl_finetuning_training_data/all_intersection_Interleaved2env"
    fi
    echo "  ROUTE_ASSIGNMENT_MODE=town"
    echo "  TRAIN_TOWNS=\"${TRAIN_TOWNS_HINT}\""
else
    echo "Use in your .env file:"
    if [[ "${SOURCE_DATASET}" == "allscenarios_b2d" ]]; then
        echo "  ROUTES_FOLDER=data/rl_finetuning_training_data/all_scenarios_b2d_train"
    else
        echo "  ROUTES_FOLDER=data/rl_finetuning_training_data/all_intersection"
    fi
    echo "  ROUTE_ASSIGNMENT_MODE=town"
    echo "  TRAIN_TOWNS=\"${TRAIN_TOWNS_HINT}\""
fi
