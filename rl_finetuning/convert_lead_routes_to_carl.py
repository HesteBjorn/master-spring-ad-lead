from __future__ import annotations

import argparse
import gzip
import math
import pathlib
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import defaultdict

import carla

SCENARIO_TYPE_MAP = {
    "NonSignalizedJunctionLeftTurnEnterFlow": "NonSignalizedJunctionLeftTurn",
    "SignalizedJunctionLeftTurnEnterFlow": "SignalizedJunctionLeftTurn",
}

UNSUPPORTED_SCENARIOS = {
    "CrossJunctionDefectTrafficLight",
    "RedLightWithoutLeadVehicle",
}

DEFAULT_COMMAND = "4"
DEFAULT_PREROLL_METERS = 35.0
DEFAULT_HOP_RESOLUTION = 1.0
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_TARGET_TOLERANCE_METERS = 2.0
DEFAULT_MAX_SEGMENT_STEPS = 300


def remap_scenario_type(old_type: str) -> str:
    if old_type in UNSUPPORTED_SCENARIOS:
        raise ValueError(
            f"Unsupported LEAD scenario type '{old_type}'. No safe CaRL equivalent is defined."
        )
    return SCENARIO_TYPE_MAP.get(old_type, old_type)


def parse_location(position_elem: ET.Element) -> carla.Location:
    return carla.Location(
        x=float(position_elem.attrib["x"]),
        y=float(position_elem.attrib["y"]),
        z=float(position_elem.attrib["z"]),
    )


def compute_route_length(route_points: list[tuple[carla.Transform, object]]) -> float:
    route_length = 0.0
    previous_location = None
    for transform, _ in route_points:
        if previous_location is not None:
            route_length += transform.location.distance(previous_location)
        previous_location = transform.location
    return round(route_length, 2)


def road_option_value(connection: object) -> str:
    try:
        return str(int(connection))
    except (TypeError, ValueError):
        value = getattr(connection, "value", None)
        if value is not None:
            return str(int(value))
    return DEFAULT_COMMAND


def choose_branch(
    candidates: list[carla.Waypoint],
    current: carla.Waypoint,
    forward_hint: carla.Vector3D,
) -> carla.Waypoint:
    if len(candidates) == 1:
        return candidates[0]

    best_candidate = candidates[0]
    best_score = float("-inf")
    for candidate in candidates:
        direction = current.transform.location - candidate.transform.location
        direction_norm = math.sqrt(
            direction.x * direction.x + direction.y * direction.y
        )
        hint_norm = math.sqrt(
            forward_hint.x * forward_hint.x + forward_hint.y * forward_hint.y
        )
        if direction_norm == 0.0 or hint_norm == 0.0:
            score = 0.0
        else:
            score = (direction.x * forward_hint.x + direction.y * forward_hint.y) / (
                direction_norm * hint_norm
            )

        # Prefer staying on the same lane if there is ambiguity.
        if (
            candidate.lane_id == current.lane_id
            and candidate.road_id == current.road_id
        ):
            score += 0.25

        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate


def backtrack_spawn_waypoint(
    start_wp: carla.Waypoint,
    forward_hint: carla.Vector3D,
    preroll_meters: float,
) -> carla.Waypoint:
    current_wp = start_wp
    traversed = 0.0
    best_non_junction_wp = start_wp if not start_wp.is_junction else None

    while traversed < preroll_meters:
        previous_wps = current_wp.previous(1.0)
        if not previous_wps:
            break
        next_wp = choose_branch(previous_wps, current_wp, forward_hint)
        traversed += next_wp.transform.location.distance(current_wp.transform.location)
        current_wp = next_wp
        if not current_wp.is_junction:
            best_non_junction_wp = current_wp

    if best_non_junction_wp is not None:
        return best_non_junction_wp
    return current_wp


def choose_forward_candidate(
    candidates: list[carla.Waypoint], target_location: carla.Location
) -> carla.Waypoint:
    best_candidate = candidates[0]
    best_distance = best_candidate.transform.location.distance(target_location)
    for candidate in candidates[1:]:
        distance = candidate.transform.location.distance(target_location)
        if distance < best_distance:
            best_candidate = candidate
            best_distance = distance
    return best_candidate


def walk_to_target(
    start_wp: carla.Waypoint,
    target_wp: carla.Waypoint,
    hop_resolution: float,
    tolerance_meters: float,
    max_segment_steps: int,
) -> list[tuple[carla.Transform, object]]:
    route: list[tuple[carla.Transform, object]] = [(start_wp.transform, 4)]
    current_wp = start_wp
    best_distance = current_wp.transform.location.distance(target_wp.transform.location)

    for _ in range(max_segment_steps):
        if best_distance <= tolerance_meters:
            break

        next_wps = current_wp.next(hop_resolution)
        if not next_wps:
            break

        next_wp = choose_forward_candidate(next_wps, target_wp.transform.location)
        next_distance = next_wp.transform.location.distance(
            target_wp.transform.location
        )

        # Avoid endless loops if a branch choice stops making progress.
        if next_distance > best_distance + 0.25:
            break

        route.append((next_wp.transform, 4))
        current_wp = next_wp
        best_distance = next_distance

    if best_distance > tolerance_meters:
        raise ValueError(
            f"Failed to walk route close enough to target waypoint. Remaining distance: {best_distance:.2f}m"
        )

    return route


def build_dense_route(
    world_map: carla.Map,
    source_positions: list[ET.Element],
    preroll_meters: float,
    hop_resolution: float,
    tolerance_meters: float,
    max_segment_steps: int,
) -> list[tuple[carla.Transform, object]]:
    projected_wps = [
        world_map.get_waypoint(
            parse_location(position),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        for position in source_positions
    ]
    if any(wp is None for wp in projected_wps):
        raise ValueError(
            "Failed to project one or more source waypoints onto a driving lane"
        )

    if len(projected_wps) >= 2:
        forward_hint = (
            projected_wps[1].transform.location - projected_wps[0].transform.location
        )
    else:
        yaw_radians = math.radians(projected_wps[0].transform.rotation.yaw)
        forward_hint = carla.Vector3D(
            x=math.cos(yaw_radians), y=math.sin(yaw_radians), z=0.0
        )

    start_wp = backtrack_spawn_waypoint(projected_wps[0], forward_hint, preroll_meters)

    dense_route: list[tuple[carla.Transform, object]] = [(start_wp.transform, 4)]
    current_wp = start_wp
    for target_wp in projected_wps:
        segment = walk_to_target(
            current_wp,
            target_wp,
            hop_resolution,
            tolerance_meters,
            max_segment_steps,
        )
        dense_route.extend(segment[1:])
        current_wp = target_wp

    if len(dense_route) < 2:
        raise ValueError(
            "Route collapsed to fewer than two dense waypoints after reconstruction"
        )

    return dense_route


def convert_route(
    route_elem: ET.Element,
    route_id: int,
    world_map: carla.Map,
    preroll_meters: float,
    hop_resolution: float,
    tolerance_meters: float,
    max_segment_steps: int,
) -> ET.Element:
    town = route_elem.attrib["town"]
    old_positions = list(route_elem.find("waypoints").iter("position"))
    if not old_positions:
        raise ValueError(f"Route in town {town} has no waypoints")

    old_scenarios = list(route_elem.find("scenarios").iter("scenario"))
    if not old_scenarios:
        raise ValueError(f"Route in town {town} has no scenarios")

    dense_route = build_dense_route(
        world_map,
        old_positions,
        preroll_meters,
        hop_resolution,
        tolerance_meters,
        max_segment_steps,
    )
    if len(dense_route) < 2:
        raise ValueError(
            f"Route in town {town} interpolated to fewer than two dense waypoints"
        )

    new_route = ET.Element("route")
    new_route.set("id", str(route_id))
    new_route.set("town", town)
    new_route.set("length", str(compute_route_length(dense_route)))
    ET.SubElement(new_route, "weathers")

    waypoints_elem = ET.SubElement(new_route, "waypoints")
    for transform, connection in dense_route:
        new_position = ET.SubElement(waypoints_elem, "position")
        new_position.set("x", f"{transform.location.x:.1f}")
        new_position.set("y", f"{transform.location.y:.1f}")
        new_position.set("z", f"{transform.location.z:.1f}")
        new_position.set("pitch", f"{transform.rotation.pitch:.1f}")
        new_position.set("yaw", f"{transform.rotation.yaw:.1f}")
        new_position.set("roll", f"{transform.rotation.roll:.1f}")
        new_position.set("command", road_option_value(connection))

    new_scenarios = ET.SubElement(new_route, "scenarios")
    for scenario_index, old_scenario in enumerate(old_scenarios, start=1):
        old_type = old_scenario.attrib["type"]
        new_type = remap_scenario_type(old_type)
        new_scenario = ET.SubElement(new_scenarios, "scenario")
        new_scenario.set("type", new_type)
        new_scenario.set("name", f"{new_type}_{scenario_index}")
        for old_child in list(old_scenario):
            new_child = ET.SubElement(new_scenario, old_child.tag)
            for key, value in old_child.attrib.items():
                new_child.set(key, value)

    return new_route


def wait_for_server(client: carla.Client, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            client.get_world()
            return
        except RuntimeError:
            time.sleep(1.0)
    raise TimeoutError(
        f"CARLA server did not become ready within {timeout_seconds} seconds"
    )


def maybe_launch_carla(
    carla_root: pathlib.Path | None, port: int
) -> subprocess.Popen[str] | None:
    if carla_root is None:
        return None
    return subprocess.Popen(
        [
            "bash",
            str(carla_root / "CarlaUE4.sh"),
            f"-carla-rpc-port={port}",
            "-nosound",
            "-RenderOffScreen",
            "-nullrhi",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True, type=pathlib.Path)
    parser.add_argument("--dest_dir", required=True, type=pathlib.Path)
    parser.add_argument("--skip_towns", nargs="*", default=[])
    parser.add_argument("--carla_host", default="127.0.0.1")
    parser.add_argument("--carla_port", default=2000, type=int)
    parser.add_argument("--carla_root", type=pathlib.Path, default=None)
    parser.add_argument("--preroll_meters", default=DEFAULT_PREROLL_METERS, type=float)
    parser.add_argument("--hop_resolution", default=DEFAULT_HOP_RESOLUTION, type=float)
    parser.add_argument(
        "--timeout_seconds", default=DEFAULT_TIMEOUT_SECONDS, type=float
    )
    parser.add_argument(
        "--target_tolerance_meters", default=DEFAULT_TARGET_TOLERANCE_METERS, type=float
    )
    parser.add_argument(
        "--max_segment_steps", default=DEFAULT_MAX_SEGMENT_STEPS, type=int
    )
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    dest_dir = args.dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(source_dir.glob("*.xml"))
    if not source_files:
        raise FileNotFoundError(f"No XML files found in {source_dir}")

    routes_by_town: dict[str, list[tuple[pathlib.Path, ET.Element]]] = defaultdict(list)
    skipped_files: list[tuple[pathlib.Path, str]] = []
    for source_file in source_files:
        tree = ET.parse(source_file)
        root = tree.getroot()
        route = root.find("route")
        if route is None:
            raise ValueError(f"{source_file} does not contain a route element")
        town = route.attrib["town"]
        if town in args.skip_towns:
            continue
        waypoints_elem = route.find("waypoints")
        scenarios_elem = route.find("scenarios")
        if waypoints_elem is None or not list(waypoints_elem.iter("position")):
            skipped_files.append((source_file, "missing waypoints"))
            continue
        if scenarios_elem is None or not list(scenarios_elem.iter("scenario")):
            skipped_files.append((source_file, "missing scenarios"))
            continue
        routes_by_town[town].append((source_file, route))

    if not routes_by_town:
        raise ValueError("No routes left after applying town filters")

    carla_process = maybe_launch_carla(
        args.carla_root.resolve() if args.carla_root else None, args.carla_port
    )
    try:
        if carla_process is not None:
            time.sleep(10.0)

        client = carla.Client(args.carla_host, args.carla_port)
        client.set_timeout(args.timeout_seconds)
        wait_for_server(client, args.timeout_seconds)

        for town, routes in sorted(routes_by_town.items()):
            world = client.load_world(town, reset_settings=False)
            world_map = world.get_map()

            routes_root = ET.Element("routes")
            route_id = 0
            for source_file, route in routes:
                try:
                    converted = convert_route(
                        route,
                        route_id,
                        world_map,
                        args.preroll_meters,
                        args.hop_resolution,
                        args.target_tolerance_meters,
                        args.max_segment_steps,
                    )
                    routes_root.append(converted)
                    route_id += 1
                except Exception as exc:  # noqa: BLE001
                    skipped_files.append((source_file, f"conversion failed: {exc}"))

            if route_id == 0:
                raise ValueError(f"No valid routes were generated for {town}")

            ET.indent(routes_root, space="  ")
            xml_bytes = ET.tostring(routes_root, encoding="utf-8", xml_declaration=True)
            output_path = dest_dir / f"route_{town}_00.xml.gz"
            with gzip.open(output_path, "wb") as output_file:
                output_file.write(xml_bytes)
    finally:
        if carla_process is not None:
            carla_process.terminate()
            try:
                carla_process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                carla_process.kill()

    for skipped_file, reason in skipped_files:
        print(f"Skipped {skipped_file}: {reason}")


if __name__ == "__main__":
    main()
