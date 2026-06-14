#!/usr/bin/env python3
"""Stage compressed training route files into plain .xml for closed-loop eval.

The training route sets ship as gzipped XML (route_TownXX_NN.xml.gz), but the
leaderboard evaluator and the slurm script generator glob for plain *.xml. This
helper gunzips a chosen town's route files into a staging directory, optionally
trimming each file to the first N routes for a cheap first scan.

Usage:
    python rl_finetuning/eval_viz/stage_routes.py \
        --src data/rl_finetuning_training_data/NonSignalizedJunctionLeftTurnEnterFlow \
        --town Town13 \
        --out outputs/eval_viz/routes_town13 \
        [--max-routes 20]
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import os
import xml.etree.ElementTree as ET

# ClearNoon weather (mirrors lead/common/weathers.py "ClearNoon"), as route XML
# attributes. Applied to every staged route so closed-loop runs use the fixed
# clear-noon weather the model trained on instead of the dim leaderboard default.
_CLEAR_NOON = {
    "route_percentage": "0",
    "cloudiness": "5.0",
    "precipitation": "0.0",
    "precipitation_deposits": "0.0",
    "wind_intensity": "10.0",
    "sun_azimuth_angle": "-1.0",
    "sun_altitude_angle": "90.0",
    "fog_density": "2.0",
    "fog_distance": "0.75",
    "fog_falloff": "0.1",
    "wetness": "0.0",
    "scattering_intensity": "1.0",
    "mie_scattering_scale": "0.03",
    "rayleigh_scattering_scale": "0.0331",
    "dust_storm": "0.0",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="Folder with route_*.xml.gz files.")
    ap.add_argument(
        "--town", default=None, help="Keep only files for this town (e.g. Town13)."
    )
    ap.add_argument("--out", required=True, help="Staging dir for plain .xml output.")
    ap.add_argument(
        "--max-routes",
        type=int,
        default=0,
        help="If >0, keep only the first N routes per file (cheap probe).",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # Accept both the gzipped ported training routes (route_TownXX_NN.xml.gz, one
    # town per file) and the plain LEAD source routes (route_NNNNNN.xml, single
    # route per file, town only in the route attribute). Town filtering happens
    # per <route> below so both layouts work.
    src_files = sorted(
        glob.glob(os.path.join(args.src, "**", "*.xml.gz"), recursive=True)
        + glob.glob(os.path.join(args.src, "**", "*.xml"), recursive=True)
    )
    if not src_files:
        raise SystemExit(f"no .xml/.xml.gz routes under {args.src}")

    total_routes = 0
    seen_hashes: dict[str, str] = {}
    for src in src_files:
        if src.endswith(".gz"):
            with gzip.open(src, "rt", encoding="utf-8") as fh:
                raw = fh.read()
        else:
            with open(src, encoding="utf-8") as fh:
                raw = fh.read()
        # The interleaved training layouts ship identical route files duplicated
        # per parallel env (e.g. route_Town13_00/01/02 are byte-identical). Stage
        # each unique route set only once so the scan does not repeat work.
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            print(
                f"[stage] skip duplicate {os.path.basename(src)} (same as {seen_hashes[digest]})"
            )
            continue
        seen_hashes[digest] = os.path.basename(src)
        tree = ET.ElementTree(ET.fromstring(raw))
        root = tree.getroot()

        # Keep only routes for the requested town. The ported dense files put the
        # town in the filename (one town per file), but the LEAD source files mix
        # towns across single-route files, so filter on each route's town attribute.
        if args.town:
            for route_elem in root.findall("route"):
                if route_elem.get("town") != args.town:
                    root.remove(route_elem)
            if not root.findall("route"):
                continue

        # The training routes carry an empty <weathers /> element. The Bench2Drive
        # parser crashes on an empty element, and an ABSENT one makes it fall back
        # to a dim partly-cloudy default (sun_altitude=70, cloudiness=50). Replace
        # the weather with explicit ClearNoon so scan + rendered videos use the
        # same fixed clear-noon weather the model trained on (lead/common/weathers.py).
        n_set = 0
        for route_elem in root.findall("route"):
            for weathers_elem in route_elem.findall("weathers"):
                route_elem.remove(weathers_elem)
            weathers = ET.SubElement(route_elem, "weathers")
            ET.SubElement(weathers, "weather", _CLEAR_NOON)
            # Keep weathers as the first child (parser is position-agnostic, but
            # this matches the original route layout).
            route_elem.remove(weathers)
            route_elem.insert(0, weathers)
            n_set += 1
        if n_set:
            print(f"[stage] set ClearNoon weather on {n_set} route(s)")

        routes = root.findall("route")
        if args.max_routes > 0 and len(routes) > args.max_routes:
            for extra in routes[args.max_routes :]:
                root.remove(extra)
            routes = root.findall("route")
        total_routes += len(routes)

        stem = os.path.basename(src)  # route_Town13_00.xml or route_000987.xml
        if stem.endswith(".gz"):
            stem = stem[: -len(".gz")]
        dst = os.path.join(args.out, stem)
        tree.write(dst, encoding="utf-8", xml_declaration=True)
        print(f"[stage] {src}  ->  {dst}  ({len(routes)} routes)")

    print(
        f"[stage] staged {len(src_files)} file(s), {total_routes} routes total -> {args.out}"
    )


if __name__ == "__main__":
    main()
