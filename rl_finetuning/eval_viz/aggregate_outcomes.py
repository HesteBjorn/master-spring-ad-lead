#!/usr/bin/env python3
"""Aggregate leaderboard checkpoint JSONs into a per-route outcome table.

Reads the standard leaderboard ``_checkpoint.records`` produced by a closed-loop
evaluation run (see scan_routes.sh) and classifies the outcome of each route.
When several models are supplied, the routes are joined on their RouteScenario id
so the per-model outcomes sit side by side. This is the table used to pick the
illustrative clips for the thesis videos:

    stop_sign clip          -> td3 outcome == stop_sign
    gap-taking success      -> td3 outcome == success
    impatient collision     -> td3 outcome == collision, reached the junction
    TFv6-vs-TD3 side-by-side -> base != success AND td3 == success (column 'divergence')

Usage:
    python rl_finetuning/eval_viz/aggregate_outcomes.py \
        --model base=outputs/eval_viz/base \
        --model td3=outputs/eval_viz/td3 \
        [--out outputs/eval_viz/outcomes.tsv]

Each model path is either a leaderboard checkpoint JSON, or a directory that is
searched recursively for ``*.json`` checkpoint files (e.g. the eval/ folder).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass

# Infraction keys, grouped by the outcome they map to. Mirrors the leaderboard
# record["infractions"] schema (each value is a list of human-readable strings).
_COLLISION_KEYS = ("collisions_vehicle", "collisions_layout", "collisions_pedestrian")
_STOP_KEYS = ("stop_infraction",)
_RED_KEYS = ("red_light",)
_BLOCKED_KEYS = ("vehicle_blocked",)
_DEV_KEYS = ("route_dev",)
_TIMEOUT_KEYS = ("route_timeout", "scenario_timeouts")

# score_route (route completion %) above which we treat the junction as reached.
# Left-turn scenarios place the interaction near the route, so a high completion
# (or any collision, which only happens once the agent meets traffic) implies the
# agent actually got to the intersection rather than failing early.
_REACHED_INTERSECTION_RC = 50.0


@dataclass
class RouteOutcome:
    model: str
    route_id: str
    town: str
    score_route: float
    score_penalty: float
    route_length: float
    duration_game: float
    status: str
    n_collision: int
    n_stop: int
    n_red: int
    n_blocked: int
    n_dev: int
    n_timeout: int
    primary: str
    reached_intersection: bool


def _count(infractions: dict, keys) -> int:
    return sum(len(infractions.get(k, []) or []) for k in keys)


def _classify(record: dict) -> RouteOutcome | None:
    route_id = record.get("route_id")
    if route_id is None:
        return None
    inf = record.get("infractions", {}) or {}
    scores = record.get("scores", {}) or {}
    score_route = float(scores.get("score_route", 0.0))
    score_penalty = float(scores.get("score_penalty", 0.0))
    status = str(record.get("status", ""))

    n_collision = _count(inf, _COLLISION_KEYS)
    n_stop = _count(inf, _STOP_KEYS)
    n_red = _count(inf, _RED_KEYS)
    n_blocked = _count(inf, _BLOCKED_KEYS)
    n_dev = _count(inf, _DEV_KEYS)
    n_timeout = _count(inf, _TIMEOUT_KEYS)

    # Primary label: the most salient terminating event takes precedence, so a
    # route that both ran a stop sign and later crashed is filed under collision.
    completed = status.startswith("Completed") and score_route >= 99.0
    if n_collision > 0:
        primary = "collision"
    elif n_stop > 0:
        primary = "stop_sign"
    elif n_red > 0:
        primary = "red_light"
    elif n_blocked > 0:
        primary = "blocked"
    elif completed and score_penalty >= 0.99:
        # Fully completed with no penalty: a clean success outranks a minor,
        # non-terminating route deviation.
        primary = "success"
    elif n_dev > 0:
        primary = "route_dev"
    elif completed:
        # Completed the route but picked up a non-fatal penalty (e.g. route_dev
        # already handled above, min-speed, etc.).
        primary = "completed_penalized"
    elif n_timeout > 0:
        primary = "timeout"
    else:
        primary = "incomplete"

    meta = record.get("meta", {}) or {}
    reached = score_route >= _REACHED_INTERSECTION_RC or n_collision > 0
    return RouteOutcome(
        model="",
        route_id=str(route_id),
        town=str(record.get("town_name", "")),
        score_route=score_route,
        score_penalty=score_penalty,
        route_length=float(meta.get("route_length", 0.0) or 0.0),
        duration_game=float(meta.get("duration_game", 0.0) or 0.0),
        status=status,
        n_collision=n_collision,
        n_stop=n_stop,
        n_red=n_red,
        n_blocked=n_blocked,
        n_dev=n_dev,
        n_timeout=n_timeout,
        primary=primary,
        reached_intersection=reached,
    )


def _load_records(path: str) -> list[dict]:
    """Return all leaderboard records found at ``path`` (file or directory)."""
    files: list[str]
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True))
    else:
        files = [path]

    records: list[dict] = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        recs = data.get("_checkpoint", {}).get("records")
        if isinstance(recs, list):
            records.extend(recs)
    return records


def _short_route_id(route_id: str) -> str:
    """RouteScenario_23687_rep0 -> 23687 for compact, joinable keys."""
    parts = route_id.split("_")
    for p in parts:
        if p.isdigit():
            return p
    return route_id


def collect(model_specs: list[str]) -> dict[str, dict[str, RouteOutcome]]:
    """Return {model_name: {short_route_id: RouteOutcome}}."""
    out: dict[str, dict[str, RouteOutcome]] = {}
    for spec in model_specs:
        if "=" not in spec:
            raise SystemExit(f"--model expects name=path, got: {spec}")
        name, path = spec.split("=", 1)
        per_route: dict[str, RouteOutcome] = {}
        for rec in _load_records(path):
            oc = _classify(rec)
            if oc is None:
                continue
            oc.model = name
            per_route[_short_route_id(oc.route_id)] = oc
        if not per_route:
            print(f"[warn] no records found for model '{name}' at {path}")
        out[name] = per_route
    return out


_LONG_COLS = (
    "model route_id town score_route reached primary "
    "n_collision n_stop n_red n_blocked n_dev n_timeout status"
).split()


def write_long(table: dict[str, dict[str, RouteOutcome]], out_path: str | None) -> None:
    lines = ["\t".join(_LONG_COLS)]
    for model, routes in table.items():
        for rid in sorted(routes, key=lambda r: (r.isdigit() is False, r)):
            o = routes[rid]
            lines.append(
                "\t".join(
                    str(x)
                    for x in (
                        model,
                        rid,
                        o.town,
                        f"{o.score_route:.1f}",
                        int(o.reached_intersection),
                        o.primary,
                        o.n_collision,
                        o.n_stop,
                        o.n_red,
                        o.n_blocked,
                        o.n_dev,
                        o.n_timeout,
                        o.status,
                    )
                )
            )
    text = "\n".join(lines)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"[ok] wrote long table -> {out_path}")
    else:
        print(text)


def write_wide_and_summary(table: dict[str, dict[str, RouteOutcome]]) -> None:
    models = list(table.keys())
    all_rids = sorted(
        {rid for routes in table.values() for rid in routes},
        key=lambda r: (r.isdigit() is False, int(r) if r.isdigit() else 0, r),
    )

    print("\n=== outcome counts per model ===")
    for model in models:
        counts: dict[str, int] = {}
        for o in table[model].values():
            counts[o.primary] = counts.get(o.primary, 0) + 1
        total = sum(counts.values())
        summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"{model:>8} (n={total}):  {summary}")

    # Side-by-side divergence column needs at least two models. Treat the first
    # as the reference (base TFv6) and the second as the policy (TD3).
    if len(models) >= 2:
        base, policy = models[0], models[1]
        print(
            f"\n=== {policy} vs {base} per route (divergence = {base}!=success & {policy}==success) ==="
        )
        header = ["route", "town", f"{base}", f"{policy}", "divergence"]
        print("\t".join(header))
        diverging = []
        for rid in all_rids:
            b = table[base].get(rid)
            p = table[policy].get(rid)
            if b is None or p is None:
                continue
            div = (b.primary != "success") and (p.primary == "success")
            if div:
                diverging.append(rid)
            print("\t".join([rid, p.town, b.primary, p.primary, "YES" if div else ""]))
        print(
            f"\n[pick] side-by-side candidates ({base} fails, {policy} succeeds): "
            f"{', '.join(diverging) if diverging else 'none'}"
        )

    # Direct picks from the policy model (last one given).
    policy = models[-1]
    print(f"\n=== clip picks from '{policy}' ===")
    picks = {
        "stop_sign (#3)": [r for r, o in table[policy].items() if o.n_stop > 0],
        "gap success (#4)": [
            r for r, o in table[policy].items() if o.primary == "success"
        ],
        "impatient collision (#5)": [
            r
            for r, o in table[policy].items()
            if o.primary == "collision" and o.reached_intersection
        ],
        "red_light": [r for r, o in table[policy].items() if o.n_red > 0],
    }
    for label, rids in picks.items():
        shown = ", ".join(sorted(rids, key=lambda r: int(r) if r.isdigit() else 0)[:12])
        print(f"  {label:>26}: n={len(rids):3d}   {shown}")


# Behavior -> predicate over a policy RouteOutcome. "divergence" is handled
# separately (it needs the base model too).
_BEHAVIORS = {
    "stop_sign": lambda o: o.n_stop > 0,
    "success": lambda o: o.primary == "success",
    "gap": lambda o: o.primary == "success",
    "collision": lambda o: o.primary == "collision" and o.reached_intersection,
    "red_light": lambda o: o.n_red > 0,
    "blocked": lambda o: o.n_blocked > 0,
}


def print_behavior_candidates(
    table: dict[str, dict[str, RouteOutcome]], behavior: str
) -> None:
    """Detailed, suitability-sorted candidate list for one behavior.

    Sorted by route length ascending (shorter routes are easier to clip), so the
    top rows are the most convenient clean examples to swap to.
    """
    models = list(table.keys())
    policy = models[-1]
    routes = table[policy]

    if behavior == "divergence":
        if len(models) < 2:
            raise SystemExit("divergence needs two --model entries (base, policy)")
        base = table[models[0]]
        cands = [
            o
            for rid, o in routes.items()
            if o.primary == "success" and rid in base and base[rid].primary != "success"
        ]
    else:
        pred = _BEHAVIORS.get(behavior)
        if pred is None:
            raise SystemExit(
                f"unknown behavior '{behavior}'. Choose: {', '.join(sorted(_BEHAVIORS))}, divergence"
            )
        cands = [o for o in routes.values() if pred(o)]

    cands.sort(key=lambda o: (o.route_length or 1e9, o.route_id))
    print(
        f"\n=== '{behavior}' candidates from '{policy}' (sorted: shortest route first) ==="
    )
    cols = [
        "route",
        "len_m",
        "dur_s",
        "rc%",
        "penalty",
        "n_stop",
        "n_coll",
        "primary",
        "status",
    ]
    print("\t".join(cols))
    for o in cands:
        print(
            "\t".join(
                str(x)
                for x in (
                    _short_route_id(o.route_id),
                    f"{o.route_length:.0f}",
                    f"{o.duration_game:.0f}",
                    f"{o.score_route:.0f}",
                    f"{o.score_penalty:.2f}",
                    o.n_stop,
                    o.n_collision,
                    o.primary,
                    o.status,
                )
            )
        )
    print(
        f"[pick] {len(cands)} candidate(s); render one with "
        f"ROUTE_ID=<route> bash rl_finetuning/eval_viz/render_clip.sh"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="name=path",
        help="Model label and path to its checkpoint JSON or eval/ dir. Repeatable. "
        "Order matters: first is the reference (base), last is the policy (td3).",
    )
    ap.add_argument("--out", default=None, help="Optional path for the long TSV table.")
    ap.add_argument(
        "--behavior",
        default=None,
        help="Print a detailed, suitability-sorted candidate list for one behavior "
        "(stop_sign, gap/success, collision, red_light, blocked, divergence) to help "
        "swap to a cleaner/shorter route, instead of the default summary.",
    )
    args = ap.parse_args()
    if not args.model:
        ap.error("provide at least one --model name=path")

    table = collect(args.model)
    if args.behavior:
        print_behavior_candidates(table, args.behavior)
        return
    write_long(table, args.out)
    write_wide_and_summary(table)


if __name__ == "__main__":
    main()
