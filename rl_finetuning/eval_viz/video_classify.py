"""Classify a leaderboard route record into a demo-video category.

Shared by monitor_routes.py (live keep/delete) and build_overview.py (catalog +
comparisons). The rules implement the agreed policy:

  * route deviation (route_dev) -> DELETE (bad footage).
  * timeouts are KEPT (valid for the TD3-vs-TFv6 comparison; "timeout is fine").
  * min_speed / outside_route_lanes are ubiquitous background noise -> ignored
    when deciding whether an infraction is "isolated".
  * meaningful infractions = {collision, stop, red_light, blocked}. A route with
    exactly one of these (and nothing else meaningful) is an isolated example;
    collisions that follow a stop/red are listed under their own combo category.
"""

from __future__ import annotations

_COLLISION_KEYS = ("collisions_vehicle", "collisions_layout", "collisions_pedestrian")
_STOP_KEYS = ("stop_infraction",)
_RED_KEYS = ("red_light",)
_BLOCKED_KEYS = ("vehicle_blocked",)
_DEV_KEYS = ("route_dev",)
_TIMEOUT_KEYS = ("route_timeout", "scenario_timeouts")

DELETE = "DELETE"  # sentinel category -> remove the route folder

# Categories worth cataloguing as standalone clips.
KEEP_CATEGORIES = (
    "success",
    "stop_isolated",
    "red_isolated",
    "collision_isolated",
    "blocked_isolated",
    "timeout",
    "stop_then_collision",
    "red_then_collision",
    "stop_red_then_collision",
    "stop_red_combo",
    "other",
)

_REACHED_RC = 50.0


def _count(inf: dict, keys) -> int:
    return sum(len(inf.get(k, []) or []) for k in keys)


def classify(record: dict) -> dict:
    """Return a flat dict: category + outcome stats + meta for one route record."""
    inf = record.get("infractions", {}) or {}
    scores = record.get("scores", {}) or {}
    meta = record.get("meta", {}) or {}
    rc = float(scores.get("score_route", 0.0))
    pen = float(scores.get("score_penalty", 0.0))
    status = str(record.get("status", ""))

    n_coll = _count(inf, _COLLISION_KEYS)
    n_stop = _count(inf, _STOP_KEYS)
    n_red = _count(inf, _RED_KEYS)
    n_blocked = _count(inf, _BLOCKED_KEYS)
    n_dev = _count(inf, _DEV_KEYS)
    n_timeout = _count(inf, _TIMEOUT_KEYS)
    completed = status.startswith("Completed") and rc >= 99.0

    if n_dev > 0:
        category = DELETE
    elif n_coll > 0:
        if n_stop > 0 and n_red > 0:
            category = "stop_red_then_collision"
        elif n_stop > 0:
            category = "stop_then_collision"
        elif n_red > 0:
            category = "red_then_collision"
        else:
            category = "collision_isolated"
    elif n_stop > 0 and n_red > 0:
        category = "stop_red_combo"
    elif n_stop > 0:
        category = "stop_isolated"
    elif n_red > 0:
        category = "red_isolated"
    elif n_blocked > 0:
        category = "blocked_isolated"
    elif n_timeout > 0:
        category = "timeout"
    elif completed:
        category = "success"
    else:
        category = "other"

    import re

    digits = re.findall(r"\d+", str(record.get("route_id", "")))
    short_id = digits[0] if digits else str(record.get("route_id", ""))

    return {
        "route_id": short_id,
        "town": str(record.get("town_name", "")),
        "category": category,
        "is_success": category == "success",
        "score_route": rc,
        "score_penalty": pen,
        "route_length": float(meta.get("route_length", 0.0) or 0.0),
        "duration_game": float(meta.get("duration_game", 0.0) or 0.0),
        "status": status,
        "n_collision": n_coll,
        "n_stop": n_stop,
        "n_red": n_red,
        "n_blocked": n_blocked,
        "n_timeout": n_timeout,
        "reached_intersection": rc >= _REACHED_RC or n_coll > 0,
    }
