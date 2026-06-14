"""Optionally shorten the Bench2Drive route (game-time) timeout for video sweeps.

The route timeout lives in scenario_runner's RouteTimeoutBehavior: it starts at
MIN_TIMEOUT=300 s and grows with route progress, so a stuck/hesitating agent runs
~300 s before timing out — producing huge clips and slow sweeps. Setting the
environment variable ROUTE_TIMEOUT_S=<seconds> caps it (e.g. 100), which makes
timeout routes end promptly and keeps their clips bounded.

This is a runtime monkeypatch (no edits to the 3rd_party submodule). It is applied
from td3_sensor_agent at import time, which the leaderboard does once at startup
(importlib.import_module) before any route scenario is built, so every route uses
the shortened timeout. No-op unless ROUTE_TIMEOUT_S is set.
"""

from __future__ import annotations

import os


def apply() -> None:
    secs = os.environ.get("ROUTE_TIMEOUT_S")
    if not secs:
        return
    try:
        value = float(secs)
    except ValueError:
        return
    try:
        # Imported lazily: scenario_runner is only on the path during leaderboard
        # runs, not e.g. during checkpoint extraction.
        from srunner.scenariomanager.timer import RouteTimeoutBehavior
    except Exception as exc:  # noqa: BLE001
        print(
            f"[route_timeout_patch] scenario_runner unavailable, not patching: {exc}",
            flush=True,
        )
        return

    RouteTimeoutBehavior.MIN_TIMEOUT = value
    # Neutralize the progress-based extension so the cap is ~fixed at MIN_TIMEOUT:
    # timeout_speed = speed_limit * PERC/100; a huge PERC makes the per-metre time
    # addition negligible.
    RouteTimeoutBehavior.TIMEOUT_ROUTE_PERC = 1.0e9
    print(
        f"[route_timeout_patch] route timeout capped at {value:.0f}s game time",
        flush=True,
    )
