from __future__ import annotations

import sys
from pathlib import Path


def ensure_carl_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    team_code = repo_root / "3rd_party" / "CaRL" / "CARLA" / "team_code"
    leaderboard_custom = (
        repo_root
        / "3rd_party"
        / "CaRL"
        / "CARLA"
        / "custom_leaderboard"
        / "leaderboard"
    )

    for path in (team_code, leaderboard_custom, repo_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.append(path_str)
