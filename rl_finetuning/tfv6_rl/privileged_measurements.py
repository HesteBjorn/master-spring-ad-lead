from __future__ import annotations

from typing import Any


def privileged_measurement_names(rl_config: Any) -> list[str]:
    names = ["remaining_time", "time_till_blocked", "perc_route_left"]
    if bool(getattr(rl_config, "use_ttc", False)):
        names.append("remaining_ttc_penalty_ticks")
    if bool(getattr(rl_config, "use_comfort_infraction", False)):
        names.extend(
            [
                "comfort_acc_lon",
                "comfort_acc_lat",
                "comfort_jerk",
                "comfort_jerk_lon",
                "comfort_yaw_rate",
                "comfort_yaw_accel",
            ]
        )
    return names


def privileged_measurement_dim(rl_config: Any) -> int:
    return len(privileged_measurement_names(rl_config))


def validate_privileged_measurement_dim(rl_config: Any) -> None:
    expected = privileged_measurement_dim(rl_config)
    cfg_dim = getattr(rl_config, "num_value_measurements", None)
    if cfg_dim is None:
        return
    if int(cfg_dim) != int(expected):
        raise ValueError(
            "num_value_measurements does not match enabled privileged measurements: "
            f"config={cfg_dim}, expected={expected} "
            f"(use_ttc={getattr(rl_config, 'use_ttc', False)}, "
            f"use_comfort_infraction={getattr(rl_config, 'use_comfort_infraction', False)})"
        )
