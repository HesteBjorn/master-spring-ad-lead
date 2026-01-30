from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lead.training.config_training import TrainingConfig


@dataclass(frozen=True)
class ObsSpec:
    key: str
    shape: tuple[int, ...]
    dtype: np.dtype


class ObsCodec:
    """Defines observation schema and pack/unpack helpers for TFv6 RL."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.specs: list[ObsSpec] = []

        # RGB image (C, H, W)
        self.specs.append(
            ObsSpec(
                key="rgb",
                shape=(3, config.final_image_height, config.final_image_width),
                dtype=np.uint8,
            )
        )
        # LiDAR BEV (C=1, H, W)
        self.specs.append(
            ObsSpec(
                key="rasterized_lidar",
                shape=(1, config.lidar_height_pixel, config.lidar_width_pixel),
                dtype=np.float32,
            )
        )

        if config.use_radars:
            radar_points = config.num_radar_sensors * config.num_radar_points_per_sensor
            self.specs.append(
                ObsSpec(
                    key="radar",
                    shape=(radar_points, 5),
                    dtype=np.float32,
                )
            )

        self.specs.extend(
            [
                ObsSpec("target_point_previous", (2,), np.float32),
                ObsSpec("target_point", (2,), np.float32),
                ObsSpec("target_point_next", (2,), np.float32),
                ObsSpec("speed", (1,), np.float32),
                ObsSpec("command", (6,), np.float32),
                ObsSpec("next_command", (6,), np.float32),
            ]
        )

        self._spec_map = {spec.key: spec for spec in self.specs}

    @property
    def keys(self) -> list[str]:
        return [spec.key for spec in self.specs]

    def pack(self, obs: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Return list of arrays in the fixed spec order."""
        packed: list[np.ndarray] = []
        for spec in self.specs:
            if spec.key not in obs:
                raise KeyError(f"Missing observation key: {spec.key}")
            arr = obs[spec.key]
            if arr.shape != spec.shape:
                raise ValueError(
                    f"Obs '{spec.key}' has shape {arr.shape}, expected {spec.shape}"
                )
            if arr.dtype != spec.dtype:
                arr = arr.astype(spec.dtype)
            packed.append(arr)
        return packed

    def unpack(self, buffers: list[memoryview | bytes]) -> dict[str, np.ndarray]:
        """Decode list of buffers into an observation dict."""
        if len(buffers) != len(self.specs):
            raise ValueError(f"Expected {len(self.specs)} buffers, got {len(buffers)}")
        obs: dict[str, np.ndarray] = {}
        for spec, buf in zip(self.specs, buffers, strict=False):
            arr = np.frombuffer(buf, dtype=spec.dtype).reshape(spec.shape)
            obs[spec.key] = arr
        return obs
