from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from lead.training.config_training import TrainingConfig


@dataclass(frozen=True)
class ActionSlices:
    route: slice | None
    waypoints: slice | None
    target_speed: slice | None


class ActionCodec:
    """Encode/decode TFv6 planning outputs to a flat PPO action vector.

    The action vector is normalized to roughly [-1, 1] for stability.
    Layout (enabled heads only):
      - route checkpoints (num_route_points_prediction * 2)
      - waypoints (num_way_points_prediction * 2)
      - target_speed (1)
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.predict_route = bool(config.predict_spatial_path)
        self.predict_waypoints = bool(config.predict_temporal_spatial_waypoints)
        self.predict_target_speed = bool(config.predict_target_speed)

        self.num_route_points = (
            config.num_route_points_prediction if self.predict_route else 0
        )
        self.num_waypoints = (
            config.num_way_points_prediction if self.predict_waypoints else 0
        )

        self.route_dim = self.num_route_points * 2
        self.waypoints_dim = self.num_waypoints * 2
        self.speed_dim = 1 if self.predict_target_speed else 0

        self.action_dim = self.route_dim + self.waypoints_dim + self.speed_dim

        idx = 0
        route_slice = None
        if self.predict_route:
            route_slice = slice(idx, idx + self.route_dim)
            idx += self.route_dim
        waypoints_slice = None
        if self.predict_waypoints:
            waypoints_slice = slice(idx, idx + self.waypoints_dim)
            idx += self.waypoints_dim
        speed_slice = None
        if self.predict_target_speed:
            speed_slice = slice(idx, idx + 1)
            idx += 1

        self.slices = ActionSlices(
            route=route_slice, waypoints=waypoints_slice, target_speed=speed_slice
        )

        # Normalization scales
        self.route_scale = torch.tensor(
            [config.max_x_meter, config.max_y_meter], dtype=torch.float32
        )
        self.waypoint_scale = torch.tensor(
            [config.max_x_meter, config.max_y_meter], dtype=torch.float32
        )
        self.speed_scale = float(config.max_speed)

    def encode(
        self,
        route: torch.Tensor | None,
        waypoints: torch.Tensor | None,
        target_speed: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode planning outputs into a flat normalized action tensor.

        Args:
            route: (B, num_route_points, 2)
            waypoints: (B, num_waypoints, 2)
            target_speed: (B,) or (B,1)
        Returns:
            action: (B, action_dim) in roughly [-1, 1]
        """
        chunks = []
        if self.predict_route:
            if route is None:
                raise ValueError("route is required by config")
            route_norm = route / self.route_scale.to(route.device)
            chunks.append(route_norm.reshape(route.shape[0], -1))
        if self.predict_waypoints:
            if waypoints is None:
                raise ValueError("waypoints is required by config")
            wp_norm = waypoints / self.waypoint_scale.to(waypoints.device)
            chunks.append(wp_norm.reshape(waypoints.shape[0], -1))
        if self.predict_target_speed:
            if target_speed is None:
                raise ValueError("target_speed is required by config")
            if target_speed.ndim == 2:
                target_speed = target_speed.squeeze(-1)
            speed_norm = (target_speed / self.speed_scale).unsqueeze(-1)
            chunks.append(speed_norm)

        action = torch.cat(chunks, dim=1)
        return torch.clamp(action, -1.0, 1.0)

    def decode(
        self, action: np.ndarray | torch.Tensor
    ) -> tuple[
        np.ndarray | torch.Tensor, np.ndarray | torch.Tensor, np.ndarray | torch.Tensor
    ]:
        """Decode a flat normalized action vector into route, waypoints, speed.

        Args:
            action: (action_dim,) or (B, action_dim)
        Returns:
            route: (B, num_route_points, 2) or None
            waypoints: (B, num_waypoints, 2) or None
            target_speed: (B,) or None
        """
        if isinstance(action, np.ndarray):
            return self._decode_numpy(action)
        return self._decode_torch(action)

    def _decode_torch(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if action.ndim == 1:
            action = action.unsqueeze(0)
        route = None
        waypoints = None
        target_speed = None

        if self.slices.route is not None:
            route_flat = action[:, self.slices.route]
            route = route_flat.view(-1, self.num_route_points, 2)
            route = route * self.route_scale.to(route.device)

        if self.slices.waypoints is not None:
            wp_flat = action[:, self.slices.waypoints]
            waypoints = wp_flat.view(-1, self.num_waypoints, 2)
            waypoints = waypoints * self.waypoint_scale.to(waypoints.device)

        if self.slices.target_speed is not None:
            target_speed = action[:, self.slices.target_speed].squeeze(-1)
            target_speed = target_speed * self.speed_scale

        return route, waypoints, target_speed

    def _decode_numpy(
        self, action: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if action.ndim == 1:
            action = action[None, :]
        route = None
        waypoints = None
        target_speed = None

        if self.slices.route is not None:
            route_flat = action[:, self.slices.route]
            route = route_flat.reshape(-1, self.num_route_points, 2)
            route = route * self.route_scale.numpy()

        if self.slices.waypoints is not None:
            wp_flat = action[:, self.slices.waypoints]
            waypoints = wp_flat.reshape(-1, self.num_waypoints, 2)
            waypoints = waypoints * self.waypoint_scale.numpy()

        if self.slices.target_speed is not None:
            target_speed = action[:, self.slices.target_speed].squeeze(-1)
            target_speed = target_speed * self.speed_scale

        return route, waypoints, target_speed

    def decode_to_control_tensors(
        self, action: np.ndarray | torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Decode action into torch tensors suitable for closed-loop control.

        Returns:
            route: (1, num_route_points, 2)
            waypoints: (1, num_waypoints, 2)
            target_speed: (1, 1)
        """
        route, waypoints, target_speed = self.decode(action)
        if isinstance(route, np.ndarray):
            route_t = torch.from_numpy(route).to(device=device, dtype=torch.float32)
        else:
            route_t = route
        if isinstance(waypoints, np.ndarray):
            waypoints_t = torch.from_numpy(waypoints).to(
                device=device, dtype=torch.float32
            )
        else:
            waypoints_t = waypoints
        if target_speed is not None:
            if isinstance(target_speed, np.ndarray):
                speed_t = torch.from_numpy(target_speed).to(
                    device=device, dtype=torch.float32
                )
            else:
                speed_t = target_speed
            speed_t = speed_t.view(-1, 1)
        else:
            speed_t = None
        return route_t, waypoints_t, speed_t
