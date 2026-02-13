from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

from lead.common.constants import LIDAR_COLOR
from lead.common.pid_controller import (
    LateralPIDController,
    PIDController,
    get_throttle,
)
from lead.expert.config_expert import ExpertConfig
from lead.inference.config_closed_loop import ClosedLoopConfig
from lead.training.config_training import TrainingConfig
from lead.visualization import viz_utils
from rl_finetuning.tfv6_rl.action_codec import ActionCodec


@dataclass
class ControlOutput:
    steer: float
    throttle: float
    brake: float


class _ClosedLoopControlAdapter:
    """Mirror EnvAgentTFv6 control logic for visualization-only control traces."""

    def __init__(
        self, config_closed_loop: ClosedLoopConfig, config_expert: ExpertConfig
    ):
        self.config_closed_loop = config_closed_loop
        self.config_expert = config_expert
        self.lateral_waypoint_controller = PIDController(
            k_p=self.config_closed_loop.turn_kp,
            k_i=self.config_closed_loop.turn_ki,
            k_d=self.config_closed_loop.turn_kd,
            n=self.config_closed_loop.turn_n,
        )
        self.longitudinal_waypoint_controller = PIDController(
            k_p=self.config_closed_loop.speed_kp,
            k_i=self.config_closed_loop.speed_ki,
            k_d=self.config_closed_loop.speed_kd,
            n=self.config_closed_loop.speed_n,
        )
        self.lateral_route_controller = LateralPIDController(self.config_closed_loop)

    def _control_from_route_and_target_speed(
        self, route: np.ndarray | None, target_speed: float | None, speed: float
    ) -> ControlOutput:
        if route is None or target_speed is None:
            return ControlOutput(0.0, 0.0, 1.0)

        target_speed = float(target_speed)
        brake = bool(
            target_speed < 0.01
            or (speed / max(target_speed, 1e-6)) > self.config_closed_loop.brake_ratio
        )
        steer = self.lateral_route_controller.step(
            route,
            float(speed),
            0.0,
            0.0,
            sensor_agent_steer_correction=self.config_closed_loop.sensor_agent_steer_correction,
        )
        throttle, brake = get_throttle(
            brake, target_speed, float(speed), self.config_expert
        )
        return ControlOutput(float(steer), float(throttle), float(brake))

    def _control_from_waypoints(
        self, waypoints: np.ndarray | None, speed: float
    ) -> ControlOutput:
        if waypoints is None or waypoints.shape[0] == 0:
            return ControlOutput(0.0, 0.0, 1.0)

        one_second = 2
        half_second = 1
        if waypoints.shape[0] >= 4:
            one_second = min(waypoints.shape[0], 4)
            half_second = max(1, one_second // 2)

        desired_speed = (
            np.linalg.norm(waypoints[half_second - 1] - waypoints[one_second - 1]) * 2.0
        )
        delta_speed = np.clip(
            desired_speed - speed, 0.0, self.config_closed_loop.wp_delta_clip
        )

        brake = (desired_speed < self.config_closed_loop.brake_speed) or (
            (speed / max(desired_speed, 1e-6)) > self.config_closed_loop.brake_ratio
        )
        throttle = self.longitudinal_waypoint_controller.step(float(delta_speed))
        throttle = float(throttle if not brake else 0.0)

        if self.config_closed_loop.tuned_aim_distance:
            aim_distance = np.clip(0.975532 * speed + 1.915288, 24, 105) / 10
        elif desired_speed < self.config_closed_loop.aim_distance_threshold:
            aim_distance = self.config_closed_loop.aim_distance_slow
        else:
            aim_distance = self.config_closed_loop.aim_distance_fast

        aim_index = waypoints.shape[0] - 1
        for idx, pred_waypoint in enumerate(waypoints):
            if np.linalg.norm(pred_waypoint) >= aim_distance:
                aim_index = idx
                break

        aim = waypoints[aim_index]
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90.0
        if speed < 0.01 or brake:
            angle = 0.0

        steer = self.lateral_waypoint_controller.step(float(angle))
        steer = float(np.clip(steer, -1.0, 1.0))
        return ControlOutput(steer, throttle, float(brake))

    def infer_control(
        self,
        route: np.ndarray | None,
        waypoints: np.ndarray | None,
        target_speed: float | None,
        speed: float,
    ) -> ControlOutput:
        control = self._control_from_route_and_target_speed(route, target_speed, speed)
        wp_control = self._control_from_waypoints(waypoints, speed)
        if self.config_closed_loop.steer_modality == "waypoint":
            control.steer = wp_control.steer
        if self.config_closed_loop.throttle_modality == "waypoint":
            control.throttle = wp_control.throttle
        if self.config_closed_loop.brake_modality == "waypoint":
            control.brake = wp_control.brake
        if control.brake > 0.0:
            control.throttle = 0.0
            if speed < 0.01:
                control.steer = 0.0
        return control


class PPORolloutVisualizer:
    def __init__(
        self,
        training_config: TrainingConfig,
        action_codec: ActionCodec,
        output_dir: str,
        num_envs: int,
        gamma: float = 0.99,
        every_n: int = 1,
        max_images: int = 0,
        image_scale: int = 3,
    ) -> None:
        self.training_config = training_config
        self.action_codec = action_codec
        self.output_dir = output_dir
        self.gamma = float(gamma)
        self.every_n = max(1, int(every_n))
        self.max_images = max(0, int(max_images))
        self.image_scale = max(1, int(image_scale))
        os.makedirs(self.output_dir, exist_ok=True)
        self.images_written = 0
        self.episode_returns = [0.0 for _ in range(max(1, num_envs))]
        self.pending_forward_return_stamps: dict[
            int, list[tuple[int, int, str, int, int]]
        ] = {}
        self.negative_reward_burst_remaining = [0 for _ in range(max(1, num_envs))]
        self.negative_reward_burst_len_terminal = 2
        self.negative_reward_burst_len_non_terminal = 10
        self.random_burst_remaining = [0 for _ in range(max(1, num_envs))]
        self.random_burst_len = 10
        self.random_burst_probability = 0.003
        self.prev_values = [None for _ in range(max(1, num_envs))]
        self.value_burst_remaining = [0 for _ in range(max(1, num_envs))]
        self.value_burst_len = 10
        # Always-on value triggers when debug viz is enabled.
        self.value_low_threshold = 0.0
        self.value_drop_threshold = 0.25

        self.config_closed_loop = ClosedLoopConfig(raise_error_on_missing_key=False)
        self.config_expert = ExpertConfig()
        self.controllers = [
            _ClosedLoopControlAdapter(self.config_closed_loop, self.config_expert)
            for _ in range(max(1, num_envs))
        ]
        self.steer_modality = self.config_closed_loop.steer_modality
        self.throttle_modality = self.config_closed_loop.throttle_modality
        self.brake_modality = self.config_closed_loop.brake_modality
        self.target_speed_active_for_control = (
            self.throttle_modality == "target_speed"
            or self.brake_modality == "target_speed"
        )
        self.waypoints_active_for_control = (
            self.steer_modality == "waypoint"
            or self.throttle_modality == "waypoint"
            or self.brake_modality == "waypoint"
        )

        self.size_width = int(
            (self.training_config.max_y_meter - self.training_config.min_y_meter)
            * self.training_config.pixels_per_meter
        )
        self.size_height = int(
            (self.training_config.max_x_meter - self.training_config.min_x_meter)
            * self.training_config.pixels_per_meter
        )
        self.ppm = self.training_config.pixels_per_meter * self.image_scale
        self.bev_h = self.size_width * self.image_scale
        self.bev_w = self.size_height * self.image_scale
        # Keep exactly the same frame convention as rasterize_lidar:
        # image row maps to y-bin, image column maps to x-bin.
        self.origin_x = int(-self.training_config.min_x_meter * self.ppm)
        self.origin_y = int(-self.training_config.min_y_meter * self.ppm)

    def _world_to_bev(self, x: float, y: float) -> tuple[int, int]:
        px = int(round(self.origin_x + x * self.ppm))
        py = int(round(self.origin_y + y * self.ppm))
        return px, py

    def _draw_polyline(
        self,
        bev: np.ndarray,
        points_xy: np.ndarray | None,
        color: tuple[int, int, int],
        radius: int,
        line_thickness: int,
    ) -> None:
        if points_xy is None or points_xy.shape[0] == 0:
            return
        prev = None
        for idx, pt in enumerate(points_xy):
            x = float(pt[0])
            y = float(pt[1])
            px, py = self._world_to_bev(x, y)
            shade = viz_utils.lighter_shade(
                color, idx, points_xy.shape[0], max_lighter=70
            )
            cv2.circle(bev, (px, py), radius=radius, color=shade, thickness=-1)
            if prev is not None:
                cv2.line(
                    bev, prev, (px, py), shade, line_thickness, lineType=cv2.LINE_AA
                )
            prev = (px, py)

    def _draw_target_marker(
        self,
        bev_img: np.ndarray,
        point_xy: np.ndarray,
        color: tuple[int, int, int],
        radius: int,
        label: str,
    ) -> tuple[bool, tuple[int, int]]:
        px, py = self._world_to_bev(float(point_xy[0]), float(point_xy[1]))
        h, w = bev_img.shape[:2]
        in_view = 0 <= px < w and 0 <= py < h
        if in_view:
            cv2.circle(bev_img, (px, py), radius, color, -1)
            cv2.putText(
                bev_img,
                label,
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                lineType=cv2.LINE_AA,
            )
            return True, (px, py)

        cx = int(np.clip(px, 0, w - 1))
        cy = int(np.clip(py, 0, h - 1))
        cv2.circle(bev_img, (cx, cy), radius + 1, color, 2)
        cv2.putText(
            bev_img,
            f"{label}(OOB)",
            (max(2, cx + 8), max(14, cy - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            lineType=cv2.LINE_AA,
        )
        return False, (cx, cy)

    def _build_bev(
        self,
        lidar: np.ndarray,
        target_point: np.ndarray,
        target_point_next: np.ndarray,
        route_sample: np.ndarray | None,
        route_mean: np.ndarray | None,
        waypoints_sample: np.ndarray | None,
        waypoints_mean: np.ndarray | None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        bev = lidar.astype(np.float32)
        if bev.ndim == 3:
            bev = bev[0]
        bev = bev / (bev.max() + 1e-6)

        start_color = np.array([255, 255, 255], dtype=np.float32)
        end_color = np.array(LIDAR_COLOR, dtype=np.float32)
        bev_img = np.zeros((*bev.shape, 3), dtype=np.float32)
        for c in range(3):
            bev_img[..., c] = start_color[c] + (end_color[c] - start_color[c]) * bev
        bev_img = bev_img.astype(np.uint8)

        bev_img = cv2.resize(
            bev_img,
            dsize=(self.bev_w, self.bev_h),
            interpolation=cv2.INTER_NEAREST,
        )

        cv2.circle(
            bev_img,
            (self.origin_x, self.origin_y),
            radius=8,
            color=(20, 20, 20),
            thickness=-1,
        )
        cv2.line(
            bev_img,
            (self.origin_x, self.origin_y),
            (self.origin_x + int(self.ppm * 2.0), self.origin_y),
            (20, 20, 20),
            2,
            lineType=cv2.LINE_AA,
        )
        ego_w = int(round(self.training_config.ego_extent_x * self.ppm))
        ego_h = int(round(self.training_config.ego_extent_y * self.ppm))
        cv2.rectangle(
            bev_img,
            (self.origin_x - ego_w, self.origin_y - ego_h),
            (self.origin_x + ego_w, self.origin_y + ego_h),
            (255, 0, 0),
            3,
            lineType=cv2.LINE_AA,
        )

        tp_in_view, tp_drawn = self._draw_target_marker(
            bev_img, target_point, color=(10, 10, 220), radius=7, label="TP"
        )
        tpn_in_view, tpn_drawn = self._draw_target_marker(
            bev_img, target_point_next, color=(30, 170, 240), radius=6, label="TP+1"
        )

        if self.steer_modality == "route":
            self._draw_polyline(
                bev_img, route_mean, color=(230, 110, 0), radius=6, line_thickness=2
            )
            self._draw_polyline(
                bev_img, route_sample, color=(20, 80, 230), radius=5, line_thickness=2
            )
        if self.waypoints_active_for_control:
            self._draw_polyline(
                bev_img,
                waypoints_mean,
                color=(170, 100, 200),
                radius=4,
                line_thickness=2,
            )
            self._draw_polyline(
                bev_img,
                waypoints_sample,
                color=(50, 190, 60),
                radius=4,
                line_thickness=2,
            )

        self._overlay_legend_on_bev(bev_img)

        return bev_img, {
            "tp_in_view": tp_in_view,
            "tp_next_in_view": tpn_in_view,
            "tp_drawn_px": tp_drawn,
            "tp_next_drawn_px": tpn_drawn,
        }

    def _overlay_legend_on_bev(self, bev: np.ndarray) -> None:
        legend_items = []
        if self.steer_modality == "route":
            legend_items.extend(
                [
                    ("Route mean (TFv6)", (230, 110, 0)),
                    ("Route sampled (PPO)", (20, 80, 230)),
                ]
            )
        if self.waypoints_active_for_control:
            legend_items.extend(
                [
                    ("Waypoints mean", (170, 100, 200)),
                    ("Waypoints sampled", (50, 190, 60)),
                ]
            )

        row_h = 32
        swatch = 16
        font_scale = 0.68
        x_right = bev.shape[1] - 8
        y_top = 10

        max_text_w = 0
        for name, _ in legend_items:
            size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            max_text_w = max(max_text_w, size[0])
        panel_w = swatch + 6 + max_text_w + 10
        panel_h = row_h * len(legend_items) + 8
        x0 = max(0, x_right - panel_w)
        y0 = y_top
        cv2.rectangle(bev, (x0, y0), (x0 + panel_w, y0 + panel_h), (245, 245, 245), -1)
        cv2.rectangle(bev, (x0, y0), (x0 + panel_w, y0 + panel_h), (190, 190, 190), 1)

        y = y0 + 22
        for name, color in legend_items:
            cv2.rectangle(
                bev, (x0 + 6, y - swatch + 2), (x0 + 6 + swatch, y + 2), color, -1
            )
            cv2.putText(
                bev,
                name,
                (x0 + 6 + swatch + 5, y + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (30, 30, 30),
                1,
                lineType=cv2.LINE_AA,
            )
            y += row_h

    def _draw_speed_distribution(
        self,
        panel: np.ndarray,
        *,
        x0: int,
        y0: int,
        w: int,
        h: int,
        mean_speed: float,
        std_speed: float,
        selected_speed: float,
        current_speed: float,
    ) -> None:
        if w < 120 or h < 120:
            return
        cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (210, 210, 210), 1)
        cv2.putText(
            panel,
            "Target-Speed Distribution",
            (x0 + 8, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            1,
            lineType=cv2.LINE_AA,
        )

        # Keep plot high in the corner and reserve a larger bottom margin for ticks/x-label.
        plot_left = x0 + 12
        plot_right = x0 + w - 10
        plot_top = y0 + 64
        plot_bottom = y0 + h - 44
        cv2.rectangle(
            panel, (plot_left, plot_top), (plot_right, plot_bottom), (235, 235, 235), -1
        )
        cv2.rectangle(
            panel, (plot_left, plot_top), (plot_right, plot_bottom), (180, 180, 180), 1
        )

        max_speed = max(
            float(self.training_config.max_speed), mean_speed + 4.0 * std_speed
        )
        max_speed = max(max_speed, selected_speed + 2.0, current_speed + 2.0, 1.0)
        xs = np.linspace(0.0, max_speed, 220)
        sigma = max(1e-3, std_speed)
        ys = np.exp(-0.5 * ((xs - mean_speed) / sigma) ** 2)
        ys = ys / (ys.max() + 1e-8)

        def speed_to_px(v: float) -> int:
            alpha = float(np.clip(v / max_speed, 0.0, 1.0))
            return int(round(plot_left + alpha * (plot_right - plot_left)))

        def dens_to_py(v: float) -> int:
            alpha = float(np.clip(v, 0.0, 1.0))
            return int(round(plot_bottom - alpha * (plot_bottom - plot_top)))

        pts = np.stack(
            [
                np.array([speed_to_px(x), dens_to_py(y)])
                for x, y in zip(xs, ys, strict=False)
            ]
        )
        cv2.polylines(
            panel,
            [pts.astype(np.int32)],
            isClosed=False,
            color=(30, 120, 220),
            thickness=2,
        )

        mean_x = speed_to_px(mean_speed)
        selected_x = speed_to_px(selected_speed)
        current_x = speed_to_px(current_speed)
        cv2.line(
            panel,
            (mean_x, plot_top),
            (mean_x, plot_bottom),
            (240, 130, 0),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            panel,
            (selected_x, plot_top),
            (selected_x, plot_bottom),
            (30, 30, 210),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            panel,
            (current_x, plot_top),
            (current_x, plot_bottom),
            (40, 150, 60),
            1,
            lineType=cv2.LINE_AA,
        )

        # Compact summary at the top so it does not collide with x-axis text.
        cv2.putText(
            panel,
            f"mean={mean_speed:.2f}  selected={selected_speed:.2f}  ego={current_speed:.2f}",
            (x0 + 8, y0 + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (30, 30, 30),
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"std={std_speed:.2f}",
            (x0 + 8, y0 + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (30, 30, 30),
            1,
            lineType=cv2.LINE_AA,
        )

        low_x = speed_to_px(max(0.0, mean_speed - std_speed))
        high_x = speed_to_px(mean_speed + std_speed)
        cv2.line(
            panel,
            (low_x, plot_top),
            (low_x, plot_bottom),
            (180, 180, 60),
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            panel,
            (high_x, plot_top),
            (high_x, plot_bottom),
            (180, 180, 60),
            1,
            lineType=cv2.LINE_AA,
        )

        ticks = np.linspace(0.0, max_speed, 5)
        for t in ticks:
            tx = speed_to_px(float(t))
            cv2.line(panel, (tx, plot_bottom), (tx, plot_bottom + 4), (80, 80, 80), 1)
            cv2.putText(
                panel,
                f"{t:.1f}",
                (tx - 10, plot_bottom + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (40, 40, 40),
                1,
                lineType=cv2.LINE_AA,
            )
        cv2.putText(
            panel,
            "speed [m/s]",
            (plot_left + max(30, (plot_right - plot_left) // 3), plot_bottom + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (40, 40, 40),
            1,
            lineType=cv2.LINE_AA,
        )

    def _draw_spatial_std_profile(
        self,
        panel: np.ndarray,
        *,
        x0: int,
        y0: int,
        w: int,
        h: int,
        route_point_std: np.ndarray | None,
        waypoint_point_std: np.ndarray | None,
    ) -> None:
        if route_point_std is None and waypoint_point_std is None:
            return

        if w < 180 or h < 130:
            return

        cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (210, 210, 210), 1)
        cv2.putText(
            panel,
            "Spatial Std Profile",
            (x0 + 8, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            1,
            lineType=cv2.LINE_AA,
        )

        plot_left = x0 + 12
        plot_right = x0 + w - 10
        plot_top = y0 + 46
        plot_bottom = y0 + h - 34
        cv2.rectangle(
            panel, (plot_left, plot_top), (plot_right, plot_bottom), (235, 235, 235), -1
        )
        cv2.rectangle(
            panel, (plot_left, plot_top), (plot_right, plot_bottom), (180, 180, 180), 1
        )

        max_points = 0
        if route_point_std is not None:
            max_points = max(max_points, int(route_point_std.shape[0]))
        if waypoint_point_std is not None:
            max_points = max(max_points, int(waypoint_point_std.shape[0]))
        max_points = max(max_points, 1)

        max_std = 0.0
        if route_point_std is not None and route_point_std.size > 0:
            max_std = max(max_std, float(route_point_std.max()))
        if waypoint_point_std is not None and waypoint_point_std.size > 0:
            max_std = max(max_std, float(waypoint_point_std.max()))
        max_std = max(max_std * 1.1, 1e-3)

        def idx_to_px(i: int, n: int) -> int:
            if n <= 1:
                return int(round((plot_left + plot_right) / 2))
            alpha = float(i) / float(n - 1)
            return int(round(plot_left + alpha * (plot_right - plot_left)))

        def std_to_py(v: float) -> int:
            alpha = float(np.clip(v / max_std, 0.0, 1.0))
            return int(round(plot_bottom - alpha * (plot_bottom - plot_top)))

        def draw_series(values: np.ndarray | None, color: tuple[int, int, int]) -> None:
            if values is None or values.size == 0:
                return
            pts = np.array(
                [
                    [idx_to_px(i, values.shape[0]), std_to_py(float(v))]
                    for i, v in enumerate(values)
                ],
                dtype=np.int32,
            )
            cv2.polylines(
                panel,
                [pts],
                isClosed=False,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            for p in pts:
                cv2.circle(panel, tuple(p), 2, color, -1, lineType=cv2.LINE_AA)

        draw_series(route_point_std, (20, 80, 230))
        draw_series(waypoint_point_std, (50, 190, 60))

        cv2.putText(
            panel,
            f"max std={max_std:.3f}",
            (plot_left, y0 + h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (40, 40, 40),
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "x: point index [-]",
            (plot_left + max(70, (plot_right - plot_left) // 2 - 20), y0 + h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (40, 40, 40),
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "y: std",
            (x0 + 8, y0 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (40, 40, 40),
            1,
            lineType=cv2.LINE_AA,
        )

        # x-axis tick marks (point index)
        x_tick_count = min(5, max_points)
        if x_tick_count >= 2:
            for i in range(x_tick_count):
                idx = int(round(i * (max_points - 1) / (x_tick_count - 1)))
                tx = idx_to_px(idx, max_points)
                cv2.line(
                    panel,
                    (tx, plot_bottom),
                    (tx, plot_bottom + 4),
                    (80, 80, 80),
                    1,
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    panel,
                    f"{idx}",
                    (tx - 8, plot_bottom + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.33,
                    (40, 40, 40),
                    1,
                    lineType=cv2.LINE_AA,
                )

        # y-axis tick marks (std value)
        y_ticks = np.linspace(0.0, max_std, 4)
        for yv in y_ticks:
            ty = std_to_py(float(yv))
            cv2.line(
                panel,
                (plot_left - 4, ty),
                (plot_left, ty),
                (80, 80, 80),
                1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                f"{yv:.2f}",
                (plot_left - 40, ty + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.31,
                (40, 40, 40),
                1,
                lineType=cv2.LINE_AA,
            )

        legend_x = x0 + 122
        legend_y = y0 + 36
        if route_point_std is not None and route_point_std.size > 0:
            cv2.rectangle(
                panel,
                (legend_x, legend_y - 8),
                (legend_x + 12, legend_y + 4),
                (20, 80, 230),
                -1,
            )
            cv2.putText(
                panel,
                "route",
                (legend_x + 16, legend_y + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (30, 30, 30),
                1,
                lineType=cv2.LINE_AA,
            )
            legend_x += 62
        if waypoint_point_std is not None and waypoint_point_std.size > 0:
            cv2.rectangle(
                panel,
                (legend_x, legend_y - 8),
                (legend_x + 12, legend_y + 4),
                (50, 190, 60),
                -1,
            )
            cv2.putText(
                panel,
                "waypoints",
                (legend_x + 16, legend_y + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (30, 30, 30),
                1,
                lineType=cv2.LINE_AA,
            )

    def _prepare_bev_for_column(
        self, bev: np.ndarray, target_h: int, target_w: int
    ) -> np.ndarray:
        # Rotate so ego heading points up in the rendered column.
        bev = np.rot90(bev, k=1)
        bev = np.ascontiguousarray(bev)
        # Crop small left/right margins to save width.
        crop = int(0.08 * bev.shape[1])
        if 2 * crop < bev.shape[1] - 20:
            bev = bev[:, crop:-crop]
        return cv2.resize(bev, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    def _put_text_lines(
        self, panel: np.ndarray, lines: list[str], start_y: int = 210
    ) -> None:
        y = start_y
        for line in lines:
            cv2.putText(
                panel,
                line,
                (14, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 20, 20),
                1,
                lineType=cv2.LINE_AA,
            )
            y += 24

    def maybe_write(
        self,
        *,
        global_step: int,
        rollout_step: int,
        update_idx: int,
        env_idx: int,
        obs: dict[str, np.ndarray],
        reward: float,
        done: bool,
        truncated: bool,
        sampled_action: np.ndarray,
        mean_action: np.ndarray,
        log_std: np.ndarray,
        value_estimate: float,
    ) -> None:
        env_slot = env_idx % len(self.episode_returns)
        if reward < 0.0:
            is_terminal_transition = bool(done) or bool(truncated)
            burst_len = (
                self.negative_reward_burst_len_terminal
                if is_terminal_transition
                else self.negative_reward_burst_len_non_terminal
            )
            self.negative_reward_burst_remaining[env_slot] = max(
                self.negative_reward_burst_remaining[env_slot],
                burst_len,
            )
        if self.random_burst_remaining[env_slot] <= 0:
            if np.random.random() < self.random_burst_probability:
                self.random_burst_remaining[env_slot] = self.random_burst_len

        prev_value = self.prev_values[env_slot]
        value_is_low = value_estimate < self.value_low_threshold
        value_dropped = (
            prev_value is not None
            and (prev_value - value_estimate) > self.value_drop_threshold
        )
        if value_is_low or value_dropped:
            self.value_burst_remaining[env_slot] = max(
                self.value_burst_remaining[env_slot],
                self.value_burst_len,
            )

        regular_cadence_due = (global_step % self.every_n) == 0
        burst_due = self.negative_reward_burst_remaining[env_slot] > 0
        value_burst_due = self.value_burst_remaining[env_slot] > 0
        random_burst_due = self.random_burst_remaining[env_slot] > 0
        if (
            not regular_cadence_due
            and not burst_due
            and not value_burst_due
            and not random_burst_due
        ):
            self.prev_values[env_slot] = value_estimate
            return
        if self.max_images > 0 and self.images_written >= self.max_images:
            self.prev_values[env_slot] = value_estimate
            return

        speed = float(obs["speed"][0])
        route_sample, waypoints_sample, target_speed_sample = self.action_codec.decode(
            sampled_action
        )
        route_mean, waypoints_mean, target_speed_mean = self.action_codec.decode(
            mean_action
        )
        route_sample = None if route_sample is None else route_sample[0]
        route_mean = None if route_mean is None else route_mean[0]
        waypoints_sample = None if waypoints_sample is None else waypoints_sample[0]
        waypoints_mean = None if waypoints_mean is None else waypoints_mean[0]
        target_speed_sample = (
            None if target_speed_sample is None else float(target_speed_sample[0])
        )
        target_speed_mean = (
            None if target_speed_mean is None else float(target_speed_mean[0])
        )

        rgb = np.transpose(obs["rgb"], (1, 2, 0)).astype(np.uint8)
        bev, target_debug = self._build_bev(
            lidar=obs["rasterized_lidar"],
            target_point=obs["target_point"],
            target_point_next=obs["target_point_next"],
            route_sample=route_sample,
            route_mean=route_mean,
            waypoints_sample=waypoints_sample,
            waypoints_mean=waypoints_mean,
        )
        rgb_h, rgb_w = rgb.shape[:2]
        # Keep legacy overall resolution basis, then widen total canvas by 50%.
        legacy_bev_w = int(round(rgb_h * (self.bev_w / max(1, self.bev_h))))
        base_final_w = legacy_bev_w + 620
        final_w = int(round(base_final_w * 1.5))
        legacy_cam_h = int(round(rgb_h * (base_final_w / max(1, rgb_w))))
        final_h = rgb_h + 8 + legacy_cam_h

        # Keep BEV column size unchanged by widening.
        left_w = int(round(final_h * (2.0 / 3.0)))
        left_w = min(left_w, int(base_final_w * 0.46))
        left_w = max(left_w, 220)
        gap = 8
        right_x = left_w + gap
        right_w = final_w - right_x
        cam_h = int(round(rgb_h * (right_w / max(1, rgb_w))))
        cam_h = min(cam_h, int(final_h * 0.52))
        cam_h = max(cam_h, 120)
        top_h = final_h - gap - cam_h
        top_h = max(top_h, 240)
        cam_h = final_h - gap - top_h

        bev_col = self._prepare_bev_for_column(bev, target_h=final_h, target_w=left_w)
        self._overlay_legend_on_bev(bev_col)

        scalar_w = int(round(right_w * 0.56))
        graph_w = right_w - gap - scalar_w
        scalar_panel = np.full((top_h, scalar_w, 3), 248, dtype=np.uint8)
        graphs_panel = np.full((top_h, graph_w, 3), 248, dtype=np.uint8)

        std = np.exp(log_std.astype(np.float32))
        target_speed_std = None
        route_point_std = None
        waypoint_point_std = None

        if self.action_codec.slices.route is not None:
            route_slice = self.action_codec.slices.route
            route_std_flat = std[route_slice]
            route_std_xy = route_std_flat.reshape(-1, 2)
            route_point_std = np.sqrt(np.mean(np.square(route_std_xy), axis=1))

        if self.action_codec.slices.waypoints is not None:
            wp_slice = self.action_codec.slices.waypoints
            wp_std_flat = std[wp_slice]
            wp_std_xy = wp_std_flat.reshape(-1, 2)
            waypoint_point_std = np.sqrt(np.mean(np.square(wp_std_xy), axis=1))

        if self.action_codec.slices.target_speed is not None:
            speed_idx = self.action_codec.slices.target_speed.start
            target_speed_std = float(std[speed_idx] * self.action_codec.speed_scale)

        self.episode_returns[env_slot] += reward
        episode_return = float(self.episode_returns[env_slot])

        text_lines = [
            f"update={update_idx} step={rollout_step}",
            (
                f"reward={reward:+.4f} cumulative_reward={episode_return:+.4f} "
                f"value={value_estimate:+.4f}"
            ),
            "forward_discounted_return=pending",
            f"std[min/mean/max]={std.min():.4f}/{std.mean():.4f}/{std.max():.4f}",
            (
                "target_point "
                f"xy=({float(obs['target_point'][0]):+.2f},{float(obs['target_point'][1]):+.2f}) "
                f"in_view={int(target_debug['tp_in_view'])}"
            ),
            (
                "target_point_next "
                f"xy=({float(obs['target_point_next'][0]):+.2f},{float(obs['target_point_next'][1]):+.2f}) "
                f"in_view={int(target_debug['tp_next_in_view'])}"
            ),
        ]
        text_start_y = 244
        line_h = 24
        if text_start_y + line_h * len(text_lines) > top_h - 10:
            text_start_y = max(24, top_h - 10 - line_h * len(text_lines))
        self._put_text_lines(scalar_panel, text_lines, start_y=text_start_y)

        if (
            self.target_speed_active_for_control
            and target_speed_mean is not None
            and target_speed_sample is not None
            and target_speed_std is not None
        ):
            graph_top_h = int(round(top_h * 0.54))
            graph_bottom_h = top_h - gap - graph_top_h
            self._draw_speed_distribution(
                graphs_panel,
                x0=0,
                y0=0,
                w=graph_w,
                h=graph_top_h,
                mean_speed=target_speed_mean,
                std_speed=target_speed_std,
                selected_speed=target_speed_sample,
                current_speed=speed,
            )
            self._draw_spatial_std_profile(
                graphs_panel,
                x0=0,
                y0=graph_top_h + gap,
                w=graph_w,
                h=graph_bottom_h,
                route_point_std=route_point_std,
                waypoint_point_std=waypoint_point_std,
            )
        else:
            self._draw_spatial_std_profile(
                graphs_panel,
                x0=0,
                y0=0,
                w=graph_w,
                h=top_h,
                route_point_std=route_point_std,
                waypoint_point_std=waypoint_point_std,
            )

        if done or truncated:
            self.episode_returns[env_slot] = 0.0

        right_top = np.full((top_h, right_w, 3), 248, dtype=np.uint8)
        right_top[:, :scalar_w] = scalar_panel
        right_top[:, scalar_w + gap : scalar_w + gap + graph_w] = graphs_panel

        rgb_resized = cv2.resize(rgb, (right_w, cam_h), interpolation=cv2.INTER_LINEAR)
        cv2.putText(
            rgb_resized,
            "CAMERA SENSOR",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

        right_col = np.full((final_h, right_w, 3), 235, dtype=np.uint8)
        right_col[:top_h] = right_top
        right_col[top_h + gap : top_h + gap + cam_h] = rgb_resized

        grid = np.full((final_h, final_w, 3), 235, dtype=np.uint8)
        grid[:, :left_w] = bev_col
        grid[:, right_x : right_x + right_w] = right_col
        cv2.putText(
            grid,
            "BEV + POLICY DEBUG",
            (12, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (15, 15, 15),
            1,
            lineType=cv2.LINE_AA,
        )

        filename = f"u{update_idx:06d}_gs{global_step:012d}_step{rollout_step:04d}_env{env_idx}.jpg"
        os.makedirs(self.output_dir, exist_ok=True)
        image_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(
            image_path,
            grid,
            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        )
        self.images_written += 1

        stamp_x = right_x + 12
        stamp_y = text_start_y + 2 * line_h
        self.pending_forward_return_stamps.setdefault(update_idx, []).append(
            (rollout_step, env_idx, image_path, stamp_x, stamp_y)
        )

        if self.negative_reward_burst_remaining[env_slot] > 0:
            self.negative_reward_burst_remaining[env_slot] -= 1
        if self.value_burst_remaining[env_slot] > 0:
            self.value_burst_remaining[env_slot] -= 1
        if self.random_burst_remaining[env_slot] > 0:
            self.random_burst_remaining[env_slot] -= 1
        self.prev_values[env_slot] = value_estimate
        return

    def stamp_forward_returns(
        self, update_idx: int, forward_returns: np.ndarray
    ) -> None:
        records = self.pending_forward_return_stamps.pop(update_idx, [])
        if not records or forward_returns.ndim != 2:
            return

        num_steps, num_envs = forward_returns.shape
        for rollout_step, env_idx, image_path, stamp_x, stamp_y in records:
            os.makedirs(self.output_dir, exist_ok=True)
            if rollout_step >= num_steps or env_idx >= num_envs:
                continue
            if not os.path.isfile(image_path):
                continue
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                continue

            forward_return = float(forward_returns[rollout_step, env_idx])
            y_top, y_bottom = stamp_y - 16, stamp_y + 8
            cv2.rectangle(
                image,
                (stamp_x - 2, y_top),
                (stamp_x + 560, y_bottom),
                (248, 248, 248),
                -1,
            )
            cv2.putText(
                image,
                f"forward_discounted_return={forward_return:+.4f}",
                (stamp_x, stamp_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 20, 20),
                1,
                lineType=cv2.LINE_AA,
            )
            cv2.imwrite(image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
