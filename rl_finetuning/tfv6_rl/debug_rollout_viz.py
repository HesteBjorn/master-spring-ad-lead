from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

from lead.common.constants import LIDAR_COLOR, TP_DEFAULT_COLOR
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


@dataclass
class _PendingForwardReturnStamp:
    rollout_step: int
    env_idx: int
    image_path: str
    stamp_x: int
    stamp_y: int


@dataclass
class _PendingEpisodeInfractionStamp:
    image_path: str
    stamp_x: int
    stamp_y: int


@dataclass
class _ScheduledBurstState:
    remaining: int = 0
    last_start_global_step: int | None = None


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
        scheduled_burst_len: int = 1000,
        max_images: int = 0,
        image_scale: int = 3,
        standstill_speed_hold_enabled: bool = False,
        standstill_speed_hold_frames: int = 1,
        standstill_speed_hold_ego_speed_threshold: float = 0.1,
        standstill_speed_hold_target_speed_threshold: float = 1.0 / 3.6,
        speed_temperature: float = 1.0,
        use_residual_policy: bool = False,
    ) -> None:
        self.training_config = training_config
        self.action_codec = action_codec
        self.speed_temperature = float(speed_temperature)
        self.output_dir = output_dir
        self.gamma = float(gamma)
        self.every_n = max(1, int(every_n))
        self.scheduled_burst_len = max(1, int(scheduled_burst_len))
        self.max_images = max(0, int(max_images))
        self.image_scale = max(1, int(image_scale))
        self.standstill_speed_hold_enabled = bool(standstill_speed_hold_enabled)
        self.standstill_speed_hold_frames = max(1, int(standstill_speed_hold_frames))
        self.standstill_speed_hold_ego_speed_threshold = float(
            standstill_speed_hold_ego_speed_threshold
        )
        self.standstill_speed_hold_target_speed_threshold = float(
            standstill_speed_hold_target_speed_threshold
        )
        self.use_residual_policy = bool(use_residual_policy)
        os.makedirs(self.output_dir, exist_ok=True)
        self.images_written = 0
        self.episode_returns = [0.0 for _ in range(max(1, num_envs))]
        self.pending_forward_return_stamps: dict[
            int, list[_PendingForwardReturnStamp]
        ] = {}
        self.pending_episode_infraction_stamps: list[
            list[_PendingEpisodeInfractionStamp]
        ] = [[] for _ in range(max(1, num_envs))]
        self.scheduled_burst_states = [
            _ScheduledBurstState() for _ in range(max(1, num_envs))
        ]
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
        # Match Visualizer origin convention exactly.
        x_extent_m = self.training_config.max_x_meter - self.training_config.min_x_meter
        self.origin_x = int(
            self.bev_w // (x_extent_m / max((-self.training_config.min_x_meter), 1.0))
        )
        self.origin_y = self.bev_h // 2

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

    def _draw_numbered_target_marker(
        self,
        bev_img: np.ndarray,
        center_xy: tuple[int, int],
        color: tuple[int, int, int],
        radius: int,
        number: int,
    ) -> None:
        """Draw a filled target circle with a centered numeric label."""
        x, y = int(center_xy[0]), int(center_xy[1])
        cv2.circle(bev_img, (x, y), radius, color, thickness=-1)
        text = str(number)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.58 if number < 10 else 0.5
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        tx = x - tw // 2
        ty = y + th // 2
        cv2.putText(
            bev_img,
            text,
            (tx, ty),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )

    def _build_bev(
        self,
        lidar: np.ndarray,
        target_point_previous: np.ndarray,
        target_point: np.ndarray,
        target_point_next: np.ndarray,
        route_sample: np.ndarray | None,
        route_mean: np.ndarray | None,
        waypoints_sample: np.ndarray | None,
        waypoints_mean: np.ndarray | None,
        route_base: np.ndarray | None = None,
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

        if self.steer_modality == "route":
            if self.use_residual_policy:
                # Residual mode: blue = base TFv6, green = base+residual mean, orange = sampled.
                self._draw_polyline(
                    bev_img, route_base, color=(20, 80, 230), radius=5, line_thickness=2
                )
                self._draw_polyline(
                    bev_img, route_mean, color=(50, 190, 60), radius=6, line_thickness=2
                )
                self._draw_polyline(
                    bev_img,
                    route_sample,
                    color=(230, 110, 0),
                    radius=5,
                    line_thickness=2,
                )
            else:
                # Normal mode: blue = TFv6 mean, orange = PPO sampled.
                self._draw_polyline(
                    bev_img, route_mean, color=(20, 80, 230), radius=6, line_thickness=2
                )
                self._draw_polyline(
                    bev_img,
                    route_sample,
                    color=(230, 110, 0),
                    radius=5,
                    line_thickness=2,
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

        tp_prev_in_view, tp_prev_drawn = self._draw_target_marker(
            bev_img,
            target_point_previous,
            color=TP_DEFAULT_COLOR,
            radius=9,
            label="TP-1",
        )
        tp_in_view, tp_drawn = self._draw_target_marker(
            bev_img,
            target_point,
            color=TP_DEFAULT_COLOR,
            radius=11,
            label="TP",
        )
        tpn_in_view, tpn_drawn = self._draw_target_marker(
            bev_img,
            target_point_next,
            color=TP_DEFAULT_COLOR,
            radius=9,
            label="TP+1",
        )
        if tp_prev_in_view:
            self._draw_numbered_target_marker(
                bev_img, tp_prev_drawn, TP_DEFAULT_COLOR, 14, 0
            )
        if tp_in_view:
            self._draw_numbered_target_marker(
                bev_img, tp_drawn, TP_DEFAULT_COLOR, 18, 1
            )
        if tpn_in_view:
            self._draw_numbered_target_marker(
                bev_img, tpn_drawn, TP_DEFAULT_COLOR, 14, 2
            )

        self._overlay_legend_on_bev(bev_img)

        return bev_img, {
            "tp_prev_in_view": tp_prev_in_view,
            "tp_in_view": tp_in_view,
            "tp_next_in_view": tpn_in_view,
            "tp_prev_drawn_px": tp_prev_drawn,
            "tp_drawn_px": tp_drawn,
            "tp_next_drawn_px": tpn_drawn,
        }

    def _overlay_legend_on_bev(self, bev: np.ndarray) -> None:
        legend_items = []
        if self.steer_modality == "route":
            if self.use_residual_policy:
                legend_items.extend(
                    [
                        ("Route base (TFv6)", (20, 80, 230)),
                        ("Route mean (base+residual)", (50, 190, 60)),
                        ("Route sampled", (230, 110, 0)),
                    ]
                )
            else:
                legend_items.extend(
                    [
                        ("Route mean (TFv6)", (20, 80, 230)),
                        ("Route sampled (PPO)", (230, 110, 0)),
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
        speed_probs: np.ndarray,
        speed_bins: np.ndarray,
        mean_speed: float,
        sampled_speed: float | None,
        current_speed: float,
        hold_indicator_text: str | None,
    ) -> None:
        if w < 220 or h < 140:
            return

        speed_probs = np.asarray(speed_probs, dtype=np.float32).reshape(-1)
        speed_bins = np.asarray(speed_bins, dtype=np.float32).reshape(-1)
        if speed_probs.size == 0 or speed_bins.size == 0:
            return
        if speed_probs.size != speed_bins.size:
            count = min(speed_probs.size, speed_bins.size)
            speed_probs = speed_probs[:count]
            speed_bins = speed_bins[:count]

        prob_sum = float(speed_probs.sum())
        if prob_sum <= 0.0:
            return
        speed_probs = speed_probs / prob_sum

        cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (210, 210, 210), 1)
        cv2.putText(
            panel,
            "Base Policy Speed Histogram",
            (x0 + 8, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            1,
            lineType=cv2.LINE_AA,
        )

        plot_left = x0 + 12
        plot_right = x0 + w - 10
        plot_top = y0 + 64
        bar_top = plot_top + 22
        plot_bottom = y0 + h - 42
        cv2.rectangle(
            panel, (plot_left, plot_top), (plot_right, plot_bottom), (235, 235, 235), -1
        )
        cv2.rectangle(
            panel, (plot_left, plot_top), (plot_right, plot_bottom), (180, 180, 180), 1
        )

        max_speed = max(
            float(self.training_config.max_speed),
            (float(np.max(speed_bins)) + 1.0) if speed_bins.size > 0 else 1.0,
            mean_speed + 1.0,
            current_speed + 1.0,
            1.0,
        )
        max_prob = max(
            0.20,
            (float(np.max(speed_probs)) * 1.18) if speed_probs.size > 0 else 0.20,
        )

        def speed_to_px(v: float) -> int:
            alpha = float(np.clip(v / max_speed, 0.0, 1.0))
            return int(round(plot_left + alpha * (plot_right - plot_left)))

        def prob_to_py(v: float) -> int:
            alpha = float(np.clip(v / max_prob, 0.0, 1.0))
            return int(round(plot_bottom - alpha * (plot_bottom - bar_top)))

        cv2.putText(
            panel,
            (
                f"base_mean={mean_speed:.2f} m/s   "
                f"sampled={sampled_speed:.2f} m/s   ego={current_speed:.2f} m/s"
                if sampled_speed is not None
                else f"base_mean={mean_speed:.2f} m/s   ego={current_speed:.2f} m/s"
            ),
            (x0 + 8, y0 + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (30, 30, 30),
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            panel,
            (x0 + 10, y0 + 50),
            (x0 + 28, y0 + 50),
            (240, 130, 0),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "base mean",
            (x0 + 34, y0 + 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (30, 30, 30),
            1,
            lineType=cv2.LINE_AA,
        )
        legend_x = x0 + 112
        if sampled_speed is not None:
            cv2.line(
                panel,
                (legend_x, y0 + 50),
                (legend_x + 18, y0 + 50),
                (200, 50, 50),
                2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                "sampled",
                (legend_x + 24, y0 + 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (30, 30, 30),
                1,
                lineType=cv2.LINE_AA,
            )
            legend_x += 86
        cv2.line(
            panel,
            (legend_x, y0 + 50),
            (legend_x + 18, y0 + 50),
            (40, 150, 60),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "ego speed",
            (legend_x + 24, y0 + 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (30, 30, 30),
            1,
            lineType=cv2.LINE_AA,
        )
        if hold_indicator_text is not None:
            cv2.putText(
                panel,
                hold_indicator_text,
                (legend_x + 98, y0 + 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (30, 30, 30),
                1,
                lineType=cv2.LINE_AA,
            )

        y_ticks = np.linspace(0.0, max_prob, 4)
        for prob in y_ticks:
            ty = prob_to_py(float(prob))
            cv2.line(
                panel,
                (plot_left, ty),
                (plot_right, ty),
                (220, 220, 220),
                1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                f"{prob:.2f}",
                (plot_left - 34, ty + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.31,
                (70, 70, 70),
                1,
                lineType=cv2.LINE_AA,
            )

        if speed_bins.size == 1:
            step = max(1.0, max_speed * 0.1)
            bin_edges = np.array(
                [
                    max(0.0, float(speed_bins[0]) - 0.5 * step),
                    min(max_speed, float(speed_bins[0]) + 0.5 * step),
                ],
                dtype=np.float32,
            )
        else:
            mids = 0.5 * (speed_bins[:-1] + speed_bins[1:])
            left_edge = max(0.0, float(speed_bins[0]) - float(mids[0] - speed_bins[0]))
            right_edge = min(
                max_speed, float(speed_bins[-1]) + float(speed_bins[-1] - mids[-1])
            )
            bin_edges = np.concatenate(
                (
                    np.array([left_edge], dtype=np.float32),
                    mids.astype(np.float32),
                    np.array([right_edge], dtype=np.float32),
                )
            )

        prob_label_y = plot_top + 14
        for idx, prob in enumerate(speed_probs):
            left = speed_to_px(float(bin_edges[idx]))
            right = speed_to_px(float(bin_edges[idx + 1]))
            prob_center_x = int(round(0.5 * (left + right)))
            self._put_centered_text(
                panel,
                f"{100.0 * float(prob):.1f}%",
                prob_center_x,
                prob_label_y,
                font_scale=0.31,
                color=(35, 35, 35),
            )

        for idx, prob in enumerate(speed_probs):
            left = speed_to_px(float(bin_edges[idx]))
            right = speed_to_px(float(bin_edges[idx + 1]))
            top = prob_to_py(float(prob))
            if right <= left:
                right = left + 1
            cv2.rectangle(
                panel,
                (left + 1, top),
                (right - 1, plot_bottom - 1),
                (30, 120, 220),
                -1,
            )
            cv2.rectangle(
                panel,
                (left + 1, top),
                (right - 1, plot_bottom - 1),
                (20, 80, 150),
                1,
            )

        mean_x = speed_to_px(mean_speed)
        sampled_x = None if sampled_speed is None else speed_to_px(sampled_speed)
        current_x = speed_to_px(current_speed)
        cv2.line(
            panel,
            (mean_x, plot_top),
            (mean_x, plot_bottom),
            (240, 130, 0),
            2,
            lineType=cv2.LINE_AA,
        )
        if sampled_x is not None:
            cv2.line(
                panel,
                (sampled_x, plot_top),
                (sampled_x, plot_bottom),
                (200, 50, 50),
                2,
                lineType=cv2.LINE_AA,
            )
        cv2.line(
            panel,
            (current_x, plot_top),
            (current_x, plot_bottom),
            (40, 150, 60),
            2,
            lineType=cv2.LINE_AA,
        )

        for speed in speed_bins:
            tx = speed_to_px(float(speed))
            cv2.line(panel, (tx, plot_bottom), (tx, plot_bottom + 4), (80, 80, 80), 1)
            cv2.putText(
                panel,
                f"{speed:.1f}",
                (tx - 10, plot_bottom + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.31,
                (40, 40, 40),
                1,
                lineType=cv2.LINE_AA,
            )
        cv2.putText(
            panel,
            "speed bin [m/s]",
            (plot_left + max(28, (plot_right - plot_left) // 3), plot_bottom + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (40, 40, 40),
            1,
            lineType=cv2.LINE_AA,
        )

    def _draw_residual_speed_panel(
        self,
        panel: np.ndarray,
        *,
        x0: int,
        y0: int,
        w: int,
        h: int,
        base_speed: float,
        adjusted_speed: float,
        sampled_speed: float | None,
        current_speed: float,
        hold_indicator_text: str | None,
        speed_std_m_s: float = 0.0,
    ) -> None:
        """Speed panel for residual RL mode.

        Vertical markers on a plain speed axis:
          • blue   — frozen TFv6 E[categorical speed]  (matches route base color)
          • green  — residual-adjusted mean (base + delta)
          • orange — sampled speed (executed)           (matches route sampled color)
          • purple — current ego speed
        An orange Gaussian envelope around the adjusted mean is drawn when
        speed_std_m_s > 0. An arrow connects base to adjusted.
        """
        if w < 220 or h < 120:
            return

        COLOR_BASE = (20, 80, 230)  # blue   — frozen TFv6 base
        COLOR_ADJ = (50, 190, 60)  # green  — residual-adjusted mean
        COLOR_SAMPLED = (230, 110, 0)  # orange — sampled / executed speed
        COLOR_EGO = (180, 0, 180)  # purple — ego speed

        cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (210, 210, 210), 1)
        cv2.putText(
            panel,
            "Speed (Residual RL)",
            (x0 + 8, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1,
            lineType=cv2.LINE_AA,
        )

        delta = adjusted_speed - base_speed
        sampled_str = (
            f"   smp={sampled_speed:.2f} m/s" if sampled_speed is not None else ""
        )
        cv2.putText(
            panel,
            (
                f"base={base_speed:.2f} m/s   adj={adjusted_speed:.2f} m/s"
                f"   d={delta:+.2f} m/s{sampled_str}   ego={current_speed:.2f} m/s"
            ),
            (x0 + 8, y0 + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (30, 30, 30),
            1,
            lineType=cv2.LINE_AA,
        )

        # Legend
        lx = x0 + 10
        legend_entries = [
            ("base (TFv6)", COLOR_BASE),
            ("adjusted mean", COLOR_ADJ),
            ("sampled", COLOR_SAMPLED),
            ("ego speed", COLOR_EGO),
        ]
        for lname, lcolor in legend_entries:
            cv2.line(
                panel,
                (lx, y0 + 50),
                (lx + 18, y0 + 50),
                lcolor,
                2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                lname,
                (lx + 24, y0 + 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (30, 30, 30),
                1,
                lineType=cv2.LINE_AA,
            )
            (tw, _), _ = cv2.getTextSize(lname, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            lx += 24 + tw + 12
        if hold_indicator_text is not None:
            cv2.putText(
                panel,
                hold_indicator_text,
                (lx, y0 + 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (30, 30, 30),
                1,
                lineType=cv2.LINE_AA,
            )

        # Plot area
        plot_left = x0 + 12
        plot_right = x0 + w - 10
        plot_top = y0 + 66
        plot_bottom = y0 + h - 32
        if plot_bottom <= plot_top + 20:
            return

        cv2.rectangle(
            panel, (plot_left, plot_top), (plot_right, plot_bottom), (235, 235, 235), -1
        )
        cv2.rectangle(
            panel, (plot_left, plot_top), (plot_right, plot_bottom), (180, 180, 180), 1
        )

        max_speed_v = max(
            float(self.training_config.max_speed),
            base_speed + 1.0,
            adjusted_speed + 1.0,
            current_speed + 1.0,
            1.0,
        )

        def spd_to_px(v: float) -> int:
            return int(
                round(
                    plot_left
                    + float(np.clip(v / max_speed_v, 0.0, 1.0))
                    * (plot_right - plot_left)
                )
            )

        # Grid lines + axis labels
        for tick_v in np.linspace(0.0, max_speed_v, 5):
            tx = spd_to_px(float(tick_v))
            cv2.line(
                panel,
                (tx, plot_top),
                (tx, plot_bottom),
                (215, 215, 215),
                1,
                lineType=cv2.LINE_AA,
            )
            cv2.line(panel, (tx, plot_bottom), (tx, plot_bottom + 4), (80, 80, 80), 1)
            cv2.putText(
                panel,
                f"{tick_v:.1f}",
                (tx - 12, plot_bottom + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.31,
                (40, 40, 40),
                1,
                lineType=cv2.LINE_AA,
            )
        cv2.putText(
            panel,
            "speed [m/s]",
            (plot_left + (plot_right - plot_left) // 3, plot_bottom + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (40, 40, 40),
            1,
            lineType=cv2.LINE_AA,
        )

        base_x = spd_to_px(base_speed)
        adj_x = spd_to_px(adjusted_speed)

        # Orange Gaussian envelope around the adjusted mean (if std is available)
        if speed_std_m_s > 0.005:
            n_pts = max(plot_right - plot_left, 1)
            speeds_v = np.linspace(0.0, max_speed_v, n_pts, dtype=np.float32)
            gauss = np.exp(
                -0.5 * ((speeds_v - adjusted_speed) / max(speed_std_m_s, 1e-4)) ** 2
            )
            peak_h = plot_bottom - plot_top - 2
            top_ys = (plot_bottom - gauss * peak_h).astype(np.int32)
            top_ys = np.clip(top_ys, plot_top, plot_bottom)
            top_pts = np.stack(
                [np.arange(plot_left, plot_left + n_pts, dtype=np.int32), top_ys],
                axis=1,
            )
            poly = np.concatenate(
                [top_pts, [[plot_right, plot_bottom], [plot_left, plot_bottom]]], axis=0
            ).astype(np.int32)
            overlay = panel.copy()
            cv2.fillPoly(overlay, [poly], color=(180, 210, 255))  # light orange fill
            cv2.addWeighted(overlay, 0.35, panel, 0.65, 0, panel)
            cv2.polylines(
                panel,
                [top_pts.reshape(-1, 1, 2)],
                False,
                COLOR_SAMPLED,
                1,
                lineType=cv2.LINE_AA,
            )

        # Vertical lines — draw ego first so policy lines render on top.
        # Draw sampled before adjusted so adjusted mean is visible on top.
        ego_x = spd_to_px(current_speed)
        smp_x = spd_to_px(sampled_speed) if sampled_speed is not None else None
        cv2.line(
            panel,
            (ego_x, plot_top),
            (ego_x, plot_bottom),
            COLOR_EGO,
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            panel,
            (base_x, plot_top),
            (base_x, plot_bottom),
            COLOR_BASE,
            2,
            lineType=cv2.LINE_AA,
        )
        if smp_x is not None:
            cv2.line(
                panel,
                (smp_x, plot_top),
                (smp_x, plot_bottom),
                COLOR_SAMPLED,
                2,
                lineType=cv2.LINE_AA,
            )
        cv2.line(
            panel,
            (adj_x, plot_top),
            (adj_x, plot_bottom),
            COLOR_ADJ,
            2,
            lineType=cv2.LINE_AA,
        )

        # Arrow from base to adjusted: makes the correction direction immediately visible
        if abs(adj_x - base_x) > 3:
            mid_y = (plot_top + plot_bottom) // 2
            cv2.arrowedLine(
                panel,
                (base_x, mid_y),
                (adj_x, mid_y),
                (140, 140, 140),
                1,
                cv2.LINE_AA,
                0,
                0.2,
            )

    def _draw_residual_coeff_panel(
        self,
        panel: np.ndarray,
        *,
        x0: int,
        y0: int,
        w: int,
        h: int,
        coeff_preds: np.ndarray,  # shape (2*(rank+1),): [means..., log_stds...]
    ) -> None:
        """Forest-plot style panel: one row per residual coefficient (route modes + speed)."""
        # Derive rank from array size: 2*(rank+1) elements → rank = len/2 - 1
        n = len(coeff_preds) // 2  # route modes + speed
        rank = n - 1
        means = coeff_preds[:n].astype(np.float32)
        log_stds = coeff_preds[n : n + n].astype(np.float32)
        stds = np.exp(log_stds)

        if w < 120 or h < 80:
            return

        COLOR_ROUTE = (20, 80, 230)  # blue  — route basis modes
        COLOR_SPEED = (230, 110, 0)  # orange — speed coefficient
        COLOR_ZERO = (160, 160, 160)  # gray  — zero reference

        # --- Title ---
        cv2.rectangle(panel, (x0, y0), (x0 + w - 1, y0 + h - 1), (210, 210, 210), 1)
        cv2.putText(
            panel,
            "Residual Coefficients",
            (x0 + 8, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

        header_h = 26
        axis_h = 28  # bottom: x-axis ticks/labels
        label_w = 52  # left: row labels
        text_w = 88  # right: μ / σ numeric text
        plot_x0 = x0 + label_w
        plot_x1 = x0 + w - text_w - 4
        plot_y0 = y0 + header_h
        plot_y1 = y0 + h - axis_h

        if plot_x1 <= plot_x0 + 20 or plot_y1 <= plot_y0 + n * 10:
            return

        row_h = max(12, (plot_y1 - plot_y0) // n)

        # TD3 residual coefficients are tanh-bounded, so keep the visual scale
        # fixed across frames instead of rescaling to the current uncertainty.
        axis_min = -1.0
        axis_max = 1.0

        def val_to_px(v: float) -> int:
            alpha = (float(v) - axis_min) / (axis_max - axis_min)
            return int(round(plot_x0 + alpha * (plot_x1 - plot_x0)))

        zero_px = val_to_px(0.0)
        rows_bottom = plot_y0 + n * row_h

        # Continuous zero reference line through all rows
        cv2.line(
            panel,
            (zero_px, plot_y0),
            (zero_px, rows_bottom),
            COLOR_ZERO,
            1,
            cv2.LINE_AA,
        )

        for i in range(n):
            row_y0 = plot_y0 + i * row_h
            row_y1 = row_y0 + row_h
            row_cy = (row_y0 + row_y1) // 2

            color = COLOR_SPEED if i == rank else COLOR_ROUTE
            label = "speed" if i == rank else f"mode {i}"
            mu = float(means[i])
            sigma = float(stds[i])

            # Subtle row separator
            if i > 0:
                cv2.line(
                    panel,
                    (plot_x0, row_y0),
                    (plot_x1, row_y0),
                    (220, 220, 220),
                    1,
                )

            # Light background tint per row
            bg = (245, 248, 255) if color == COLOR_ROUTE else (255, 248, 242)
            cv2.rectangle(
                panel, (plot_x0, row_y0 + 2), (plot_x1 - 1, row_y1 - 2), bg, -1
            )

            # ±σ filled band (blended lighter version of the row color)
            band_h4 = max(3, row_h // 4)
            band_y0 = row_cy - band_h4
            band_y1 = row_cy + band_h4
            sx0 = max(plot_x0, min(plot_x1, val_to_px(mu - sigma)))
            sx1 = max(plot_x0, min(plot_x1, val_to_px(mu + sigma)))
            if sx1 > sx0:
                fill = tuple(int(c * 0.25 + 248 * 0.75) for c in color)
                cv2.rectangle(panel, (sx0, band_y0), (sx1, band_y1), fill, -1)
                cv2.rectangle(panel, (sx0, band_y0), (sx1, band_y1), color, 1)

            # Mean vertical line (slightly taller than band)
            mu_px = max(plot_x0, min(plot_x1, val_to_px(mu)))
            cv2.line(
                panel,
                (mu_px, band_y0 - 4),
                (mu_px, band_y1 + 4),
                color,
                2,
                cv2.LINE_AA,
            )

            # Mean dot with white outline
            cv2.circle(panel, (mu_px, row_cy), 4, color, -1, cv2.LINE_AA)
            cv2.circle(panel, (mu_px, row_cy), 4, (255, 255, 255), 1, cv2.LINE_AA)

            # Row label (left side)
            cv2.putText(
                panel,
                label,
                (x0 + 4, row_cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (40, 40, 40),
                1,
                cv2.LINE_AA,
            )

            # Numeric text (right side): two lines, μ and σ
            mu_sign = "+" if mu >= 0 else ""
            cv2.putText(
                panel,
                f"mu={mu_sign}{mu:.3f}",
                (plot_x1 + 6, row_cy - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.33,
                (40, 40, 40),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                f"std={sigma:.3f}",
                (plot_x1 + 6, row_cy + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.33,
                (40, 40, 40),
                1,
                cv2.LINE_AA,
            )

        # --- X-axis ---
        axis_y = rows_bottom
        cv2.line(panel, (plot_x0, axis_y), (plot_x1, axis_y), (120, 120, 120), 1)
        for tv in [axis_min, -0.5, 0.0, 0.5, axis_max]:
            tx = val_to_px(tv)
            if plot_x0 <= tx <= plot_x1:
                cv2.line(panel, (tx, axis_y), (tx, axis_y + 4), (80, 80, 80), 1)
                tick_label = "0" if tv == 0.0 else f"{tv:+.2f}"
                (lw, _), _ = cv2.getTextSize(
                    tick_label, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1
                )
                cv2.putText(
                    panel,
                    tick_label,
                    (tx - lw // 2, axis_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.30,
                    (40, 40, 40),
                    1,
                    cv2.LINE_AA,
                )
        cv2.putText(
            panel,
            "coeff value",
            (plot_x0, axis_y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
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

    def _overlay_attention_heatmap(
        self,
        bev_col: np.ndarray,
        attention_weights: np.ndarray,
        alpha: float = 0.45,
    ) -> np.ndarray:
        """Overlay residual attention heatmap on an already-column-prepared BEV image.

        attention_weights: float32 array (num_heads, 145) — output of
        _last_residual_attention_weights[env_idx].  The first 120 entries are the
        12×10 spatial BEV token grid (12 = x/forward, 10 = y/lateral, stride 32 px).
        Entries 120:145 are status tokens (ego vel/accel, command, target points,
        past positions/speeds) — included in cross-attention but not visualised here.

        The overlay is applied with the same rot90 + crop transform as
        _prepare_bev_for_column so the heatmap aligns with the rendered BEV.
        """
        n_spatial = 120  # 12 (x/forward) × 10 (y/lateral) backbone anchor cells
        n_x, n_y = 12, 10  # backbone feature grid: x=forward, y=lateral

        # Slice to spatial tokens only for the BEV heatmap; status tokens are not spatial.
        # Average across heads.
        w = attention_weights[:, :n_spatial].mean(axis=0)  # (120,)
        w_min, w_max = w.min(), w.max()
        w = (w - w_min) / (w_max - w_min + 1e-6)

        # Reshape: (n_x=12, n_y=10) then transpose to (n_y=10, n_x=12) = (rows, cols)
        # in the original BEV image space (rows=y/lateral, cols=x/forward).
        hmap = w.reshape(n_x, n_y).T  # (10, 12)

        # Upscale to full BEV pixel dims (bev_h × bev_w).
        hmap_full = cv2.resize(
            hmap, (self.bev_w, self.bev_h), interpolation=cv2.INTER_LINEAR
        )

        # Apply the same spatial transform as _prepare_bev_for_column.
        hmap_rot = np.rot90(hmap_full, k=1)
        hmap_rot = np.ascontiguousarray(hmap_rot)
        crop = int(0.08 * hmap_rot.shape[1])
        if 2 * crop < hmap_rot.shape[1] - 20:
            hmap_rot = hmap_rot[:, crop:-crop]
        hmap_resized = cv2.resize(
            hmap_rot,
            (bev_col.shape[1], bev_col.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        # Color grade: transparent (low) → yellow → red (high).
        # BGR: yellow=(0,255,255), red=(0,0,255). Only green channel varies.
        t = hmap_resized.astype(np.float32)  # [0, 1]
        hmap_color = np.stack(
            [
                np.zeros_like(t),  # B = 0
                (255.0 * (1.0 - t)),  # G: 255 at low, 0 at high
                np.full_like(t, 255.0),  # R = 255 always
            ],
            axis=2,
        ).astype(np.float32)

        # Per-pixel alpha: low attention → transparent, high → colored.
        alpha_map = (t * alpha)[:, :, np.newaxis]
        blended = (
            (bev_col.astype(np.float32) * (1.0 - alpha_map) + hmap_color * alpha_map)
            .clip(0, 255)
            .astype(np.uint8)
        )

        # Colorbar: same grade blended against mid-gray background at each alpha level.
        bar_w, bar_h = 12, min(100, blended.shape[0] - 40)
        bar_x, bar_y = 6, 22
        t_bar = np.linspace(1.0, 0.0, bar_h, dtype=np.float32)  # top=hi, bottom=lo
        bar_colors = np.stack(
            [
                np.zeros_like(t_bar),
                255.0 * (1.0 - t_bar),
                np.full_like(t_bar, 255.0),
            ],
            axis=1,
        ).reshape(bar_h, 1, 3)  # (bar_h, 1, 3) BGR float
        bar_alphas = (t_bar * alpha).reshape(bar_h, 1, 1)
        bg = np.full((bar_h, 1, 3), 128, dtype=np.float32)
        bar_blended = (
            (bg * (1.0 - bar_alphas) + bar_colors * bar_alphas)
            .clip(0, 255)
            .astype(np.uint8)
        )
        bar_blended = np.repeat(bar_blended, bar_w, axis=1)
        blended[bar_y : bar_y + bar_h, bar_x : bar_x + bar_w] = bar_blended

        cv2.putText(
            blended,
            f"{w_max:.5f}",
            (bar_x + bar_w + 3, bar_y + 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            blended,
            f"{w_min:.5f}",
            (bar_x + bar_w + 3, bar_y + bar_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            blended,
            "attn",
            (bar_x, bar_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return blended

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

    def _put_centered_text(
        self,
        panel: np.ndarray,
        text: str,
        center_x: int,
        baseline_y: int,
        *,
        font_scale: float,
        color: tuple[int, int, int],
    ) -> None:
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        text_x = int(round(center_x - text_w / 2.0))
        cv2.putText(
            panel,
            text,
            (text_x, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            lineType=cv2.LINE_AA,
        )

    def _stamp_text_line(
        self, image: np.ndarray, *, stamp_x: int, stamp_y: int, text: str
    ) -> None:
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        y_top = stamp_y - text_h - baseline - 3
        y_bottom = stamp_y + baseline + 3
        cv2.rectangle(
            image,
            (stamp_x - 2, y_top),
            (stamp_x + text_w + 6, y_bottom),
            (248, 248, 248),
            -1,
        )
        cv2.putText(
            image,
            text,
            (stamp_x, stamp_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            1,
            lineType=cv2.LINE_AA,
        )

    def _format_route_completion(self, route_completion: float | None) -> str:
        if route_completion is None or not np.isfinite(route_completion):
            return "route_completion=N/A"
        return f"route_completion={float(route_completion):.1f}%"

    def _format_infraction_reason(self, infraction_type: str | None) -> str:
        if infraction_type is None:
            return "None"
        normalized = str(infraction_type).strip()
        if normalized == "" or normalized == "finished_route":
            return "None"
        return normalized.replace("_", " ")

    def _standstill_hold_indicator_text(
        self, *, speed_hold_frames: int, speed_hold_target_speed: float | None
    ) -> str | None:
        if not self.standstill_speed_hold_enabled or speed_hold_frames <= 1:
            return None
        if speed_hold_target_speed is None or not np.isfinite(speed_hold_target_speed):
            return None
        return f"[Holding {speed_hold_target_speed:.1f} m/s x{speed_hold_frames}]"

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
        base_mean_action: np.ndarray | None = None,
        target_speed_logits: np.ndarray | None = None,
        residual_coeff_preds: np.ndarray | None = None,
        residual_attention_weights: np.ndarray | None = None,
        log_std: np.ndarray,
        value_estimate: float,
        route_completion: float | None,
        speed_hold_frames: int,
        speed_hold_target_speed: float | None,
        q_estimate: float | None = None,
    ) -> None:
        env_slot = env_idx % len(self.episode_returns)
        scheduled_burst_state = self.scheduled_burst_states[env_slot]
        should_start_scheduled_burst = (
            scheduled_burst_state.last_start_global_step is None
        )
        if (
            not should_start_scheduled_burst
            and global_step > 0
            and (global_step % self.every_n) == 0
            and scheduled_burst_state.last_start_global_step != global_step
        ):
            should_start_scheduled_burst = True
        if should_start_scheduled_burst:
            scheduled_burst_state.remaining = max(
                scheduled_burst_state.remaining,
                self.scheduled_burst_len,
            )
            scheduled_burst_state.last_start_global_step = global_step
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

        scheduled_burst_due = scheduled_burst_state.remaining > 0
        value_burst_due = self.value_burst_remaining[env_slot] > 0
        random_burst_due = self.random_burst_remaining[env_slot] > 0
        if not scheduled_burst_due and not value_burst_due and not random_burst_due:
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
        # Decode base TFv6 action (residual mode only: pure frozen-model prediction).
        route_base = None
        if base_mean_action is not None:
            route_base_decoded, _, _ = self.action_codec.decode(base_mean_action)
            route_base = None if route_base_decoded is None else route_base_decoded[0]
        hold_indicator_text = self._standstill_hold_indicator_text(
            speed_hold_frames=speed_hold_frames,
            speed_hold_target_speed=speed_hold_target_speed,
        )

        rgb = np.transpose(obs["rgb"], (1, 2, 0)).astype(np.uint8)
        # obs["rgb"] is stored in RGB order; OpenCV rendering/writes expect BGR.
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bev, _ = self._build_bev(
            lidar=obs["rasterized_lidar"],
            target_point_previous=obs["target_point_previous"],
            target_point=obs["target_point"],
            target_point_next=obs["target_point_next"],
            route_sample=route_sample,
            route_mean=route_mean,
            waypoints_sample=waypoints_sample,
            waypoints_mean=waypoints_mean,
            route_base=route_base,
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
        if (
            residual_attention_weights is not None
            and residual_attention_weights.shape[-1] >= 120
        ):
            bev_col = self._overlay_attention_heatmap(
                bev_col, residual_attention_weights
            )
        self._overlay_legend_on_bev(bev_col)

        scalar_w = int(round(right_w * 0.56))
        graph_w = right_w - gap - scalar_w
        scalar_panel = np.full((top_h, scalar_w, 3), 248, dtype=np.uint8)
        graphs_panel = np.full((top_h, graph_w, 3), 248, dtype=np.uint8)

        std = np.exp(log_std.astype(np.float32))
        route_point_std = None
        waypoint_point_std = None
        base_speed_probs = None

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

        if target_speed_logits is not None:
            logits = np.asarray(target_speed_logits, dtype=np.float32).reshape(-1)
            if logits.size > 0:
                logits = logits / self.speed_temperature
                logits = logits - float(np.max(logits))
                exp_logits = np.exp(logits)
                denom = float(exp_logits.sum())
                if denom > 0.0:
                    base_speed_probs = exp_logits / denom

        self.episode_returns[env_slot] += reward
        episode_return = float(self.episode_returns[env_slot])

        # Small env-id label in the top-right corner of the scalar panel so the user
        # can identify which route/environment instance this frame came from.
        _env_label = f"env={env_idx}"
        (_env_lw, _env_lh), _ = cv2.getTextSize(
            _env_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
        )
        cv2.putText(
            scalar_panel,
            _env_label,
            (scalar_w - _env_lw - 6, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )

        text_lines = [
            (
                f"update={update_idx} step={rollout_step} "
                f"{self._format_route_completion(route_completion)}"
            ),
            (
                f"reward={reward:+.4f} cumulative_reward={episode_return:+.4f} "
                + (
                    f"Q1={q_estimate:+.4f}"
                    if q_estimate is not None
                    else f"value={value_estimate:+.4f}"
                )
            ),
            "forward_discounted_return=pending",
            "forward_infraction_reason=pending",
            f"std[min/mean/max]={std.min():.4f}/{std.mean():.4f}/{std.max():.4f}",
        ]
        line_h = 24
        text_bottom_margin = 16
        text_start_y = max(
            24,
            top_h - text_bottom_margin - line_h * (len(text_lines) - 1),
        )
        self._put_text_lines(scalar_panel, text_lines, start_y=text_start_y)

        if (
            self.use_residual_policy
            and base_speed_probs is not None
            and target_speed_mean is not None
        ):
            # Residual mode: replace histogram with a clean speed-line panel.
            # base_speed = E[frozen TFv6 categorical] in m/s;
            # target_speed_mean is the residual-adjusted mean (base + delta) in m/s.
            speed_bins_np = np.asarray(
                self.training_config.target_speed_classes, dtype=np.float32
            )
            base_speed_m_s = float(np.sum(base_speed_probs * speed_bins_np))
            # diag_log_std[-1] is the effective speed log-std in normalized action space.
            speed_std_m_s = float(
                np.exp(float(log_std[-1])) * float(self.training_config.max_speed)
            )
            speed_panel_h = max(0, text_start_y - 18)
            self._draw_residual_speed_panel(
                scalar_panel,
                x0=0,
                y0=0,
                w=scalar_w,
                h=speed_panel_h,
                base_speed=base_speed_m_s,
                adjusted_speed=target_speed_mean,
                sampled_speed=target_speed_sample,
                current_speed=speed,
                hold_indicator_text=hold_indicator_text,
                speed_std_m_s=speed_std_m_s,
            )
        elif (
            self.target_speed_active_for_control
            and target_speed_mean is not None
            and base_speed_probs is not None
        ):
            speed_panel_h = max(0, text_start_y - 18)
            self._draw_speed_distribution(
                scalar_panel,
                x0=0,
                y0=0,
                w=scalar_w,
                h=speed_panel_h,
                speed_probs=base_speed_probs,
                speed_bins=np.asarray(
                    self.training_config.target_speed_classes, dtype=np.float32
                ),
                mean_speed=target_speed_mean,
                sampled_speed=target_speed_sample,
                current_speed=speed,
                hold_indicator_text=hold_indicator_text,
            )

        if self.use_residual_policy and residual_coeff_preds is not None:
            self._draw_residual_coeff_panel(
                graphs_panel,
                x0=0,
                y0=0,
                w=graph_w,
                h=top_h,
                coeff_preds=residual_coeff_preds,
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

        filename = (
            f"u{update_idx:06d}_gs{global_step:012d}_"
            f"step{rollout_step:04d}_env{env_idx}.jpg"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        image_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(
            image_path,
            grid,
            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        )
        self.images_written += 1

        stamp_x = right_x + 12
        forward_return_y = text_start_y + 2 * line_h
        forward_infraction_y = text_start_y + 3 * line_h
        self.pending_forward_return_stamps.setdefault(update_idx, []).append(
            _PendingForwardReturnStamp(
                rollout_step=rollout_step,
                env_idx=env_idx,
                image_path=image_path,
                stamp_x=stamp_x,
                stamp_y=forward_return_y,
            )
        )
        self.pending_episode_infraction_stamps[env_slot].append(
            _PendingEpisodeInfractionStamp(
                image_path=image_path,
                stamp_x=stamp_x,
                stamp_y=forward_infraction_y,
            )
        )

        if scheduled_burst_state.remaining > 0:
            scheduled_burst_state.remaining -= 1
        if self.value_burst_remaining[env_slot] > 0:
            self.value_burst_remaining[env_slot] -= 1
        if self.random_burst_remaining[env_slot] > 0:
            self.random_burst_remaining[env_slot] -= 1
        self.prev_values[env_slot] = value_estimate
        return

    def stamp_episode_forward_returns(self, step_to_return: dict[int, float]) -> None:
        """Stamp forward discounted returns for a completed TD3 episode.

        In TD3, update_idx == global_step and there is no batch rollout buffer,
        so PPO's stamp_forward_returns cannot be used. Call this at episode end
        with a dict mapping each global_step of the episode to its discounted
        return to episode end (computed backward). Only steps that have pending
        stamp records (i.e., frames that were written during that step) are
        stamped; all others are silently skipped.
        """
        for update_idx, forward_return in step_to_return.items():
            records = self.pending_forward_return_stamps.pop(update_idx, [])
            for record in records:
                if not os.path.isfile(record.image_path):
                    continue
                image = cv2.imread(record.image_path, cv2.IMREAD_COLOR)
                if image is None:
                    continue
                self._stamp_text_line(
                    image,
                    stamp_x=record.stamp_x,
                    stamp_y=record.stamp_y,
                    text=f"forward_discounted_return={forward_return:+.4f}",
                )
                cv2.imwrite(
                    record.image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 92]
                )

    def stamp_forward_returns(
        self, update_idx: int, forward_returns: np.ndarray
    ) -> None:
        records = self.pending_forward_return_stamps.pop(update_idx, [])
        if not records or forward_returns.ndim != 2:
            return

        num_steps, num_envs = forward_returns.shape
        for record in records:
            os.makedirs(self.output_dir, exist_ok=True)
            if record.rollout_step >= num_steps or record.env_idx >= num_envs:
                continue
            if not os.path.isfile(record.image_path):
                continue
            image = cv2.imread(record.image_path, cv2.IMREAD_COLOR)
            if image is None:
                continue

            forward_return = float(forward_returns[record.rollout_step, record.env_idx])
            self._stamp_text_line(
                image,
                stamp_x=record.stamp_x,
                stamp_y=record.stamp_y,
                text=f"forward_discounted_return={forward_return:+.4f}",
            )
            cv2.imwrite(record.image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    def note_episode_outcome(
        self, *, env_idx: int, terminal_infraction_type: str | None
    ) -> None:
        env_slot = env_idx % len(self.pending_episode_infraction_stamps)
        records = self.pending_episode_infraction_stamps[env_slot]
        if not records:
            return

        infraction_reason = self._format_infraction_reason(terminal_infraction_type)
        for record in records:
            if not os.path.isfile(record.image_path):
                continue
            image = cv2.imread(record.image_path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            self._stamp_text_line(
                image,
                stamp_x=record.stamp_x,
                stamp_y=record.stamp_y,
                text=f"forward_infraction_reason={infraction_reason}",
            )
            cv2.imwrite(record.image_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        records.clear()
