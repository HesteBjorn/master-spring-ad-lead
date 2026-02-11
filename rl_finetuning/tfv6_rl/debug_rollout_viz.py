from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

from lead.common.constants import CARLA_NAVIGATION_COMMAND_STR_MAP, LIDAR_COLOR
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
        every_n: int = 1,
        max_images: int = 0,
        image_scale: int = 3,
    ) -> None:
        self.training_config = training_config
        self.action_codec = action_codec
        self.output_dir = output_dir
        self.every_n = max(1, int(every_n))
        self.max_images = max(0, int(max_images))
        self.image_scale = max(1, int(image_scale))
        os.makedirs(self.output_dir, exist_ok=True)
        self.images_written = 0

        self.config_closed_loop = ClosedLoopConfig(raise_error_on_missing_key=False)
        self.config_expert = ExpertConfig()
        self.controllers = [
            _ClosedLoopControlAdapter(self.config_closed_loop, self.config_expert)
            for _ in range(max(1, num_envs))
        ]
        self.command_names = {
            int(k.value): v for k, v in CARLA_NAVIGATION_COMMAND_STR_MAP.items()
        }
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

        return bev_img, {
            "tp_in_view": tp_in_view,
            "tp_next_in_view": tpn_in_view,
            "tp_drawn_px": tp_drawn,
            "tp_next_drawn_px": tpn_drawn,
        }

    def _add_legend(self, panel: np.ndarray) -> None:
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
        legend_items.extend(
            [
                ("Target point", (10, 10, 220)),
                ("Target point next", (30, 170, 240)),
            ]
        )
        y = 26
        for name, color in legend_items:
            cv2.rectangle(panel, (14, y - 12), (34, y + 8), color, -1)
            cv2.putText(
                panel,
                name,
                (42, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (30, 30, 30),
                1,
                lineType=cv2.LINE_AA,
            )
            y += 28

    def _draw_speed_distribution(
        self,
        panel: np.ndarray,
        *,
        mean_speed: float,
        std_speed: float,
        selected_speed: float,
        current_speed: float,
    ) -> None:
        x0, y0 = 330, 18
        w, h = 270, 170
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
        plot_top = y0 + 48
        plot_bottom = y0 + h - 42
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
            (plot_left + 82, plot_bottom + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (40, 40, 40),
            1,
            lineType=cv2.LINE_AA,
        )

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
        if rollout_step % self.every_n != 0:
            return
        if self.max_images > 0 and self.images_written >= self.max_images:
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

        controller = self.controllers[env_idx % len(self.controllers)]
        ctrl_sample = controller.infer_control(
            route_sample, waypoints_sample, target_speed_sample, speed
        )
        ctrl_mean = controller.infer_control(
            route_mean, waypoints_mean, target_speed_mean, speed
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
        bev = cv2.resize(bev, (int(bev.shape[1] * rgb_h / bev.shape[0]), rgb_h))

        panel = np.full((rgb_h, 620, 3), 248, dtype=np.uint8)
        self._add_legend(panel)

        command_idx = int(np.argmax(obs["command"])) + 1
        next_command_idx = int(np.argmax(obs["next_command"])) + 1
        command_name = self.command_names.get(command_idx, f"Cmd {command_idx}")
        next_command_name = self.command_names.get(
            next_command_idx, f"Cmd {next_command_idx}"
        )
        std = np.exp(log_std.astype(np.float32))
        target_speed_std = None
        if self.action_codec.slices.target_speed is not None:
            speed_idx = self.action_codec.slices.target_speed.start
            target_speed_std = float(std[speed_idx] * self.action_codec.speed_scale)

        speed_head_enabled = bool(self.training_config.predict_target_speed)
        wp_head_enabled = bool(self.training_config.predict_temporal_spatial_waypoints)
        route_head_enabled = bool(self.training_config.predict_spatial_path)
        text_lines = [
            f"update={update_idx} rollout_step={rollout_step} global_step={global_step} env={env_idx}",
            f"reward={reward:+.4f} done={int(done)} trunc={int(truncated)} value={value_estimate:+.4f}",
            f"speed={speed:.3f} m/s | cmd={command_name} | next={next_command_name}",
            (
                "heads enabled: "
                f"route={int(route_head_enabled)} "
                f"waypoints={int(wp_head_enabled)} "
                f"target_speed={int(speed_head_enabled)}"
            ),
            (
                "controller feed: "
                f"steer={self.steer_modality} "
                f"throttle={self.throttle_modality} "
                f"brake={self.brake_modality}"
            ),
            (
                "active policy signal: "
                f"{'target_speed_scalar' if self.target_speed_active_for_control else 'temporal_waypoints'}"
            ),
            (
                f"std[min/mean/max]={std.min():.4f}/{std.mean():.4f}/{std.max():.4f}"
                f" | action_dim={sampled_action.shape[0]}"
            ),
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
            (
                "sample ctrl "
                f"(steer,throttle,brake)=({ctrl_sample.steer:+.3f},{ctrl_sample.throttle:.3f},{ctrl_sample.brake:.1f})"
            ),
            (
                "mean ctrl   "
                f"(steer,throttle,brake)=({ctrl_mean.steer:+.3f},{ctrl_mean.throttle:.3f},{ctrl_mean.brake:.1f})"
            ),
        ]
        if self.target_speed_active_for_control:
            text_lines.append(
                f"sample target_speed={target_speed_sample:.3f} m/s"
                if target_speed_sample is not None
                else "sample target_speed=None"
            )
            text_lines.append(
                (
                    f"mean target_speed={target_speed_mean:.3f} m/s "
                    f"(std={target_speed_std:.3f} m/s)"
                )
                if (target_speed_mean is not None and target_speed_std is not None)
                else "mean target_speed=None"
            )
        else:
            text_lines.append("target-speed plot disabled (waypoint control active)")
        self._put_text_lines(panel, text_lines, start_y=210)

        if (
            self.target_speed_active_for_control
            and target_speed_mean is not None
            and target_speed_sample is not None
            and target_speed_std is not None
        ):
            self._draw_speed_distribution(
                panel,
                mean_speed=target_speed_mean,
                std_speed=target_speed_std,
                selected_speed=target_speed_sample,
                current_speed=speed,
            )

        grid = np.concatenate([rgb, bev, panel], axis=1)
        filename = f"u{update_idx:06d}_gs{global_step:012d}_step{rollout_step:04d}_env{env_idx}.jpg"
        cv2.imwrite(
            os.path.join(self.output_dir, filename),
            grid,
            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        )
        self.images_written += 1
