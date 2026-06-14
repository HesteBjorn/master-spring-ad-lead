"""Clean inference-time clip renderer for the thesis demonstration videos.

Three balanced panels per frame, nothing else:

    +----------------------------------------------+
    |                  CAMERA                       |
    +----------------------+-----------------------+
    |   LiDAR BEV + ROUTE  |       SPEED           |
    +----------------------+-----------------------+

Design choices (all driven by thesis-figure clarity):
  * BEV is zoomed around the ego and rotated so the ego heading / route point UP,
    matching the forward direction of the camera above it.
  * Only the frozen-TFv6 route is drawn (blue, "TFv6 route"); the speed-only
    residual leaves the route unchanged, so no separate residual route is shown.
  * Speed is shown as horizontal "volume" bars that fill proportionally: the
    frozen-TFv6 target speed, the TD3-corrected target speed, and the ego speed.

Geometry (origin, pixels-per-meter, canvas size) is borrowed from the training
PPORolloutVisualizer so the BEV scale matches the rest of the work, but all
drawing here is self-contained. Frames are written frame_000000.jpg ... so a
one-line ffmpeg call stitches them.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch

from rl_finetuning.tfv6_rl.debug_rollout_viz import PPORolloutVisualizer

# BGR colours.
_BG = (245, 245, 245)
_PANEL = (248, 248, 248)
_LABEL_BG = (32, 32, 32)
_LABEL_FG = (255, 255, 255)
_ROUTE_BLUE = (230, 120, 20)  # TFv6 route (blue in BGR)
_BAR_BASE = (230, 120, 20)  # TFv6 base speed bar (blue)
_BAR_CORR = (60, 190, 60)  # TD3 corrected speed bar (green)
_BAR_EGO = (150, 150, 150)  # ego speed bar (grey)
_LIDAR_DARK = np.array([60, 60, 60], dtype=np.float32)


class InferenceClipVisualizer:
    """Render one clean composite frame per inference step into ``output_dir``."""

    def __init__(
        self,
        *,
        training_config,
        action_codec,
        output_dir: str,
        residual: bool = True,
        speed_temperature: float = 1.0,
        image_scale: int = 3,
        canvas_width: int = 1280,
        # Taller bottom row -> panel nearer square, so two side-by-side land near
        # a ~2:1 pair rather than a thin ~2.5:1 ultrawide strip.
        bottom_fraction: float = 0.60,
        zoom_forward_m: float = 44.0,
        zoom_back_m: float = 14.0,
        zoom_side_m: float = 26.0,
    ) -> None:
        # Internal visualizer instance used purely for config-correct geometry
        # (origin, ppm, canvas size). No training drawing routines are called.
        self._viz = PPORolloutVisualizer(
            training_config=training_config,
            action_codec=action_codec,
            output_dir=output_dir,
            num_envs=1,
            image_scale=image_scale,
            use_residual_policy=residual,
        )
        self.training_config = training_config
        self.action_codec = action_codec
        self.residual = bool(residual)
        self.speed_temperature = float(speed_temperature)
        self.output_dir = output_dir
        self.canvas_width = int(canvas_width)
        self.bottom_fraction = float(bottom_fraction)
        self.zoom_forward_m = float(zoom_forward_m)
        self.zoom_back_m = float(zoom_back_m)
        self.zoom_side_m = float(zoom_side_m)
        self.max_speed = float(getattr(training_config, "max_speed", 16.0))
        os.makedirs(self.output_dir, exist_ok=True)
        self._frame = 0
        self._render_failed = False

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _np(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().to("cpu", dtype=torch.float32).numpy()
        return np.asarray(x, dtype=np.float32)

    @classmethod
    def _unbatch(cls, x):
        arr = cls._np(x)
        if arr is None:
            return None
        return arr[0] if arr.ndim >= 1 and arr.shape[0] == 1 else arr

    def _decode_route_speed(self, action):
        if action is None:
            return None, None
        route, _wp, target_speed = self.action_codec.decode(self._unbatch(action))
        route = None if route is None else np.asarray(route).reshape(-1, 2)
        ts = (
            None
            if target_speed is None
            else float(np.asarray(target_speed).reshape(-1)[0])
        )
        return route, ts

    def _base_speed_from_logits(self, logits):
        if logits is None:
            return None
        logits = self._unbatch(logits).reshape(-1).astype(np.float32)
        if logits.size == 0:
            return None
        z = logits / self.speed_temperature
        z = z - float(np.max(z))
        p = np.exp(z)
        s = float(p.sum())
        if s <= 0.0:
            return None
        probs = p / s
        bins = np.asarray(self.training_config.target_speed_classes, dtype=np.float32)
        n = min(probs.size, bins.size)
        return float(np.sum(probs[:n] * bins[:n]))

    def _label(self, img, text, x, y, scale=0.5):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(img, (x, y), (x + tw + 12, y + th + 10), _LABEL_BG, -1)
        cv2.putText(
            img,
            text,
            (x + 6, y + th + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            _LABEL_FG,
            1,
            lineType=cv2.LINE_AA,
        )

    def _w2px(self, x, y):
        return (
            int(round(self._viz.origin_x + x * self._viz.ppm)),
            int(round(self._viz.origin_y + y * self._viz.ppm)),
        )

    # --------------------------------------------------------------- BEV panel
    def _render_bev(self, lidar, route_world):
        """LiDAR raster + ego + TFv6 route, zoomed around ego and rotated up."""
        bev = np.squeeze(self._np(lidar)).astype(np.float32)
        if bev.ndim == 3:
            bev = bev[0]
        bev = bev / (bev.max() + 1e-6)
        img = (255.0 + (_LIDAR_DARK - 255.0) * bev[..., None]).astype(np.uint8)
        img = cv2.resize(
            img, (self._viz.bev_w, self._viz.bev_h), interpolation=cv2.INTER_NEAREST
        )

        # TFv6 route polyline (blue), drawn in the original (forward = +x) frame.
        if route_world is not None and route_world.shape[0] > 0:
            pts = [self._w2px(float(p[0]), float(p[1])) for p in route_world]
            for a, b in zip(pts[:-1], pts[1:], strict=True):
                cv2.line(img, a, b, _ROUTE_BLUE, 3, lineType=cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, p, 3, _ROUTE_BLUE, -1, lineType=cv2.LINE_AA)

        # Ego marker: identical to the training debug-viz (origin dot + forward
        # heading line + blue ego box). +x (forward) becomes "up" after rotation.
        ppm = self._viz.ppm
        ox, oy = self._viz.origin_x, self._viz.origin_y
        cv2.circle(img, (ox, oy), 8, (20, 20, 20), -1)
        cv2.line(
            img,
            (ox, oy),
            (ox + int(ppm * 2.0), oy),
            (20, 20, 20),
            2,
            lineType=cv2.LINE_AA,
        )
        ego_w = int(round(self.training_config.ego_extent_x * ppm))
        ego_h = int(round(self.training_config.ego_extent_y * ppm))
        cv2.rectangle(
            img,
            (ox - ego_w, oy - ego_h),
            (ox + ego_w, oy + ego_h),
            (255, 0, 0),
            3,
            lineType=cv2.LINE_AA,
        )

        # Zoom: crop a window around the ego (more ahead than behind).
        c_lo = int(round(ox - self.zoom_back_m * ppm))
        c_hi = int(round(ox + self.zoom_forward_m * ppm))
        r_lo = int(round(oy - self.zoom_side_m * ppm))
        r_hi = int(round(oy + self.zoom_side_m * ppm))
        h, w = img.shape[:2]
        c_lo, c_hi = max(0, c_lo), min(w, c_hi)
        r_lo, r_hi = max(0, r_lo), min(h, r_hi)
        crop = img[r_lo:r_hi, c_lo:c_hi]
        if crop.size == 0:
            crop = img

        # Rotate so forward (+x, right) points up, matching the camera.
        return cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # ------------------------------------------------------------- speed bars
    def _draw_speed_bars(self, panel, base_speed, corrected_speed, ego_speed):
        """Vertical level-meter bars that fill bottom-up by speed."""
        h, w = panel.shape[:2]
        cv2.putText(
            panel,
            "Speed",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (25, 25, 25),
            2,
            lineType=cv2.LINE_AA,
        )

        bars = []
        if self.residual and base_speed is not None:
            bars.append(("TFv6", base_speed, _BAR_BASE))
            bars.append(("TD3", corrected_speed, _BAR_CORR))
        elif corrected_speed is not None:
            bars.append(("TFv6", corrected_speed, _BAR_BASE))
        bars.append(("Ego", ego_speed, _BAR_EGO))

        mx = max(self.max_speed, 1.0)
        plot_top = 56
        plot_bottom = h - 56
        n = len(bars)
        bar_w = min(72, int((w - 60) / (n * 1.6)))
        slot = (w - 30) / n
        # y-axis: 0 and max ticks on the left edge of the plot.
        cv2.line(panel, (30, plot_top), (30, plot_bottom), (180, 180, 180), 1)
        cv2.putText(
            panel,
            f"{mx:.0f}",
            (4, plot_top + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (120, 120, 120),
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "0",
            (16, plot_bottom),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (120, 120, 120),
            1,
            lineType=cv2.LINE_AA,
        )

        for i, (name, val, color) in enumerate(bars):
            v = 0.0 if val is None else float(val)
            cx = int(40 + slot * (i + 0.5))
            x0, x1 = cx - bar_w // 2, cx + bar_w // 2
            cv2.rectangle(panel, (x0, plot_top), (x1, plot_bottom), (225, 225, 225), -1)
            cv2.rectangle(panel, (x0, plot_top), (x1, plot_bottom), (170, 170, 170), 1)
            fill = int(round(np.clip(v / mx, 0.0, 1.0) * (plot_bottom - plot_top)))
            cv2.rectangle(panel, (x0, plot_bottom - fill), (x1, plot_bottom), color, -1)
            cv2.putText(
                panel,
                f"{v:.1f}",
                (cx - 16, plot_top - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (30, 30, 30),
                1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                name,
                (cx - 18, plot_bottom + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (30, 30, 30),
                1,
                lineType=cv2.LINE_AA,
            )

        # Delta annotation for the residual correction.
        if self.residual and base_speed is not None and corrected_speed is not None:
            d = corrected_speed - base_speed
            cv2.putText(
                panel,
                f"residual delta = {d:+.1f} m/s",
                (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (40, 150, 40) if d >= 0 else (40, 40, 180),
                1,
                lineType=cv2.LINE_AA,
            )

    # ----------------------------------------------------------------- render
    def render(self, **kwargs) -> None:
        """Fault-tolerant entry point: a viz bug must never crash the eval run."""
        if self._render_failed:
            return
        try:
            self._render_frame(**kwargs)
        except Exception as exc:  # noqa: BLE001 — diagnostics only, keep driving
            if not self._render_failed:
                self._render_failed = True
                import traceback

                print(f"[clipviz] render disabled after error: {exc}", flush=True)
                traceback.print_exc()

    def _render_frame(
        self,
        *,
        data: dict,
        corrected_action: torch.Tensor,
        base_action: torch.Tensor | None = None,
        target_speed_logits: torch.Tensor | None = None,
        **_ignored,
    ) -> None:
        ego_speed = float(self._np(data["speed"]).reshape(-1)[0])
        route_corr, ts_corr = self._decode_route_speed(corrected_action)
        route_base, _ = self._decode_route_speed(base_action)
        base_speed = self._base_speed_from_logits(target_speed_logits)

        # Only the TFv6 route is shown (speed-only residual leaves route unchanged).
        tfv6_route = route_base if route_base is not None else route_corr
        bev = self._render_bev(data["rasterized_lidar"], tfv6_route)
        if self._frame == 0:
            print(
                f"[clipviz] frame0 bev={bev.shape} route={None if tfv6_route is None else tfv6_route.shape}",
                flush=True,
            )

        # Camera (RGB channels-first -> HWC BGR).
        rgb = self._unbatch(data["rgb"]).astype(np.uint8)
        if rgb.ndim == 3 and rgb.shape[0] in (1, 3):
            rgb = np.transpose(rgb, (1, 2, 0))
        cam = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # --- Layout: camera banner on top, BEV + speed on the bottom row ---
        pad, gap = 10, 8
        inner_w = self.canvas_width - 2 * pad
        cam_h = int(round(inner_w * cam.shape[0] / max(1, cam.shape[1])))
        bottom_h = int(round(inner_w * self.bottom_fraction))
        # LiDAR panel gets ~half the figure width; speed panel takes the rest.
        bev_w_slot = inner_w // 2
        speed_w = inner_w - gap - bev_w_slot
        canvas_h = pad + cam_h + gap + bottom_h + pad
        canvas = np.full((canvas_h, self.canvas_width, 3), _BG, dtype=np.uint8)

        cam_r = cv2.resize(cam, (inner_w, cam_h), interpolation=cv2.INTER_LINEAR)
        canvas[pad : pad + cam_h, pad : pad + inner_w] = cam_r
        self._label(canvas, "Camera", pad + 6, pad + 6, scale=0.75)

        by = pad + cam_h + gap
        # Fit the BEV into its half-width slot preserving aspect (no distortion).
        bev_slot = np.full((bottom_h, bev_w_slot, 3), _BG, dtype=np.uint8)
        scale = min(bev_w_slot / bev.shape[1], bottom_h / bev.shape[0])
        rw, rh = int(bev.shape[1] * scale), int(bev.shape[0] * scale)
        bev_r = cv2.resize(bev, (rw, rh), interpolation=cv2.INTER_NEAREST)
        ox0 = (bev_w_slot - rw) // 2
        oy0 = (bottom_h - rh) // 2
        bev_slot[oy0 : oy0 + rh, ox0 : ox0 + rw] = bev_r
        canvas[by : by + bottom_h, pad : pad + bev_w_slot] = bev_slot
        self._label(canvas, "LiDAR BEV + TFv6 route", pad + 6, by + 6, scale=0.75)

        sx = pad + bev_w_slot + gap
        panel = np.full((bottom_h, speed_w, 3), _PANEL, dtype=np.uint8)
        self._draw_speed_bars(panel, base_speed, ts_corr, ego_speed)
        canvas[by : by + bottom_h, sx : sx + speed_w] = panel

        cv2.imwrite(
            os.path.join(self.output_dir, f"frame_{self._frame:06d}.jpg"),
            canvas,
            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        )
        self._frame += 1
