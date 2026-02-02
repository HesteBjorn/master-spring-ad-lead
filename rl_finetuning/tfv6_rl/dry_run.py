from __future__ import annotations

import argparse
import lzma
import pickle
from pathlib import Path

import cv2
import laspy
import numpy as np
import torch

from lead.common import common_utils
from lead.common import constants as viz_constants
from lead.common.constants import RadarDataIndex
from lead.data_loader import carla_dataset_utils
from lead.data_loader.carla_dataset_utils import rasterize_lidar
from rl_finetuning.tfv6_rl.action_codec import ActionCodec
from rl_finetuning.tfv6_rl.obs_codec import ObsCodec
from rl_finetuning.tfv6_rl.policy_tfv6_ppo import TFv6PPOPolicy, load_training_config


def build_dummy_obs(
    obs_codec: ObsCodec,
    training_config,
    batch_size: int,
    device: torch.device,
):
    obs = {}
    for spec in obs_codec.specs:
        shape = (batch_size,) + spec.shape
        if spec.key == "rgb":
            arr = np.random.randint(0, 255, size=shape, dtype=np.uint8)
        elif spec.key == "rasterized_lidar":
            arr = np.random.rand(*shape).astype(np.float32)
        elif spec.key == "radar":
            # Radar schema: [x, y, z, v, sensor_id]
            arr = np.zeros(shape, dtype=np.float32)
            arr[..., RadarDataIndex.X] = np.random.uniform(
                training_config.min_x_meter,
                training_config.max_x_meter,
                size=shape[:-1],
            )
            arr[..., RadarDataIndex.Y] = np.random.uniform(
                training_config.min_y_meter,
                training_config.max_y_meter,
                size=shape[:-1],
            )
            arr[..., RadarDataIndex.V] = np.random.uniform(
                0.0, training_config.max_speed, size=shape[:-1]
            )
            sensor_ids = np.random.randint(
                0,
                training_config.num_radar_sensors,
                size=shape[:-1],
                dtype=np.int64,
            )
            arr[..., RadarDataIndex.SENSOR_ID] = sensor_ids.astype(np.float32)
        elif spec.key in ("command", "next_command"):
            arr = np.zeros(shape, dtype=np.float32)
            # set a random one-hot command per batch
            for b in range(batch_size):
                idx = np.random.randint(0, 6)
                arr[b, idx] = 1.0
        else:
            arr = np.random.randn(*shape).astype(np.float32)
        obs[spec.key] = torch.tensor(arr, device=device)
    return obs


def _find_route_dir(data_root: Path) -> Path:
    if data_root.is_dir() and (data_root / "rgb").exists():
        return data_root
    for scenario_dir in sorted(data_root.iterdir()):
        if not scenario_dir.is_dir():
            continue
        for route_dir in sorted(scenario_dir.iterdir()):
            if (route_dir / "rgb").exists():
                return route_dir
    raise FileNotFoundError(f"No route folder found under {data_root}")


def _load_meta(meta_path: Path) -> dict:
    with lzma.open(meta_path, "rb") as f:
        return pickle.load(f)


def _load_rgb_image(rgb_path: Path, config) -> np.ndarray:
    img = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if (
        img.shape[0] != config.final_image_height
        or img.shape[1] != config.final_image_width
    ):
        img = cv2.resize(
            img,
            (config.final_image_width, config.final_image_height),
            interpolation=cv2.INTER_LINEAR,
        )

    if config.horizontal_fov_reduction > 0:
        crop_pixels = config.horizontal_fov_reduction
        img = img[:, crop_pixels:-crop_pixels]
        img = cv2.resize(
            img,
            (config.final_image_width, config.final_image_height),
            interpolation=cv2.INTER_LINEAR,
        )

    if config.num_used_cameras != config.num_available_cameras:
        n = config.num_available_cameras
        w = img.shape[1] // n
        rgb_slices = []
        for i, use in enumerate(config.used_cameras):
            if use:
                s, e = i * w, (i + 1) * w
                rgb_slices.append(img[:, s:e])
        img = np.concatenate(rgb_slices, axis=1)

    return np.transpose(img, (2, 0, 1))


def _load_lidar_raster(lidar_path: Path, config) -> np.ndarray:
    las = laspy.read(str(lidar_path))
    lidar_pc = las.xyz
    lidar_time = las["time"]
    lidar_pc = lidar_pc[lidar_time < config.training_used_lidar_steps]
    lidar_pc = common_utils.align_lidar(
        lidar_pc, np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.0
    )
    raster = rasterize_lidar(config=config, lidar=lidar_pc)
    return raster[None].astype(np.float32)


def _load_radar(radar_path: Path, config) -> np.ndarray | None:
    if not config.use_radars:
        return None
    radars = np.load(str(radar_path), allow_pickle=True)
    radar_dict = {
        f"radar{i}": radars[f"radar{i}"] for i in range(1, config.num_radar_sensors + 1)
    }
    radar_list = carla_dataset_utils.preprocess_radar_input(config, radar_dict)
    return np.concatenate(radar_list, axis=0).astype(np.float32)


def _resolve_tp_key(meta: dict, prefix: str, tp_pop_distance: float) -> str:
    direct_key = f"{prefix}{tp_pop_distance}"
    if direct_key in meta:
        return direct_key
    if prefix.rstrip("_") in meta:
        return prefix.rstrip("_")
    candidates = []
    for k in meta.keys():
        if k.startswith(prefix):
            try:
                suffix = float(k[len(prefix) :])
            except ValueError:
                continue
            candidates.append((abs(suffix - tp_pop_distance), k))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    raise KeyError(
        f"Could not resolve key for prefix {prefix} (tp_pop_distance={tp_pop_distance})"
    )


def _extract_target_points(
    meta: dict, config
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    noisy_version = "_gps" if config.use_noisy_tp else ""
    key_tp = _resolve_tp_key(
        meta, f"next{noisy_version}_target_points_", config.tp_pop_distance
    )
    key_cmd = _resolve_tp_key(
        meta, f"next{noisy_version}_commands_", config.tp_pop_distance
    )

    next_tp_list = meta[key_tp]
    next_cmd_list = meta[key_cmd]

    filtered_tp_list = []
    filtered_command_list = []
    for pt, cmd in zip(next_tp_list, next_cmd_list, strict=False):
        if (
            len(next_tp_list) == 2
            or not filtered_tp_list
            or not np.allclose(pt[:2], filtered_tp_list[-1][:2])
        ):
            filtered_tp_list.append(pt)
            filtered_command_list.append(cmd)
    next_tp_list = filtered_tp_list
    next_cmd_list = filtered_command_list

    ego_yaw = meta["theta"]
    if config.use_noisy_tp:
        if config.use_kalman_filter_for_gps:
            ego_pos = np.array(meta["filtered_pos_global"][:2])
        else:
            ego_pos = np.array(meta["noisy_pos_global"][:2])
    else:
        ego_pos = np.array(meta["pos_global"][:2])

    def transform(point: list[float]) -> np.ndarray:
        return common_utils.inverse_conversion_2d(np.array(point), ego_pos, ego_yaw)

    if len(next_tp_list) > 2:
        tp_prev = transform(next_tp_list[0][:2])
        tp = transform(next_tp_list[1][:2])
        tp_next = transform(next_tp_list[2][:2])
    else:
        tp_prev = transform(next_tp_list[0][:2])
        tp = transform(next_tp_list[1][:2])
        tp_next = transform(next_tp_list[1][:2])

    command = carla_dataset_utils.command_to_one_hot(next_cmd_list[0]).astype(
        np.float32
    )
    next_command = carla_dataset_utils.command_to_one_hot(next_cmd_list[1]).astype(
        np.float32
    )
    return tp_prev, tp, tp_next, command, next_command


def build_real_obs(route_dir: Path, frame_idx: int, config, device: torch.device):
    rgb_path = route_dir / "rgb" / f"{frame_idx:04d}.jpg"
    if not rgb_path.exists():
        rgb_path = route_dir / "rgb" / f"{frame_idx:04d}.png"
    lidar_path = route_dir / "lidar" / f"{frame_idx:04d}.laz"
    if not lidar_path.exists():
        lidar_path = route_dir / "lidar" / f"{frame_idx:04d}.las"
    radar_path = route_dir / "radar" / f"{frame_idx:04d}.npz"
    meta_path = route_dir / "metas" / f"{frame_idx:04d}.pkl"

    meta = _load_meta(meta_path)
    rgb = _load_rgb_image(rgb_path, config)
    raster = _load_lidar_raster(lidar_path, config)
    radar = _load_radar(radar_path, config)

    tp_prev, tp, tp_next, command, next_command = _extract_target_points(meta, config)

    obs = {
        "rgb": rgb.astype(np.uint8),
        "rasterized_lidar": raster.astype(np.float32),
        "target_point_previous": tp_prev.astype(np.float32),
        "target_point": tp.astype(np.float32),
        "target_point_next": tp_next.astype(np.float32),
        "speed": np.array([meta["speed"]], dtype=np.float32),
        "command": command,
        "next_command": next_command,
    }
    if radar is not None:
        obs["radar"] = radar.astype(np.float32)

    obs_torch = {
        key: torch.tensor(val, device=device).unsqueeze(0) for key, val in obs.items()
    }
    return obs, obs_torch


def _build_bev_image(
    rasterized_lidar: np.ndarray, config, scale_factor: int = 4
) -> np.ndarray:
    bev = rasterized_lidar.squeeze()
    bev = bev / (bev.max() + 1e-6)
    start_color = np.array([255, 255, 255], dtype=np.float32)
    end_color = np.array(viz_constants.LIDAR_COLOR, dtype=np.float32)
    bev_img = np.zeros((*bev.shape, 3), dtype=np.float32)
    for c in range(3):
        bev_img[..., c] = start_color[c] + (end_color[c] - start_color[c]) * bev
    bev_img = bev_img.astype(np.uint8)
    h, w = bev_img.shape[:2]
    return cv2.resize(
        bev_img, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_NEAREST
    )


def _meters_to_pixel(points: np.ndarray, config, scale_factor: int = 4) -> np.ndarray:
    ppm = config.pixels_per_meter * scale_factor
    origin_x = int(-config.min_x_meter * ppm)
    origin_y = int((config.max_y_meter - config.min_y_meter) * ppm / 2)
    xs = points[:, 0] * ppm + origin_x
    ys = points[:, 1] * ppm + origin_y
    return np.stack([xs, ys], axis=1).astype(np.int32)


def _draw_points(img: np.ndarray, points: np.ndarray, color, radius: int = 4):
    for x, y in points:
        cv2.circle(img, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)


def _draw_path(img: np.ndarray, points: np.ndarray, color, radius: int = 4):
    _draw_points(img, points, color, radius)


def _draw_polyline(img: np.ndarray, points: np.ndarray, color, thickness: int = 1):
    if points.shape[0] < 2:
        return
    for i in range(points.shape[0] - 1):
        x1, y1 = int(points[i][0]), int(points[i][1])
        x2, y2 = int(points[i + 1][0]), int(points[i + 1][1])
        cv2.line(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)


def _draw_crosses(
    img: np.ndarray, points: np.ndarray, color, size: int = 6, thickness: int = 2
):
    for x, y in points:
        x_i, y_i = int(x), int(y)
        cv2.line(
            img,
            (x_i - size, y_i - size),
            (x_i + size, y_i + size),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            img,
            (x_i - size, y_i + size),
            (x_i + size, y_i - size),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )


def _draw_ego_box(img: np.ndarray, config, scale_factor: int = 4):
    ppm = config.pixels_per_meter * scale_factor
    origin_x = int(-config.min_x_meter * ppm)
    origin_y = int((config.max_y_meter - config.min_y_meter) * ppm / 2)
    w = int(config.ego_extent_x * 2 * ppm)
    h = int(config.ego_extent_y * 2 * ppm)
    x1 = int(origin_x - w / 2)
    y1 = int(origin_y - h / 2)
    x2 = int(origin_x + w / 2)
    y2 = int(origin_y + h / 2)
    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        viz_constants.TP_DEFAULT_COLOR,
        thickness=2,
        lineType=cv2.LINE_AA,
    )


def visualize_predictions(
    rgb: np.ndarray,
    rasterized_lidar: np.ndarray,
    config,
    pred_route: np.ndarray | None,
    pred_waypoints: np.ndarray | None,
    tp_prev: np.ndarray,
    tp: np.ndarray,
    tp_next: np.ndarray,
    sampled_routes: list[np.ndarray],
    sampled_wps: list[np.ndarray],
    target_speed: float | None,
    sample_speeds: list[float],
    output_path: Path,
    show_sampled_waypoints: bool,
):
    bev = _build_bev_image(rasterized_lidar, config)
    scale = 4

    _draw_ego_box(bev, config, scale_factor=scale)

    tp_points = np.stack([tp_prev, tp, tp_next], axis=0)
    tp_pix = _meters_to_pixel(tp_points, config, scale)
    _draw_crosses(bev, tp_pix, viz_constants.TP_DEFAULT_COLOR, size=6, thickness=2)

    if pred_route is not None:
        route_pix = _meters_to_pixel(pred_route, config, scale)
        _draw_path(bev, route_pix, viz_constants.PREDICTION_ROUTE_COLOR, radius=4)
    if pred_waypoints is not None:
        wp_pix = _meters_to_pixel(pred_waypoints, config, scale)
        _draw_points(bev, wp_pix, viz_constants.PREDICTION_WAYPOINT_COLOR, radius=5)

    sample_colors = [
        viz_constants.rgb(255, 0, 255),
        viz_constants.rgb(0, 255, 255),
        viz_constants.rgb(255, 128, 0),
        viz_constants.rgb(128, 255, 0),
        viz_constants.rgb(0, 128, 255),
        viz_constants.rgb(255, 0, 128),
    ]
    for i, (sr, sw) in enumerate(zip(sampled_routes, sampled_wps, strict=False)):
        color = sample_colors[i % len(sample_colors)]
        if sr is not None:
            route_pix = _meters_to_pixel(sr, config, scale)
            _draw_polyline(bev, route_pix, color, thickness=1)
        if show_sampled_waypoints and sw is not None:
            wp_pix = _meters_to_pixel(sw, config, scale)
            _draw_points(bev, wp_pix, color, radius=3)

    rgb_vis = np.transpose(rgb, (1, 2, 0))
    rgb_vis = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)
    rgb_vis = cv2.resize(
        rgb_vis, (bev.shape[1], bev.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    panel = np.concatenate([rgb_vis, bev], axis=1)

    if target_speed is not None:
        txt = (
            f"target_speed_mean={target_speed:.2f} m/s | "
            f"samples={','.join(f'{s:.2f}' for s in sample_speeds)}"
        )
        cv2.putText(
            panel,
            txt,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            3,
        )
        cv2.putText(
            panel,
            txt,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
        )

    cv2.imwrite(str(output_path), panel)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True, type=str, help="Path to TFv6 checkpoint folder"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--sample-type", type=str, default="mean", choices=["mean", "sample"]
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to CARLA data root (e.g., data/carla_leaderboard2/data).",
    )
    parser.add_argument(
        "--route-dir",
        type=str,
        default=None,
        help="Path to a specific route folder under data_root.",
    )
    parser.add_argument(
        "--frame", type=int, default=0, help="Frame index to visualize."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dry_run_vis.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=6,
        help="Number of stochastic action samples to visualize.",
    )
    parser.add_argument(
        "--show-sampled-waypoints",
        type=lambda x: str(x).lower() in ("1", "true", "yes", "y"),
        default=False,
        help="If true, draw sampled waypoints as dots (default: False).",
    )
    parser.add_argument(
        "--log-std-init",
        type=float,
        default=None,
        help="Override PPO log_std initialization for visualization (e.g. -4.0).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    training_config = load_training_config(args.checkpoint)
    obs_codec = ObsCodec(training_config)
    action_codec = ActionCodec(training_config)

    policy = TFv6PPOPolicy(
        observation_space=None,
        action_space=None,
        tfv6_checkpoint=args.checkpoint,
        tfv6_prefix="model",
        device=device,
    ).to(device)
    if args.log_std_init is not None:
        with torch.no_grad():
            policy.log_std.fill_(float(args.log_std_init))

    if args.data_root or args.route_dir:
        data_root = Path(args.data_root) if args.data_root else None
        route_dir = Path(args.route_dir) if args.route_dir else None
        if route_dir is None:
            if data_root is None:
                raise ValueError("Either --data-root or --route-dir must be provided.")
            route_dir = _find_route_dir(data_root)
        obs_np, obs = build_real_obs(route_dir, args.frame, training_config, device)

        with torch.no_grad():
            with torch.amp.autocast(
                device_type=device.type,
                dtype=training_config.torch_float_type,
                enabled=training_config.use_mixed_precision_training
                and device.type == "cuda",
            ):
                predictions = policy.tfv6(obs)

        route = (
            predictions.pred_route.detach().cpu().float().numpy()[0]
            if predictions.pred_route is not None
            else None
        )
        waypoints = (
            predictions.pred_future_waypoints.detach().cpu().float().numpy()[0]
            if predictions.pred_future_waypoints is not None
            else None
        )
        target_speed = (
            float(
                predictions.pred_target_speed_scalar.detach().cpu().float().numpy()[0]
            )
            if predictions.pred_target_speed_scalar is not None
            else None
        )

        action_mean = action_codec.encode(
            predictions.pred_route,
            predictions.pred_future_waypoints,
            predictions.pred_target_speed_scalar,
        ).float()
        log_std = policy.log_std.unsqueeze(0).expand_as(action_mean)
        log_std = policy._apply_noise_ramp(log_std)
        dist = policy.action_dist.proba_distribution(action_mean, log_std)

        sampled_routes = []
        sampled_wps = []
        sample_speeds = []
        for _ in range(args.num_samples):
            act = torch.clamp(dist.sample(), -1.0, 1.0)
            r, w, s = action_codec.decode(act.cpu())
            if r is not None:
                if isinstance(r, torch.Tensor):
                    r = r.detach().cpu().numpy()
                sampled_routes.append(r[0])
            else:
                sampled_routes.append(None)
            if w is not None:
                if isinstance(w, torch.Tensor):
                    w = w.detach().cpu().numpy()
                sampled_wps.append(w[0])
            else:
                sampled_wps.append(None)
            if s is not None:
                if isinstance(s, torch.Tensor):
                    s = s.detach().cpu().numpy()
                sample_speeds.append(float(s[0]))

        output_path = Path(args.output)
        visualize_predictions(
            rgb=obs_np["rgb"],
            rasterized_lidar=obs_np["rasterized_lidar"],
            config=training_config,
            pred_route=route,
            pred_waypoints=waypoints,
            tp_prev=obs_np["target_point_previous"],
            tp=obs_np["target_point"],
            tp_next=obs_np["target_point_next"],
            sampled_routes=sampled_routes,
            sampled_wps=sampled_wps,
            target_speed=target_speed,
            sample_speeds=sample_speeds,
            output_path=output_path,
            show_sampled_waypoints=args.show_sampled_waypoints,
        )
        print(f"[dry_run] Wrote visualization to {output_path}")
        return

    obs = build_dummy_obs(obs_codec, training_config, args.batch_size, device)

    with torch.no_grad():
        actions, logprob, entropy, values, _, mu, sigma, _, _, _, _ = policy.forward(
            obs, sample_type=args.sample_type
        )

    print("[dry_run] batch_size", args.batch_size)
    print("[dry_run] obs keys", list(obs.keys()))
    print("[dry_run] action_dim", actions.shape[-1])
    print("[dry_run] actions shape", tuple(actions.shape))
    print("[dry_run] logprob shape", tuple(logprob.shape))
    print("[dry_run] values shape", tuple(values.shape))

    route, waypoints, target_speed = action_codec.decode(actions.cpu())
    if route is not None:
        print("[dry_run] route shape", route.shape)
    if waypoints is not None:
        print("[dry_run] waypoints shape", waypoints.shape)
    if target_speed is not None:
        print("[dry_run] target_speed shape", target_speed.shape)

    print("[dry_run] OK")


if __name__ == "__main__":
    main()
