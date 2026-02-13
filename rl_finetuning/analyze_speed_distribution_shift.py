#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

from lead.tfv6.tfv6 import TFv6
from rl_finetuning.tfv6_rl.dry_run import build_real_obs
from rl_finetuning.tfv6_rl.policy_tfv6_ppo import load_training_config

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _discover_route_dirs(data_root: Path) -> list[Path]:
    route_dirs: list[Path] = []
    for rgb_dir in data_root.rglob("rgb"):
        route_dir = rgb_dir.parent
        if (route_dir / "metas").exists() and (route_dir / "lidar").exists():
            # Radar is optional at discovery time; build_real_obs handles config.use_radars.
            route_dirs.append(route_dir)
    return sorted(route_dirs)


def _normalize_state_dict_for_tfv6(
    raw_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    # Finetuned PPO checkpoints commonly store keys with "tfv6." prefix.
    if any(k.startswith("tfv6.") for k in raw_state_dict.keys()):
        return {
            key.replace("tfv6.", "", 1): value
            for key, value in raw_state_dict.items()
            if key.startswith("tfv6.")
        }
    return raw_state_dict


def _load_tfv6_model(
    checkpoint_file: Path,
    config,
    device: torch.device,
) -> TFv6:
    model = TFv6(device, config).to(device).eval()
    raw = torch.load(checkpoint_file, map_location=device, weights_only=True)
    state_dict = _normalize_state_dict_for_tfv6(raw)
    model.load_state_dict(state_dict, strict=True)
    return model


def _collate_obs(obs_list: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = obs_list[0].keys()
    return {key: torch.cat([obs[key] for obs in obs_list], dim=0) for key in keys}


def _extract_path_tensor(predictions) -> torch.Tensor | None:
    if getattr(predictions, "pred_route", None) is not None:
        return predictions.pred_route
    if getattr(predictions, "pred_future_waypoints", None) is not None:
        return predictions.pred_future_waypoints
    return None


def _build_path_plot_points(
    path_points_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Model path points are in ego coordinates (x forward, y lateral).
    # For plotting with ego front upward, use horizontal=y and vertical=-x.
    x_forward = path_points_xy[:, 0]
    y_lateral = path_points_xy[:, 1]
    return y_lateral, -x_forward


def _sample_for_scatter(
    old: np.ndarray, new: np.ndarray, delta: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if old.shape[0] <= max_points:
        return old, new, delta
    rng = np.random.default_rng(0)
    keep = rng.choice(old.shape[0], size=max_points, replace=False)
    return old[keep], new[keep], delta[keep]


def _smooth_hist2d(hist: np.ndarray, passes: int = 2) -> np.ndarray:
    # Lightweight smoothing without extra dependencies (separable [1,2,1] kernel).
    kernel = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    kernel = kernel / kernel.sum()
    out = hist.astype(np.float32, copy=True)
    for _ in range(max(0, int(passes))):
        out = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=out
        )
        out = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=out
        )
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate old-vs-finetuned target-speed distribution shift over route data."
    )
    parser.add_argument(
        "--old-checkpoint",
        required=True,
        type=Path,
        help="Path to base TF_v6 model checkpoint file (.pth).",
    )
    parser.add_argument(
        "--finetuned-checkpoint",
        required=True,
        type=Path,
        help="Path to finetuned policy checkpoint file (.pth).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/carla_leaderboard2/data"),
        help="Root containing route folders (with rgb/lidar/metas/radar).",
    )
    parser.add_argument(
        "--config-checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Checkpoint folder containing config.json used to build TF_v6."
            " Defaults to parent folder of --old-checkpoint."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference across frames.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/local_evaluation"),
        help="Directory to save plot + stats.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="speed_distribution_shift_all_routes",
        help="Output file stem (without extension).",
    )
    parser.add_argument(
        "--max-scatter-points",
        type=int,
        default=50000,
        help="Maximum number of points used for scatter rendering.",
    )
    parser.add_argument(
        "--path-heatmap-bins",
        type=int,
        default=100,
        help="Number of bins per axis for path heatmaps.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config_dir = args.config_checkpoint_dir
    if config_dir is None:
        config_dir = args.old_checkpoint.parent

    if not (config_dir / "config.json").exists():
        raise FileNotFoundError(
            f"config.json not found in {config_dir}. Pass --config-checkpoint-dir explicitly."
        )

    if not args.old_checkpoint.exists():
        raise FileNotFoundError(f"Missing old checkpoint: {args.old_checkpoint}")
    if not args.finetuned_checkpoint.exists():
        raise FileNotFoundError(
            f"Missing finetuned checkpoint: {args.finetuned_checkpoint}"
        )
    if not args.data_root.exists():
        raise FileNotFoundError(f"Missing data root: {args.data_root}")

    route_dirs = _discover_route_dirs(args.data_root)
    if not route_dirs:
        raise FileNotFoundError(
            f"No route folders found under {args.data_root} (expected rgb/lidar/metas dirs)."
        )

    device = torch.device(args.device)
    training_config = load_training_config(str(config_dir))

    print(f"[speed-shift] loading old model: {args.old_checkpoint}", flush=True)
    old_model = _load_tfv6_model(args.old_checkpoint, training_config, device)

    print(
        f"[speed-shift] loading finetuned model: {args.finetuned_checkpoint}",
        flush=True,
    )
    finetuned_model = _load_tfv6_model(
        args.finetuned_checkpoint, training_config, device
    )

    all_old: list[float] = []
    all_new: list[float] = []
    per_route: list[dict[str, float | int | str]] = []
    failed_frames: list[tuple[str, int, str]] = []
    all_old_path_points: list[np.ndarray] = []
    all_new_path_points: list[np.ndarray] = []

    with torch.no_grad():
        for route_idx, route_dir in enumerate(route_dirs, start=1):
            metas = sorted((route_dir / "metas").glob("*.pkl"))
            frame_indices = [int(path.stem) for path in metas]

            route_old: list[float] = []
            route_new: list[float] = []

            i = 0
            while i < len(frame_indices):
                batch_indices = frame_indices[i : i + args.batch_size]
                obs_batch: list[dict[str, torch.Tensor]] = []
                for frame_idx in batch_indices:
                    try:
                        _, obs = build_real_obs(
                            route_dir, frame_idx, training_config, device
                        )
                        obs_batch.append(obs)
                    except Exception as exc:  # noqa: BLE001
                        failed_frames.append((str(route_dir), frame_idx, str(exc)))

                if obs_batch:
                    obs_cat = _collate_obs(obs_batch)
                    autocast_enabled = (
                        training_config.use_mixed_precision_training
                        and device.type == "cuda"
                    )
                    with torch.amp.autocast(
                        device_type=device.type,
                        dtype=training_config.torch_float_type,
                        enabled=autocast_enabled,
                    ):
                        pred_old = old_model(obs_cat, skip_perception_heads=True)
                        pred_new = finetuned_model(obs_cat, skip_perception_heads=True)
                    speeds_old = (
                        pred_old.pred_target_speed_scalar.detach().cpu().numpy()
                    )
                    speeds_new = (
                        pred_new.pred_target_speed_scalar.detach().cpu().numpy()
                    )
                    route_old.extend(speeds_old.astype(np.float32).tolist())
                    route_new.extend(speeds_new.astype(np.float32).tolist())

                    old_paths = _extract_path_tensor(pred_old)
                    new_paths = _extract_path_tensor(pred_new)
                    if old_paths is not None:
                        old_points = (
                            old_paths.detach().cpu().float().numpy().reshape(-1, 2)
                        )
                        all_old_path_points.append(old_points.astype(np.float32))
                    if new_paths is not None:
                        new_points = (
                            new_paths.detach().cpu().float().numpy().reshape(-1, 2)
                        )
                        all_new_path_points.append(new_points.astype(np.float32))

                i += args.batch_size

            if route_old:
                old_arr = np.array(route_old, dtype=np.float32)
                new_arr = np.array(route_new, dtype=np.float32)
                delta = new_arr - old_arr
                per_route.append(
                    {
                        "route_dir": str(route_dir),
                        "n": int(len(old_arr)),
                        "old_mean": float(old_arr.mean()),
                        "new_mean": float(new_arr.mean()),
                        "mean_shift_new_minus_old": float(delta.mean()),
                        "abs_mean_shift": float(np.abs(delta).mean()),
                        "rmse_shift": float(np.sqrt(np.mean(delta**2))),
                    }
                )
                all_old.extend(route_old)
                all_new.extend(route_new)

            print(
                f"[speed-shift] route {route_idx}/{len(route_dirs)} "
                f"frames={len(frame_indices)} ok={len(route_old)}",
                flush=True,
            )

    if not all_old:
        raise RuntimeError("No successful samples collected from discovered routes.")

    old = np.array(all_old, dtype=np.float32)
    new = np.array(all_new, dtype=np.float32)
    delta = new - old

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / f"{args.output_stem}.png"
    scatter_path = args.output_dir / f"{args.output_stem}_scatter.png"
    path_heatmap_path = args.output_dir / f"{args.output_stem}_path_heatmaps.png"
    stats_path = args.output_dir / f"{args.output_stem}.txt"
    per_route_path = args.output_dir / f"{args.output_stem}_per_route.csv"

    bins = np.linspace(0, max(float(old.max()), float(new.max()), 1.0), 40)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(
        old, bins=bins, alpha=0.55, density=True, color="#4C78A8", label="TF_v6"
    )
    axes[0].hist(
        new,
        bins=bins,
        alpha=0.55,
        density=True,
        color="#F58518",
        label="Finetuned Policy",
    )
    axes[0].axvline(old.mean(), color="#4C78A8", linestyle="--", linewidth=1)
    axes[0].axvline(new.mean(), color="#F58518", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted target speed (m/s)")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"aggregated across {len(per_route)} routes")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].hist(delta, bins=40, alpha=0.9, density=True, color="#54A24B")
    axes[1].axvline(delta.mean(), color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Finetuned Policy - TF_v6 (m/s)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("shift distribution")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    scatter_old, scatter_new, scatter_delta = _sample_for_scatter(
        old, new, delta, max(1000, int(args.max_scatter_points))
    )
    speed_min = min(float(old.min()), float(new.min()))
    speed_max = max(float(old.max()), float(new.max()))
    fig_scatter, axes_scatter = plt.subplots(1, 2, figsize=(12, 5))
    axes_scatter[0].scatter(
        scatter_old,
        scatter_new,
        s=4,
        alpha=0.18,
        c="#2F4B7C",
        linewidths=0,
    )
    axes_scatter[0].plot(
        [speed_min, speed_max], [speed_min, speed_max], "--", color="black", linewidth=1
    )
    axes_scatter[0].set_xlabel("TF_v6 speed (m/s)")
    axes_scatter[0].set_ylabel("Finetuned speed (m/s)")
    axes_scatter[0].set_title("same-sample prediction relation")
    axes_scatter[0].grid(alpha=0.25, linewidth=0.5)

    hb = axes_scatter[1].hexbin(
        old,
        new,
        C=np.abs(delta),
        reduce_C_function=np.mean,
        gridsize=65,
        mincnt=1,
        cmap="magma",
    )
    axes_scatter[1].plot(
        [speed_min, speed_max], [speed_min, speed_max], "--", color="white", linewidth=1
    )
    axes_scatter[1].set_xlabel("TF_v6 speed (m/s)")
    axes_scatter[1].set_ylabel("Finetuned speed (m/s)")
    axes_scatter[1].set_title("where absolute change is largest")
    cb = fig_scatter.colorbar(hb, ax=axes_scatter[1])
    cb.set_label("mean |new - old| (m/s)")
    fig_scatter.tight_layout()
    fig_scatter.savefig(scatter_path, dpi=170)
    plt.close(fig_scatter)

    if all_old_path_points and all_new_path_points:
        old_path_points = np.concatenate(all_old_path_points, axis=0)
        new_path_points = np.concatenate(all_new_path_points, axis=0)
        old_px, old_py = _build_path_plot_points(old_path_points)
        new_px, new_py = _build_path_plot_points(new_path_points)

        all_x = np.concatenate([old_px, new_px], axis=0)
        all_y = np.concatenate([old_py, new_py], axis=0)
        x_min, x_max = np.percentile(all_x, [1.0, 99.0])
        y_min, y_max = np.percentile(all_y, [1.0, 99.0])
        x_pad = 0.05 * max(1e-3, (x_max - x_min))
        y_pad = 0.10 * max(1e-3, (y_max - y_min))
        x_edges = np.linspace(
            x_min - x_pad, x_max + x_pad, int(args.path_heatmap_bins) + 1
        )
        y_edges = np.linspace(
            y_min - y_pad, y_max + y_pad, int(args.path_heatmap_bins) + 1
        )

        old_hist_raw, _, _ = np.histogram2d(old_py, old_px, bins=[y_edges, x_edges])
        new_hist_raw, _, _ = np.histogram2d(new_py, new_px, bins=[y_edges, x_edges])
        old_hist = np.log1p(_smooth_hist2d(old_hist_raw, passes=2))
        new_hist = np.log1p(_smooth_hist2d(new_hist_raw, passes=2))
        old_vmax = max(1e-6, float(np.percentile(old_hist, 99.2)))
        new_vmax = max(1e-6, float(np.percentile(new_hist, 99.2)))

        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
        fig_paths, axes_paths = plt.subplots(
            1,
            2,
            figsize=(6.5, 5),
            gridspec_kw={"wspace": 0.04},
            constrained_layout=True,
        )
        im_old = axes_paths[0].imshow(
            old_hist,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="Blues",
            interpolation="bilinear",
            vmax=old_vmax,
        )
        axes_paths[0].set_title("TF_v6 path density")
        axes_paths[0].set_xlabel("lateral y (m)")
        axes_paths[0].set_ylabel("forward x (m, up)")
        axes_paths[0].axhline(0.0, color="white", linewidth=0.7, alpha=0.8)
        axes_paths[0].axvline(0.0, color="white", linewidth=0.7, alpha=0.8)
        old_levels = np.linspace(old_hist.min(), old_vmax, 6)[1:]
        axes_paths[0].contour(
            old_hist,
            levels=old_levels,
            extent=extent,
            origin="lower",
            colors="white",
            linewidths=0.5,
            alpha=0.45,
        )
        fig_paths.colorbar(im_old, ax=axes_paths[0], fraction=0.038, pad=0.015)

        im_new = axes_paths[1].imshow(
            new_hist,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="Oranges",
            interpolation="bilinear",
            vmax=new_vmax,
        )
        axes_paths[1].set_title("PPO finetuned path density")
        axes_paths[1].set_xlabel("lateral y (m)")
        axes_paths[1].set_ylabel("forward x (m, up)")
        axes_paths[1].axhline(0.0, color="white", linewidth=0.7, alpha=0.8)
        axes_paths[1].axvline(0.0, color="white", linewidth=0.7, alpha=0.8)
        new_levels = np.linspace(new_hist.min(), new_vmax, 6)[1:]
        axes_paths[1].contour(
            new_hist,
            levels=new_levels,
            extent=extent,
            origin="lower",
            colors="white",
            linewidths=0.5,
            alpha=0.45,
        )
        fig_paths.colorbar(im_new, ax=axes_paths[1], fraction=0.038, pad=0.015)
        fig_paths.savefig(path_heatmap_path, dpi=170)
        plt.close(fig_paths)

    with per_route_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "route_dir,n,old_mean,new_mean,mean_shift_new_minus_old,abs_mean_shift,rmse_shift\n"
        )
        for row in per_route:
            handle.write(
                f"{row['route_dir']},{row['n']},{row['old_mean']:.6f},"
                f"{row['new_mean']:.6f},{row['mean_shift_new_minus_old']:.6f},"
                f"{row['abs_mean_shift']:.6f},{row['rmse_shift']:.6f}\n"
            )

    lines: list[str] = []
    lines.append(f"old_checkpoint={args.old_checkpoint}")
    lines.append(f"finetuned_checkpoint={args.finetuned_checkpoint}")
    lines.append(f"config_checkpoint_dir={config_dir}")
    lines.append(f"data_root={args.data_root}")
    lines.append(f"n_routes_total={len(route_dirs)}")
    lines.append(f"n_routes_used={len(per_route)}")
    lines.append(f"n_samples_total={len(old)}")
    lines.append(f"n_failed_frames={len(failed_frames)}")
    lines.append(f"old_mean={old.mean():.6f}")
    lines.append(f"new_mean={new.mean():.6f}")
    lines.append(f"avg_difference_new_minus_old={delta.mean():.6f}")
    lines.append(f"avg_abs_difference={np.abs(delta).mean():.6f}")
    lines.append(f"rmse_difference={np.sqrt(np.mean(delta**2)):.6f}")
    lines.append(f"old_std={old.std():.6f}")
    lines.append(f"new_std={new.std():.6f}")
    for q in [1, 5, 25, 50, 75, 95, 99]:
        lines.append(
            f"old_p{q}={np.percentile(old, q):.6f} "
            f"new_p{q}={np.percentile(new, q):.6f} "
            f"delta_p{q}={np.percentile(delta, q):.6f}"
        )

    sorted_routes = sorted(
        per_route,
        key=lambda row: abs(float(row["mean_shift_new_minus_old"])),
        reverse=True,
    )
    lines.append("top_routes_by_abs_mean_shift:")
    for row in sorted_routes[:8]:
        lines.append(
            f"{row['route_dir']} | n={row['n']} | "
            f"shift={row['mean_shift_new_minus_old']:.6f} | "
            f"old={row['old_mean']:.6f} | new={row['new_mean']:.6f}"
        )

    if failed_frames:
        lines.append("failed_examples:")
        for route_dir, frame_idx, err in failed_frames[:20]:
            lines.append(f"{route_dir} frame={frame_idx} err={err[:120]}")
    lines.append(f"plot_speed_distribution={plot_path}")
    lines.append(f"plot_speed_scatter={scatter_path}")
    if all_old_path_points and all_new_path_points:
        lines.append(f"plot_path_heatmaps={path_heatmap_path}")
        lines.append(
            f"path_points_old={sum(int(points.shape[0]) for points in all_old_path_points)}"
        )
        lines.append(
            f"path_points_new={sum(int(points.shape[0]) for points in all_new_path_points)}"
        )
    else:
        lines.append("path_plots_skipped=true (no path head predictions available)")

    stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[speed-shift] plot_file={plot_path}")
    print(f"[speed-shift] scatter_file={scatter_path}")
    if all_old_path_points and all_new_path_points:
        print(f"[speed-shift] path_heatmap_file={path_heatmap_path}")
    print(f"[speed-shift] stats_file={stats_path}")
    print(f"[speed-shift] per_route_file={per_route_path}")


if __name__ == "__main__":
    main()
