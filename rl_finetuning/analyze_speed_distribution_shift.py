#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jsonpickle
import matplotlib
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from torch import nn

from lead.inference.config_closed_loop import ClosedLoopConfig
from lead.tfv6.tfv6 import TFv6
from rl_finetuning.tfv6_rl.action_codec import ActionCodec, infer_action_head_usage
from rl_finetuning.tfv6_rl.action_noise_codec import build_route_basis
from rl_finetuning.tfv6_rl.dry_run import build_real_obs
from rl_finetuning.tfv6_rl.policy_tfv6_ppo import TFv6PPOPolicy, load_training_config

matplotlib.use("Agg")
import matplotlib.pyplot as plt

USE_TEX = False
SERIF = ["cmr10", "CMU Serif", "Latin Modern Roman", "STIXGeneral", "DejaVu Serif"]
PALETTE = {
    "blue": "#4477AA",
    "red": "#EE6677",
    "green": "#228833",
    "yellow": "#CCBB44",
    "cyan": "#66CCEE",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
}
CYCLE = [
    PALETTE["blue"],
    PALETTE["red"],
    PALETTE["green"],
    PALETTE["purple"],
    PALETTE["cyan"],
    PALETTE["yellow"],
]
BASE_FONT_SIZE = 11


def _apply_result_graphs_style() -> None:
    matplotlib.rcParams.update(
        {
            "text.usetex": USE_TEX,
            "font.family": "serif",
            "font.serif": SERIF,
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "axes.formatter.use_mathtext": True,
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": BASE_FONT_SIZE + 1,
            "axes.labelsize": BASE_FONT_SIZE,
            "xtick.labelsize": BASE_FONT_SIZE - 1,
            "ytick.labelsize": BASE_FONT_SIZE - 1,
            "legend.fontsize": BASE_FONT_SIZE - 1,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#444444",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.grid": True,
            "grid.color": "#CCCCCC",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "lines.linewidth": 1.8,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "figure.figsize": (5.9, 3.6),
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.constrained_layout.use": True,
            "axes.prop_cycle": matplotlib.cycler(color=CYCLE),
        }
    )


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


class LegacyResidualPPOModel(nn.Module):
    """Compatibility path for older residual PPO checkpoints.

    These checkpoints store residual_queries + residual_head.* instead of the newer
    residual_cnn/status_proj/residual_out modules.  The residual action is still a
    low-rank correction added to frozen TFv6 route and target-speed predictions.
    """

    def __init__(
        self,
        checkpoint_file: Path,
        state_dict: dict[str, torch.Tensor],
        training_config,
        rl_config,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.checkpoint_file = checkpoint_file
        self.training_config = training_config
        self.rl_config = rl_config
        self.device = device
        self.tfv6 = _load_tfv6_model(checkpoint_file, training_config, device)

        self.speed_temperature = float(getattr(rl_config, "speed_temperature", 1.0))
        self.residual_route_rank = int(getattr(rl_config, "residual_route_rank", 3))
        self.residual_alpha = float(getattr(rl_config, "residual_alpha", 0.15))
        self.residual_alpha_speed = float(
            getattr(rl_config, "residual_alpha_speed", self.residual_alpha)
        )
        self.disable_residual_route = bool(
            getattr(rl_config, "disable_residual_route", False)
        )

        self.action_codec = self._build_residual_action_codec(training_config)
        route_basis = build_route_basis(
            num_route_points=self.action_codec.num_route_points,
            route_dim=self.action_codec.route_dim,
            rank=self.residual_route_rank,
        )
        self.register_buffer("residual_route_basis", route_basis, persistent=False)
        speed_bins = torch.tensor(
            training_config.target_speed_classes, dtype=torch.float32
        ) / float(training_config.max_speed)
        self.register_buffer("residual_speed_bins", speed_bins, persistent=False)

        queries = state_dict["residual_queries"].detach().clone().float()
        self.residual_queries = nn.Parameter(queries)
        self.residual_head = self._build_legacy_residual_head(state_dict)
        residual_head_state = {
            key.replace("residual_head.", "", 1): value
            for key, value in state_dict.items()
            if key.startswith("residual_head.")
        }
        self.residual_head.load_state_dict(residual_head_state, strict=True)
        self.to(device)
        self.eval()

    @staticmethod
    def _build_residual_action_codec(training_config) -> ActionCodec:
        closed_loop_config = ClosedLoopConfig(raise_error_on_missing_key=False)
        use_route, _use_waypoints, use_target_speed = infer_action_head_usage(
            training_config,
            steer_modality=closed_loop_config.steer_modality,
            throttle_modality=closed_loop_config.throttle_modality,
            brake_modality=closed_loop_config.brake_modality,
        )
        return ActionCodec(
            training_config,
            use_route=use_route,
            use_waypoints=False,
            use_target_speed=use_target_speed,
        )

    @staticmethod
    def _build_legacy_residual_head(
        state_dict: dict[str, torch.Tensor],
    ) -> nn.Sequential:
        in_dim = int(state_dict["residual_head.0.weight"].shape[0])
        hidden_dim = int(state_dict["residual_head.1.bias"].shape[0])
        out_dim = int(state_dict["residual_head.4.bias"].shape[0])
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def _legacy_residual_features(self, base_action: torch.Tensor) -> torch.Tensor:
        kv = getattr(self.tfv6.planning_decoder, "kv", None)
        if kv is None:
            raise RuntimeError("Planning decoder context tokens not available.")
        kv = kv.detach().float()
        mean_pooled = kv.mean(dim=1)
        queries = self.residual_queries.to(device=kv.device, dtype=kv.dtype)
        scale = math.sqrt(float(queries.shape[-1]))
        scores = torch.einsum("qd,bnd->bqn", queries, kv) / scale
        weights = scores.softmax(dim=-1)
        attention_pooled = torch.einsum("bqn,bnd->bqd", weights, kv)
        spatial = torch.cat([mean_pooled, attention_pooled.flatten(start_dim=1)], dim=1)
        return torch.cat([spatial, base_action.float()], dim=1).float()

    def predict(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        predictions = self.tfv6(obs, skip_perception_heads=True)
        route = predictions.pred_route
        if route is None:
            raise RuntimeError("Legacy residual policy requires pred_route from TFv6.")
        if predictions.pred_target_speed_distribution is None:
            raise RuntimeError(
                "Legacy residual policy requires TFv6 target-speed logits."
            )

        route = route.float()
        batch_size = route.shape[0]
        route_scale = self.action_codec.route_scale.to(
            device=route.device, dtype=route.dtype
        )
        route_norm = (route / route_scale).reshape(batch_size, -1)

        logits = predictions.pred_target_speed_distribution.float()
        speed_probs = torch.softmax(logits / self.speed_temperature, dim=-1)
        speed_bins = self.residual_speed_bins.to(
            device=speed_probs.device, dtype=speed_probs.dtype
        )
        speed_expected = (speed_probs * speed_bins).sum(dim=-1, keepdim=True)
        base_action = torch.cat([route_norm, speed_expected], dim=1)

        residual_features = self._legacy_residual_features(base_action)
        coeff_means = self.residual_head(residual_features)
        rank = self.residual_route_rank
        route_coeff_means = coeff_means[:, :rank]
        speed_coeff_mean = coeff_means[:, rank : rank + 1]
        basis = self.residual_route_basis.to(
            device=route_coeff_means.device, dtype=route_coeff_means.dtype
        )
        delta_route_mean = (
            torch.zeros_like(route_norm)
            if self.disable_residual_route
            else self.residual_alpha * route_coeff_means @ basis.t()
        )
        delta_speed_mean = self.residual_alpha_speed * speed_coeff_mean
        action_mean = torch.cat(
            [route_norm + delta_route_mean, speed_expected + delta_speed_mean], dim=1
        )
        decoded_route, decoded_waypoints, target_speed = self.action_codec.decode(
            action_mean.float()
        )
        path = decoded_route if decoded_route is not None else decoded_waypoints
        if target_speed is None:
            raise RuntimeError(
                f"Legacy residual policy {self.checkpoint_file} did not decode speed."
            )
        return target_speed, path


@dataclass
class LoadedPredictor:
    checkpoint_file: Path
    kind: str
    model: torch.nn.Module

    def predict(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.kind == "ppo_residual_legacy":
            return self.model.predict(obs)

        if self.kind == "ppo_residual":
            outputs = self.model(obs, sample_type="mean")
            action_mean = outputs[5]
            route, waypoints, target_speed = self.model.action_codec.decode(
                action_mean.float()
            )
            path = route if route is not None else waypoints
            if target_speed is None:
                raise RuntimeError(
                    f"Residual policy {self.checkpoint_file} did not decode a target speed."
                )
            return target_speed, path

        predictions = self.model(obs, skip_perception_heads=True)
        return predictions.pred_target_speed_scalar, _extract_path_tensor(predictions)


def _resolve_checkpoint_file(path: Path, prefix: str = "model") -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    checkpoints = sorted(path.glob(f"{prefix}*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No {prefix}*.pth checkpoint files found in {path}")
    return checkpoints[-1]


def _load_checkpoint_state(checkpoint_file: Path, device: torch.device | str):
    return torch.load(checkpoint_file, map_location=device, weights_only=True)


def _state_dict_keys(state) -> list[str]:
    if not isinstance(state, dict):
        return []
    return [str(key) for key in state.keys()]


def _is_ppo_residual_state(state, rl_config) -> bool:
    config_says_residual = bool(getattr(rl_config, "use_residual_policy", False))
    residual_prefixes = (
        "residual_",
        "residual_out.",
        "residual_cnn.",
        "residual_status_proj.",
        "residual_query_attn.",
        "residual_cross_attn.",
    )
    keys = _state_dict_keys(state)
    state_says_residual = any(
        key.startswith(residual_prefixes)
        or key in {"residual_queries", "residual_log_std"}
        for key in keys
    )
    return config_says_residual or state_says_residual


def _is_td3_residual_state(state) -> bool:
    keys = set(_state_dict_keys(state))
    return any(key.startswith("actor_") for key in keys) and any(
        key.startswith("qf") for key in keys
    )


def _is_legacy_ppo_residual_state(state) -> bool:
    keys = set(_state_dict_keys(state))
    return "residual_queries" in keys and any(
        key.startswith("residual_head.") for key in keys
    )


def _has_tfv6_prefix(state) -> bool:
    return any(key.startswith("tfv6.") for key in _state_dict_keys(state))


def _load_rl_config_for_checkpoint(checkpoint_file: Path):
    config_file = checkpoint_file.parent / "config.json"
    if not config_file.exists():
        return None

    text = config_file.read_text(encoding="utf-8")
    loaded = jsonpickle.decode(text)
    if isinstance(loaded, dict):
        return SimpleNamespace(**loaded)
    return loaded


def _select_tfv6_checkpoint_dir(rl_config, fallback_dir: Path) -> Path:
    configured = getattr(rl_config, "tfv6_checkpoint", None)
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            candidate = candidate.parent
        if candidate.exists():
            return candidate
    return fallback_dir


def _set_config_attr(config, name: str, value) -> None:
    try:
        setattr(config, name, value)
    except Exception:
        if hasattr(config, "__dict__"):
            config.__dict__[name] = value
        else:
            raise


def _prepare_residual_state_for_policy(rl_config, state):
    residual_out_bias = (
        state.get("residual_out.bias") if isinstance(state, dict) else None
    )
    if residual_out_bias is None:
        return state

    prepared = dict(state)
    out_dim = int(residual_out_bias.shape[0])
    configured_rank = int(getattr(rl_config, "residual_route_rank", out_dim - 1))
    mean_dim = configured_rank + 1

    if out_dim == mean_dim:
        return prepared

    # Older CNN residual checkpoints used residual_out to emit
    # [route_means..., speed_mean, route_log_stds..., speed_log_std].
    # Current TFv6PPOPolicy expects only the means and keeps log-std separately.
    if out_dim == 2 * mean_dim and "residual_log_std" not in prepared:
        print(
            "[speed-shift] detected legacy residual_out layout "
            f"with {mean_dim} means + {mean_dim} log_stds; "
            "using mean rows only for deterministic analysis.",
            flush=True,
        )
        prepared["residual_out.weight"] = prepared["residual_out.weight"][
            :mean_dim
        ].clone()
        prepared["residual_out.bias"] = prepared["residual_out.bias"][:mean_dim].clone()
        return prepared

    if "residual_log_std" in prepared and out_dim == int(
        prepared["residual_log_std"].numel()
    ):
        checkpoint_rank = out_dim - 1
        if checkpoint_rank != configured_rank:
            print(
                "[speed-shift] residual_route_rank mismatch: "
                f"config={configured_rank}, checkpoint={checkpoint_rank}; "
                "using checkpoint shape for analysis.",
                flush=True,
            )
            _set_config_attr(rl_config, "residual_route_rank", checkpoint_rank)
        return prepared

    raise RuntimeError(
        "Cannot infer residual_out layout from checkpoint: "
        f"config residual_route_rank={configured_rank}, residual_out_dim={out_dim}, "
        f"has residual_log_std={'residual_log_std' in prepared}."
    )


def _load_policy_state_compatible(
    policy: TFv6PPOPolicy, state_dict: dict[str, torch.Tensor]
) -> None:
    incompatible = policy.load_state_dict(state_dict, strict=False)
    allowed_missing = {"residual_log_std"}
    missing = [key for key in incompatible.missing_keys if key not in allowed_missing]
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Residual PPO checkpoint is not compatible with the reconstructed policy. "
            f"missing={missing}, unexpected={unexpected}"
        )


def _load_predictor(
    checkpoint_path: Path,
    training_config,
    device: torch.device,
    fallback_tfv6_checkpoint_dir: Path,
) -> LoadedPredictor:
    checkpoint_file = _resolve_checkpoint_file(checkpoint_path)
    state_cpu = _load_checkpoint_state(checkpoint_file, "cpu")
    rl_config = _load_rl_config_for_checkpoint(checkpoint_file)

    if _is_td3_residual_state(state_cpu):
        raise NotImplementedError(
            "This analysis script detected a TD3-style residual checkpoint. "
            "It currently supports plain TFv6 and PPO residual sensor-agent checkpoints."
        )

    if _is_ppo_residual_state(state_cpu, rl_config):
        if rl_config is None:
            raise FileNotFoundError(
                f"Residual PPO checkpoint {checkpoint_file} needs its training config.json "
                "in the same folder so the residual policy can be reconstructed."
            )
        if _is_legacy_ppo_residual_state(state_cpu):
            model = LegacyResidualPPOModel(
                checkpoint_file, state_cpu, training_config, rl_config, device
            )
            return LoadedPredictor(checkpoint_file, "ppo_residual_legacy", model)

        prepared_state = _prepare_residual_state_for_policy(rl_config, state_cpu)
        tfv6_checkpoint_dir = _select_tfv6_checkpoint_dir(
            rl_config, fallback_tfv6_checkpoint_dir
        )
        policy = TFv6PPOPolicy(
            tfv6_checkpoint=str(tfv6_checkpoint_dir),
            device=device,
            rl_config=rl_config,
            train_planning_decoder_only=getattr(
                rl_config, "train_planning_decoder_only", True
            ),
        ).to(device)
        _load_policy_state_compatible(policy, prepared_state)
        policy.eval()
        return LoadedPredictor(checkpoint_file, "ppo_residual", policy)

    model = _load_tfv6_model(checkpoint_file, training_config, device)
    kind = "ppo_tfv6_state" if _has_tfv6_prefix(state_cpu) else "tfv6"
    return LoadedPredictor(checkpoint_file, kind, model)


def _collate_obs(obs_list: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = obs_list[0].keys()
    return {key: torch.cat([obs[key] for obs in obs_list], dim=0) for key in keys}


def _prepare_obs_for_inference(
    obs: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    if device.type == "cuda":
        return obs
    return {
        key: value.float() if torch.is_floating_point(value) else value
        for key, value in obs.items()
    }


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
    # For plotting with ego front upward, use horizontal=y and vertical=x.
    x_forward = path_points_xy[:, 0]
    y_lateral = path_points_xy[:, 1]
    return y_lateral, x_forward


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
        description=(
            "Aggregate old-vs-finetuned target-speed distribution shift over route data. "
            "The finetuned checkpoint can be plain TFv6 weights or a PPO residual policy."
        )
    )
    parser.add_argument(
        "--old-checkpoint",
        required=True,
        type=Path,
        help="Path to base TF_v6 checkpoint file or checkpoint folder.",
    )
    parser.add_argument(
        "--finetuned-checkpoint",
        required=True,
        type=Path,
        help="Path to finetuned policy checkpoint file or checkpoint folder.",
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
            " Defaults to --old-checkpoint when it is a folder, otherwise its parent."
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
    _apply_result_graphs_style()
    args = _parse_args()

    config_dir = args.config_checkpoint_dir
    if config_dir is None:
        config_dir = (
            args.old_checkpoint
            if args.old_checkpoint.is_dir()
            else args.old_checkpoint.parent
        )

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
    old_predictor = _load_predictor(
        args.old_checkpoint, training_config, device, config_dir
    )
    print(
        f"[speed-shift] old kind={old_predictor.kind} "
        f"checkpoint={old_predictor.checkpoint_file}",
        flush=True,
    )

    print(
        f"[speed-shift] loading finetuned model: {args.finetuned_checkpoint}",
        flush=True,
    )
    finetuned_predictor = _load_predictor(
        args.finetuned_checkpoint, training_config, device, config_dir
    )
    print(
        f"[speed-shift] finetuned kind={finetuned_predictor.kind} "
        f"checkpoint={finetuned_predictor.checkpoint_file}",
        flush=True,
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
                    obs_cat = _prepare_obs_for_inference(
                        _collate_obs(obs_batch), device
                    )
                    autocast_enabled = (
                        training_config.use_mixed_precision_training
                        and device.type == "cuda"
                    )
                    with torch.amp.autocast(
                        device_type=device.type,
                        dtype=training_config.torch_float_type,
                        enabled=autocast_enabled,
                    ):
                        speeds_old_t, old_paths = old_predictor.predict(obs_cat)
                        speeds_new_t, new_paths = finetuned_predictor.predict(obs_cat)
                    speeds_old = speeds_old_t.detach().cpu().float().numpy().reshape(-1)
                    speeds_new = speeds_new_t.detach().cpu().float().numpy().reshape(-1)
                    route_old.extend(speeds_old.astype(np.float32).tolist())
                    route_new.extend(speeds_new.astype(np.float32).tolist())
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
        old,
        bins=bins,
        alpha=0.55,
        density=True,
        color=PALETTE["blue"],
        label="TF_v6",
        zorder=2,
    )
    axes[0].hist(
        new,
        bins=bins,
        alpha=0.55,
        density=True,
        color=PALETTE["red"],
        label="Finetuned Policy",
        zorder=2,
    )
    axes[0].axvline(
        old.mean(), color=PALETTE["blue"], linestyle="--", linewidth=1, zorder=3
    )
    axes[0].axvline(
        new.mean(), color=PALETTE["red"], linestyle="--", linewidth=1, zorder=3
    )
    axes[0].set_xlabel("Predicted target speed (m/s)")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"aggregated across {len(per_route)} routes")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].hist(
        delta, bins=40, alpha=0.9, density=True, color=PALETTE["green"], zorder=2
    )
    axes[1].axvline(delta.mean(), color="black", linestyle="--", linewidth=1, zorder=3)
    axes[1].set_xlabel("Finetuned Policy - TF_v6 (m/s)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("shift distribution")

    fig.savefig(plot_path)
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
        c=PALETTE["blue"],
        linewidths=0,
        zorder=3,
    )
    axes_scatter[0].plot(
        [speed_min, speed_max],
        [speed_min, speed_max],
        "--",
        color="black",
        linewidth=1,
        zorder=4,
    )
    axes_scatter[0].set_xlabel("TF_v6 speed (m/s)")
    axes_scatter[0].set_ylabel("Finetuned speed (m/s)")
    axes_scatter[0].set_title("same-sample prediction relation")
    axes_scatter[0].grid(alpha=0.25, linewidth=0.5, zorder=0)

    hb = axes_scatter[1].hexbin(
        old,
        new,
        C=np.abs(delta),
        reduce_C_function=np.mean,
        gridsize=65,
        mincnt=1,
        cmap="magma",
        zorder=3,
    )
    axes_scatter[1].plot(
        [speed_min, speed_max],
        [speed_min, speed_max],
        "--",
        color="white",
        linewidth=1,
        zorder=4,
    )
    axes_scatter[1].set_xlabel("TF_v6 speed (m/s)")
    axes_scatter[1].set_ylabel("Finetuned speed (m/s)")
    axes_scatter[1].set_title("where absolute change is largest")
    cb = fig_scatter.colorbar(hb, ax=axes_scatter[1])
    cb.set_label("mean |new - old| (m/s)")
    fig_scatter.savefig(scatter_path)
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
        old_vmax = max(1e-6, float(np.percentile(old_hist, 98.8)))
        new_vmax = max(1e-6, float(np.percentile(new_hist, 98.8)))

        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
        old_cmap = LinearSegmentedColormap.from_list(
            "result_graphs_blue_density", ["#FFFFFF", PALETTE["blue"]]
        )
        new_cmap = LinearSegmentedColormap.from_list(
            "result_graphs_red_density", ["#FFFFFF", PALETTE["red"]]
        )
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
            cmap=old_cmap,
            interpolation="bilinear",
            vmax=old_vmax,
            zorder=2,
        )
        axes_paths[0].set_title("TF_v6 path density")
        axes_paths[0].set_xlabel("lateral y (m)")
        axes_paths[0].set_ylabel("forward x (m)")
        axes_paths[0].axhline(0.0, color="white", linewidth=0.7, alpha=0.8, zorder=3)
        axes_paths[0].axvline(0.0, color="white", linewidth=0.7, alpha=0.8, zorder=3)
        old_levels = np.linspace(old_hist.min(), old_vmax, 6)[1:]
        axes_paths[0].contour(
            old_hist,
            levels=old_levels,
            extent=extent,
            origin="lower",
            colors="white",
            linewidths=0.5,
            alpha=0.45,
            zorder=4,
        )
        fig_paths.colorbar(im_old, ax=axes_paths[0], fraction=0.038, pad=0.015)

        im_new = axes_paths[1].imshow(
            new_hist,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=new_cmap,
            interpolation="bilinear",
            vmax=new_vmax,
            zorder=2,
        )
        axes_paths[1].set_title("PPO finetuned path density")
        axes_paths[1].set_xlabel("lateral y (m)")
        axes_paths[1].set_ylabel("forward x (m)")
        axes_paths[1].axhline(0.0, color="white", linewidth=0.7, alpha=0.8, zorder=3)
        axes_paths[1].axvline(0.0, color="white", linewidth=0.7, alpha=0.8, zorder=3)
        new_levels = np.linspace(new_hist.min(), new_vmax, 6)[1:]
        axes_paths[1].contour(
            new_hist,
            levels=new_levels,
            extent=extent,
            origin="lower",
            colors="white",
            linewidths=0.5,
            alpha=0.45,
            zorder=4,
        )
        fig_paths.colorbar(im_new, ax=axes_paths[1], fraction=0.038, pad=0.015)
        fig_paths.savefig(path_heatmap_path)
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
    lines.append(f"old_checkpoint_resolved={old_predictor.checkpoint_file}")
    lines.append(f"old_predictor_kind={old_predictor.kind}")
    lines.append(f"finetuned_checkpoint={args.finetuned_checkpoint}")
    lines.append(f"finetuned_checkpoint_resolved={finetuned_predictor.checkpoint_file}")
    lines.append(f"finetuned_predictor_kind={finetuned_predictor.kind}")
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
