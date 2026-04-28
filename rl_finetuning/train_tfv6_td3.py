"""
TD3 training loop for TFv6 residual RL. Single-process, single-GPU, off-policy.
Adapted from CleanRL td3_continuous_action.py, verified against SB3 TD3 and
ResFiT (Ankile et al. 2024, amazon-science/residual-offpolicy-rl).

Key algorithmic details:
  - Twin Q-networks (Fujimoto 2018) — min target for Bellman update
  - Delayed actor updates every ``policy_delay`` critic steps
  - Target policy smoothing with clipped Gaussian noise
  - Critic warmup (ResFiT): actor frozen for ``critic_warmup_steps`` critic
    updates after ``learning_starts``
  - Layer norm in Q-head (RLPD, Ball 2023) — already in PPO value_head arch
  - SB3-style handle_timeout_termination: dones=terminated (not OR truncated)
  - Q-function acts in coefficient space [rank+1], not full action space [21]
  - base_action stored in replay buffer (ResFiT) for coefficient recovery
"""

from __future__ import annotations

import argparse
import os
import pathlib
import random
import time
from collections import deque
from copy import deepcopy

try:
    import gymnasium as gym
    from gymnasium.envs.registration import register
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "gymnasium is required. Install with `pip install gymnasium`."
    ) from exc

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import zmq
from tensorboardX import SummaryWriter
from torch import nn, optim

from rl_finetuning.tfv6_rl.path_utils import ensure_carl_paths

ensure_carl_paths()

from rl_finetuning.tfv6_rl.policy_tfv6_td3 import (  # noqa: E402
    TFv6ResidualActorTD3,
    TFv6ResidualQNetworkTD3,
)
from rl_finetuning.tfv6_rl.residual_backbone import TFv6ResidualBackbone  # noqa: E402
from rl_finetuning.tfv6_rl_config import GlobalConfig  # noqa: E402

jsonpickle_numpy.register_handlers()
jsonpickle.set_encoder_options("json", sort_keys=True, indent=4)
torch.set_num_threads(1)


def strtobool(v):
    return str(v).lower() in ("yes", "y", "true", "t", "1", "True")


def none_or_str(value):
    if value == "None":
        return None
    return value


# ── Reward / episode stat helpers (mirrored from train_tfv6_ppo.py) ──────────

_REWARD_COMPONENT_KEYS = [
    "speed_penalty",
    "ttc_frac",
    "comfort_penalty",
    "lane_centering",
    "outside_lanes_frac",
]

_INFRACTION_TYPES = [
    "finished_route",
    "collision",
    "ran_red_light",
    "ran_stop_sign",
    "route_deviation",
    "route_deviation_2",
    "ego_blocked",
    "vehicle_too_close",
    "timeout",
    "off_road_term",
]

# Log infraction fracs every N completed episodes — mirrors PPO's per-update logging.
_INFRACTION_LOG_EVERY_N = 50


def _iter_episode_stats(info: dict):
    """Yield (return, length) pairs from completed episode info."""

    def _iter_from_episode_dict(episode_info: dict, done_mask=None):
        returns = np.asarray(episode_info["r"])
        lengths = np.asarray(episode_info["l"])
        if returns.ndim == 0:
            yield float(returns.item()), int(lengths.item())
            return
        flat_returns = returns.reshape(-1)
        flat_lengths = lengths.reshape(-1)
        flat_done_mask = (
            np.asarray(done_mask, dtype=bool).reshape(-1)
            if done_mask is not None
            else flat_lengths > 0
        )
        for idx, is_done in enumerate(flat_done_mask):
            if is_done:
                yield float(flat_returns[idx]), int(flat_lengths[idx])

    if "final_info" in info:
        for single_info in info["final_info"]:
            if single_info is None:
                continue
            if "episode" in single_info:
                yield from _iter_from_episode_dict(single_info["episode"])
            elif "tfv6_episode" in single_info:
                yield from _iter_from_episode_dict(single_info["tfv6_episode"])
        return
    if "episode" in info:
        yield from _iter_from_episode_dict(info["episode"], info.get("_episode"))
        return
    if "tfv6_episode" in info:
        yield from _iter_from_episode_dict(
            info["tfv6_episode"], info.get("_tfv6_episode")
        )


def _iter_reward_components(info: dict):
    """Yield reward component dicts for completed tfv6 episodes."""

    def _scalar(v):
        return float(np.asarray(v).reshape(-1)[0])

    if "final_info" in info:
        for single_info in info["final_info"]:
            if single_info is None:
                continue
            ep = single_info.get("tfv6_episode")
            if ep is not None and "speed_penalty" in ep:
                yield {k: _scalar(ep.get(k, 0.0)) for k in _REWARD_COMPONENT_KEYS}
        return
    ep = info.get("tfv6_episode")
    if ep is not None and "speed_penalty" in ep:
        arrays = {
            k: np.asarray(ep.get(k, 0.0)).reshape(-1) for k in _REWARD_COMPONENT_KEYS
        }
        n = arrays[_REWARD_COMPONENT_KEYS[0]].shape[0]
        done_mask = info.get("_tfv6_episode")
        flat_done = (
            np.asarray(done_mask, dtype=bool).reshape(-1)
            if done_mask is not None
            else np.ones(n, dtype=bool)
        )
        for idx in range(n):
            if flat_done[idx]:
                yield {k: float(arrays[k][idx]) for k in _REWARD_COMPONENT_KEYS}


# ── Replay buffer ─────────────────────────────────────────────────────────────


class DictReplayBuffer:
    """Pre-allocated numpy replay buffer for Dict observation spaces.

    Follows SB3's ``handle_timeout_termination`` convention:
      ``dones[t] = terminated`` (NOT ``terminated or truncated``)
      ``timeouts[t] = truncated``

    This ensures the Bellman target correctly bootstraps Q(next_obs) for
    truncated episodes (environment time limit) while not bootstrapping for
    true terminal states (collision / route completion).
    """

    def __init__(
        self,
        capacity: int,
        obs_space,
        action_dim: int,
        coeff_dim: int,
        device: torch.device,
    ) -> None:
        self.capacity = capacity
        self.pos = 0
        self.full = False
        self.device = device

        self.obs_buf: dict[str, np.ndarray] = {}
        self.next_obs_buf: dict[str, np.ndarray] = {}
        for key, space in obs_space.spaces.items():
            dtype = np.uint8 if space.dtype == np.uint8 else np.float32
            self.obs_buf[key] = np.zeros((capacity,) + space.shape, dtype=dtype)
            self.next_obs_buf[key] = np.zeros((capacity,) + space.shape, dtype=dtype)

        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.action_coeffs = np.zeros((capacity, coeff_dim), dtype=np.float32)
        self.base_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.timeouts = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        obs: dict[str, np.ndarray],
        next_obs: dict[str, np.ndarray],
        action: np.ndarray,
        action_coeffs: np.ndarray,
        base_action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        for key in self.obs_buf:
            self.obs_buf[key][self.pos] = obs[key]
            self.next_obs_buf[key][self.pos] = next_obs[key]
        self.actions[self.pos] = action
        self.action_coeffs[self.pos] = action_coeffs
        self.base_actions[self.pos] = base_action
        self.rewards[self.pos] = reward
        # SB3-style: done = terminated only (not truncated), timeout = truncated.
        self.dones[self.pos] = float(terminated)
        self.timeouts[self.pos] = float(truncated)
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size: int):
        """Sample a random minibatch, return tensors on self.device."""
        max_idx = self.capacity if self.full else self.pos
        indices = np.random.randint(0, max_idx, size=batch_size)

        obs_tensors: dict[str, torch.Tensor] = {}
        next_obs_tensors: dict[str, torch.Tensor] = {}
        for key in self.obs_buf:
            dtype = (
                torch.uint8 if self.obs_buf[key].dtype == np.uint8 else torch.float32
            )
            obs_tensors[key] = torch.tensor(
                self.obs_buf[key][indices], dtype=dtype, device=self.device
            )
            next_obs_tensors[key] = torch.tensor(
                self.next_obs_buf[key][indices], dtype=dtype, device=self.device
            )

        return {
            "obs": obs_tensors,
            "next_obs": next_obs_tensors,
            "actions": torch.tensor(self.actions[indices], device=self.device),
            "action_coeffs": torch.tensor(
                self.action_coeffs[indices], device=self.device
            ),
            "base_actions": torch.tensor(
                self.base_actions[indices], device=self.device
            ),
            "rewards": torch.tensor(self.rewards[indices], device=self.device),
            "dones": torch.tensor(self.dones[indices], device=self.device),
            "timeouts": torch.tensor(self.timeouts[indices], device=self.device),
        }

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def save(self, folder: str) -> None:
        """Atomically save buffer state to buffer_latest.npz in folder.

        Writes to a .tmp file first, then renames — safe against SIGTERM mid-write.
        Saves only filled slots (arr[:fill]) uncompressed — on NVMe this takes ~1-11s
        for 10K-100K transitions, well within CARLA's 900s ZMQ timeout.
        """
        out_path = os.path.join(folder, "buffer_latest.npz")
        tmp_path = os.path.join(folder, "buffer_latest.tmp.npz")
        fill = self.capacity if self.full else self.pos
        data: dict[str, np.ndarray] = {
            "_pos": np.array([self.pos], dtype=np.int64),
            "_full": np.array([self.full]),
            "_capacity": np.array([self.capacity], dtype=np.int64),
            "actions": self.actions[:fill],
            "action_coeffs": self.action_coeffs[:fill],
            "base_actions": self.base_actions[:fill],
            "rewards": self.rewards[:fill],
            "dones": self.dones[:fill],
            "timeouts": self.timeouts[:fill],
        }
        for key, arr in self.obs_buf.items():
            data[f"obs_{key}"] = arr[:fill]
        for key, arr in self.next_obs_buf.items():
            data[f"next_obs_{key}"] = arr[:fill]
        np.savez(tmp_path, **data)
        os.replace(tmp_path, out_path)

    def load(self, folder: str) -> bool:
        """Load buffer state from buffer_latest.npz in folder.

        Returns True on success, False if file is missing or incompatible
        (capacity mismatch, unknown obs keys). On failure the buffer stays empty.
        """
        path = os.path.join(folder, "buffer_latest.npz")
        if not os.path.exists(path):
            return False
        try:
            d = np.load(path)
        except Exception as exc:
            print(
                f"[td3] Warning: failed to open buffer file ({exc}); starting empty.",
                flush=True,
            )
            return False

        saved_capacity = int(d["_capacity"][0])
        if saved_capacity != self.capacity:
            print(
                f"[td3] Warning: buffer capacity mismatch "
                f"(saved={saved_capacity}, current={self.capacity}); starting empty.",
                flush=True,
            )
            return False

        missing = [
            k
            for k in list(self.obs_buf) + list(self.next_obs_buf)
            if f"obs_{k}" not in d or f"next_obs_{k}" not in d
        ]
        if missing:
            print(
                f"[td3] Warning: buffer missing obs keys {missing}; starting empty.",
                flush=True,
            )
            return False

        try:
            self.pos = int(d["_pos"][0])
            self.full = bool(d["_full"][0])
            fill = self.capacity if self.full else self.pos
            for key in self.obs_buf:
                self.obs_buf[key][:fill] = d[f"obs_{key}"]
            for key in self.next_obs_buf:
                self.next_obs_buf[key][:fill] = d[f"next_obs_{key}"]
            self.actions[:fill] = d["actions"]
            self.action_coeffs[:fill] = d["action_coeffs"]
            self.base_actions[:fill] = d["base_actions"]
            self.rewards[:fill] = d["rewards"]
            self.dones[:fill] = d["dones"]
            self.timeouts[:fill] = d["timeouts"]
        except Exception as exc:
            # Reset to empty on partial load failure.
            self.pos = 0
            self.full = False
            print(
                f"[td3] Warning: buffer load failed mid-copy ({exc}); starting empty.",
                flush=True,
            )
            return False

        return True


def _polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """Soft-update: target = (1-tau)*target + tau*source for all parameters."""
    with torch.no_grad():
        for p_src, p_tgt in zip(source.parameters(), target.parameters(), strict=True):
            p_tgt.data.mul_(1.0 - tau).add_(p_src.data, alpha=tau)


# ── Argument parsing ──────────────────────────────────────────────────────────


def parse_args(config):
    # fmt: off
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--exp_name', type=str, default=config.exp_name)
    parser.add_argument('--gym_id', type=str, default=config.gym_id)
    parser.add_argument('--tfv6_checkpoint', type=str, required=True,
                        help='Path to TFv6 checkpoint folder (config.json + model_*.pth).')
    parser.add_argument('--tfv6_prefix', type=str, default='model',
                        help='Prefix of TFv6 model files inside checkpoint folder.')
    parser.add_argument('--seed', type=int, default=config.seed)
    parser.add_argument('--total_timesteps', type=int, default=config.total_timesteps)
    parser.add_argument('--torch_deterministic', type=lambda x: bool(strtobool(x)),
                        default=config.torch_deterministic, nargs='?', const=True)
    parser.add_argument('--allow_tf32', type=lambda x: bool(strtobool(x)),
                        default=config.allow_tf32, nargs='?', const=True)
    parser.add_argument('--benchmark', type=lambda x: bool(strtobool(x)),
                        default=config.benchmark, nargs='?', const=True)
    parser.add_argument('--matmul_precision', type=str, default=config.matmul_precision)
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)),
                        default=config.cuda, nargs='?', const=True)
    parser.add_argument('--track', type=lambda x: bool(strtobool(x)),
                        default=config.track, nargs='?', const=True)
    parser.add_argument('--wandb_project_name', type=str, default=config.wandb_project_name)
    parser.add_argument('--wandb_entity', type=str, default=config.wandb_entity)

    # ── TD3 algorithm ────────────────────────────────────────────────────────
    parser.add_argument('--buffer_size', type=int,
                        default=getattr(config, 'buffer_size', 100_000))
    parser.add_argument('--learning_starts', type=int,
                        default=getattr(config, 'learning_starts', 5_000),
                        help='Steps before any learning begins (random/pure-noise warmup).')
    parser.add_argument('--td3_batch_size', type=int,
                        default=getattr(config, 'td3_batch_size', 256),
                        help='Minibatch size sampled from replay buffer per update.')
    parser.add_argument('--gamma', type=float, default=config.gamma)
    parser.add_argument('--tau', type=float,
                        default=getattr(config, 'tau', 0.005),
                        help='Polyak averaging coefficient for target networks.')
    parser.add_argument('--policy_delay', type=int,
                        default=getattr(config, 'policy_delay', 2),
                        help='Actor updated every policy_delay critic steps.')
    parser.add_argument('--exploration_noise', type=float,
                        default=getattr(config, 'exploration_noise', 0.1),
                        help='Std of Gaussian exploration noise added to actor output at rollout.')
    parser.add_argument('--target_policy_noise', type=float,
                        default=getattr(config, 'target_policy_noise', 0.2),
                        help='Std of target policy smoothing noise.')
    parser.add_argument('--target_noise_clip', type=float,
                        default=getattr(config, 'target_noise_clip', 0.5),
                        help='Clip bound for target policy smoothing noise.')
    parser.add_argument('--critic_warmup_steps', type=int,
                        default=getattr(config, 'critic_warmup_steps', 10_000),
                        help='Number of critic-only updates after learning_starts before actor training begins.')
    parser.add_argument('--actor_lr', type=float,
                        default=getattr(config, 'actor_lr', 1e-4))
    parser.add_argument('--critic_lr', type=float,
                        default=getattr(config, 'critic_lr', 1e-3))
    parser.add_argument('--max_grad_norm', type=float, default=config.max_grad_norm)

    # ── Residual policy ──────────────────────────────────────────────────────
    parser.add_argument('--use_residual_policy', type=lambda x: bool(strtobool(x)),
                        default=getattr(config, 'use_residual_policy', True),
                        nargs='?', const=True)
    parser.add_argument('--residual_route_rank', type=int,
                        default=getattr(config, 'residual_route_rank', 2))
    parser.add_argument('--residual_alpha', type=float,
                        default=getattr(config, 'residual_alpha', 0.15))
    parser.add_argument('--residual_alpha_speed', type=float,
                        default=getattr(config, 'residual_alpha_speed', 0.15))
    parser.add_argument('--disable_residual_route', type=lambda x: bool(strtobool(x)),
                        default=getattr(config, 'disable_residual_route', False),
                        nargs='?', const=True)
    parser.add_argument('--skip_perception_heads', type=lambda x: bool(strtobool(x)),
                        default=getattr(config, 'skip_perception_heads', True),
                        nargs='?', const=True)
    parser.add_argument('--speed_temperature', type=float,
                        default=getattr(config, 'speed_temperature', 1.0))
    parser.add_argument('--use_value_measurements', type=lambda x: bool(strtobool(x)),
                        default=getattr(config, 'use_value_measurements', True),
                        nargs='?', const=True)
    parser.add_argument('--num_value_measurements', type=int,
                        default=getattr(config, 'num_value_measurements', 0))

    # ── Reward ───────────────────────────────────────────────────────────────
    parser.add_argument('--reward_type', type=str, default=config.reward_type)
    parser.add_argument('--consider_tl', type=lambda x: bool(strtobool(x)),
                        default=config.consider_tl, nargs='?', const=True)
    parser.add_argument('--speeding_infraction', type=lambda x: bool(strtobool(x)),
                        default=config.speeding_infraction, nargs='?', const=True)
    parser.add_argument('--use_termination_hint', type=lambda x: bool(strtobool(x)),
                        default=config.use_termination_hint, nargs='?', const=True)
    parser.add_argument('--use_rl_termination_hint', type=lambda x: bool(strtobool(x)),
                        default=config.use_rl_termination_hint, nargs='?', const=True)
    parser.add_argument('--use_perc_progress', type=lambda x: bool(strtobool(x)),
                        default=config.use_perc_progress, nargs='?', const=True)
    parser.add_argument('--use_leave_route_done', type=lambda x: bool(strtobool(x)),
                        default=config.use_leave_route_done, nargs='?', const=True)
    parser.add_argument('--use_outside_route_lanes', type=lambda x: bool(strtobool(x)),
                        default=config.use_outside_route_lanes, nargs='?', const=True)
    parser.add_argument('--use_off_road_term', type=lambda x: bool(strtobool(x)),
                        default=config.use_off_road_term, nargs='?', const=True)
    parser.add_argument('--off_road_term_perc', type=float, default=config.off_road_term_perc)
    parser.add_argument('--use_ttc', type=lambda x: bool(strtobool(x)),
                        default=config.use_ttc, nargs='?', const=True)
    parser.add_argument('--penalize_yellow_light', type=lambda x: bool(strtobool(x)),
                        default=config.penalize_yellow_light, nargs='?', const=True)
    parser.add_argument('--use_comfort_infraction', type=lambda x: bool(strtobool(x)),
                        default=config.use_comfort_infraction, nargs='?', const=True)
    parser.add_argument('--use_single_reward', type=lambda x: bool(strtobool(x)),
                        default=config.use_single_reward, nargs='?', const=True)
    parser.add_argument('--use_new_stop_sign_detector', type=lambda x: bool(strtobool(x)),
                        default=config.use_new_stop_sign_detector, nargs='?', const=True)
    parser.add_argument('--terminal_hint', type=float, default=config.terminal_hint)
    parser.add_argument('--terminal_penalty_warmup_n', type=int, default=0)
    parser.add_argument('--lane_distance_violation_threshold', type=float,
                        default=config.lane_distance_violation_threshold)
    parser.add_argument('--lane_dist_penalty_softener', type=float,
                        default=config.lane_dist_penalty_softener)
    parser.add_argument('--comfort_penalty_factor', type=float,
                        default=config.comfort_penalty_factor)
    parser.add_argument('--use_survival_reward', type=lambda x: bool(strtobool(x)),
                        default=config.use_survival_reward, nargs='?', const=True)

    # ── Infrastructure ───────────────────────────────────────────────────────
    parser.add_argument('--logdir', type=str, default=config.logdir)
    parser.add_argument('--save_every', type=int, default=getattr(config, 'save_every', 10_000),
                        help='Save model checkpoint and replay buffer every N global steps.')
    parser.add_argument('--load_file', type=none_or_str, nargs='?', default=config.load_file,
                        help='TD3 checkpoint file to resume from (model_latest_*.pth).')
    parser.add_argument('--ports', nargs='+', default=config.ports, type=int,
                        help='ZMQ port(s) for CARLA env(s).')
    parser.add_argument('--gpu_ids', nargs='+', default=config.gpu_ids, type=int)
    parser.add_argument('--run_dir', type=str, default='',
                        help='Concrete run folder. Used for debug artifacts.')
    parser.add_argument('--debug_viz', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True,
                        help='If true, dump rollout debug visualizations to run_dir/debug_viz.')
    parser.add_argument('--debug_viz_every_n', type=int, default=1,
                        help='Start a full debug-viz burst at startup and every N global steps.')
    parser.add_argument('--debug_viz_burst_len', type=int, default=1000,
                        help='Number of consecutive frames to write for each scheduled debug-viz burst.')
    parser.add_argument('--debug_viz_max_images', type=int, default=0,
                        help='Maximum number of debug images to write (0 means unlimited).')
    parser.add_argument('--debug_viz_image_scale', type=int, default=3,
                        help='Scale factor for BEV rendering in debug visualizations.')
    parser.add_argument('--debug', type=lambda x: bool(strtobool(x)),
                        default=False, nargs='?', const=True)

    # ── Shared env args ──────────────────────────────────────────────────────
    parser.add_argument('--normalize_rewards', type=lambda x: bool(strtobool(x)),
                        default=config.normalize_rewards, nargs='?', const=True)
    parser.add_argument('--use_new_bev_obs', type=lambda x: bool(strtobool(x)),
                        default=config.use_new_bev_obs, nargs='?', const=True)
    parser.add_argument('--obs_num_channels', type=int, default=config.obs_num_channels)
    parser.add_argument('--map_folder', type=str, default=config.map_folder)
    parser.add_argument('--pixels_per_meter', type=float, default=config.pixels_per_meter)
    parser.add_argument('--route_width', type=int, default=config.route_width)
    parser.add_argument('--bev_semantics_width', type=int, default=config.bev_semantics_width)
    parser.add_argument('--bev_semantics_height', type=int, default=config.bev_semantics_height)
    parser.add_argument('--pixels_ev_to_bottom', type=int, default=config.pixels_ev_to_bottom)
    parser.add_argument('--use_history', type=lambda x: bool(strtobool(x)),
                        default=config.use_history, nargs='?', const=True)
    parser.add_argument('--use_green_wave', type=lambda x: bool(strtobool(x)),
                        default=config.use_green_wave, nargs='?', const=True)
    parser.add_argument('--render_green_tl', type=lambda x: bool(strtobool(x)),
                        default=config.render_green_tl, nargs='?', const=True)
    parser.add_argument('--num_route_points_rendered', type=int,
                        default=config.num_route_points_rendered)
    parser.add_argument('--render_shoulder', type=lambda x: bool(strtobool(x)),
                        default=config.render_shoulder, nargs='?', const=True)
    parser.add_argument('--use_shoulder_channel', type=lambda x: bool(strtobool(x)),
                        default=config.use_shoulder_channel, nargs='?', const=True)
    parser.add_argument('--render_speed_lines', type=lambda x: bool(strtobool(x)),
                        default=config.render_speed_lines, nargs='?', const=True)
    parser.add_argument('--render_yellow_time', type=lambda x: bool(strtobool(x)),
                        default=config.render_yellow_time, nargs='?', const=True)
    parser.add_argument('--use_positional_encoding', type=lambda x: bool(strtobool(x)),
                        default=config.use_positional_encoding, nargs='?', const=True)
    parser.add_argument('--condition_outside_junction', type=lambda x: bool(strtobool(x)),
                        default=config.condition_outside_junction, nargs='?', const=True)
    parser.add_argument('--eval_time', type=float, default=config.eval_time)
    parser.add_argument('--terminal_reward', type=float, default=config.terminal_reward)
    parser.add_argument('--min_thresh_lat_dist', type=float, default=config.min_thresh_lat_dist)
    parser.add_argument('--weight_decay', type=float, default=config.weight_decay)
    parser.add_argument('--adam_eps', type=float, default=config.adam_eps)
    parser.add_argument('--beta_1', type=float, default=config.beta_1)
    parser.add_argument('--beta_2', type=float, default=config.beta_2)
    parser.add_argument('--standstill_speed_hold_enabled',
                        type=lambda x: bool(strtobool(x)),
                        default=getattr(config, 'standstill_speed_hold_enabled', False),
                        nargs='?', const=True)
    parser.add_argument('--standstill_speed_hold_frames', type=int,
                        default=getattr(config, 'standstill_speed_hold_frames', 3))
    parser.add_argument('--standstill_speed_hold_ego_speed_threshold', type=float,
                        default=getattr(config, 'standstill_speed_hold_ego_speed_threshold', 0.1))
    parser.add_argument('--standstill_speed_hold_target_speed_threshold', type=float,
                        default=getattr(config, 'standstill_speed_hold_target_speed_threshold', 1.0 / 3.6))

    args, unknown = parser.parse_known_args()
    if unknown:
        print("Unknown arguments:", unknown)
    # fmt: on
    return args


# ── Env factory ───────────────────────────────────────────────────────────────


def make_env(gym_id, args, run_name, port, config):
    def thunk():
        render_mode = "rgb_array"
        env = gym.make(gym_id, port=port, config=config, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        if config.normalize_rewards:
            env = gym.wrappers.NormalizeReward(env, gamma=config.gamma)
            env = gym.wrappers.TransformReward(
                env, lambda reward: np.clip(reward, -10, 10)
            )
        return env

    return thunk


# ── Checkpoint helpers ────────────────────────────────────────────────────────


def _save_td3_checkpoint(
    folder: str,
    stem: str,
    backbone,
    actor,
    qf1,
    qf2,
    target_backbone,
    actor_target,
    qf1_target,
    qf2_target,
    actor_optimizer,
    critic_optimizer,
    global_step: int,
    config,
) -> None:
    """Save TD3 checkpoint as two files: model_*.pth and optimizer_*.pth."""
    os.makedirs(folder, exist_ok=True)
    model_path = os.path.join(folder, f"model_{stem}.pth")
    optimizer_path = os.path.join(folder, f"optimizer_{stem}.pth")

    # Backbone learnable state (skip TFv6 — large, always reloaded from checkpoint dir).
    backbone_learnable = {
        "residual_cnn": backbone.residual_cnn.state_dict(),
        "residual_status_proj": backbone.residual_status_proj.state_dict(),
    }
    target_backbone_learnable = {
        "residual_cnn": target_backbone.residual_cnn.state_dict(),
        "residual_status_proj": target_backbone.residual_status_proj.state_dict(),
    }

    torch.save(
        {
            "backbone": backbone_learnable,
            "actor_residual_out": actor.residual_out.state_dict(),
            "qf1_q_head": qf1.q_head.state_dict(),
            "qf2_q_head": qf2.q_head.state_dict(),
            "target_backbone": target_backbone_learnable,
            "actor_target_residual_out": actor_target.residual_out.state_dict(),
            "qf1_target_q_head": qf1_target.q_head.state_dict(),
            "qf2_target_q_head": qf2_target.q_head.state_dict(),
            "global_step": global_step,
            "config": jsonpickle.encode(config),
        },
        model_path,
    )
    torch.save(
        {
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
        },
        optimizer_path,
    )


def _load_td3_checkpoint(
    load_file: str,
    backbone,
    actor,
    qf1,
    qf2,
    target_backbone,
    actor_target,
    qf1_target,
    qf2_target,
    actor_optimizer,
    critic_optimizer,
    device: torch.device,
) -> int:
    """Load TD3 checkpoint. Returns global_step to resume from."""
    ckpt = torch.load(load_file, map_location=device)

    bb = ckpt["backbone"]
    backbone.residual_cnn.load_state_dict(bb["residual_cnn"])
    backbone.residual_status_proj.load_state_dict(bb["residual_status_proj"])
    actor.residual_out.load_state_dict(ckpt["actor_residual_out"])
    qf1.q_head.load_state_dict(ckpt["qf1_q_head"])
    qf2.q_head.load_state_dict(ckpt["qf2_q_head"])

    tbb = ckpt["target_backbone"]
    target_backbone.residual_cnn.load_state_dict(tbb["residual_cnn"])
    target_backbone.residual_status_proj.load_state_dict(tbb["residual_status_proj"])
    actor_target.residual_out.load_state_dict(ckpt["actor_target_residual_out"])
    qf1_target.q_head.load_state_dict(ckpt["qf1_target_q_head"])
    qf2_target.q_head.load_state_dict(ckpt["qf2_target_q_head"])

    optimizer_file = load_file.replace("model_", "optimizer_")
    if os.path.exists(optimizer_file):
        opt_ckpt = torch.load(optimizer_file, map_location=device)
        try:
            actor_optimizer.load_state_dict(opt_ckpt["actor_optimizer"])
            critic_optimizer.load_state_dict(opt_ckpt["critic_optimizer"])
        except (ValueError, KeyError) as exc:
            print(
                f"[td3] warning: optimizer state mismatch ({exc}). Using fresh optimizers.",
                flush=True,
            )

    global_step = int(ckpt.get("global_step", 0)) + 1
    return global_step


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    register(
        id="CARLAEnvTFv6-v0",
        entry_point="rl_finetuning.tfv6_rl.env_gym_tfv6:CARLAEnvTFv6",
        max_episode_steps=None,
    )
    config = GlobalConfig()
    config.gym_id = "CARLAEnvTFv6-v0"
    config.use_exploration_suggest = False
    args = parse_args(config)

    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}"
    exp_folder = os.path.join(args.logdir, args.exp_name)
    pathlib.Path(exp_folder).mkdir(parents=True, exist_ok=True)

    if args.track:
        wandb_folder = os.path.join(exp_folder, "wandb")
        pathlib.Path(wandb_folder).mkdir(parents=True, exist_ok=True)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            allow_val_change=True,
            save_code=False,
            mode="online",
            resume="auto",
            dir=wandb_folder,
            settings=wandb.Settings(
                _disable_stats=True, _disable_meta=True, start_method="fork"
            ),
        )

    writer = SummaryWriter(exp_folder)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n{}".format(
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
        ),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = (
        torch.device(f"cuda:{args.gpu_ids[0]}")
        if torch.cuda.is_available() and args.cuda
        else torch.device("cpu")
    )
    print(f"[td3] device={device}", flush=True)

    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.benchmark = args.benchmark
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    torch.set_float32_matmul_precision(args.matmul_precision)

    # Load config from checkpoint if resuming.
    if args.load_file is not None:
        load_folder = pathlib.Path(args.load_file).parent.resolve()
        config_path = os.path.join(load_folder, "config.json")
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                loaded_config = jsonpickle.decode(f.read())
            config.__dict__.update(loaded_config.__dict__)

    config.initialize(**vars(args))

    # Write config for this run.
    json_config = jsonpickle.encode(config)
    with open(os.path.join(exp_folder, "config.json"), "w", encoding="utf-8") as f:
        f.write(json_config)

    # ── Send config to CARLA env via ZMQ (same pattern as PPO) ──────────────
    context = zmq.Context()
    for port in args.ports:
        socket = context.socket(zmq.PAIR)
        comm_folder = pathlib.Path(__file__).parent / "tfv6_rl" / "comm_files"
        comm_folder.mkdir(parents=True, exist_ok=True)
        communication_file = comm_folder / str(port)
        socket.bind(f"ipc://{communication_file}.conf_lock")
        socket.send_string(jsonpickle.encode(config))
        _ = socket.recv_string()
        socket.close()
    context.term()
    print("[td3] Sent CONFIG to env(s)", flush=True)

    # ── Environment ──────────────────────────────────────────────────────────
    env = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args, run_name, port, config) for port in args.ports]
    )
    assert isinstance(env.single_action_space, gym.spaces.Box)
    action_dim = env.single_action_space.shape[0]

    # ── Networks ─────────────────────────────────────────────────────────────
    backbone = TFv6ResidualBackbone(
        tfv6_checkpoint=args.tfv6_checkpoint,
        tfv6_prefix=args.tfv6_prefix,
        device=device,
        rl_config=config,
    ).to(device)

    target_backbone = deepcopy(backbone)
    target_backbone.to(device)
    target_backbone.eval()
    for p in target_backbone.parameters():
        p.requires_grad_(False)

    actor = TFv6ResidualActorTD3(backbone, config).to(device)
    qf1 = TFv6ResidualQNetworkTD3(backbone, config).to(device)
    qf2 = TFv6ResidualQNetworkTD3(backbone, config).to(device)

    actor_target = TFv6ResidualActorTD3(target_backbone, config).to(device)
    actor_target.eval()
    for p in actor_target.parameters():
        p.requires_grad_(False)

    qf1_target = TFv6ResidualQNetworkTD3(target_backbone, config).to(device)
    qf2_target = TFv6ResidualQNetworkTD3(target_backbone, config).to(device)
    actor_target.load_state_dict(actor.state_dict())
    qf1_target.q_head.load_state_dict(qf1.q_head.state_dict())
    qf2_target.q_head.load_state_dict(qf2.q_head.state_dict())
    for p in qf1_target.q_head.parameters():
        p.requires_grad_(False)
    for p in qf2_target.q_head.parameters():
        p.requires_grad_(False)

    coeff_dim = backbone.residual_route_rank + 1

    # Actor: only the linear residual output head. Features are detached before
    # this head during actor updates (ResFiT / DrQ-v2 style: encoder trained by
    # critic loss, not actor loss).
    actor_optimizer = optim.Adam(
        actor.residual_out.parameters(),
        lr=args.actor_lr,
        eps=args.adam_eps,
        betas=(args.beta_1, args.beta_2),
    )
    # Critic: Q-heads + shared CNN encoder. The encoder trains from critic
    # gradients starting at learning_starts, giving meaningful features before
    # actor updates begin after critic_warmup_steps critic-only updates.
    critic_optimizer = optim.Adam(
        list(qf1.q_head.parameters())
        + list(qf2.q_head.parameters())
        + list(backbone.residual_cnn.parameters())
        + list(backbone.residual_status_proj.parameters()),
        lr=args.critic_lr,
        eps=args.adam_eps,
        betas=(args.beta_1, args.beta_2),
    )

    actor_trainable = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    qf_trainable = sum(
        p.numel()
        for p in list(qf1.q_head.parameters()) + list(qf2.q_head.parameters())
        if p.requires_grad
    )
    print(
        f"[td3] actor trainable params={actor_trainable} "
        f"(backbone CNN+proj + residual_out)",
        flush=True,
    )
    print(f"[td3] critic trainable params={qf_trainable} (2×q_head)", flush=True)

    # ── Replay buffer ─────────────────────────────────────────────────────────
    # Store pre-encoded frozen TFv6 features instead of raw sensor obs.
    # Reduces per-transition size from ~3.5 MB to ~32 KB (56× reduction).
    replay_buffer = DictReplayBuffer(
        capacity=args.buffer_size,
        obs_space=backbone.feature_obs_space(),
        action_dim=action_dim,
        coeff_dim=coeff_dim,
        device=device,
    )

    # ── Checkpoint resume ─────────────────────────────────────────────────────
    global_step = 0
    if args.load_file is not None:
        global_step = _load_td3_checkpoint(
            args.load_file,
            backbone,
            actor,
            qf1,
            qf2,
            target_backbone,
            actor_target,
            qf1_target,
            qf2_target,
            actor_optimizer,
            critic_optimizer,
            device,
        )
        print(
            f"[td3] Resumed from {args.load_file}, global_step={global_step}",
            flush=True,
        )
        writer.add_scalar("charts/restart", 1, global_step)
        t_buf = time.time()
        buf_loaded = replay_buffer.load(exp_folder)
        if buf_loaded:
            print(
                f"[td3] Loaded buffer ({len(replay_buffer)} transitions) "
                f"in {time.time() - t_buf:.1f}s",
                flush=True,
            )
        else:
            print(
                "[td3] No compatible buffer checkpoint; starting with empty buffer.",
                flush=True,
            )

    # ── Debug visualizer ──────────────────────────────────────────────────────
    debug_viz = None
    if args.debug_viz:
        from rl_finetuning.tfv6_rl.debug_rollout_viz import PPORolloutVisualizer

        debug_root = (
            args.run_dir if args.run_dir else os.path.join(exp_folder, "latest")
        )
        debug_dir = os.path.join(debug_root, "debug_viz")
        debug_viz = PPORolloutVisualizer(
            training_config=backbone.training_config,
            action_codec=backbone.action_codec,
            output_dir=debug_dir,
            num_envs=len(args.ports),
            gamma=config.gamma,
            every_n=args.debug_viz_every_n,
            scheduled_burst_len=args.debug_viz_burst_len,
            max_images=args.debug_viz_max_images,
            image_scale=args.debug_viz_image_scale,
            standstill_speed_hold_enabled=getattr(
                config, "standstill_speed_hold_enabled", False
            ),
            standstill_speed_hold_frames=getattr(
                config, "standstill_speed_hold_frames", 1
            ),
            standstill_speed_hold_ego_speed_threshold=getattr(
                config, "standstill_speed_hold_ego_speed_threshold", 0.1
            ),
            standstill_speed_hold_target_speed_threshold=getattr(
                config, "standstill_speed_hold_target_speed_threshold", 1.0 / 3.6
            ),
            speed_temperature=backbone.speed_temperature,
            use_residual_policy=True,
        )
        print(
            f"[td3][debug_viz] enabled path={debug_dir} "
            f"every_n={args.debug_viz_every_n} burst_len={args.debug_viz_burst_len} "
            f"max_images={args.debug_viz_max_images}",
            flush=True,
        )

    # ── Initial observation ───────────────────────────────────────────────────
    print("[td3] Waiting for first observation...", flush=True)
    reset_obs, _ = env.reset(seed=[args.seed])
    print("[td3] Received first observation. Starting TD3 loop.", flush=True)

    def _obs_to_device(obs_np: dict) -> dict[str, torch.Tensor]:
        result = {}
        for key, space in env.single_observation_space.spaces.items():
            dtype = torch.uint8 if space.dtype == np.uint8 else torch.float32
            result[key] = torch.tensor(obs_np[key], device=device, dtype=dtype)
        return result

    next_obs = _obs_to_device(reset_obs)
    avg_returns = deque(
        maxlen=900
    )  # ~100 PPO-update-equivalents at 2048 steps/update, 226 steps/episode
    recent_rewards = deque(maxlen=1000)
    start_time = time.time()
    num_envs = len(args.ports)
    route_completion_by_env = [0.0] * num_envs
    terminal_infraction_by_env = [""] * num_envs

    # Infraction accumulator — reset each time infractions are logged.
    infraction_counts: dict[str, int] = {k: 0 for k in _INFRACTION_TYPES}
    episodes_since_infraction_log: int = 0
    # Persistent actor loss — retained across steps so logging doesn't miss it.
    last_actor_loss_val: float = float("nan")
    # Per-episode (global_step, reward) pairs for debug-viz forward-return stamping.
    _viz_episode_step_rewards: list[tuple[int, float]] = []

    # Precompute exploration-noise log_std arrays for debug viz (constant throughout training).
    # These represent the actual spread of executed actions due to exploration noise.
    _viz_action_log_std = np.full(
        backbone.action_codec.route_dim + 1, -10.0, dtype=np.float32
    )
    if not backbone.disable_residual_route:
        _viz_action_log_std[: backbone.action_codec.route_dim] = np.log(
            backbone.residual_alpha * args.exploration_noise + 1e-8
        )
    _viz_action_log_std[-1] = np.log(
        backbone.residual_alpha_speed * args.exploration_noise + 1e-8
    )
    _viz_coeff_log_std = np.full(
        coeff_dim, np.log(args.exploration_noise + 1e-8), dtype=np.float32
    )

    # ── TD3 training loop ─────────────────────────────────────────────────────
    start_step = global_step
    _sps_step = start_step
    _sps_time = start_time
    _t_fwd = _t_env = _t_buf = _t_train = 0.0
    for global_step in range(start_step, args.total_timesteps):
        # ── COLLECT ──────────────────────────────────────────────────────────
        # Single unified collection path throughout (including warmup).
        # The zero-initialized actor produces coeff ≈ 0 before training starts,
        # so the executed action is base_policy + exploration_noise_in_coeff_space —
        # equivalent to Ankile's warmup (base + small noise) but without a separate
        # broken branch that adds noise in full 21-dim action space.
        # disable_residual_route is always respected via coeffs_to_action.
        actor.eval()
        _t0 = time.perf_counter()
        with torch.no_grad():
            coeff_tensor = actor.forward_coeffs(next_obs)  # sets backbone state
            base_action_mean = backbone._last_base_action_mean
            noise = torch.randn_like(coeff_tensor) * args.exploration_noise
            coeff_tensor_noisy = coeff_tensor + noise
            action_tensor = actor.coeffs_to_action(coeff_tensor_noisy)
            clean_coeff_tensor = coeff_tensor  # deterministic (no noise) for viz
            coeff_tensor = coeff_tensor_noisy

            # Q-estimate for debug viz (optional, before env step).
            q_estimate_for_viz: float | None = None
            if debug_viz is not None and global_step >= args.learning_starts:
                priv_viz = next_obs.get("privileged_measurements")
                q1_val = qf1.forward_with_cached_backbone(coeff_tensor, priv_viz)
                q_estimate_for_viz = float(q1_val[0].item())

        action_np = action_tensor[0].cpu().numpy()
        coeff_np = coeff_tensor[0].cpu().numpy()
        clean_coeff_np = clean_coeff_tensor[0].cpu().numpy()
        base_action_np = base_action_mean[0].cpu().numpy()
        mean_action_np = actor.coeffs_to_action(clean_coeff_tensor)[0].cpu().numpy()
        target_speed_logits_np = (
            backbone._last_target_speed_logits[0].cpu().numpy()
            if backbone._last_target_speed_logits is not None
            else None
        )

        # Extract obs features from cache left by actor.forward_coeffs (no extra TFv6 run).
        obs_features = backbone.encode_obs_to_features()
        if "privileged_measurements" in next_obs:
            obs_features["privileged_measurements"] = (
                next_obs["privileged_measurements"][0].cpu().numpy()
            )
        _t_fwd += time.perf_counter() - _t0

        _t0 = time.perf_counter()
        step_result = env.step(action_np[None])
        next_obs_np, reward_np, terminated_np, truncated_np, info = step_result
        reward = float(reward_np[0])
        terminated = bool(terminated_np[0])
        truncated = bool(truncated_np[0])

        # Extract route completion and infraction for logging/viz.
        route_completion_by_env[0] = float(
            info.get("route_completion", [0.0])[0]
            if "route_completion" in info
            else 0.0
        )
        if "infraction_type" in info:
            infraction_val = np.asarray(info["infraction_type"], dtype=object)[0]
            if (terminated or truncated) and infraction_val is not None:
                terminal_infraction_by_env[0] = str(infraction_val or "")
        if "final_info" in info:
            for idx, single_info in enumerate(info["final_info"]):
                if single_info is not None:
                    terminal_infraction_by_env[idx] = str(
                        single_info.get("infraction_type", "") or ""
                    )

        next_obs_device = _obs_to_device(next_obs_np)

        # Determine the obs to encode for next_obs features.
        # For truncated episodes, gymnasium auto-resets, so next_obs_device is the
        # reset obs — we need the actual final obs from info["final_observation"].
        if truncated and "final_observation" in info:
            final_obs_np = info["final_observation"]
            next_obs_for_encode: dict[str, torch.Tensor] = {}
            for key, space in env.single_observation_space.spaces.items():
                dtype = torch.uint8 if space.dtype == np.uint8 else torch.float32
                if key in final_obs_np:
                    arr = np.asarray(final_obs_np[key][0])
                    next_obs_for_encode[key] = torch.tensor(
                        arr, device=device, dtype=dtype
                    ).unsqueeze(0)
                else:
                    next_obs_for_encode[key] = next_obs_device[key][:1]
        else:
            next_obs_for_encode = next_obs_device

        with torch.no_grad():
            next_obs_features = backbone.encode_obs_to_features(next_obs_for_encode)
        if "privileged_measurements" in next_obs_for_encode:
            next_obs_features["privileged_measurements"] = (
                next_obs_for_encode["privileged_measurements"][0].cpu().numpy()
            )

        replay_buffer.add(
            obs=obs_features,
            next_obs=next_obs_features,
            action=action_np,
            action_coeffs=coeff_np,
            base_action=base_action_np,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
        _t_env += time.perf_counter() - _t0

        # Reset obs (SyncVectorEnv auto-resets on done).
        next_obs = next_obs_device

        # Debug viz.
        if debug_viz is not None:
            _viz_episode_step_rewards.append((global_step, reward))
            obs_for_viz = {k: v.cpu().numpy() for k, v in next_obs.items()}
            debug_viz.maybe_write(
                global_step=global_step,
                rollout_step=global_step,
                update_idx=global_step,
                env_idx=0,
                obs={k: v[0] for k, v in obs_for_viz.items()},
                reward=reward,
                done=terminated,
                truncated=truncated,
                sampled_action=np.clip(action_np, -1.0, 1.0),
                mean_action=np.clip(mean_action_np, -1.0, 1.0),
                base_mean_action=np.clip(base_action_np, -1.0, 1.0),
                target_speed_logits=target_speed_logits_np,
                residual_coeff_preds=np.concatenate(
                    [clean_coeff_np, _viz_coeff_log_std]
                ),
                residual_attention_weights=None,
                log_std=_viz_action_log_std,
                value_estimate=0.0,
                route_completion=route_completion_by_env[0],
                speed_hold_frames=0,
                speed_hold_target_speed=None,
                q_estimate=q_estimate_for_viz,
            )
            if terminated or truncated:
                # Stamp infraction reason on all written frames from this episode.
                debug_viz.note_episode_outcome(
                    env_idx=0,
                    terminal_infraction_type=terminal_infraction_by_env[0],
                )
                # Compute discounted returns backward and stamp written frames.
                G = 0.0
                step_to_return: dict[int, float] = {}
                for gs, r in reversed(_viz_episode_step_rewards):
                    G = r + args.gamma * G
                    step_to_return[gs] = G
                debug_viz.stamp_episode_forward_returns(step_to_return)
                _viz_episode_step_rewards = []

        recent_rewards.append(reward)

        # Episode stats.
        for ep_return, ep_length in _iter_episode_stats(info):
            avg_returns.append(ep_return)
            print(
                f"[td3] global_step={global_step} episodic_return={ep_return:.4f}",
                flush=True,
            )
            writer.add_scalar("charts/episodic_return", ep_return, global_step)
            writer.add_scalar("charts/episodic_length", ep_length, global_step)
            if avg_returns:
                writer.add_scalar(
                    "charts/windowed_avg_return", np.mean(avg_returns), global_step
                )
            # Infraction tracking (mirrors PPO: every episode gets one type).
            infraction = terminal_infraction_by_env[0] or "finished_route"
            if infraction in infraction_counts:
                infraction_counts[infraction] += 1
            episodes_since_infraction_log += 1
            # Log and reset every N episodes — same formula as PPO:
            # fraction = count_k / total_episodes (sum of all type counts).
            if episodes_since_infraction_log >= _INFRACTION_LOG_EVERY_N:
                total_infraction_episodes = sum(infraction_counts.values())
                if total_infraction_episodes > 0:
                    for k in _INFRACTION_TYPES:
                        writer.add_scalar(
                            f"infractions/{k}",
                            infraction_counts[k] / total_infraction_episodes,
                            global_step,
                        )
                infraction_counts = {k: 0 for k in _INFRACTION_TYPES}
                episodes_since_infraction_log = 0
        for rc in _iter_reward_components(info):
            for k in _REWARD_COMPONENT_KEYS:
                writer.add_scalar(f"reward_components/{k}", rc[k], global_step)

        # ── UPDATE ────────────────────────────────────────────────────────────
        if (
            global_step >= args.learning_starts
            and len(replay_buffer) >= args.td3_batch_size
        ):
            critic_updates_completed = global_step - args.learning_starts + 1
            _t0 = time.perf_counter()
            batch = replay_buffer.sample(args.td3_batch_size)
            _t_buf += time.perf_counter() - _t0

            # ── Critic update ─────────────────────────────────────────────────
            _t0 = time.perf_counter()
            actor.train()
            with torch.no_grad():
                # Target policy smoothing: clipped Gaussian noise on target actor.
                next_coeffs = actor_target.forward_coeffs_from_features(
                    batch["next_obs"]
                )
                noise = (
                    torch.randn_like(next_coeffs) * args.target_policy_noise
                ).clamp(-args.target_noise_clip, args.target_noise_clip)
                next_coeffs_smoothed = (next_coeffs + noise).clamp(-1.0, 1.0)

                # Twin Q targets: min of two target Q-networks.
                next_priv = batch["next_obs"].get("privileged_measurements")
                q1_next = qf1_target(batch["next_obs"], next_coeffs_smoothed, next_priv)
                q2_next = qf2_target(batch["next_obs"], next_coeffs_smoothed, next_priv)
                min_q_next = torch.min(q1_next, q2_next)

                # Handle timeout termination (SB3-style):
                # For truncated episodes (timeouts) the episode is not truly terminal,
                # so we still bootstrap. dones=terminated handles this correctly.
                target_q = (
                    batch["rewards"] + (1.0 - batch["dones"]) * args.gamma * min_q_next
                )

            # Run online Q-networks (actor forward sets backbone state).
            # We run actor.forward_coeffs to set backbone._last_base_action_mean,
            # then pass stored batch["action_coeffs"] to the Q-networks.
            # This avoids re-running TFv6 for actor+qf1+qf2 separately.
            _log_coeffs = actor.forward_coeffs_from_features(
                batch["obs"]
            )  # sets backbone state; captured for logging
            priv = batch["obs"].get("privileged_measurements")
            q1_pred = qf1.forward_with_cached_backbone(batch["action_coeffs"], priv)
            q2_pred = qf2.forward_with_cached_backbone(batch["action_coeffs"], priv)

            critic_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_grad_norm = nn.utils.clip_grad_norm_(
                list(qf1.q_head.parameters())
                + list(qf2.q_head.parameters())
                + list(backbone.residual_cnn.parameters())
                + list(backbone.residual_status_proj.parameters()),
                args.max_grad_norm,
            )
            critic_optimizer.step()

            # Keep critic targets synchronized during critic-only warmup so
            # Bellman bootstrapping does not rely on stale/random target heads.
            _polyak_update(
                backbone.residual_cnn,
                target_backbone.residual_cnn,
                args.tau,
            )
            _polyak_update(
                backbone.residual_status_proj,
                target_backbone.residual_status_proj,
                args.tau,
            )
            _polyak_update(qf1.q_head, qf1_target.q_head, args.tau)
            _polyak_update(qf2.q_head, qf2_target.q_head, args.tau)

            # ── Actor update (delayed + after critic warmup) ──────────────────
            actor_grad_norm: float = float("nan")
            if (
                critic_updates_completed > args.critic_warmup_steps
                and critic_updates_completed % args.policy_delay == 0
            ):
                # Features detached: encoder trained by critic loss only (ResFiT style).
                pred_coeffs = actor.forward_coeffs_from_features(
                    batch["obs"], detach_features=True
                )
                actor_loss = -qf1.forward_with_cached_backbone(
                    pred_coeffs, priv, detach_features=True
                ).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_grad_norm = float(
                    nn.utils.clip_grad_norm_(
                        actor.residual_out.parameters(), args.max_grad_norm
                    )
                )
                actor_optimizer.step()
                last_actor_loss_val = float(actor_loss.item())

                # Polyak update target networks.
                _polyak_update(actor, actor_target, args.tau)
                _polyak_update(qf1.q_head, qf1_target.q_head, args.tau)
                _polyak_update(qf2.q_head, qf2_target.q_head, args.tau)

            _t_train += time.perf_counter() - _t0

            # ── Logging ───────────────────────────────────────────────────────
            if global_step % 100 == 0:
                _now = time.time()
                _window = max(global_step - _sps_step, 1)
                sps = _window / (_now - _sps_time + 1e-9)
                _sps_step, _sps_time = global_step, _now
                writer.add_scalar("timing/fwd_s", _t_fwd / _window, global_step)
                writer.add_scalar("timing/env_s", _t_env / _window, global_step)
                writer.add_scalar("timing/buf_s", _t_buf / _window, global_step)
                writer.add_scalar("timing/train_s", _t_train / _window, global_step)
                _t_fwd = _t_env = _t_buf = _t_train = 0.0
                writer.add_scalar(
                    "losses/critic_loss", float(critic_loss.item()), global_step
                )
                writer.add_scalar(
                    "losses/q1_value", float(q1_pred.mean().item()), global_step
                )
                writer.add_scalar(
                    "losses/q2_value", float(q2_pred.mean().item()), global_step
                )
                writer.add_scalar(
                    "losses/target_q_mean", float(target_q.mean().item()), global_step
                )
                writer.add_scalar(
                    "losses/q_overestimation",
                    float(q1_pred.mean().item()) - float(target_q.mean().item()),
                    global_step,
                )
                writer.add_scalar(
                    "grads/critic_grad_norm", float(critic_grad_norm), global_step
                )
                _speed_correction = (
                    backbone.residual_alpha_speed
                    * _log_coeffs[:, backbone.residual_route_rank].detach()
                    * backbone.action_codec.speed_scale
                )
                writer.add_scalar(
                    "residual/speed_correction_mean",
                    float(_speed_correction.mean().item()),
                    global_step,
                )
                writer.add_scalar(
                    "residual/speed_correction_mean_std",
                    float(_speed_correction.std().item()),
                    global_step,
                )
                writer.add_scalar("charts/SPS", sps, global_step)
                writer.add_scalar("charts/buffer_size", len(replay_buffer), global_step)
                writer.add_scalar(
                    "charts/actor_lr",
                    actor_optimizer.param_groups[0]["lr"],
                    global_step,
                )
                writer.add_scalar(
                    "charts/critic_lr",
                    critic_optimizer.param_groups[0]["lr"],
                    global_step,
                )
                writer.add_scalar("charts/restart", 0, global_step)
                if recent_rewards:
                    writer.add_scalar(
                        "charts/mean_reward", np.mean(recent_rewards), global_step
                    )
                if not np.isnan(last_actor_loss_val):
                    writer.add_scalar(
                        "losses/actor_loss", last_actor_loss_val, global_step
                    )
                if not np.isnan(actor_grad_norm):
                    writer.add_scalar(
                        "grads/actor_grad_norm", actor_grad_norm, global_step
                    )
                    _rrank = getattr(args, "residual_route_rank", 1)
                    writer.add_scalar(
                        "residual/speed_coeff_mean",
                        float(pred_coeffs[:, _rrank].mean().item()),
                        global_step,
                    )
                    writer.add_scalar(
                        "residual/speed_coeff_std",
                        float(pred_coeffs[:, _rrank].std().item()),
                        global_step,
                    )
                    if _rrank > 0 and not backbone.disable_residual_route:
                        writer.add_scalar(
                            "residual/route_coeff_mean",
                            float(pred_coeffs[:, :_rrank].mean().item()),
                            global_step,
                        )

                if global_step % 1000 == 0:
                    print(
                        f"[td3] step={global_step} critic_loss={critic_loss.item():.4f} "
                        f"actor_loss={last_actor_loss_val:.4f} SPS={sps:.3f}",
                        flush=True,
                    )

        # ── Checkpoint save ────────────────────────────────────────────────────
        if global_step > 0 and global_step % args.save_every == 0:
            step_str = f"latest_{global_step:012d}"
            _save_td3_checkpoint(
                exp_folder,
                step_str,
                backbone,
                actor,
                qf1,
                qf2,
                target_backbone,
                actor_target,
                qf1_target,
                qf2_target,
                actor_optimizer,
                critic_optimizer,
                global_step,
                config,
            )
            # Also write config.json in exp_folder (updated step count).
            with open(
                os.path.join(exp_folder, "config.json"), "w", encoding="utf-8"
            ) as f:
                f.write(jsonpickle.encode(config))
            t_buf = time.time()
            replay_buffer.save(exp_folder)
            print(
                f"[td3] Saved checkpoint + buffer ({len(replay_buffer)} transitions) "
                f"at step {global_step} in {time.time() - t_buf:.1f}s",
                flush=True,
            )

    # ── Final checkpoint ──────────────────────────────────────────────────────
    _save_td3_checkpoint(
        exp_folder,
        f"latest_{global_step:012d}",
        backbone,
        actor,
        qf1,
        qf2,
        target_backbone,
        actor_target,
        qf1_target,
        qf2_target,
        actor_optimizer,
        critic_optimizer,
        global_step,
        config,
    )
    replay_buffer.save(exp_folder)
    env.close()
    writer.close()


if __name__ == "__main__":
    main()
