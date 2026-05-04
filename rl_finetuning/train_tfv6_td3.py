"""
TD3 training loop for TFv6 residual RL. Single-process, single-GPU, off-policy.
Adapted from CleanRL td3_continuous_action.py, verified against SB3 TD3 and
ResFiT (Ankile et al. 2024, amazon-science/residual-offpolicy-rl).

Key algorithmic details:
  - Twin Q-networks (Fujimoto 2018) — min target for Bellman update
  - Delayed actor updates from ``utd_actor`` relative to critic UTD
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
from concurrent.futures import ThreadPoolExecutor
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
            if space.dtype == np.uint8:
                dtype = np.uint8
            elif int(np.prod(space.shape)) > 10_000:
                dtype = np.float16  # large feature maps (e.g. bev: 512×10×12) stored as fp16 to halve RAM
            else:
                dtype = np.float32
            self.obs_buf[key] = np.zeros((capacity,) + space.shape, dtype=dtype)
            self.next_obs_buf[key] = np.zeros((capacity,) + space.shape, dtype=dtype)

        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.action_coeffs = np.zeros((capacity, coeff_dim), dtype=np.float32)
        self.base_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.timeouts = np.zeros((capacity, 1), dtype=np.float32)
        self.n_step_gammas = np.ones(
            (capacity, 1), dtype=np.float32
        )  # overwritten by add()

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
        n_step_gamma: float = 0.0,
    ) -> int:
        insert_idx = self.pos
        for key in self.obs_buf:
            self.obs_buf[key][insert_idx] = obs[key]
            self.next_obs_buf[key][insert_idx] = next_obs[key]
        self.actions[insert_idx] = action
        self.action_coeffs[insert_idx] = action_coeffs
        self.base_actions[insert_idx] = base_action
        self.rewards[insert_idx] = reward
        # SB3-style: done = terminated only (not truncated), timeout = truncated.
        self.dones[insert_idx] = float(terminated)
        self.timeouts[insert_idx] = float(truncated)
        self.n_step_gammas[insert_idx] = n_step_gamma
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True
        return insert_idx

    def sample_numpy(self, batch_size: int) -> dict:
        """CPU-only: draw random indices and fancy-index all arrays into contiguous numpy chunks.

        Sorting the indices improves cache locality on large BEV arrays (ascending
        memory reads are friendlier to the hardware prefetcher than random jumps).
        Safe to call from a background thread — reads only, no writes.
        """
        max_idx = self.capacity if self.full else self.pos
        indices = np.random.randint(0, max_idx, size=batch_size)
        indices.sort()
        return {
            "obs": {k: self.obs_buf[k][indices] for k in self.obs_buf},
            "next_obs": {k: self.next_obs_buf[k][indices] for k in self.next_obs_buf},
            "actions": self.actions[indices],
            "action_coeffs": self.action_coeffs[indices],
            "base_actions": self.base_actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
            "timeouts": self.timeouts[indices],
            "n_step_gamma": self.n_step_gammas[indices],
        }

    def numpy_to_tensors(self, np_batch: dict) -> dict:
        """GPU part: move a numpy batch returned by sample_numpy to self.device.

        Must be called from the main thread to avoid CUDA context contention.
        """

        def _t(arr: np.ndarray, key: str | None = None) -> torch.Tensor:
            if arr.dtype == np.uint8:
                dtype = torch.uint8
            elif key == "bev" and arr.dtype == np.float16:
                dtype = torch.float16
            else:
                dtype = torch.float32
            return torch.tensor(arr, dtype=dtype, device=self.device)

        return {
            "obs": {k: _t(v, k) for k, v in np_batch["obs"].items()},
            "next_obs": {k: _t(v, k) for k, v in np_batch["next_obs"].items()},
            "actions": _t(np_batch["actions"]),
            "action_coeffs": _t(np_batch["action_coeffs"]),
            "base_actions": _t(np_batch["base_actions"]),
            "rewards": _t(np_batch["rewards"]),
            "dones": _t(np_batch["dones"]),
            "timeouts": _t(np_batch["timeouts"]),
            "n_step_gamma": _t(np_batch["n_step_gamma"]),
        }

    def sample(self, batch_size: int) -> dict:
        """Sample a random minibatch, return tensors on self.device."""
        return self.numpy_to_tensors(self.sample_numpy(batch_size))

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
            "n_step_gammas": self.n_step_gammas[:fill],
        }
        for key, arr in self.obs_buf.items():
            data[f"obs_{key}"] = arr[:fill]
        for key, arr in self.next_obs_buf.items():
            data[f"next_obs_{key}"] = arr[:fill]
        np.savez(tmp_path, **data)
        os.replace(tmp_path, out_path)

    def load(self, folder: str, default_gamma: float = 0.99) -> bool:
        """Load buffer state from buffer_latest.npz in folder.

        Returns True on success, False if file is missing or incompatible
        (capacity mismatch, unknown obs keys). On failure the buffer stays empty.
        ``default_gamma`` is used to fill n_step_gammas for old buffers that were
        saved before n_step_gammas was added (gives correct 1-step TD behavior).
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
            if "n_step_gammas" in d:
                self.n_step_gammas[:fill] = d["n_step_gammas"]
            else:
                self.n_step_gammas[:fill] = default_gamma
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


# ── N-step accumulator ────────────────────────────────────────────────────────


class NStepAccumulator:
    """Per-env n-step return accumulator for TD3.

    Maintains a sliding window of n consecutive transitions. Once the window
    is full, the oldest transition is flushed with an accumulated n-step reward
    and n-step next_obs. At episode end (terminated or truncated) all pending
    transitions are flushed with their available lookahead.

    With n=1, push() is a transparent pass-through: the transition is returned
    immediately with n_step_gamma = gamma, identical to standard 1-step TD.
    """

    def __init__(self, n: int, gamma: float) -> None:
        self.n = n
        self.gamma = gamma
        self._pending: deque = deque()

    def push(self, transition: dict) -> list[dict]:
        """Add a transition; return a list of transitions ready for the buffer.

        Each returned dict has all original keys plus ``n_step_gamma``
        (gamma^k where k is the actual lookahead used, i.e. gamma^n in the
        steady state and gamma^k < gamma^n near episode boundaries).
        """
        self._pending.append(transition)
        if transition["terminated"] or transition["truncated"]:
            return self._flush_all()
        if len(self._pending) >= self.n:
            return [self._flush_oldest()]
        return []

    def _flush_oldest(self) -> dict:
        """Flush the oldest pending transition with accumulated n-step reward."""
        result = dict(self._pending[0])
        n_step_reward = 0.0
        discount = 1.0
        for trans in self._pending:
            n_step_reward += discount * trans["reward"]
            discount *= self.gamma
        result["reward"] = n_step_reward
        result["next_obs"] = self._pending[-1]["next_obs"]
        result["terminated"] = self._pending[-1]["terminated"]
        result["truncated"] = self._pending[-1]["truncated"]
        result["n_step_gamma"] = discount  # gamma^n normally, gamma^k at boundaries
        self._pending.popleft()
        return result

    def _flush_all(self) -> list[dict]:
        results = []
        while self._pending:
            results.append(self._flush_oldest())
        return results


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
    parser.add_argument('--utd_actor', type=float,
                        default=getattr(config, 'utd_actor', 0.5),
                        help='Actor updates per collected environment step. '
                             'For UTD=1 and UTD_ACTOR=0.5, actor updates every '
                             '2 critic updates independent of num_envs.')
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
    parser.add_argument('--bc_regularization_coeff', type=float,
                        default=getattr(config, 'bc_regularization_coeff', 0.0),
                        help='L2 regularization weight on actor coefficients (TD3+BC-style). '
                             'reg_loss = gamma * |Q̄| * coeff².mean(). gamma is the maximum '
                             'fraction of Q the reg can cost (achieved when coeff=±1 everywhere). '
                             'Recommended: 0.1.')
    parser.add_argument('--utd_ratio', type=float,
                        default=getattr(config, 'utd_ratio', 1),
                        help='Critic updates per collected environment step (Update-To-Data ratio). '
                             'The trainer multiplies this by num_envs internally.')
    parser.add_argument('--n_step_returns', type=int,
                        default=getattr(config, 'n_step_returns', 1),
                        help='N-step return horizon. 1=standard 1-step TD (default).')

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
    parser.add_argument('--speed_history_len', type=int,
                        default=getattr(config, 'speed_history_len', 0),
                        help='Number of recent speed values appended to actor+critic inputs (0=disabled).')

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

    global_step = int(ckpt.get("global_step", 0))
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

    if args.utd_ratio <= 0:
        raise ValueError("--utd_ratio must be > 0")
    if args.utd_actor <= 0:
        raise ValueError("--utd_actor must be > 0")
    if args.utd_actor > args.utd_ratio:
        raise ValueError("--utd_actor must be <= --utd_ratio")

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
    _env_factories = [
        make_env(args.gym_id, args, run_name, port, config) for port in args.ports
    ]
    env = gym.vector.AsyncVectorEnv(_env_factories)
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

    coeff_dim = actor.effective_rank + 1

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
        ) + len(args.ports)
        print(
            f"[td3] Resumed from {args.load_file}, global_step={global_step}",
            flush=True,
        )
        if global_step > 0:
            writer.add_scalar("charts/restart", 1, global_step)
            writer.flush()
        t_buf = time.time()
        buf_loaded = replay_buffer.load(exp_folder, default_gamma=args.gamma)
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
        debug_viz.random_burst_probability = 0.0  # disable random bursts in TD3
        print(
            f"[td3][debug_viz] enabled path={debug_dir} "
            f"every_n={args.debug_viz_every_n} burst_len={args.debug_viz_burst_len} "
            f"max_images={args.debug_viz_max_images}",
            flush=True,
        )

    # ── Initial observation ───────────────────────────────────────────────────
    print("[td3] Waiting for first observation...", flush=True)
    reset_obs, _ = env.reset(seed=[args.seed + i for i in range(len(args.ports))])
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
    critic_updates_per_step = int(round(args.utd_ratio * num_envs))
    actor_update_period = int(round(args.utd_ratio / args.utd_actor))
    if critic_updates_per_step <= 0:
        raise ValueError("utd_ratio * num_envs must be >= 1")
    if abs(critic_updates_per_step - args.utd_ratio * num_envs) > 1e-6:
        raise ValueError("utd_ratio * num_envs must be an integer")
    if actor_update_period <= 0:
        raise ValueError("utd_ratio / utd_actor must be >= 1")
    if abs(actor_update_period - args.utd_ratio / args.utd_actor) > 1e-6:
        raise ValueError("utd_ratio / utd_actor must be an integer")
    route_completion_by_env = [0.0] * num_envs
    terminal_infraction_by_env = [""] * num_envs
    # Per-env history of replay buffer indices for terminal_penalty_warmup.
    # With num_envs>1 the buffer interleaves transitions from different envs, so
    # walking back by raw index would contaminate the wrong env's transitions.
    _env_buf_history: list[deque] = [
        deque(maxlen=max(1, args.terminal_penalty_warmup_n + args.n_step_returns - 1))
        for _ in range(num_envs)
    ]
    # Per-env n-step accumulators. With n_step_returns=1 these are pass-throughs.
    nstep_accumulators = [
        NStepAccumulator(n=args.n_step_returns, gamma=args.gamma)
        for _ in range(num_envs)
    ]

    # Infraction accumulator — reset each time infractions are logged.
    infraction_counts: dict[str, int] = {k: 0 for k in _INFRACTION_TYPES}
    episodes_since_infraction_log: int = 0
    # Persistent actor scalars — retained across steps so logging doesn't miss them
    # (actor updates on odd global steps; logging fires on even multiples of 100).
    last_actor_loss_val: float = float("nan")
    last_actor_policy_loss_val: float = float("nan")
    last_actor_reg_loss_val: float = float("nan")
    last_actor_grad_norm: float = float("nan")
    # Per-episode (global_step, reward) pairs for debug-viz forward-return stamping.
    # One list per env so returns are stamped independently.
    _viz_episode_step_rewards: list[list[tuple[int, float]]] = [
        [] for _ in range(num_envs)
    ]

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

    # ── TD3 training loop ──────────────────────────────────────────────────
    # ── Prime async pipeline ──────────────────────────────────────────────────
    # Run the first actor forward and fire step_async before the loop so that
    # the first iteration can immediately overlap UPDATE with CARLA rendering.
    actor.eval()
    with torch.no_grad():
        _prime_coeff = actor.forward_coeffs(next_obs)
        _prime_base_action = backbone._last_base_action_mean
        _prime_noise = torch.randn_like(_prime_coeff) * args.exploration_noise
        _prime_coeff_noisy = (_prime_coeff + _prime_noise).clamp(-1.0, 1.0)
        _prime_action = actor.coeffs_to_action(_prime_coeff_noisy)
        _prime_clean_coeff = _prime_coeff
        _prime_coeff = _prime_coeff_noisy
    action_np_pending = _prime_action.cpu().numpy()
    coeff_np_pending = _prime_coeff.cpu().numpy()
    clean_coeff_np_pending = _prime_clean_coeff.cpu().numpy()
    base_action_np_pending = _prime_base_action.cpu().numpy()
    mean_action_np_pending = actor.coeffs_to_action(_prime_clean_coeff).cpu().numpy()
    target_speed_logits_np_pending = (
        backbone._last_target_speed_logits.cpu().numpy()
        if backbone._last_target_speed_logits is not None
        else None
    )
    obs_features_pending = []
    for _i in range(num_envs):
        _feats = backbone.encode_obs_to_features(env_idx=_i)
        if "privileged_measurements" in next_obs:
            _feats["privileged_measurements"] = (
                next_obs["privileged_measurements"][_i].cpu().numpy()
            )
        if "speed_history" in next_obs:
            _feats["speed_history"] = next_obs["speed_history"][_i].cpu().numpy()
        obs_features_pending.append(_feats)
    env.step_async(action_np_pending)

    # ── Buffer prefetch pool ──────────────────────────────────────────────────
    # One worker per UTD slot — all batches are sampled in parallel during the
    # CARLA render window so every GPU training iteration starts with zero wait.
    _buf_prefetch_pool = ThreadPoolExecutor(max_workers=critic_updates_per_step)
    _buf_prefetch_futures: list = [None] * critic_updates_per_step

    # ── TD3 training loop ─────────────────────────────────────────────────────
    start_step = global_step
    _sps_step = start_step
    _sps_time = start_time
    _t_fwd = _t_env = _t_buf = _t_train = 0.0
    critic_updates_done = (
        max(0, (start_step - args.learning_starts) // num_envs)
        * critic_updates_per_step
    )
    global_step = start_step
    while global_step < args.total_timesteps:
        # ── UPDATE (while CARLA renders from previous step_async) ─────────────
        if (
            global_step >= args.learning_starts
            and len(replay_buffer) >= args.td3_batch_size
        ):
            for utd_idx in range(critic_updates_per_step):
                critic_updates_done += 1
                _t0 = time.perf_counter()
                _future = _buf_prefetch_futures[utd_idx]
                _np_batch = (
                    _future.result()
                    if _future is not None
                    else replay_buffer.sample_numpy(args.td3_batch_size)
                )
                # Refill this slot immediately; it will be ready for the next cycle.
                _buf_prefetch_futures[utd_idx] = _buf_prefetch_pool.submit(
                    replay_buffer.sample_numpy, args.td3_batch_size
                )
                batch = replay_buffer.numpy_to_tensors(_np_batch)
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
                    next_speed_hist = batch["next_obs"].get("speed_history")
                    q1_next = qf1_target(
                        batch["next_obs"],
                        next_coeffs_smoothed,
                        next_priv,
                        next_speed_hist,
                    )
                    q2_next = qf2_target(
                        batch["next_obs"],
                        next_coeffs_smoothed,
                        next_priv,
                        next_speed_hist,
                    )
                    min_q_next = torch.min(q1_next, q2_next)

                    # Handle timeout termination (SB3-style):
                    # For truncated episodes (timeouts) the episode is not truly terminal,
                    # so we still bootstrap. dones=terminated handles this correctly.
                    # n_step_gamma = gamma^n in the steady state; gamma^k at episode
                    # boundaries where k < n transitions were available.
                    target_q = (
                        batch["rewards"]
                        + (1.0 - batch["dones"]) * batch["n_step_gamma"] * min_q_next
                    )

                # Run online Q-networks (actor forward sets backbone state).
                # We run actor.forward_coeffs to set backbone._last_base_action_mean,
                # then pass stored batch["action_coeffs"] to the Q-networks.
                # This avoids re-running TFv6 for actor+qf1+qf2 separately.
                _log_coeffs = actor.forward_coeffs_from_features(
                    batch["obs"]
                )  # sets backbone state; captured for logging
                priv = batch["obs"].get("privileged_measurements")
                speed_hist = batch["obs"].get("speed_history")
                q1_pred = qf1.forward_with_cached_backbone(
                    batch["action_coeffs"], priv, speed_history=speed_hist
                )
                q2_pred = qf2.forward_with_cached_backbone(
                    batch["action_coeffs"], priv, speed_history=speed_hist
                )

                critic_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(
                    q2_pred, target_q
                )

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
                    critic_updates_done > args.critic_warmup_steps
                    and critic_updates_done % actor_update_period == 0
                ):
                    # Features detached: encoder trained by critic loss only (ResFiT style).
                    pred_coeffs = actor.forward_coeffs_from_features(
                        batch["obs"], detach_features=True
                    )
                    actor_policy_loss = -qf1.forward_with_cached_backbone(
                        pred_coeffs,
                        priv,
                        detach_features=True,
                        speed_history=speed_hist,
                    ).mean()
                    q_scale = (
                        actor_policy_loss.detach().abs()
                    )  # |Q̄| from current batch, no grad
                    actor_reg_loss = (
                        args.bc_regularization_coeff
                        * q_scale
                        * pred_coeffs.pow(2).mean()
                    )
                    actor_loss = actor_policy_loss + actor_reg_loss

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_grad_norm = float(
                        nn.utils.clip_grad_norm_(
                            actor.residual_out.parameters(), args.max_grad_norm
                        )
                    )
                    actor_optimizer.step()
                    last_actor_loss_val = float(actor_loss.item())
                    last_actor_policy_loss_val = float(actor_policy_loss.item())
                    last_actor_reg_loss_val = float(actor_reg_loss.item())
                    last_actor_grad_norm = float(actor_grad_norm)

                    # Actor target owns a backbone structurally, but the actor
                    # optimizer only updates residual_out. Keep backbone/Q targets
                    # on the critic-step update schedule above.
                    _polyak_update(
                        actor.residual_out, actor_target.residual_out, args.tau
                    )

                _t_train += time.perf_counter() - _t0

            # ── Logging ───────────────────────────────────────────────────────
            if global_step // 100 > (global_step - num_envs) // 100:
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
                    * _log_coeffs[:, actor.effective_rank].detach()
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
                    writer.add_scalar(
                        "losses/actor_policy_loss",
                        last_actor_policy_loss_val,
                        global_step,
                    )
                    writer.add_scalar(
                        "losses/actor_reg_loss", last_actor_reg_loss_val, global_step
                    )
                if not np.isnan(last_actor_grad_norm):
                    writer.add_scalar(
                        "grads/actor_grad_norm", last_actor_grad_norm, global_step
                    )
                    _eff_rank = actor.effective_rank
                    writer.add_scalar(
                        "residual/speed_coeff_mean",
                        float(pred_coeffs[:, _eff_rank].mean().item()),
                        global_step,
                    )
                    writer.add_scalar(
                        "residual/speed_coeff_std",
                        float(pred_coeffs[:, _eff_rank].std().item()),
                        global_step,
                    )
                    if _eff_rank > 0:
                        writer.add_scalar(
                            "residual/route_coeff_mean",
                            float(pred_coeffs[:, :_eff_rank].mean().item()),
                            global_step,
                        )

                if global_step // 1000 > (global_step - num_envs) // 1000:
                    print(
                        f"[td3] step={global_step} critic_loss={critic_loss.item():.4f} "
                        f"actor_loss={last_actor_loss_val:.4f} SPS={sps:.3f}",
                        flush=True,
                    )

        # ── COLLECT ──────────────────────────────────────────────────────────
        # Collect the result of the step_async fired at the end of the previous
        # iteration (or the pre-loop warmup on the first iteration).
        _t0 = time.perf_counter()
        next_obs_np, reward_np, terminated_np, truncated_np, info = env.step_wait()
        _t_env += time.perf_counter() - _t0

        # Extract route completion and infraction for all envs.
        if "route_completion" in info:
            rc_arr = info["route_completion"]
            for i in range(num_envs):
                route_completion_by_env[i] = float(rc_arr[i])
        if "infraction_type" in info:
            infraction_arr = np.asarray(info["infraction_type"], dtype=object)
            for i in range(num_envs):
                if (terminated_np[i] or truncated_np[i]) and infraction_arr[
                    i
                ] is not None:
                    terminal_infraction_by_env[i] = str(infraction_arr[i] or "")
        if "final_info" in info:
            for idx, single_info in enumerate(info["final_info"]):
                if single_info is not None:
                    terminal_infraction_by_env[idx] = str(
                        single_info.get("infraction_type", "") or ""
                    )

        next_obs_device = _obs_to_device(next_obs_np)
        next_obs = next_obs_device

        # Actor forward on next_obs: computes next action and populates backbone
        # cache so obs_features_next can be extracted without an extra TFv6 pass.
        _t0 = time.perf_counter()
        actor.eval()
        with torch.no_grad():
            coeff_tensor = actor.forward_coeffs(
                next_obs
            )  # sets backbone state; (num_envs, coeff_dim)
            base_action_mean = backbone._last_base_action_mean
            noise = torch.randn_like(coeff_tensor) * args.exploration_noise
            coeff_tensor_noisy = (coeff_tensor + noise).clamp(-1.0, 1.0)
            action_tensor = actor.coeffs_to_action(coeff_tensor_noisy)
            clean_coeff_tensor = coeff_tensor  # deterministic (no noise) for viz
            coeff_tensor = coeff_tensor_noisy

            # Q-estimate for debug viz (env 0 only).
            q_estimate_for_viz: float | None = None
            if debug_viz is not None and global_step >= args.learning_starts:
                priv_viz = next_obs.get("privileged_measurements")
                speed_hist_viz = next_obs.get("speed_history")
                q1_val = qf1.forward_with_cached_backbone(
                    coeff_tensor, priv_viz, speed_history=speed_hist_viz
                )
                q_estimate_for_viz = float(q1_val[0].item())

        action_np = action_tensor.cpu().numpy()  # (num_envs, action_dim)
        coeff_np = coeff_tensor.cpu().numpy()  # (num_envs, coeff_dim)
        clean_coeff_np = clean_coeff_tensor.cpu().numpy()  # (num_envs, coeff_dim)
        base_action_np = base_action_mean.cpu().numpy()  # (num_envs, action_dim)
        mean_action_np = actor.coeffs_to_action(clean_coeff_tensor).cpu().numpy()
        target_speed_logits_np = (
            backbone._last_target_speed_logits.cpu().numpy()
            if backbone._last_target_speed_logits is not None
            else None
        )  # (num_envs, num_speed_bins) or None

        # Extract obs_features_next from backbone cache (no extra TFv6 run).
        # These serve as (a) obs_features for the next pending transition, and
        # (b) next_obs_features for the current transition (non-truncated envs).
        # Must be extracted before truncation overrides overwrite the cache.
        obs_features_next = []
        for i in range(num_envs):
            feats = backbone.encode_obs_to_features(env_idx=i)
            if "privileged_measurements" in next_obs:
                feats["privileged_measurements"] = (
                    next_obs["privileged_measurements"][i].cpu().numpy()
                )
            if "speed_history" in next_obs:
                feats["speed_history"] = next_obs["speed_history"][i].cpu().numpy()
            obs_features_next.append(feats)

        # Build next_obs_features_batch: start from the cache-extracted features,
        # then override truncated envs with their true final obs (Bellman backup).
        # The two loops must be separate: encoding a truncated env's final obs
        # overwrites the backbone cache, corrupting subsequent non-truncated extractions.
        next_obs_features_batch = list(obs_features_next)
        for i in range(num_envs):
            if truncated_np[i] and "final_observation" in info:
                final_obs_np = info["final_observation"]
                final_obs_i: dict[str, torch.Tensor] = {}
                for key, space in env.single_observation_space.spaces.items():
                    dtype = torch.uint8 if space.dtype == np.uint8 else torch.float32
                    if key in final_obs_np:
                        arr = np.asarray(final_obs_np[key][i])
                        final_obs_i[key] = torch.tensor(
                            arr, device=device, dtype=dtype
                        ).unsqueeze(0)
                    else:
                        final_obs_i[key] = next_obs_device[key][i : i + 1]
                with torch.no_grad():
                    feats = backbone.encode_obs_to_features(final_obs_i, env_idx=0)
                if "privileged_measurements" in final_obs_i:
                    feats["privileged_measurements"] = (
                        final_obs_i["privileged_measurements"][0].cpu().numpy()
                    )
                if "speed_history" in final_obs_i:
                    feats["speed_history"] = (
                        final_obs_i["speed_history"][0].cpu().numpy()
                    )
                next_obs_features_batch[i] = feats

        # Buffer add: pass each pending transition through the n-step accumulator
        # before committing to the replay buffer. With n_step_returns=1 the
        # accumulator is a transparent pass-through.
        for i in range(num_envs):
            transition = {
                "obs": obs_features_pending[i],
                "next_obs": next_obs_features_batch[i],
                "action": action_np_pending[i],
                "action_coeffs": coeff_np_pending[i],
                "base_action": base_action_np_pending[i],
                "reward": float(reward_np[i]),
                "terminated": bool(terminated_np[i]),
                "truncated": bool(truncated_np[i]),
            }
            for flush_dict in nstep_accumulators[i].push(transition):
                transition_idx = replay_buffer.add(
                    obs=flush_dict["obs"],
                    next_obs=flush_dict["next_obs"],
                    action=flush_dict["action"],
                    action_coeffs=flush_dict["action_coeffs"],
                    base_action=flush_dict["base_action"],
                    reward=flush_dict["reward"],
                    terminated=flush_dict["terminated"],
                    truncated=flush_dict["truncated"],
                    n_step_gamma=flush_dict["n_step_gamma"],
                )
                _env_buf_history[i].append(transition_idx)
            # Terminal penalty warmup: ramp applied retroactively to the N transitions
            # preceding a penalized terminal frame, using per-env index history so that
            # interleaved multi-env buffer positions don't contaminate other envs.
            if args.terminal_penalty_warmup_n > 0 and args.terminal_hint > 0.0:
                collision_threshold = -0.5 * args.terminal_hint
                if terminated_np[i] and reward_np[i] < collision_threshold:
                    history = list(
                        _env_buf_history[i]
                    )  # oldest … terminal (last entry)
                    warmup_n = args.terminal_penalty_warmup_n
                    for k, idx in enumerate(reversed(history[:-1]), start=1):
                        fraction = float(warmup_n - k) / float(warmup_n)
                        if fraction <= 0.0:
                            break
                        replay_buffer.rewards[idx, 0] -= args.terminal_hint * fraction
            if terminated_np[i] or truncated_np[i]:
                _env_buf_history[i].clear()

        # Kick off the next CARLA render before Python housekeeping.
        env.step_async(action_np)
        # All UTD slots were refilled by the loop above, so in steady state this
        # is a no-op. It only fires on the very first entry or after a skipped UPDATE.
        if (
            global_step >= args.learning_starts
            and len(replay_buffer) >= args.td3_batch_size
        ):
            for _slot in range(critic_updates_per_step):
                if _buf_prefetch_futures[_slot] is None:
                    _buf_prefetch_futures[_slot] = _buf_prefetch_pool.submit(
                        replay_buffer.sample_numpy, args.td3_batch_size
                    )
        _t_fwd += time.perf_counter() - _t0

        # Advance pending state for the next cycle.
        obs_features_pending = obs_features_next
        action_np_pending = action_np
        coeff_np_pending = coeff_np
        clean_coeff_np_pending = clean_coeff_np
        base_action_np_pending = base_action_np
        mean_action_np_pending = mean_action_np
        target_speed_logits_np_pending = target_speed_logits_np

        # Debug viz — one pass per env.
        # Action variables use *_pending (the action that caused the current reward).
        if debug_viz is not None:
            obs_for_viz = {k: v.cpu().numpy() for k, v in next_obs.items()}
            for i in range(num_envs):
                _viz_episode_step_rewards[i].append((global_step, float(reward_np[i])))
                debug_viz.maybe_write(
                    global_step=global_step,
                    rollout_step=global_step,
                    update_idx=global_step,
                    env_idx=i,
                    obs={k: v[i] for k, v in obs_for_viz.items()},
                    reward=float(reward_np[i]),
                    done=bool(terminated_np[i]),
                    truncated=bool(truncated_np[i]),
                    sampled_action=np.clip(action_np_pending[i], -1.0, 1.0),
                    mean_action=np.clip(mean_action_np_pending[i], -1.0, 1.0),
                    base_mean_action=np.clip(base_action_np_pending[i], -1.0, 1.0),
                    target_speed_logits=(
                        target_speed_logits_np_pending[i]
                        if target_speed_logits_np_pending is not None
                        else None
                    ),
                    residual_coeff_preds=np.concatenate(
                        [clean_coeff_np_pending[i], _viz_coeff_log_std]
                    ),
                    residual_attention_weights=None,
                    log_std=_viz_action_log_std,
                    value_estimate=0.0,
                    route_completion=route_completion_by_env[i],
                    speed_hold_frames=0,
                    speed_hold_target_speed=None,
                    q_estimate=q_estimate_for_viz if i == 0 else None,
                )
                if terminated_np[i] or truncated_np[i]:
                    debug_viz.note_episode_outcome(
                        env_idx=i,
                        terminal_infraction_type=terminal_infraction_by_env[i],
                    )
                    G = 0.0
                    step_to_return: dict[int, float] = {}
                    for gs, r in reversed(_viz_episode_step_rewards[i]):
                        G = r + args.gamma * G
                        step_to_return[gs] = G
                    debug_viz.stamp_episode_forward_returns(step_to_return)
                    _viz_episode_step_rewards[i] = []

        recent_rewards.extend(reward_np.tolist())

        # Episode stats — use _iter_episode_stats which handles both gymnasium info
        # formats (final_info per-env dict and legacy top-level info["episode"]).
        # Infraction is attributed to whichever env has a non-empty infraction string;
        # simultaneous multi-env episode endings are rare so this is accurate enough.
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
            infraction = next(
                (
                    terminal_infraction_by_env[i]
                    for i in range(num_envs)
                    if terminal_infraction_by_env[i]
                ),
                "finished_route",
            )
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

        # Clear per-env infraction state so it doesn't bleed into future episodes.
        for i in range(num_envs):
            if terminated_np[i] or truncated_np[i]:
                terminal_infraction_by_env[i] = ""

        # ── Checkpoint save ────────────────────────────────────────────────────
        if (
            global_step > 0
            and global_step // args.save_every
            > (global_step - num_envs) // args.save_every
        ):
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
            buffer_save_s = time.time() - t_buf
            print(
                f"[td3] Saved checkpoint + buffer ({len(replay_buffer)} transitions) "
                f"at step {global_step}; saving buffer to file took {buffer_save_s:.1f}s",
                flush=True,
            )

        global_step += num_envs

    # Drain the pending step_async so env.close() does not hang.
    try:
        env.step_wait()
    except Exception:
        pass
    _buf_prefetch_pool.shutdown(wait=False, cancel_futures=True)

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
