"""
Self-contained PPO training algorithm. Adapted from CleanRL https://github.com/vwxyzjn/cleanrl
"""

# ruff: noqa: E402
import argparse
import datetime
import gc
import math
import os
import pathlib
import random
import re
import time
from collections import deque

try:
    import gymnasium as gym
    from gymnasium.envs.registration import register
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "gymnasium is required for rl_finetuning/train_tfv6_ppo.py. "
        "Install with `pip install gymnasium` (or `pip install -r requirements.txt`)."
    ) from exc
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import zmq
from pytictoc import TicToc
from tensorboardX import SummaryWriter
from torch import nn, optim
from tqdm import tqdm

from rl_finetuning.tfv6_rl.path_utils import ensure_carl_paths

ensure_carl_paths()

import rl_utils as rl_u  # noqa: E402

from rl_finetuning.tfv6_rl.policy_tfv6_ppo import TFv6PPOPolicy  # noqa: E402
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


def _is_cuda_device(device: torch.device | str) -> bool:
    if isinstance(device, torch.device):
        return device.type == "cuda"
    return str(device).startswith("cuda")


def _log_cuda_memory_snapshot(
    rank: int, device: torch.device | str, tag: str, enabled: bool
) -> None:
    if not enabled or rank != 0:
        return
    if not torch.cuda.is_available() or not _is_cuda_device(device):
        return

    device_obj = device if isinstance(device, torch.device) else torch.device(device)
    torch.cuda.synchronize(device_obj)
    allocated = torch.cuda.memory_allocated(device_obj) / (1024**2)
    reserved = torch.cuda.memory_reserved(device_obj) / (1024**2)
    max_allocated = torch.cuda.max_memory_allocated(device_obj) / (1024**2)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{ts}][debug_memory] rank={rank} tag={tag} "
        f"allocated_mib={allocated:.1f} reserved_mib={reserved:.1f} "
        f"max_allocated_mib={max_allocated:.1f}",
        flush=True,
    )


def _iter_episode_stats(info: dict):
    """Yield (return, length) pairs from vector or scalar episode info."""

    def _iter_from_episode_dict(episode_info: dict, done_mask=None):
        returns = np.asarray(episode_info["r"])
        lengths = np.asarray(episode_info["l"])

        if returns.ndim == 0:
            yield float(returns.item()), int(lengths.item())
            return

        flat_returns = returns.reshape(-1)
        flat_lengths = lengths.reshape(-1)
        if done_mask is not None:
            flat_done_mask = np.asarray(done_mask, dtype=bool).reshape(-1)
        else:
            # Fallback if no explicit done mask is available.
            flat_done_mask = flat_lengths > 0

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
        yield from _iter_from_episode_dict(
            info["episode"], done_mask=info.get("_episode")
        )
        return

    if "tfv6_episode" in info:
        yield from _iter_from_episode_dict(
            info["tfv6_episode"], done_mask=info.get("_tfv6_episode")
        )


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


def _iter_reward_components(info: dict):
    """Yield reward component dicts for completed tfv6 episodes.

    Mirrors the scalar/array handling of _iter_episode_stats/_iter_from_episode_dict.
    """

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


def _grad_norm_from_term(
    term: torch.Tensor, params: list[torch.nn.Parameter]
) -> torch.Tensor:
    if not params:
        return term.detach().new_tensor(0.0)
    grads = torch.autograd.grad(
        term,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    sq_norm = term.detach().new_tensor(0.0)
    for grad in grads:
        if grad is None:
            continue
        sq_norm = sq_norm + grad.detach().float().pow(2).sum()
    return torch.sqrt(sq_norm + 1e-30)


def save(model, optimizer, config, folder, model_file, optimizer_file):
    model_file = os.path.join(folder, model_file)
    torch.save(model.module.state_dict(), model_file)

    if optimizer is not None:
        optimizer_file = os.path.join(folder, optimizer_file)
        torch.save(optimizer.state_dict(), optimizer_file)

    json_config = jsonpickle.encode(config)
    with open(os.path.join(folder, "config.json"), "w", encoding="utf-8") as f2:
        f2.write(json_config)


def _resolve_checkpoint_state_dict(checkpoint_obj):
    if not isinstance(checkpoint_obj, dict):
        raise ValueError(
            f"Unsupported checkpoint type for value head init: {type(checkpoint_obj)}"
        )

    for key in ("model_state_dict", "state_dict", "model"):
        candidate = checkpoint_obj.get(key)
        if isinstance(candidate, dict):
            return candidate

    return checkpoint_obj


def _load_value_head_from_checkpoint(
    policy: nn.Module,
    checkpoint_path: str,
    device: torch.device | str,
    rank: int,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state_dict = _resolve_checkpoint_state_dict(checkpoint)
    target_state_dict = policy.state_dict()

    value_head_state_dict = {}
    num_source_value_head = 0
    num_missing_target = 0
    num_shape_mismatch = 0

    for source_key, source_value in source_state_dict.items():
        normalized_key = (
            source_key[7:] if source_key.startswith("module.") else source_key
        )
        if not normalized_key.startswith("value_head."):
            continue

        num_source_value_head += 1
        target_tensor = target_state_dict.get(normalized_key)
        if target_tensor is None:
            num_missing_target += 1
            continue
        if tuple(target_tensor.shape) != tuple(source_value.shape):
            num_shape_mismatch += 1
            continue

        value_head_state_dict[normalized_key] = source_value

    if not value_head_state_dict:
        raise ValueError(
            f"No compatible value_head parameters found in checkpoint: {checkpoint_path}"
        )

    policy.load_state_dict(value_head_state_dict, strict=False)

    if rank == 0:
        print(
            "[value_head_init] "
            f"source={checkpoint_path} "
            f"loaded={len(value_head_state_dict)} "
            f"source_value_head_keys={num_source_value_head} "
            f"missing_target={num_missing_target} "
            f"shape_mismatch={num_shape_mismatch}",
            flush=True,
        )


def parse_args(config):
    # fmt: off
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--rdzv_addr', default='localhost', type=str, help='Master address for the TCP store.')
    parser.add_argument('--exp_name', type=str, default=config.exp_name, help='the name of this experiment')
    parser.add_argument('--gym_id', type=str, default=config.gym_id, help='the id of the gym environment')
    parser.add_argument('--tfv6_checkpoint',
                      type=str,
                      required=True,
                      help='Path to TFv6 checkpoint folder (contains config.json and model_*.pth).')
    parser.add_argument('--tfv6_prefix',
                      type=str,
                      default='model',
                      help='Prefix of TFv6 model files inside checkpoint folder.')
    parser.add_argument('--train_planning_decoder_only',
                      type=lambda x: bool(strtobool(x)),
                      default=config.train_planning_decoder_only,
                      nargs='?',
                      const=True,
                      help='If true, only TFv6 planning decoder (plus PPO heads) is trainable.')
    parser.add_argument('--skip_perception_heads',
                      type=lambda x: bool(strtobool(x)),
                      default=config.skip_perception_heads,
                      nargs='?',
                      const=True,
                      help='If true, skip TFv6 semantic/depth/box/BEV heads during RL forward.')
    parser.add_argument('--tcp_store_port', type=int, required=True, help='port for the key value store')
    parser.add_argument('--learning_rate',
                      type=float,
                      default=config.learning_rate,
                      help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=config.seed, help='seed of the experiment')
    parser.add_argument('--total_timesteps',
                      type=int,
                      default=config.total_timesteps,
                      help='total timesteps of the experiments')
    parser.add_argument('--torch_deterministic',
                      type=lambda x: bool(strtobool(x)),
                      default=config.torch_deterministic,
                      nargs='?',
                      const=True,
                      help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--allow_tf32',
                      type=lambda x: bool(strtobool(x)),
                      default=config.allow_tf32,
                      nargs='?',
                      const=True,
                      help='whether to use tensor cores, lower numeric precision but faster speed')
    parser.add_argument('--benchmark',
                      type=lambda x: bool(strtobool(x)),
                      default=config.benchmark,
                      nargs='?',
                      const=True,
                      help='Whether to benchmark different algorithms on hardware to find the fastest')
    parser.add_argument('--matmul_precision',
                      type=str,
                      default=config.matmul_precision,
                      help=' Options highest=float32, high=tf32, medium=bfloat16')
    parser.add_argument('--debug_shapes',
                      type=lambda x: bool(strtobool(x)),
                      default=False,
                      nargs='?',
                      const=True,
                      help='if toggled, print observation/action shapes on first step')
    parser.add_argument('--heartbeat_steps',
                      type=int,
                      default=0,
                      help='If >0, print a heartbeat every N rollout steps on rank 0.')
    parser.add_argument('--debug_memory',
                      type=lambda x: bool(strtobool(x)),
                      default=False,
                      nargs='?',
                      const=True,
                      help='If true, print CUDA memory snapshots around minibatch forward/backward.')
    parser.add_argument('--run_dir',
                      type=str,
                      default='',
                      help='Concrete run folder from launcher script. Used for debug artifacts.')
    parser.add_argument('--debug_viz',
                      type=lambda x: bool(strtobool(x)),
                      default=False,
                      nargs='?',
                      const=True,
                      help='If true, dump rollout debug visualizations to run_dir/debug_viz.')
    parser.add_argument('--debug_viz_every_n',
                      type=int,
                      default=1,
                      help='Write one debug visualization every N rollout steps.')
    parser.add_argument('--debug_viz_max_images',
                      type=int,
                      default=0,
                      help='Maximum number of debug images to write (0 means unlimited).')
    parser.add_argument('--debug_viz_image_scale',
                      type=int,
                      default=3,
                      help='Scale factor for BEV rendering in debug visualizations.')

    parser.add_argument('--cuda',
                      type=lambda x: bool(strtobool(x)),
                      default=config.cuda,
                      nargs='?',
                      const=True,
                      help='if toggled, cuda will be enabled by default')
    parser.add_argument('--track',
                      type=lambda x: bool(strtobool(x)),
                      default=config.track,
                      nargs='?',
                      const=True,
                      help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb_project_name',
                      type=str,
                      default=config.wandb_project_name,
                      help='the wandb project name')
    parser.add_argument('--wandb_entity',
                      type=str,
                      default=config.wandb_entity,
                      help='the entity (team) of wandb project')
    parser.add_argument('--capture_video',
                      type=lambda x: bool(strtobool(x)),
                      default=config.capture_video,
                      nargs='?',
                      const=True,
                      help='whether to capture videos of the agent performances (check out `videos` folder)')

    # Algorithm specific arguments
    parser.add_argument('--total_batch_size',
                      type=int,
                      default=config.total_batch_size,
                      help='the total amount of data collected at every step across all environments')
    parser.add_argument('--total_minibatch_size',
                      type=int,
                      default=config.total_minibatch_size,
                      help='the total minibatch sized used for training (across all environments)')
    parser.add_argument('--lr_schedule',
                      default=config.lr_schedule,
                      type=str,
                      help='Which lr schedule to use. Options: (linear, kl, none, step, cosine, cosine_restart)')
    parser.add_argument('--gae',
                      type=lambda x: bool(strtobool(x)),
                      default=config.gae,
                      nargs='?',
                      const=True,
                      help='Use GAE for advantage computation')
    parser.add_argument('--gamma', type=float, default=config.gamma, help='the discount factor gamma')
    parser.add_argument('--gae_lambda',
                      type=float,
                      default=config.gae_lambda,
                      help='the lambda for the general advantage estimation')
    parser.add_argument('--update_epochs',
                      type=int,
                      default=config.update_epochs,
                      help='the K epochs to update the policy')
    parser.add_argument('--norm_adv',
                      type=lambda x: bool(strtobool(x)),
                      default=config.norm_adv,
                      nargs='?',
                      const=True,
                      help='Toggles advantages normalization')
    parser.add_argument('--clip_coef', type=float, default=config.clip_coef, help='the surrogate clipping coefficient')
    parser.add_argument('--clip_vloss',
                      type=lambda x: bool(strtobool(x)),
                      default=config.clip_vloss,
                      nargs='?',
                      const=True,
                      help='Toggles whether or not to use a clipped loss for the value function, as per the paper.')
    parser.add_argument('--ent_coef', type=float, default=config.ent_coef, help='coefficient of the entropy')
    parser.add_argument('--vf_coef', type=float, default=config.vf_coef, help='coefficient of the value function')
    parser.add_argument('--max_grad_norm',
                      type=float,
                      default=config.max_grad_norm,
                      help='the maximum norm for the gradient clipping')
    parser.add_argument('--target_kl', type=float, default=config.target_kl, help='the target KL divergence threshold')
    parser.add_argument('--visualize',
                      type=lambda x: bool(strtobool(x)),
                      default=config.visualize,
                      nargs='?',
                      const=True,
                      help='if toggled, Game will render on screen')
    parser.add_argument('--logdir', type=str, default=config.logdir, help='The directory to log the data into.')
    parser.add_argument('--load_file',
                      type=none_or_str,
                      nargs='?',
                      default=config.load_file,
                      help='model weights for initialization')
    parser.add_argument('--value_head_init_file',
                      type=none_or_str,
                      nargs='?',
                      default=getattr(config, 'value_head_init_file', None),
                      help='Optional RL checkpoint file used to initialize only value_head.* parameters.')
    parser.add_argument('--ports',
                      nargs='+',
                      default=config.ports,
                      type=int,
                      help='Ports of the carla_gym wrapper to connect to.'
                      'It requires to submit a port for every envs'
                      '#ports == --nproc_per_node')
    parser.add_argument('--gpu_ids',
                      nargs='+',
                      default=config.gpu_ids,
                      type=int,
                      help='Which GPUs to train on. Index 0 indicates GPU for rank 0 etc.')
    parser.add_argument('--num_envs_per_proc',
                      type=int,
                      default=config.num_envs_per_proc,
                      help='Number of environments per process. Only considered for dd_ppo.py')
    parser.add_argument('--compile_model',
                      type=lambda x: bool(strtobool(x)),
                      default=config.compile_model,
                      nargs='?',
                      const=True,
                      help='whether to compile the model with torch.compile.')
    parser.add_argument('--expl_coef', type=float, default=config.expl_coef, help='coefficient of the exploration')
    parser.add_argument('--adam_eps', type=float, default=config.adam_eps, help='eps parameter of adam optimizer')
    parser.add_argument('--cpu_collect',
                      type=lambda x: bool(strtobool(x)),
                      default=config.cpu_collect,
                      nargs='?',
                      const=True,
                      help='If true than the agent will be run on the cpu during data collection. Can be faster.')
    parser.add_argument('--stream_obs_minibatches_from_cpu_to_save_gpu_memory',
                      type=lambda x: bool(strtobool(x)),
                      default=getattr(config, 'stream_obs_minibatches_from_cpu_to_save_gpu_memory', False),
                      nargs='?',
                      const=True,
                      help='If true, keep rollout observations on CPU and move only active PPO observation minibatches to GPU during training.')
    parser.add_argument('--use_exploration_suggest',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_exploration_suggest,
                      nargs='?',
                      const=True,
                      help='Whether to use the exploration loss from roach.')
    parser.add_argument('--use_correlated_noise',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_correlated_noise,
                      nargs='?',
                      const=True,
                      help='Use correlated Gaussian noise for plan sampling.')
    parser.add_argument('--correlated_noise_rho',
                      type=float,
                      default=config.correlated_noise_rho,
                      help='Correlation strength for plan noise (0-1).')
    parser.add_argument('--noise_ramp',
                      type=lambda x: bool(strtobool(x)),
                      default=config.noise_ramp,
                      nargs='?',
                      const=True,
                      help='If true, reduce noise on the first point to anchor the route.')
    parser.add_argument('--use_speed_limit_as_max_speed',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_speed_limit_as_max_speed,
                      nargs='?',
                      const=True,
                      help='If true rr_maximum_speed will be overwritten to the current speed limit affecting the ego '
                      'vehicle.')
    parser.add_argument('--beta_min_a_b_value',
                      type=float,
                      default=config.beta_min_a_b_value,
                      help='Nugget that gets added to the softplus output of the network.'
                      'Aims to prevent degenerate distributions with the Beta.')
    parser.add_argument('--log_std_init', '--log-std-init',
                      type=float,
                      default=config.log_std_init,
                      help='Initial bias value for the state-dependent log_std head.')
    parser.add_argument('--log_std_init_route', '--log-std-init-route',
                      type=float,
                      default=getattr(config, 'log_std_init_route', None),
                      help='Optional route-specific log_std_init bias (overrides log_std_init for route group).')
    parser.add_argument('--log_std_init_speed', '--log-std-init-speed',
                      type=float,
                      default=getattr(config, 'log_std_init_speed', None),
                      help='Optional target-speed-specific log_std_init bias (overrides log_std_init for target_speed group).')
    parser.add_argument('--log_std_min', '--log-std-min',
                      type=float,
                      default=config.log_std_min,
                      help='Minimum log_std value after clamping.')
    parser.add_argument('--log_std_max', '--log-std-max',
                      type=float,
                      default=config.log_std_max,
                      help='Maximum log_std value after clamping.')
    parser.add_argument('--log_std_head_lr_mult',
                      type=float,
                      default=config.log_std_head_lr_mult,
                      help='Learning-rate multiplier for the log_std prediction head.')
    parser.add_argument('--value_head_lr_mult',
                      type=float,
                      default=getattr(config, 'value_head_lr_mult', 1.0),
                      help='Learning-rate multiplier for the critic (value_head + value_queries). '
                           'Use >1 to let the critic bootstrap faster than the actor LR allows.')
    parser.add_argument('--disable_learned_noise_head',
                      type=lambda x: bool(strtobool(x)),
                      default=getattr(config, 'disable_learned_noise_head', True),
                      nargs='?',
                      const=True,
                      help='If true, freeze action_noise_head parameters and keep constant-noise behavior.')
    parser.add_argument('--critic_updates_shared_features',
                      type=lambda x: bool(strtobool(x)),
                      default=getattr(config, 'critic_updates_shared_features', True),
                      nargs='?',
                      const=True,
                      help='If false, detach critic input features so value loss does not update shared TFv6 features.')
    parser.add_argument('--action_noise_dist',
                      type=str,
                      default=getattr(config, 'action_noise_dist', 'gaussian'),
                      choices=['gaussian', 'beta'],
                      help='Action noise distribution family for PPO exploration.')
    parser.add_argument('--speed_samping_style', '--speed_sampling_style',
                      dest='speed_sampling_style',
                      type=str,
                      default=getattr(config, 'speed_sampling_style', 'native_categorical'),
                      choices=['mean_gaussian', 'mean_gassian', 'native_categorical'],
                      help='How to sample target speed during PPO. native_categorical samples TFv6 speed bins directly; mean_gaussian keeps the old scalar-mean + noise path.')
    parser.add_argument('--standstill_speed_hold_enabled',
                      type=lambda x: bool(strtobool(x)),
                      default=getattr(config, 'standstill_speed_hold_enabled', True),
                      nargs='?',
                      const=True,
                      help='Repeat a just-sampled target-speed action for a few extra frames when starting from standstill.')
    parser.add_argument('--standstill_speed_hold_frames',
                      type=int,
                      default=getattr(config, 'standstill_speed_hold_frames', 3),
                      help='Number of consecutive control frames to apply the standstill speed hold for, including the initial sampled frame.')
    parser.add_argument('--standstill_speed_hold_ego_speed_threshold',
                      type=float,
                      default=getattr(config, 'standstill_speed_hold_ego_speed_threshold', 0.1),
                      help='Apply standstill speed hold only when ego speed is at or below this threshold in m/s.')
    parser.add_argument('--standstill_speed_hold_target_speed_threshold',
                      type=float,
                      default=getattr(config, 'standstill_speed_hold_target_speed_threshold', 1.0 / 3.6),
                      help='Require sampled target speed to exceed this threshold in m/s before starting the standstill speed hold.')
    parser.add_argument('--speed_temperature',
                      type=float,
                      default=getattr(config, 'speed_temperature', 1.0),
                      help='Temperature for speed categorical distribution. T>1 softens the distribution for more exploration.')
    parser.add_argument('--route_sampling_technique',
                      type=str,
                      default=getattr(config, 'route_sampling_technique', 'spline_curvature_perturbation'),
                      choices=['spline_curvature_perturbation', 'legacy_noise_sampling', 'lowrank_diag_route_sampling'],
                      help='Route stochastic sampling technique. Defaults to spline curvature perturbation.')
    parser.add_argument('--heading_amplitude1_std_init',
                      type=float,
                      default=getattr(config, 'heading_amplitude1_std_init', 0.07),
                      help='Initial std for first heading perturbation amplitude in spline curvature sampling.')
    parser.add_argument('--heading_amplitude2_std_init',
                      type=float,
                      default=getattr(config, 'heading_amplitude2_std_init', 0.03),
                      help='Initial std for second heading perturbation amplitude in spline curvature sampling.')
    parser.add_argument('--path_std_base_frac',
                      type=float,
                      default=getattr(config, 'path_std_base_frac', 0.15),
                      help='Base perturbation strength on the first points for spline curvature sampling.')
    parser.add_argument('--lowrank_route_rank',
                      type=int,
                      default=getattr(config, 'lowrank_route_rank', 6),
                      help='Rank k for lowrank_diag_route_sampling route covariance.')
    parser.add_argument('--lowrank_route_std_init',
                      type=float,
                      default=getattr(config, 'lowrank_route_std_init', 0.015),
                      help='Low-rank mode scale for lowrank_diag_route_sampling.')
    parser.add_argument('--use_kl_to_reference',
                      type=lambda x: bool(strtobool(x)),
                      default=getattr(config, 'use_kl_to_reference', True),
                      nargs='?',
                      const=True,
                      help='Apply KL regularization to a frozen TFv6 reference policy (base checkpoint outputs).')
    parser.add_argument('--kl_to_reference_coef',
                      type=float,
                      default=getattr(config, 'kl_to_reference_coef', 1e-4),
                      help='Coefficient for the KL-to-reference regularization term.')
    parser.add_argument('--use_residual_policy',
                      type=lambda x: bool(strtobool(x)),
                      default=getattr(config, 'use_residual_policy', False),
                      nargs='?',
                      const=True,
                      help='Freeze entire TFv6 and learn a small basis-function residual on top. '
                           'Replaces direct decoder finetuning. Forces use_kl_to_reference=False.')
    parser.add_argument('--residual_route_rank',
                      type=int,
                      default=getattr(config, 'residual_route_rank', 2),
                      help='Number of smooth basis modes for route correction (2=lateral, 4=+longitudinal).')
    parser.add_argument('--residual_alpha',
                      type=float,
                      default=getattr(config, 'residual_alpha', 0.15),
                      help='Max route correction per basis mode in normalized space (alpha * tanh(coeff)).')
    parser.add_argument('--residual_alpha_speed',
                      type=float,
                      default=getattr(config, 'residual_alpha_speed', 0.15),
                      help='Max speed correction in normalized space (fraction of max_speed).')
    parser.add_argument('--use_rpo',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_rpo,
                      nargs='?',
                      const=True,
                      help='Whether to use robust policy optimization (add noise to mean during training)')
    parser.add_argument('--rpo_alpha',
                      type=float,
                      default=config.rpo_alpha,
                      help='Noise added during training is of uniform shape [-rpo_alpha, rpo_alpha]')
    parser.add_argument('--use_new_bev_obs',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_new_bev_obs,
                      nargs='?',
                      const=True,
                      help='Whether to use the new bev observation or the old roach one.')
    parser.add_argument('--obs_num_channels',
                      type=int,
                      default=config.obs_num_channels,
                      help='Number of channels is the BEV image.')
    parser.add_argument('--map_folder',
                      type=str,
                      default=config.map_folder,
                      help='The map folder to use with the new representation. Needs to align with pixels_per_meter')
    parser.add_argument('--pixels_per_meter',
                      type=float,
                      default=config.pixels_per_meter,
                      help='Pixels per meter in the new BEV image.')
    parser.add_argument('--route_width',
                      type=int,
                      default=config.route_width,
                      help='Width of the rendered route in pixel. Only affects new obs.')
    parser.add_argument('--reward_type',
                      type=str,
                      default=config.reward_type,
                      help='Reward function to be used during training. Options: roach, simple_reward')
    parser.add_argument('--consider_tl',
                      type=lambda x: bool(strtobool(x)),
                      default=config.consider_tl,
                      nargs='?',
                      const=True,
                      help='If set to false traffic light infractions are turned off. Used in simple reward')
    parser.add_argument('--eval_time', type=float, default=config.eval_time, help='Time until the model times out.')
    parser.add_argument('--terminal_reward',
                      type=float,
                      default=config.terminal_reward,
                      help='Time until the model times out.')
    parser.add_argument('--normalize_rewards',
                      type=lambda x: bool(strtobool(x)),
                      default=config.normalize_rewards,
                      nargs='?',
                      const=True,
                      help=' Whether to use gymnasiums reward normalization.')
    parser.add_argument('--speeding_infraction',
                      type=lambda x: bool(strtobool(x)),
                      default=config.speeding_infraction,
                      nargs='?',
                      const=True,
                      help='Whether to terminate the route if the agent drives too fast. Only simple reward.')
    parser.add_argument('--min_thresh_lat_dist',
                      type=float,
                      default=config.min_thresh_lat_dist,
                      help='Time until the model times out.')
    parser.add_argument('--num_route_points_rendered',
                      type=int,
                      default=config.num_route_points_rendered,
                      help='Number of route points rendered into the BEV seg observation.')
    parser.add_argument('--use_green_wave',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_green_wave,
                      nargs='?',
                      const=True,
                      help='If True in some routes all TL that the agent encounters are set to green.')
    parser.add_argument('--image_encoder',
                      type=str,
                      default=config.image_encoder,
                      help='Which image cnn encoder to use. Either roach, roach_ln, or timm model name')
    parser.add_argument('--use_comfort_infraction',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_comfort_infraction,
                      nargs='?',
                      const=True,
                      help='Whether to apply a soft penalty if comfort limits are exceeded.')
    parser.add_argument('--use_layer_norm',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_layer_norm,
                      nargs='?',
                      const=True,
                      help='Whether to use LayerNorm before ReLU in MLPs.')
    parser.add_argument('--use_vehicle_close_penalty',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_vehicle_close_penalty,
                      nargs='?',
                      const=True,
                      help='Whether to use a penalty for being too close to the front vehicle.')
    parser.add_argument('--render_green_tl',
                      type=lambda x: bool(strtobool(x)),
                      default=config.render_green_tl,
                      nargs='?',
                      const=True,
                      help='Whether to render green traffic lights into the observation.')
    parser.add_argument('--distribution',
                      type=str,
                      default=config.distribution,
                      help='Distribution used for the action space. Options beta, normal, beta_uni_mix')
    parser.add_argument('--weight_decay',
                      type=float,
                      default=config.weight_decay,
                      help='Weight decay applied to optimizer. AdamW is used when > 0.0')
    parser.add_argument('--use_dd_ppo_preempt',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_dd_ppo_preempt,
                      nargs='?',
                      const=True,
                      help='Whether to use the dd-ppo preemption technique to early stop stragglers.')
    parser.add_argument('--use_termination_hint',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_termination_hint,
                      nargs='?',
                      const=True,
                      help='Whether to give a penalty depending on vehicle speed when crashing or running red light.')
    parser.add_argument('--use_perc_progress',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_perc_progress,
                      nargs='?',
                      const=True,
                      help='Whether to multiply RC reward by percentage away from lane center.')
    parser.add_argument('--use_min_speed_infraction',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_min_speed_infraction,
                      nargs='?',
                      const=True,
                      help='Whether to penalize the agent for driving slower than other agents on avg.')
    parser.add_argument('--use_leave_route_done',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_leave_route_done,
                      nargs='?',
                      const=True,
                      help='Whether to terminate the route when leaving the precomputed path.')
    parser.add_argument('--use_temperature',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_temperature,
                      nargs='?',
                      const=True,
                      help='Whether to scale the output distribution values with a predicted temperature.')
    parser.add_argument('--obs_num_measurements',
                      type=int,
                      default=config.obs_num_measurements,
                      help='Number of scalar measurements in observation.')
    parser.add_argument('--use_extra_control_inputs',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_extra_control_inputs,
                      nargs='?',
                      const=True,
                      help='Whether to use extra control inputs such as integral of past steering.')
    parser.add_argument('--condition_outside_junction',
                      type=lambda x: bool(strtobool(x)),
                      default=config.condition_outside_junction,
                      nargs='?',
                      const=True,
                      help=' Whether to render the route outside junctions.')
    parser.add_argument('--use_layer_norm_policy_head',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_layer_norm_policy_head,
                      nargs='?',
                      const=True,
                      help='Applicable if use_layer_norm=True, whether to also apply layernorm to the policy head.'
                      'Can be useful to remove to allow the policy to predict large values (for a, b of Beta).')
    parser.add_argument('--use_hl_gauss_value_loss',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_hl_gauss_value_loss,
                      nargs='?',
                      const=True,
                      help='Whether to use the histogram loss gauss to train the value head via classification '
                      '(instead of regression + L2)')
    parser.add_argument('--use_outside_route_lanes',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_outside_route_lanes,
                      nargs='?',
                      const=True,
                      help='Whether to terminate the route when invading opposing lanes or sidewalks.')
    parser.add_argument('--use_max_change_penalty',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_max_change_penalty,
                      nargs='?',
                      const=True,
                      help='Whether to apply a soft penalty when the action changes too fast.')
    parser.add_argument('--use_lstm',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_lstm,
                      nargs='?',
                      const=True,
                      help=' Whether to use an LSTM after the feature encoder.')
    parser.add_argument('--terminal_hint',
                      type=float,
                      default=config.terminal_hint,
                      help='Reward at the end of the episode when colliding, the number will be subtracted.')
    parser.add_argument('--terminal_penalty_warmup_n',
                      type=int,
                      default=0,
                      help='Number of steps before a collision terminal to apply linearly increasing negative '
                           'reward shaping. At step T-k the added reward is -terminal_hint*(N-k)/N, '
                           'giving a ramp from 0 at T-N up to -terminal_hint*(N-1)/N at T-1. '
                           '0 disables the feature.')
    parser.add_argument('--penalize_yellow_light',
                      type=lambda x: bool(strtobool(x)),
                      default=config.penalize_yellow_light,
                      nargs='?',
                      const=True,
                      help='Whether to penalize running a yellow light.')
    parser.add_argument('--use_target_point',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_target_point,
                      nargs='?',
                      const=True,
                      help='Whether to input a target point in the measurements.')
    parser.add_argument('--use_value_measurements',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_value_measurements,
                      nargs='?',
                      const=True,
                      help='Whether to use value measurements (otherwise all are set to 0)')
    parser.add_argument('--bev_semantics_width',
                      type=int,
                      default=config.bev_semantics_width,
                      help='Numer of pixels the bev_semantics is wide')
    parser.add_argument('--bev_semantics_height',
                      type=int,
                      default=config.bev_semantics_height,
                      help='Numer of pixels the bev_semantics is high')
    parser.add_argument('--pixels_ev_to_bottom',
                      type=int,
                      default=config.pixels_ev_to_bottom,
                      help='Numer of pixels from the vehicle to the bottom.')
    parser.add_argument('--use_history',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_history,
                      nargs='?',
                      const=True,
                      help='Whether to use historic vehicle and pedestrian observations in BEV obs')
    parser.add_argument('--use_off_road_term',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_off_road_term,
                      nargs='?',
                      const=True,
                      help='Whether to terminate when the agent drives off the drivable area')
    parser.add_argument('--off_road_term_perc',
                      type=float,
                      default=config.off_road_term_perc,
                      help='Percentage of agent overlap with off-road, that triggers the termination')
    parser.add_argument('--beta_1', type=float, default=config.beta_1, help='Beta 1 parameter of adam')
    parser.add_argument('--beta_2', type=float, default=config.beta_2, help='Beta 2 parameter of adam')
    parser.add_argument('--render_speed_lines',
                      type=lambda x: bool(strtobool(x)),
                      default=config.render_speed_lines,
                      nargs='?',
                      const=True,
                      help='Whether to render the speed lines for moving objects')
    parser.add_argument('--use_new_stop_sign_detector',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_new_stop_sign_detector,
                      nargs='?',
                      const=True,
                      help='Whether to use a different stop sign detector that prevents the policy from cheating by '
                      'changing lanes.')
    parser.add_argument('--use_positional_encoding',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_positional_encoding,
                      nargs='?',
                      const=True,
                      help='Whether to add positional encoding to the image')
    parser.add_argument('--use_ttc',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_ttc,
                      nargs='?',
                      const=True,
                      help='Whether to use TTC in the reward.')
    parser.add_argument('--num_value_measurements',
                      type=int,
                      default=config.num_value_measurements,
                      help='Number of measurements exclusive to the value head.')
    parser.add_argument('--render_yellow_time',
                      type=lambda x: bool(strtobool(x)),
                      default=config.render_yellow_time,
                      nargs='?',
                      const=True,
                      help='Whether to indicate the remaining time to red in yellow light rendering.')
    parser.add_argument('--use_single_reward',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_single_reward,
                      nargs='?',
                      const=True,
                      help='Whether to only use RC als reward source in simple reward')
    parser.add_argument('--use_rl_termination_hint',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_rl_termination_hint,
                      nargs='?',
                      const=True,
                      help='Whether to include rl infraction for termination hints')
    parser.add_argument('--render_shoulder',
                      type=lambda x: bool(strtobool(x)),
                      default=config.render_shoulder,
                      nargs='?',
                      const=True,
                      help='Whether to render shoulder lanes as roads.')
    parser.add_argument('--use_shoulder_channel',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_shoulder_channel,
                      nargs='?',
                      const=True,
                      help='Whether to use an extra channel for shoulder lanes.')
    parser.add_argument('--lane_distance_violation_threshold',
                      type=float,
                      default=config.lane_distance_violation_threshold,
                      help='Grace distance in m at which no lane perc penalty is applied')
    parser.add_argument('--lane_dist_penalty_softener',
                      type=float,
                      default=config.lane_dist_penalty_softener,
                      help='If smaller than 1 reduces lane distance penalty')
    parser.add_argument('--comfort_penalty_factor',
                      type=float,
                      default=config.comfort_penalty_factor,
                      help='Max comfort penalty if all comfort metrics are violated.')
    parser.add_argument('--use_survival_reward',
                      type=lambda x: bool(strtobool(x)),
                      default=config.use_survival_reward,
                      nargs='?',
                      const=True,
                      help='Whether to add a constant reward every frame')

    args, unknown = parser.parse_known_args()
    print('Unkown Arguments', unknown)
    # fmt: on
    return args


def make_env(gym_id, args, run_name, port, config):
    def thunk():
        if args.visualize:
            render_mode = "human"  # Only works well with num envs 1 right now
        else:
            render_mode = "rgb_array"

        env = gym.make(gym_id, port=port, config=config, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if args.capture_video:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        if config.normalize_rewards:
            env = gym.wrappers.NormalizeReward(env, gamma=config.gamma)
            env = gym.wrappers.TransformReward(
                env, lambda reward: np.clip(reward, -10, 10)
            )
        return env

    return thunk


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

    # Torchrun initialization
    # Use torchrun for starting because it has proper error handling. Local rank will be set automatically
    rank = int(os.environ["RANK"])  # Rank across all processes
    local_rank = int(os.environ["LOCAL_RANK"])  # Rank on Node
    world_size = int(os.environ["WORLD_SIZE"])  # Number of processes
    print(
        f"RANK, LOCAL_RANK and WORLD_SIZE in environ: {rank}/{local_rank}/{world_size}"
    )

    local_batch_size = args.total_batch_size // world_size
    local_bs_per_env = local_batch_size // args.num_envs_per_proc
    local_minibatch_size = args.total_minibatch_size // world_size
    num_minibatches = local_batch_size // local_minibatch_size

    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}"
    if rank == 0:
        exp_folder = os.path.join(args.logdir, f"{args.exp_name}")
        wandb_folder = os.path.join(exp_folder, "wandb")
        pathlib.Path(exp_folder).mkdir(parents=True, exist_ok=True)
        pathlib.Path(wandb_folder).mkdir(parents=True, exist_ok=True)
        if args.track:
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
                ),  # Can get large if we log all the cpu cores.
            )

        writer = SummaryWriter(exp_folder)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n{}".format(
                "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
            ),
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Is cuda available?:", torch.cuda.is_available())

    # Load the config before overwriting values with current arguments
    if args.load_file is not None:
        load_folder = pathlib.Path(args.load_file).parent.resolve()
        with open(os.path.join(load_folder, "config.json"), encoding="utf-8") as f:
            json_config = f.read()
        # 4 ms, might need to move outside the agent.
        loaded_config = jsonpickle.decode(json_config)
        # Overwrite all properties that were set in the saved config.
        config.__dict__.update(loaded_config.__dict__)

    # Configure config. Converts all arguments into config attributes
    config.initialize(**vars(args))

    if config.use_dd_ppo_preempt:
        # Gloo is used for the parallelization strategy where multiple processes with 1 env run on 1 GPU because NCCL
        # does not support this use case currently.
        num_rollouts_done_store = torch.distributed.TCPStore(
            args.rdzv_addr, args.tcp_store_port, world_size, rank == 0
        )
        torch.distributed.init_process_group(
            backend="gloo"
            if (args.cpu_collect or args.num_envs_per_proc == 1)
            else "nccl",
            store=num_rollouts_done_store,
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=45),
        )
        num_rollouts_done_store.set("num_done", "0")
        print(f"Rank:{rank}, TCP_Store_Port: {args.tcp_store_port}")
    else:
        torch.distributed.init_process_group(
            backend="gloo"
            if (args.cpu_collect or args.num_envs_per_proc == 1)
            else "nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=45),
        )

    device_id = args.gpu_ids[rank]
    print(device_id)
    if device_id < 0:
        print("ERROR! Device id must be positive.")

    train_device = (
        torch.device(f"cuda:{args.gpu_ids[rank]}")
        if torch.cuda.is_available() and args.cuda
        else torch.device("cpu")
    )
    device = train_device

    if torch.cuda.is_available() and args.cuda:
        torch.cuda.device(device)

    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.benchmark = args.benchmark
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    torch.set_float32_matmul_precision(args.matmul_precision)

    if rank == 0:
        json_config = jsonpickle.encode(config)

        with open(os.path.join(exp_folder, "config.json"), "w", encoding="utf-8") as f2:
            f2.write(json_config)

    # NOTE: need to update the config with the argparse arguments before creating the gym environment because the gym env
    # will make a copy of the config and send it to the carla leaderboard process.
    # Sends the updated config to the different environments.
    context = zmq.Context()
    socket_list = []  # So that the garbage collector doesn't clean up sockets before we are done.
    for i in range(args.num_envs_per_proc):
        socket_list.append(context.socket(zmq.PAIR))

        current_folder = pathlib.Path(__file__).parent.resolve()
        comm_folder = current_folder / "tfv6_rl" / "comm_files"
        comm_folder.mkdir(parents=True, exist_ok=True)
        communication_file = comm_folder / str(
            args.ports[args.num_envs_per_proc * local_rank + i]
        )
        socket_list[i].bind(f"ipc://{communication_file}.conf_lock")
        json_config = jsonpickle.encode(config)
        socket_list[i].send_string(json_config)
        _ = socket_list[i].recv_string()
        socket_list[i].close()

    context.term()
    del socket_list

    print(f"Rank {rank}, sent CONFIG files")

    if args.num_envs_per_proc > 1:
        env = gym.vector.AsyncVectorEnv(
            [
                make_env(
                    args.gym_id,
                    args,
                    run_name,
                    args.ports[args.num_envs_per_proc * local_rank + i],
                    config,
                )
                for i in range(args.num_envs_per_proc)
            ]
        )
    else:
        # We don't want to spawn subprocesses for a single environment, so we use Sync==Sequential Vector env.
        env = gym.vector.SyncVectorEnv(
            [
                make_env(
                    args.gym_id,
                    args,
                    run_name,
                    args.ports[args.num_envs_per_proc * local_rank + i],
                    config,
                )
            ]
        )
    assert isinstance(env.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )
    agent = TFv6PPOPolicy(
        env.single_observation_space,
        env.single_action_space,
        tfv6_checkpoint=args.tfv6_checkpoint,
        tfv6_prefix=args.tfv6_prefix,
        device=device,
        rl_config=config,
        train_planning_decoder_only=args.train_planning_decoder_only,
    ).to(device)
    if rank == 0:
        tfv6_trainable = sum(
            p.numel() for p in agent.tfv6.parameters() if p.requires_grad
        )
        tfv6_total = sum(p.numel() for p in agent.tfv6.parameters())
        backbone_trainable = sum(
            p.numel() for p in agent.tfv6.backbone.parameters() if p.requires_grad
        )
        planning_trainable = sum(
            p.numel()
            for p in agent.tfv6.planning_decoder.parameters()
            if p.requires_grad
        )
        print(
            "[debug_amp] "
            f"autocast_enabled={agent.autocast_enabled} "
            f"autocast_dtype={agent.autocast_dtype} "
            f"training_cfg_mixed_precision={agent.training_config.use_mixed_precision_training} "
            f"train_planning_decoder_only={agent.train_planning_decoder_only}",
            flush=True,
        )
        print(
            "[debug_trainable] "
            f"tfv6_trainable={tfv6_trainable}/{tfv6_total} "
            f"backbone_trainable={backbone_trainable} "
            f"planning_decoder_trainable={planning_trainable}",
            flush=True,
        )
    _log_cuda_memory_snapshot(rank, device, "after_policy_init", args.debug_memory)

    if config.compile_model:
        agent = torch.compile(agent)

    if args.value_head_init_file is not None:
        if args.load_file is None:
            _load_value_head_from_checkpoint(
                policy=agent,
                checkpoint_path=args.value_head_init_file,
                device=device,
                rank=rank,
            )
        elif rank == 0:
            print(
                "[value_head_init] ignoring --value_head_init_file because --load_file is set.",
                flush=True,
            )

    start_step = 0
    if args.load_file is not None:
        load_file_name = os.path.basename(args.load_file)
        algo_step = re.findall(r"\d+", load_file_name)
        if len(algo_step) > 0:
            start_step = int(algo_step[0]) + 1  # That step was already finished.
            print("Start training from step:", start_step)
        agent.load_state_dict(
            torch.load(args.load_file, map_location=device), strict=True
        )

    print(f"Rank {rank}, START DistributedDataParallel")
    agent = torch.nn.parallel.DistributedDataParallel(
        agent,
        device_ids=None,
        output_device=None,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    print(f"Rank:{rank}, Created DistributedDataParallel")
    # If we are resuming training use last learning rate from config.
    # If we start a fresh training set the current learning rate according to arguments.
    if args.load_file is None:
        config.current_learning_rate = args.learning_rate

    if rank == 0:
        model_parameters = filter(lambda p: p.requires_grad, agent.parameters())
        num_params = sum(np.prod(p.size()) for p in model_parameters)

        print("Total trainable parameters: ", num_params)

    trainable_named_params = [
        (name, param) for name, param in agent.named_parameters() if param.requires_grad
    ]
    log_std_named_params = [
        (name, param)
        for name, param in trainable_named_params
        if "action_noise_head." in name
    ]
    value_head_named_params = [
        (name, param)
        for name, param in trainable_named_params
        if "value_head." in name or name == "value_queries"
    ]
    value_head_param_ids = {id(p) for _, p in value_head_named_params}
    log_std_param_ids = {id(p) for _, p in log_std_named_params}
    base_named_params = [
        (name, param)
        for name, param in trainable_named_params
        if id(param) not in value_head_param_ids and id(param) not in log_std_param_ids
    ]

    optimizer_param_groups = []
    if base_named_params:
        optimizer_param_groups.append(
            {
                "params": [param for _, param in base_named_params],
                "lr": config.current_learning_rate,
                "is_log_std_head": False,
                "is_value_head": False,
            }
        )
    if log_std_named_params:
        optimizer_param_groups.append(
            {
                "params": [param for _, param in log_std_named_params],
                "lr": config.current_learning_rate * config.log_std_head_lr_mult,
                "is_log_std_head": True,
                "is_value_head": False,
            }
        )
    if value_head_named_params:
        optimizer_param_groups.append(
            {
                "params": [param for _, param in value_head_named_params],
                "lr": config.current_learning_rate * config.value_head_lr_mult,
                "is_log_std_head": False,
                "is_value_head": True,
            }
        )

    if rank == 0 and log_std_named_params:
        print(
            "[optimizer] "
            f"log_std_head_lr_mult={config.log_std_head_lr_mult} "
            f"log_std_head_trainable_params={sum(p.numel() for _, p in log_std_named_params)}",
            flush=True,
        )
    if rank == 0 and value_head_named_params:
        print(
            "[optimizer] "
            f"value_head_lr_mult={config.value_head_lr_mult} "
            f"value_head_trainable_params={sum(p.numel() for _, p in value_head_named_params)}",
            flush=True,
        )

    if config.weight_decay > 0.0:
        optimizer = optim.AdamW(
            optimizer_param_groups,
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
            betas=(config.beta_1, config.beta_2),
        )
    else:
        optimizer = optim.Adam(
            optimizer_param_groups,
            eps=config.adam_eps,
            betas=(config.beta_1, config.beta_2),
        )

    print(f"Rank:{rank}, Created Optimizer")

    # Load optimizer
    if args.load_file is not None:
        try:
            optimizer.load_state_dict(
                torch.load(
                    args.load_file.replace("model_", "optimizer_"), map_location=device
                )
            )
            print(f"Rank:{rank}, Load model")
        except ValueError as exc:
            print(
                f"Rank:{rank}, warning: optimizer state mismatch while loading resume checkpoint ({exc}). "
                "Continuing with freshly initialized optimizer.",
                flush=True,
            )
        if rank == 0:
            writer.add_scalar(
                "charts/restart", 1, config.global_step
            )  # Log that a restart happened

    if config.cpu_collect:
        device = torch.device("cpu")

    obs_storage_on_cpu = bool(
        getattr(
            config,
            "stream_obs_minibatches_from_cpu_to_save_gpu_memory",
            False,
        )
    )
    obs_storage_device = torch.device("cpu") if obs_storage_on_cpu else device
    obs_storage_pin_memory = (
        obs_storage_device.type == "cpu" and train_device.type == "cuda"
    )

    # ALGO Logic: Storage setup
    obs = {}
    for key, space in env.single_observation_space.spaces.items():
        dtype = torch.uint8 if space.dtype == np.uint8 else torch.float32
        obs_kwargs = {}
        if obs_storage_pin_memory:
            obs_kwargs["pin_memory"] = True
        obs[key] = torch.zeros(
            (local_bs_per_env, args.num_envs_per_proc) + space.shape,
            device=obs_storage_device,
            dtype=dtype,
            **obs_kwargs,
        )
    actions = torch.zeros(
        (local_bs_per_env, args.num_envs_per_proc) + env.single_action_space.shape,
        device=device,
    )
    old_mus = torch.zeros(
        (local_bs_per_env, args.num_envs_per_proc) + env.single_action_space.shape,
        device=device,
    )
    old_ref_mus = torch.zeros(
        (local_bs_per_env, args.num_envs_per_proc) + env.single_action_space.shape,
        device=device,
    )
    noise_pred_dim = agent.module.noise_pred_dim
    old_noise_preds = torch.zeros(
        (local_bs_per_env, args.num_envs_per_proc, noise_pred_dim),
        device=device,
    )
    old_diag_log_stds = torch.zeros(
        (local_bs_per_env, args.num_envs_per_proc) + env.single_action_space.shape,
        device=device,
    )
    use_native_categorical_speed = bool(
        agent.module.action_noise_codec.use_native_categorical_speed
    )
    target_speed_logits_dim = (
        len(agent.module.training_config.target_speed_classes)
        if use_native_categorical_speed
        else 0
    )
    old_target_speed_logits = None
    if use_native_categorical_speed:
        old_target_speed_logits = torch.zeros(
            (local_bs_per_env, args.num_envs_per_proc, target_speed_logits_dim),
            device=device,
        )
    old_ref_target_speed_logits = None
    if use_native_categorical_speed and getattr(config, "use_kl_to_reference", False):
        old_ref_target_speed_logits = torch.zeros(
            (local_bs_per_env, args.num_envs_per_proc, target_speed_logits_dim),
            device=device,
        )
    logprobs = torch.zeros((local_bs_per_env, args.num_envs_per_proc), device=device)
    rewards = torch.zeros((local_bs_per_env, args.num_envs_per_proc), device=device)
    dones = torch.zeros((local_bs_per_env, args.num_envs_per_proc), device=device)
    values = torch.zeros((local_bs_per_env, args.num_envs_per_proc), device=device)
    exp_n_steps = np.zeros((local_bs_per_env, args.num_envs_per_proc), dtype=np.int32)
    exp_suggest = np.zeros((local_bs_per_env, args.num_envs_per_proc), dtype=np.int32)

    # TRY NOT TO MODIFY: start the game
    if rank == 0:
        print(
            "[startup] Waiting for first observation from leaderboard env...",
            flush=True,
        )
    reset_obs = env.reset(
        seed=[
            args.seed + rank * args.num_envs_per_proc + i
            for i in range(args.num_envs_per_proc)
        ]
    )
    if rank == 0:
        print("[startup] Received first observation, starting PPO loop.", flush=True)
    next_obs = {}
    for key, space in env.single_observation_space.spaces.items():
        dtype = torch.uint8 if space.dtype == np.uint8 else torch.float32
        next_obs[key] = torch.tensor(reset_obs[0][key], device=device, dtype=dtype)
    if args.debug_shapes and rank == 0:
        obs_shapes = {k: tuple(v.shape) for k, v in next_obs.items()}
        print(
            f"[debug_shapes] obs_shapes={obs_shapes}, action_dim={env.single_action_space.shape}"
        )
    next_done = torch.zeros(args.num_envs_per_proc, device=device)
    next_lstm_state = (
        torch.zeros(
            config.num_lstm_layers,
            args.num_envs_per_proc,
            config.features_dim,
            device=device,
        ),
        torch.zeros(
            config.num_lstm_layers,
            args.num_envs_per_proc,
            config.features_dim,
            device=device,
        ),
    )
    num_updates = args.total_timesteps // args.total_batch_size
    local_processed_samples = 0
    start_time = time.time()
    # TODO This doesn't affect our agent, but it should probably be eval for data collection and train for training
    agent.train()

    if rank == 0:
        avg_returns = deque(maxlen=100)
    # from matplotlib import pyplot as plt
    # pyplots = []
    # fig = plt.figure(figsize=(config.obs_num_channels + 1, config.obs_num_channels + 1))
    # for i in range(1, config.obs_num_channels + 1):
    #   fig.add_subplot(4, 4, i)
    #   pyplots.append(plt.imshow(np.zeros((192, 192)), cmap='gray', vmin=0, vmax=255))
    # plt.show(block=False)

    if config.use_hl_gauss_value_loss:
        hl_gauss_bins = rl_u.hl_gaus_bins(
            config.hl_gauss_vmin,
            config.hl_gauss_vmax,
            config.hl_gauss_bucket_size,
            device,
        )

    print(f"Rank:{rank}, Created obs", flush=True)

    debug_viz = None
    if args.debug_viz and rank == 0:
        from rl_finetuning.tfv6_rl.debug_rollout_viz import PPORolloutVisualizer

        debug_root = (
            args.run_dir if args.run_dir else os.path.join(exp_folder, "latest")
        )
        debug_dir = os.path.join(debug_root, "debug_viz")
        debug_viz = PPORolloutVisualizer(
            training_config=agent.module.training_config
            if isinstance(agent, torch.nn.parallel.DistributedDataParallel)
            else agent.training_config,
            action_codec=agent.module.action_codec
            if isinstance(agent, torch.nn.parallel.DistributedDataParallel)
            else agent.action_codec,
            output_dir=debug_dir,
            num_envs=args.num_envs_per_proc,
            gamma=config.gamma,
            every_n=args.debug_viz_every_n,
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
            speed_temperature=agent.module.speed_temperature
            if isinstance(agent, torch.nn.parallel.DistributedDataParallel)
            else agent.speed_temperature,
            use_residual_policy=getattr(config, "use_residual_policy", False),
        )
        print(
            f"[debug_viz] enabled path={debug_dir} "
            f"every_n={args.debug_viz_every_n} max_images={args.debug_viz_max_images}",
            flush=True,
        )

    for update in tqdm(range(start_step, num_updates), disable=rank != 0):
        if config.cpu_collect:
            device = "cpu"
            agent.to(device)
        # Free all data from last interation.
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

        if config.use_dd_ppo_preempt:
            num_rollouts_done_store.set("num_done", "0")

        # Buffers we use to store returns and aggregate them later to rank 0 for logging.
        total_returns = torch.zeros(
            world_size, device=device, dtype=torch.float32, requires_grad=False
        )
        total_lengths = torch.zeros(
            world_size, device=device, dtype=torch.float32, requires_grad=False
        )
        num_total_returns = torch.zeros(
            world_size, device=device, dtype=torch.int32, requires_grad=False
        )
        total_rc = {
            k: torch.zeros(
                world_size, device=device, dtype=torch.float32, requires_grad=False
            )
            for k in _REWARD_COMPONENT_KEYS
        }
        infraction_counts = {
            k: torch.zeros(
                world_size, device=device, dtype=torch.float32, requires_grad=False
            )
            for k in _INFRACTION_TYPES
        }

        if config.use_lstm:
            initial_lstm_state = (
                next_lstm_state[0].clone(),
                next_lstm_state[1].clone(),
            )

        # Annealing the rate if instructed to do so.
        if config.lr_schedule == "linear":
            frac = 1.0 - (update - 1.0) / num_updates
            config.current_learning_rate = frac * config.learning_rate
        elif config.lr_schedule == "step":
            frac = update / num_updates
            lr_multiplier = 1.0
            for change_percentage in config.lr_schedule_step_perc:
                if frac > change_percentage:
                    lr_multiplier *= config.lr_schedule_step_factor
            config.current_learning_rate = lr_multiplier * config.learning_rate
        elif config.lr_schedule == "cosine":
            frac = update / num_updates
            config.current_learning_rate = (
                0.5 * config.learning_rate * (1 + math.cos(frac * math.pi))
            )
        elif config.lr_schedule == "cosine_restart":
            frac = update / (num_updates + 1)  # + 1 so it doesn't become 100 %
            for idx, frac_restart in enumerate(config.lr_schedule_cosine_restarts):
                if frac >= frac_restart:
                    current_idx = idx
            base_frac = config.lr_schedule_cosine_restarts[current_idx]
            length_current_interval = (
                config.lr_schedule_cosine_restarts[current_idx + 1]
                - config.lr_schedule_cosine_restarts[current_idx]
            )
            frac_current_iter = (frac - base_frac) / length_current_interval
            config.current_learning_rate = (
                0.5 * config.learning_rate * (1 + math.cos(frac_current_iter * math.pi))
            )

        for param_group in optimizer.param_groups:
            if param_group.get("is_log_std_head", False):
                lr_mult = config.log_std_head_lr_mult
            elif param_group.get("is_value_head", False):
                lr_mult = config.value_head_lr_mult
            else:
                lr_mult = 1.0
            param_group["lr"] = config.current_learning_rate * lr_mult

        t0 = TicToc()  # Data collect
        t1 = TicToc()  # Forward pass
        t2 = TicToc()  # Env step
        t3 = TicToc()  # Pre-processing
        t4 = TicToc()  # Train inter
        t5 = TicToc()  # Logging
        t0.tic()
        inference_times = []
        env_times = []
        for step in range(0, local_bs_per_env):
            config.global_step += 1 * world_size * args.num_envs_per_proc
            local_processed_samples += 1 * world_size * args.num_envs_per_proc
            if (
                args.heartbeat_steps > 0
                and rank == 0
                and step % args.heartbeat_steps == 0
            ):
                print(
                    f"[heartbeat] update {update}/{num_updates - 1} "
                    f"step {step}/{local_bs_per_env - 1} "
                    f"global_step={config.global_step}",
                    flush=True,
                )

            for key in obs.keys():
                # Avoid async GPU->temporary CPU->buffer transfers here. With pinned
                # CPU rollout storage, assigning from `next_obs[key].to(...,
                # non_blocking=True)` can leave the stored observation one step stale.
                obs[key][step].copy_(next_obs[key], non_blocking=False)
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                t1.tic()
                (
                    action,
                    logprob,
                    _,
                    value,
                    _,
                    mu,
                    noise_pred,
                    _,
                    base_target_speed_logits,
                    diag_log_std,
                    next_lstm_state,
                ) = agent.forward(next_obs, lstm_state=next_lstm_state, done=next_done)
                if config.use_hl_gauss_value_loss:
                    value_pdf = F.softmax(value, dim=1)
                    value = torch.sum(value_pdf * hl_gauss_bins.unsqueeze(0), dim=1)
                if getattr(config, "use_kl_to_reference", False):
                    ref_mu, ref_target_speed_logits = (
                        agent.module.get_reference_policy_outputs(next_obs)
                    )
                else:
                    ref_mu = mu
                    ref_target_speed_logits = base_target_speed_logits
                inference_times.append(t1.tocvalue())
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            old_mus[step] = mu
            old_ref_mus[step] = ref_mu
            old_noise_preds[step] = noise_pred
            old_diag_log_stds[step] = diag_log_std
            if old_target_speed_logits is not None:
                if base_target_speed_logits is None:
                    raise RuntimeError(
                        "native_categorical speed sampling expects target-speed logits."
                    )
                old_target_speed_logits[step] = base_target_speed_logits
            if old_ref_target_speed_logits is not None:
                if ref_target_speed_logits is None:
                    raise RuntimeError(
                        "Reference TFv6 must provide target-speed logits when "
                        "native_categorical speed sampling is enabled."
                    )
                old_ref_target_speed_logits[step] = ref_target_speed_logits

            # TRY NOT TO MODIFY: execute the game and log data.
            t2.tic()
            next_obs, reward, termination, truncation, info = env.step(
                action.cpu().numpy()
            )
            env_times.append(t2.tocvalue())
            # for i in range(config.obs_num_channels):
            #   pyplots[i].set_data(next_obs['bev_semantics'][0, i])
            # plt.pause(0.0001)

            done = np.logical_or(
                termination, truncation
            )  # Not treated separately in original PPO
            rewards[step] = torch.tensor(reward, device=device, dtype=torch.float32)
            final_info_list = info.get("final_info", [])
            route_completion_by_env = np.full(
                args.num_envs_per_proc, np.nan, dtype=np.float32
            )
            speed_hold_frames_by_env = np.zeros(args.num_envs_per_proc, dtype=np.int32)
            speed_hold_target_speed_by_env = np.full(
                args.num_envs_per_proc, np.nan, dtype=np.float32
            )
            terminal_infraction_by_env = [""] * args.num_envs_per_proc
            if "route_completion" in info:
                route_completion_info = np.asarray(
                    info["route_completion"], dtype=np.float32
                ).reshape(-1)
                num_route_values = min(
                    route_completion_info.shape[0], args.num_envs_per_proc
                )
                route_completion_by_env[:num_route_values] = route_completion_info[
                    :num_route_values
                ]
            if "speed_hold_frames" in info:
                speed_hold_frames_info = np.asarray(
                    info["speed_hold_frames"], dtype=np.int32
                ).reshape(-1)
                num_hold_frame_values = min(
                    speed_hold_frames_info.shape[0], args.num_envs_per_proc
                )
                speed_hold_frames_by_env[:num_hold_frame_values] = (
                    speed_hold_frames_info[:num_hold_frame_values]
                )
            if "speed_hold_target_speed" in info:
                speed_hold_target_speed_info = np.asarray(
                    info["speed_hold_target_speed"], dtype=np.float32
                ).reshape(-1)
                num_hold_target_values = min(
                    speed_hold_target_speed_info.shape[0], args.num_envs_per_proc
                )
                speed_hold_target_speed_by_env[:num_hold_target_values] = (
                    speed_hold_target_speed_info[:num_hold_target_values]
                )
            if "infraction_type" in info:
                infraction_info = np.asarray(
                    info["infraction_type"], dtype=object
                ).reshape(-1)
                num_infraction_values = min(
                    infraction_info.shape[0], args.num_envs_per_proc
                )
                for idx in range(num_infraction_values):
                    if done[idx] and infraction_info[idx] is not None:
                        terminal_infraction_by_env[idx] = str(
                            infraction_info[idx] or ""
                        )
            for idx, single_info in enumerate(final_info_list):
                if idx >= args.num_envs_per_proc or single_info is None:
                    continue
                if "route_completion" in single_info:
                    route_completion_by_env[idx] = float(
                        single_info["route_completion"]
                    )
                if "speed_hold_frames" in single_info:
                    speed_hold_frames_by_env[idx] = int(
                        single_info["speed_hold_frames"]
                    )
                if "speed_hold_target_speed" in single_info:
                    speed_hold_target_speed_by_env[idx] = float(
                        single_info["speed_hold_target_speed"]
                    )
                terminal_infraction_by_env[idx] = str(
                    single_info.get("infraction_type", "") or ""
                )

            if debug_viz is not None:
                for env_idx in range(args.num_envs_per_proc):
                    obs_for_viz = {
                        key: obs[key][step, env_idx].detach().cpu().numpy()
                        for key in env.single_observation_space.spaces.keys()
                    }
                    _last_base = agent.module._last_base_action_mean
                    base_mean_action_np = (
                        np.clip(_last_base[env_idx].cpu().numpy(), -1.0, 1.0)
                        if (
                            getattr(config, "use_residual_policy", False)
                            and _last_base is not None
                            and env_idx < _last_base.shape[0]
                        )
                        else None
                    )
                    debug_viz.maybe_write(
                        global_step=config.global_step,
                        rollout_step=step,
                        update_idx=update,
                        env_idx=env_idx,
                        obs=obs_for_viz,
                        reward=float(reward[env_idx]),
                        done=bool(done[env_idx]),
                        truncated=bool(truncation[env_idx]),
                        sampled_action=np.clip(
                            action[env_idx].detach().cpu().numpy(), -1.0, 1.0
                        ),
                        mean_action=np.clip(
                            mu[env_idx].detach().cpu().numpy(), -1.0, 1.0
                        ),
                        base_mean_action=base_mean_action_np,
                        target_speed_logits=(
                            None
                            if base_target_speed_logits is None
                            else base_target_speed_logits[env_idx]
                            .detach()
                            .cpu()
                            .numpy()
                        ),
                        residual_coeff_preds=(
                            noise_pred[env_idx].detach().cpu().numpy()
                            if getattr(config, "use_residual_policy", False)
                            else None
                        ),
                        log_std=diag_log_std[env_idx].detach().cpu().numpy(),
                        value_estimate=float(value[env_idx].item()),
                        route_completion=float(route_completion_by_env[env_idx]),
                        speed_hold_frames=int(speed_hold_frames_by_env[env_idx]),
                        speed_hold_target_speed=float(
                            speed_hold_target_speed_by_env[env_idx]
                        ),
                    )

            next_done = torch.tensor(done, device=device, dtype=torch.float32)
            next_obs = {
                key: torch.tensor(
                    next_obs[key],
                    device=device,
                    dtype=(
                        torch.uint8
                        if env.single_observation_space.spaces[key].dtype == np.uint8
                        else torch.float32
                    ),
                )
                for key in env.single_observation_space.spaces.keys()
            }
            if "final_info" in info.keys():
                for idx, single_info in enumerate(final_info_list):
                    if single_info is not None:
                        if config.use_exploration_suggest:
                            # Exploration loss
                            exp_n_steps[step, idx] = single_info["n_steps"]
                            exp_suggest[step, idx] = single_info["suggest"]
            if debug_viz is not None:
                for idx in range(args.num_envs_per_proc):
                    if bool(done[idx]):
                        debug_viz.note_episode_outcome(
                            env_idx=idx,
                            terminal_infraction_type=terminal_infraction_by_env[idx],
                        )

            for ep_return, ep_length in _iter_episode_stats(info):
                print(
                    f"rank: {rank}, config.global_step={config.global_step}, "
                    f"episodic_return={ep_return}"
                )
                total_returns[rank] += ep_return
                total_lengths[rank] += ep_length
                num_total_returns[rank] += 1
            for rc in _iter_reward_components(info):
                for k in _REWARD_COMPONENT_KEYS:
                    total_rc[k][rank] += float(rc[k])
            for idx in range(args.num_envs_per_proc):
                if bool(done[idx]):
                    infraction = terminal_infraction_by_env[idx] or "finished_route"
                    if infraction in infraction_counts:
                        infraction_counts[infraction][rank] += 1.0

            if config.use_dd_ppo_preempt:
                num_done = int(num_rollouts_done_store.get("num_done"))
                min_steps = int(config.dd_ppo_min_perc * local_bs_per_env)
                if (
                    num_done / world_size
                ) > config.dd_ppo_preempt_threshold and step > min_steps:
                    print(f"Rank:{rank}, Preempt at step: {step}, Num done: {num_done}")
                    break  # End data collection early the other workers are finished.

        t0.toc(msg=f"Rank:{rank}, Data collection.")
        print(f"Rank:{rank}, Avg forward time {sum(inference_times)}")
        print(f"Rank:{rank}, Avg env time {sum(env_times)}")
        t3.tic()

        if config.use_dd_ppo_preempt:
            num_rollouts_done_store.add("num_done", 1)

        # In case of a dd-ppo preempt this can be smaller than local batch size
        num_collected_steps = step + 1

        # Terminal penalty warmup: add linearly increasing negative rewards to the N steps
        # preceding each collision terminal frame. This provides denser reward signal than
        # relying solely on the sparse terminal event.
        # At step T-k: add -terminal_hint * (N-k)/N  (ramp: 0 at T-N, ~-terminal_hint at T-1)
        # Stops at episode boundaries so only steps in the same episode are shaped.
        if args.terminal_penalty_warmup_n > 0 and args.terminal_hint > 0.0:
            warmup_n = args.terminal_penalty_warmup_n
            # Move to CPU for index-heavy boundary checks, copy back afterwards.
            rewards_cpu = rewards[:num_collected_steps].cpu().clone()
            dones_cpu = dones[:num_collected_steps].cpu()
            collision_threshold = -0.5 * args.terminal_hint
            for env_idx in range(args.num_envs_per_proc):
                for t in range(num_collected_steps):
                    if rewards_cpu[t, env_idx].item() >= collision_threshold:
                        continue  # not a collision terminal frame
                    for k in range(1, warmup_n + 1):
                        t_prev = t - k
                        if t_prev < 0:
                            break  # reached start of buffer
                        if dones_cpu[t_prev, env_idx].item() > 0.5:
                            break  # crossed into a previous episode
                        fraction = float(warmup_n - k) / float(warmup_n)
                        rewards_cpu[t_prev, env_idx] -= args.terminal_hint * fraction
            rewards[:num_collected_steps] = rewards_cpu.to(rewards.device)

        # bootstrap value if not done
        with torch.no_grad():
            if config.use_hl_gauss_value_loss:
                next_value = agent.module.get_value(
                    next_obs, next_lstm_state, next_done
                )
                value_pdf = F.softmax(next_value, dim=1)
                next_value = torch.sum(value_pdf * hl_gauss_bins.unsqueeze(0), dim=1)
            else:
                next_value = agent.module.get_value(
                    next_obs, next_lstm_state, next_done
                ).squeeze(1)
            if args.gae:
                advantages = torch.zeros_like(rewards, device=device)
                lastgaelam = 0.0
                for t in reversed(range(num_collected_steps)):
                    if t == local_bs_per_env - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + args.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards, device=device)
                for t in reversed(range(num_collected_steps)):
                    if t == local_bs_per_env - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        if debug_viz is not None and rank == 0:
            debug_viz.stamp_forward_returns(
                update_idx=update,
                forward_returns=returns[:num_collected_steps].detach().cpu().numpy(),
            )

        if config.cpu_collect:
            device = train_device
            agent.to(device)

        exploration_suggests = np.zeros(
            (num_collected_steps, args.num_envs_per_proc), dtype=np.int32
        )
        if config.use_exploration_suggest:
            for step in range(num_collected_steps):
                for idx in range(args.num_envs_per_proc):
                    n_steps = exp_n_steps[step, idx]
                    if n_steps > 0:
                        n_start = max(0, step - n_steps)
                        exploration_suggests[n_start:step, idx] = exp_suggest[step, idx]

        b_obs = {}
        for key, space in env.single_observation_space.spaces.items():
            b_obs[key] = obs[key][:num_collected_steps].reshape((-1,) + space.shape)
        b_logprobs = logprobs[:num_collected_steps].reshape(-1)
        b_actions = actions[:num_collected_steps].reshape(
            (-1,) + env.single_action_space.shape
        )
        b_dones = dones[:num_collected_steps].reshape(
            -1
        )  # TODO check if pre-emption trick causes problems with LSTM.
        b_advantages = advantages[:num_collected_steps].reshape(-1)
        b_returns = returns[:num_collected_steps].reshape(-1)
        b_values = values[:num_collected_steps].reshape(-1)
        b_old_mus = old_mus[:num_collected_steps].reshape(
            (-1,) + env.single_action_space.shape
        )
        b_old_ref_mus = old_ref_mus[:num_collected_steps].reshape(
            (-1,) + env.single_action_space.shape
        )
        b_old_noise_preds = old_noise_preds[:num_collected_steps].reshape(
            (-1, noise_pred_dim)
        )
        b_old_diag_log_stds = old_diag_log_stds[:num_collected_steps].reshape(
            (-1,) + env.single_action_space.shape
        )
        b_old_target_speed_logits = None
        if old_target_speed_logits is not None:
            b_old_target_speed_logits = old_target_speed_logits[
                :num_collected_steps
            ].reshape((-1, target_speed_logits_dim))
        b_old_ref_target_speed_logits = None
        if old_ref_target_speed_logits is not None:
            b_old_ref_target_speed_logits = old_ref_target_speed_logits[
                :num_collected_steps
            ].reshape((-1, target_speed_logits_dim))
        b_exploration_suggests = exploration_suggests[:num_collected_steps].reshape(-1)

        # When the data was collected on the CPU, move it to GPU before training
        if config.cpu_collect:
            b_logprobs = b_logprobs.to(device)
            b_actions = b_actions.to(device)
            b_dones = b_dones.to(device)
            b_advantages = b_advantages.to(device)
            b_returns = b_returns.to(device)
            b_values = b_values.to(device)
            b_old_mus = b_old_mus.to(device)
            b_old_ref_mus = b_old_ref_mus.to(device)
            b_old_noise_preds = b_old_noise_preds.to(device)
            b_old_diag_log_stds = b_old_diag_log_stds.to(device)
            if b_old_target_speed_logits is not None:
                b_old_target_speed_logits = b_old_target_speed_logits.to(device)
            if b_old_ref_target_speed_logits is not None:
                b_old_ref_target_speed_logits = b_old_ref_target_speed_logits.to(device)

        # Synchronize all processes
        torch.distributed.barrier()
        # Aggregate returns to GPU 0 for logging and storing the best model.
        # Gloo doesn't support AVG, so we implement it via sum / num returns
        torch.distributed.all_reduce(total_returns, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_lengths, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(
            num_total_returns, op=torch.distributed.ReduceOp.SUM
        )
        for k in _REWARD_COMPONENT_KEYS:
            torch.distributed.all_reduce(total_rc[k], op=torch.distributed.ReduceOp.SUM)
        for k in _INFRACTION_TYPES:
            torch.distributed.all_reduce(
                infraction_counts[k], op=torch.distributed.ReduceOp.SUM
            )

        if rank == 0:
            num_total_returns_all_processes = torch.sum(num_total_returns)
            # Only can log return if there was any episode that finished
            if num_total_returns_all_processes > 0:
                total_returns_all_processes = torch.sum(total_returns)
                total_lengths_all_processes = torch.sum(total_lengths)
                avg_return = (
                    total_returns_all_processes / num_total_returns_all_processes
                )
                avg_return = avg_return.item()
                avg_length = (
                    total_lengths_all_processes / num_total_returns_all_processes
                )
                avg_length = avg_length.item()

                avg_returns.append(avg_return)
                windowed_avg_return = sum(avg_returns) / len(avg_returns)

                writer.add_scalar(
                    "charts/episodic_return", avg_return, config.global_step
                )
                writer.add_scalar(
                    "charts/windowed_avg_return",
                    windowed_avg_return,
                    config.global_step,
                )
                writer.add_scalar(
                    "charts/episodic_length", avg_length, config.global_step
                )
                for k in _REWARD_COMPONENT_KEYS:
                    avg_rc = (
                        torch.sum(total_rc[k]) / num_total_returns_all_processes
                    ).item()
                    writer.add_scalar(
                        f"reward_components/{k}", avg_rc, config.global_step
                    )
                total_infraction_episodes = sum(
                    torch.sum(infraction_counts[k]).item() for k in _INFRACTION_TYPES
                )
                if total_infraction_episodes > 0:
                    for k in _INFRACTION_TYPES:
                        frac = (
                            torch.sum(infraction_counts[k]).item()
                            / total_infraction_episodes
                        )
                        writer.add_scalar(f"infractions/{k}", frac, config.global_step)
                if windowed_avg_return >= config.max_training_score:
                    config.max_training_score = windowed_avg_return
                    # Same model could reach multiple high scores
                    if config.best_iteration != update:
                        config.best_iteration = update
                        save(agent, None, config, exp_folder, "model_best.pth", None)

        # Optimizing the policy and value network
        if config.use_lstm:
            assert args.num_envs_per_proc % num_minibatches == 0
            assert not config.use_dd_ppo_preempt

            envsperbatch = args.num_envs_per_proc // num_minibatches
            envinds = np.arange(args.num_envs_per_proc)
            flatinds = np.arange(local_batch_size).reshape(
                local_bs_per_env, args.num_envs_per_proc
            )

        b_inds_original = np.arange(num_collected_steps * args.num_envs_per_proc)

        if config.use_dd_ppo_preempt:
            b_inds_original = np.resize(b_inds_original, (local_batch_size,))

        clipfracs = []

        # Free all data accumulated from data collection.
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
            if torch.cuda.is_available() and _is_cuda_device(device):
                device_obj = (
                    device if isinstance(device, torch.device) else torch.device(device)
                )
                torch.cuda.reset_peak_memory_stats(device_obj)

        t3.toc(msg=f"Rank:{rank}, Data pre-processing.")
        t4.tic()
        _log_cuda_memory_snapshot(
            rank, device, f"update={update}_train_start", args.debug_memory
        )
        latest_epoch = 0
        grad_probe_noise_entropy = None
        grad_probe_noise_policy = None
        kl_to_reference_values = []
        noise_head_params = [
            p for p in agent.module.action_noise_head.parameters() if p.requires_grad
        ]
        for epoch in range(args.update_epochs):
            latest_epoch = epoch
            approx_kl_divs = []
            if config.use_lstm:
                np.random.shuffle(envinds)
            else:
                p = np.random.permutation(len(b_inds_original))
                b_inds = b_inds_original[p]

            total_steps = local_batch_size
            step_size = local_minibatch_size
            if config.use_lstm:
                total_steps = args.num_envs_per_proc
                step_size = envsperbatch

            for start in range(0, total_steps, step_size):
                if config.use_lstm:
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    lstm_state = (
                        initial_lstm_state[0][:, mbenvinds],
                        initial_lstm_state[1][:, mbenvinds],
                    )
                    mb_inds = flatinds[
                        :, mbenvinds
                    ].ravel()  # be really careful about the index
                else:
                    end = start + local_minibatch_size
                    mb_inds = b_inds[start:end]
                    lstm_state = None

                if config.use_exploration_suggest:
                    b_exploration_suggests_sampled = b_exploration_suggests[mb_inds]
                else:
                    b_exploration_suggests_sampled = None
                b_obs_sampled = {}
                for key in b_obs.keys():
                    sampled_obs = b_obs[key][mb_inds]
                    if sampled_obs.device != device:
                        sampled_obs = sampled_obs.to(
                            device,
                            non_blocking=obs_storage_pin_memory,
                        )
                    b_obs_sampled[key] = sampled_obs
                should_log_memory = args.debug_memory and start == 0
                if should_log_memory:
                    _log_cuda_memory_snapshot(
                        rank,
                        device,
                        f"update={update}_epoch={epoch}_pre_forward",
                        True,
                    )
                # Don't need action, so we don't unscale
                (
                    _,
                    newlogprob,
                    entropy,
                    newvalue,
                    exploration_loss,
                    _,
                    new_noise_pred,
                    distribution,
                    new_target_speed_logits,
                    _diag_log_std,
                    _,
                ) = agent.forward(
                    b_obs_sampled,
                    actions=b_actions[mb_inds],
                    exploration_suggests=b_exploration_suggests_sampled,
                    lstm_state=lstm_state,
                    done=b_dones[mb_inds],
                )
                if should_log_memory:
                    _log_cuda_memory_snapshot(
                        rank,
                        device,
                        f"update={update}_epoch={epoch}_post_forward",
                        True,
                    )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    with torch.no_grad():
                        # Distributed mean
                        advantage_mean = mb_advantages.mean()
                        torch.distributed.all_reduce(
                            advantage_mean, op=torch.distributed.ReduceOp.SUM
                        )
                        advantage_mean = advantage_mean / world_size

                        # Distributed std
                        advantage_std = torch.sum(
                            torch.square(mb_advantages - advantage_mean)
                        )
                        torch.distributed.all_reduce(
                            advantage_std, op=torch.distributed.ReduceOp.SUM
                        )
                        advantage_std = advantage_std / (
                            world_size * torch.numel(mb_advantages) - 1
                        )  # -1 is bessel's correction
                        advantage_std = torch.sqrt(advantage_std)

                        mb_advantages = (mb_advantages - advantage_mean) / (
                            advantage_std + 1e-8
                        )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                if args.clip_vloss:
                    # Value clipping is not implemented with HL_Gauss loss
                    assert config.use_hl_gauss_value_loss is False
                    newvalue = newvalue.view(-1)
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    if config.use_hl_gauss_value_loss:
                        target_pdf = rl_u.hl_gaus_pdf(
                            b_returns[mb_inds],
                            config.hl_gauss_std,
                            config.hl_gauss_vmin,
                            config.hl_gauss_vmax,
                            config.hl_gauss_bucket_size,
                        )
                        v_loss = F.cross_entropy(newvalue, target_pdf)
                    else:
                        newvalue = newvalue.view(-1)
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                kl_to_reference_loss = torch.zeros((), device=device)
                if getattr(config, "use_kl_to_reference", False):
                    ref_mu_sampled = b_old_ref_mus[mb_inds]
                    # Keep the same dispersion and only anchor the action mean to the
                    # frozen TFv6 reference policy.
                    ref_distribution, _ = (
                        agent.module.action_noise_codec.proba_distribution(
                            ref_mu_sampled,
                            new_noise_pred.detach(),
                            from_head=False,
                            speed_logits=(
                                None
                                if b_old_ref_target_speed_logits is None
                                else b_old_ref_target_speed_logits[mb_inds]
                            ),
                        )
                    )
                    kl_to_ref = agent.module.action_noise_codec.kl_divergence(
                        distribution, ref_distribution
                    )
                    kl_to_reference_loss = kl_to_ref.mean()
                    kl_to_reference_values.append(kl_to_reference_loss.detach())
                loss = (
                    pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef
                )
                if getattr(config, "use_kl_to_reference", False):
                    loss = loss + config.kl_to_reference_coef * kl_to_reference_loss
                if config.use_exploration_suggest:
                    loss = loss + args.expl_coef * exploration_loss

                if grad_probe_noise_entropy is None and len(noise_head_params) > 0:
                    entropy_term = -config.ent_coef * entropy_loss
                    grad_probe_noise_entropy = _grad_norm_from_term(
                        entropy_term, noise_head_params
                    )
                    grad_probe_noise_policy = _grad_norm_from_term(
                        pg_loss, noise_head_params
                    )

                optimizer.zero_grad()
                loss.backward()
                if should_log_memory:
                    _log_cuda_memory_snapshot(
                        rank,
                        device,
                        f"update={update}_epoch={epoch}_post_backward",
                        True,
                    )
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                if should_log_memory:
                    _log_cuda_memory_snapshot(
                        rank,
                        device,
                        f"update={update}_epoch={epoch}_post_step",
                        True,
                    )

                old_mu_sampled = b_old_mus[mb_inds]
                old_noise_preds_sampled = b_old_noise_preds[mb_inds]
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    # approx_kl = ((ratio - 1) - logratio).mean()

                    if getattr(config, "use_residual_policy", False):
                        # Residual mode: distributions are diagonal Gaussians over
                        # (rank+1) basis coefficients. Reconstruct old dist from
                        # stored head output: [:rank+1] = coeff means, [rank+1:] = log_stds.
                        _r = config.residual_route_rank
                        old_coeff_means = old_noise_preds_sampled[:, : _r + 1]
                        old_coeff_log_stds = old_noise_preds_sampled[:, _r + 1 :]
                        old_normal = torch.distributions.Normal(
                            old_coeff_means, torch.exp(old_coeff_log_stds)
                        )
                        kl_div = torch.distributions.kl_divergence(
                            old_normal, distribution.distribution
                        )
                        if kl_div.ndim > 1:
                            kl_div = kl_div.sum(dim=1)
                        approx_kl_divs.append(kl_div.mean())
                    else:
                        # We compute approx KL according to roach
                        old_distribution, _ = (
                            agent.module.action_noise_codec.proba_distribution(
                                old_mu_sampled,
                                old_noise_preds_sampled,
                                from_head=False,
                                speed_logits=(
                                    None
                                    if b_old_target_speed_logits is None
                                    else b_old_target_speed_logits[mb_inds]
                                ),
                            )
                        )
                        kl_div = agent.module.action_noise_codec.kl_divergence(
                            old_distribution, distribution
                        )
                        approx_kl_divs.append(kl_div.mean())

                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean()]

            approx_kl = torch.mean(torch.stack(approx_kl_divs))
            # Gloo doesn't support AVG, so we implement it via sum / world size
            torch.distributed.barrier()
            torch.distributed.all_reduce(approx_kl, op=torch.distributed.ReduceOp.SUM)
            approx_kl = approx_kl / world_size
            if args.target_kl is not None and config.lr_schedule == "kl":
                if approx_kl > args.target_kl:
                    if config.lr_schedule_step is not None:
                        config.kl_early_stop += 1
                        if config.kl_early_stop >= config.lr_schedule_step:
                            config.current_learning_rate *= 0.5
                            config.kl_early_stop = 0

                    break

        del b_obs  # Remove large array
        t4.toc(msg=f"Rank:{rank}, Training.")
        t5.tic()

        config.latest_iteration = update
        # Avg value to log over all Environments
        # Sync with 3 envs takes 4 ms.
        torch.distributed.barrier()
        # Gloo doesn't support AVG, so we implement it via sum / world size
        torch.distributed.all_reduce(v_loss, op=torch.distributed.ReduceOp.SUM)
        v_loss = v_loss / world_size

        torch.distributed.all_reduce(pg_loss, op=torch.distributed.ReduceOp.SUM)
        pg_loss = pg_loss / world_size

        torch.distributed.all_reduce(entropy_loss, op=torch.distributed.ReduceOp.SUM)
        entropy_loss = entropy_loss / world_size

        if config.use_exploration_suggest:
            torch.distributed.all_reduce(
                exploration_loss, op=torch.distributed.ReduceOp.SUM
            )
            exploration_loss = exploration_loss / world_size

        torch.distributed.all_reduce(old_approx_kl, op=torch.distributed.ReduceOp.SUM)
        old_approx_kl = old_approx_kl / world_size

        torch.distributed.all_reduce(approx_kl, op=torch.distributed.ReduceOp.SUM)
        approx_kl = approx_kl / world_size

        if len(kl_to_reference_values) > 0:
            kl_to_reference_log = torch.mean(torch.stack(kl_to_reference_values))
        else:
            kl_to_reference_log = torch.zeros((), device=device)
        torch.distributed.all_reduce(
            kl_to_reference_log, op=torch.distributed.ReduceOp.SUM
        )
        kl_to_reference_log = kl_to_reference_log / world_size

        b_values = b_values[b_inds_original]
        torch.distributed.all_reduce(b_values, op=torch.distributed.ReduceOp.SUM)
        b_values = b_values / world_size

        b_returns = b_returns[b_inds_original]
        torch.distributed.all_reduce(b_returns, op=torch.distributed.ReduceOp.SUM)
        b_returns = b_returns / world_size

        b_advantages = b_advantages[b_inds_original]
        torch.distributed.all_reduce(b_advantages, op=torch.distributed.ReduceOp.SUM)
        b_advantages = b_advantages / world_size

        clipfracs = torch.mean(torch.stack(clipfracs))
        torch.distributed.all_reduce(clipfracs, op=torch.distributed.ReduceOp.SUM)
        clipfracs = clipfracs / world_size

        if rank == 0:
            save(
                agent,
                optimizer,
                config,
                exp_folder,
                f"model_latest_{update:09d}.pth",
                f"optimizer_latest_{update:09d}.pth",
            )
            frac = update / num_updates
            if config.current_eval_interval_idx < len(config.eval_intervals):
                if frac >= config.eval_intervals[config.current_eval_interval_idx]:
                    save(
                        agent,
                        None,
                        config,
                        exp_folder,
                        f"model_eval_{update:09d}.pth",
                        None,
                    )
                    config.current_eval_interval_idx += 1

            # Cleanup file from last epoch
            for file in os.listdir(exp_folder):
                if file.startswith("model_latest_") and file.endswith(".pth"):
                    if file != f"model_latest_{update:09d}.pth":
                        old_model_file = os.path.join(exp_folder, file)
                        if os.path.isfile(old_model_file):
                            os.remove(old_model_file)
                if file.startswith("optimizer_latest_") and file.endswith(".pth"):
                    if file != f"optimizer_latest_{update:09d}.pth":
                        old_model_file = os.path.join(exp_folder, file)
                        if os.path.isfile(old_model_file):
                            os.remove(old_model_file)

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar(
                "charts/learning_rate",
                optimizer.param_groups[0]["lr"],
                config.global_step,
            )
            writer.add_scalar("losses/value_loss", v_loss.item(), config.global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), config.global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), config.global_step)
            if getattr(config, "use_kl_to_reference", False):
                writer.add_scalar(
                    "losses/kl_to_reference",
                    kl_to_reference_log.item(),
                    config.global_step,
                )
                writer.add_scalar(
                    "losses/kl_to_reference_contrib",
                    (config.kl_to_reference_coef * kl_to_reference_log).item(),
                    config.global_step,
                )
            if config.use_exploration_suggest:
                writer.add_scalar(
                    "losses/exploration", exploration_loss.item(), config.global_step
                )
            writer.add_scalar(
                "losses/old_approx_kl", old_approx_kl.item(), config.global_step
            )
            writer.add_scalar("losses/approx_kl", approx_kl.item(), config.global_step)
            writer.add_scalar("losses/clipfrac", clipfracs.item(), config.global_step)
            writer.add_scalar(
                "losses/explained_variance", explained_var, config.global_step
            )
            writer.add_scalar("losses/latest_epoch", latest_epoch, config.global_step)
            writer.add_scalar(
                "charts/discounted_returns", b_returns.mean().item(), config.global_step
            )
            writer.add_scalar(
                "charts/advantages", b_advantages.mean().item(), config.global_step
            )
            if (
                grad_probe_noise_entropy is not None
                and grad_probe_noise_policy is not None
            ):
                grad_probe_noise_entropy_log = grad_probe_noise_entropy.detach().clone()
                grad_probe_noise_policy_log = grad_probe_noise_policy.detach().clone()
                torch.distributed.all_reduce(
                    grad_probe_noise_entropy_log, op=torch.distributed.ReduceOp.SUM
                )
                torch.distributed.all_reduce(
                    grad_probe_noise_policy_log, op=torch.distributed.ReduceOp.SUM
                )
                grad_probe_noise_entropy_log = grad_probe_noise_entropy_log / world_size
                grad_probe_noise_policy_log = grad_probe_noise_policy_log / world_size
                writer.add_scalar(
                    "grads/noise_head_from_entropy_norm",
                    grad_probe_noise_entropy_log.item(),
                    config.global_step,
                )
                writer.add_scalar(
                    "grads/noise_head_from_policy_norm",
                    grad_probe_noise_policy_log.item(),
                    config.global_step,
                )
                writer.add_scalar(
                    "grads/noise_head_entropy_to_policy_ratio",
                    (
                        grad_probe_noise_entropy_log
                        / (grad_probe_noise_policy_log + 1e-12)
                    ).item(),
                    config.global_step,
                )
            # Policy uncertainty diagnostics from rollout-time log_std samples.
            noise_codec = agent.module.action_noise_codec
            rollout_log_std = b_old_diag_log_stds[
                :, : noise_codec.continuous_action_dim
            ]
            rollout_std = torch.exp(rollout_log_std)
            if rollout_log_std.numel() > 0:
                writer.add_scalar(
                    "policy/log_std_mean",
                    rollout_log_std.mean().item(),
                    config.global_step,
                )
                writer.add_scalar(
                    "policy/log_std_min",
                    rollout_log_std.min().item(),
                    config.global_step,
                )
                writer.add_scalar(
                    "policy/log_std_max",
                    rollout_log_std.max().item(),
                    config.global_step,
                )
                writer.add_scalar(
                    "policy/std_mean", rollout_std.mean().item(), config.global_step
                )
                writer.add_scalar(
                    "policy/std_min", rollout_std.min().item(), config.global_step
                )
                writer.add_scalar(
                    "policy/std_max", rollout_std.max().item(), config.global_step
                )

            action_codec = agent.module.action_codec
            if action_codec.predict_route:
                route_slice = action_codec.slices.route
                writer.add_scalar(
                    "policy/std_route_mean",
                    rollout_std[:, route_slice].mean().item(),
                    config.global_step,
                )
            if action_codec.predict_waypoints:
                waypoints_slice = action_codec.slices.waypoints
                writer.add_scalar(
                    "policy/std_waypoints_mean",
                    rollout_std[:, waypoints_slice].mean().item(),
                    config.global_step,
                )
            if action_codec.predict_target_speed:
                if noise_codec.use_native_categorical_speed:
                    if b_old_target_speed_logits is None:
                        raise RuntimeError(
                            "Missing target-speed logits for native_categorical policy "
                            "logging."
                        )
                    speed_categorical = torch.distributions.Categorical(
                        logits=b_old_target_speed_logits
                    )
                    writer.add_scalar(
                        "policy/target_speed_cat_entropy",
                        speed_categorical.entropy().mean().item(),
                        config.global_step,
                    )
                    speed_values = noise_codec.speed_values.to(
                        device=b_old_target_speed_logits.device,
                        dtype=b_old_target_speed_logits.dtype,
                    )
                    speed_probs = speed_categorical.probs
                    expected_speed = (speed_probs * speed_values.unsqueeze(0)).sum(
                        dim=1
                    ) * action_codec.speed_scale
                    writer.add_scalar(
                        "policy/target_speed_expected",
                        expected_speed.mean().item(),
                        config.global_step,
                    )
                else:
                    speed_slice = action_codec.slices.target_speed
                    writer.add_scalar(
                        "policy/std_target_speed",
                        rollout_std[:, speed_slice].mean().item(),
                        config.global_step,
                    )
            if getattr(config, "use_residual_policy", False):
                # Layout: [route_coeff_0..rank-1 means, speed_mean,
                #          route_coeff_0..rank-1 log_stds, speed_log_std]
                _rrank = getattr(config, "residual_route_rank", 1)
                for i in range(_rrank):
                    writer.add_scalar(
                        f"residual/route_coeff_{i}_mean",
                        b_old_noise_preds[:, i].mean().item(),
                        config.global_step,
                    )
                writer.add_scalar(
                    "residual/speed_coeff_mean",
                    b_old_noise_preds[:, _rrank].mean().item(),
                    config.global_step,
                )
                for i in range(_rrank):
                    writer.add_scalar(
                        f"residual/route_coeff_{i}_log_std",
                        b_old_noise_preds[:, _rrank + 1 + i].mean().item(),
                        config.global_step,
                    )
                writer.add_scalar(
                    "residual/speed_coeff_log_std",
                    b_old_noise_preds[:, 2 * _rrank + 1].mean().item(),
                    config.global_step,
                )
            else:
                for idx, group_name in enumerate(noise_codec.noise_pred_names):
                    writer.add_scalar(
                        f"policy/noise_pred_{group_name}",
                        b_old_noise_preds[:, idx].mean().item(),
                        config.global_step,
                    )
            if noise_codec.distribution_type == "beta":
                # Beta-specific diagnostics: grouped concentration and implied alpha/beta.
                grouped_conc = b_old_noise_preds
                writer.add_scalar(
                    "policy/beta/concentration_mean",
                    grouped_conc.mean().item(),
                    config.global_step,
                )
                writer.add_scalar(
                    "policy/beta/concentration_min",
                    grouped_conc.min().item(),
                    config.global_step,
                )
                writer.add_scalar(
                    "policy/beta/concentration_max",
                    grouped_conc.max().item(),
                    config.global_step,
                )
                for idx, group_name in enumerate(noise_codec.group_names):
                    writer.add_scalar(
                        f"policy/beta/concentration_{group_name}",
                        grouped_conc[:, idx].mean().item(),
                        config.global_step,
                    )

                diag_conc = noise_codec.expand_group_values(grouped_conc)
                mu_clamped = torch.clamp(b_old_mus, -1.0 + 1e-4, 1.0 - 1e-4)
                mu01 = torch.clamp((mu_clamped + 1.0) * 0.5, 1e-4, 1.0 - 1e-4)
                alpha = mu01 * diag_conc
                beta = (1.0 - mu01) * diag_conc
                writer.add_scalar(
                    "policy/beta/alpha_mean", alpha.mean().item(), config.global_step
                )
                writer.add_scalar(
                    "policy/beta/alpha_min", alpha.min().item(), config.global_step
                )
                writer.add_scalar(
                    "policy/beta/alpha_max", alpha.max().item(), config.global_step
                )
                writer.add_scalar(
                    "policy/beta/beta_mean", beta.mean().item(), config.global_step
                )
                writer.add_scalar(
                    "policy/beta/beta_min", beta.min().item(), config.global_step
                )
                writer.add_scalar(
                    "policy/beta/beta_max", beta.max().item(), config.global_step
                )
            # Always log a reward signal, even if no episode terminates.
            writer.add_scalar(
                "charts/mean_reward", rewards.mean().item(), config.global_step
            )
            # Adjusted so it doesn't count the first epoch which is slower than the rest (converges faster)
            print("SPS:", int(local_processed_samples / (time.time() - start_time)))
            writer.add_scalar(
                "charts/SPS",
                int(local_processed_samples / (time.time() - start_time)),
                config.global_step,
            )
            writer.add_scalar("charts/restart", 0, config.global_step)

        t5.toc(msg=f"Rank:{rank}, Logging")

    env.close()
    if rank == 0:
        writer.close()

        save(
            agent,
            optimizer,
            config,
            exp_folder,
            "model_final.pth",
            "optimizer_final.pth",
        )
        wandb.finish(exit_code=0, quiet=True)
        print("Done training.")


if __name__ == "__main__":
    main()
