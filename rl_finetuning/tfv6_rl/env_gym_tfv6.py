from __future__ import annotations

import os
import pathlib

import gymnasium as gym
import numpy as np
import zmq
from gymnasium import spaces

from lead.inference.config_closed_loop import ClosedLoopConfig
from rl_finetuning.tfv6_rl.action_codec import ActionCodec, infer_action_head_usage
from rl_finetuning.tfv6_rl.obs_codec import ObsCodec
from rl_finetuning.tfv6_rl.policy_tfv6_ppo import load_training_config


class CARLAEnvTFv6(gym.Env):
    """Gym env for TFv6 PPO training via ZMQ."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, port, config, render_mode="rgb_array"):
        self.num_recv = 0
        self._counter_resynced = False
        self.port = port
        self.initialized = False
        self.rl_config = config
        self.episode_return = 0.0
        self.episode_length = 0
        self._ep_reward_components = {
            "speed_penalty": 0.0,
            "ttc_frac": 0.0,
            "comfort_penalty": 0.0,
            "lane_centering": 0.0,
            "outside_lanes_frac": 0.0,
        }

        tfv6_checkpoint = getattr(config, "tfv6_checkpoint", None)
        if tfv6_checkpoint is None:
            raise ValueError("tfv6_checkpoint must be set in config")
        self.training_config = load_training_config(tfv6_checkpoint)
        self.closed_loop_config = ClosedLoopConfig(raise_error_on_missing_key=False)

        self.obs_codec = ObsCodec(self.training_config, rl_config=self.rl_config)
        use_route, use_waypoints, use_target_speed = infer_action_head_usage(
            self.training_config,
            steer_modality=self.closed_loop_config.steer_modality,
            throttle_modality=self.closed_loop_config.throttle_modality,
            brake_modality=self.closed_loop_config.brake_modality,
        )
        self.action_codec = ActionCodec(
            self.training_config,
            use_route=use_route,
            use_waypoints=use_waypoints,
            use_target_speed=use_target_speed,
        )

        obs_spaces: dict[str, spaces.Box] = {}
        for spec in self.obs_codec.specs:
            low = 0.0
            high = 255.0
            if spec.dtype != np.uint8:
                low = -np.inf
                high = np.inf
                if spec.key == "rasterized_lidar":
                    low = 0.0
                    high = 1.0
                if spec.key == "command" or spec.key == "next_command":
                    low = 0.0
                    high = 1.0
            obs_spaces[spec.key] = spaces.Box(
                low, high, shape=spec.shape, dtype=spec.dtype
            )

        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(self.action_codec.action_dim,), dtype=np.float32
        )

        self.metadata["render_fps"] = self.rl_config.frame_rate
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)

    def _check_or_resync_counter(self, num_sent: int) -> None:
        if self.num_recv == num_sent:
            return
        # AsyncVectorEnv creates/closes a dummy env during startup. That can advance
        # the leaderboard-side sender counter before the real worker consumes frames.
        # Allow a single startup resync, then keep strict checks afterward.
        if not self._counter_resynced and num_sent >= self.num_recv:
            self.num_recv = int(num_sent)
            self._counter_resynced = True
            print(
                f"[env_gym_tfv6] Counter resync on port {self.port}: num_recv -> {self.num_recv}",
                flush=True,
            )
            return
        raise ValueError(
            "Communication breakdown, Leaderboard sent more frames than client consumed."
            f" num_recv: {self.num_recv}, num_sent: {num_sent}"
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not self.initialized:
            current_folder = pathlib.Path(__file__).parent.resolve()
            comm_folder = os.path.join(current_folder, "comm_files")
            pathlib.Path(comm_folder).mkdir(parents=True, exist_ok=True)
            communication_file = os.path.join(comm_folder, str(self.port))
            self.socket.bind(f"ipc://{communication_file}.lock")
            msg = self.socket.recv_string()
            print(msg)
            self.initialized = True

        data = self.socket.recv_multipart(copy=False)
        self.num_recv += 1
        self.episode_return = 0.0
        self.episode_length = 0
        self._ep_reward_components = {
            "speed_penalty": 0.0,
            "ttc_frac": 0.0,
            "comfort_penalty": 0.0,
            "lane_centering": 0.0,
            "outside_lanes_frac": 0.0,
        }

        obs_buffers = data[: len(self.obs_codec.specs)]
        observation = self.obs_codec.unpack(obs_buffers)

        idx = len(self.obs_codec.specs)
        info = {
            "n_steps": np.frombuffer(data[idx + 3], dtype=np.int32),
            "suggest": np.frombuffer(data[idx + 4], dtype=np.int32),
            "route_completion": np.frombuffer(data[idx + 5], dtype=np.float32),
            "speed_hold_frames": np.frombuffer(data[idx + 6], dtype=np.int32),
            "speed_hold_target_speed": np.frombuffer(data[idx + 7], dtype=np.float32),
            "infraction_type": bytes(data[idx + 8]).decode("utf-8"),
        }
        num_sent = np.frombuffer(data[idx + 14], dtype=np.uint64).item()
        self._check_or_resync_counter(num_sent)

        return observation, info

    def step(self, action):
        self.socket.send(action.tobytes(), copy=False)
        data = self.socket.recv_multipart(copy=False)
        self.num_recv += 1

        obs_buffers = data[: len(self.obs_codec.specs)]
        observation = self.obs_codec.unpack(obs_buffers)

        idx = len(self.obs_codec.specs)
        reward = np.frombuffer(data[idx], dtype=np.float32).item()
        termination = np.frombuffer(data[idx + 1], dtype=bool).item()
        truncation = np.frombuffer(data[idx + 2], dtype=bool).item()

        info = {
            "n_steps": np.frombuffer(data[idx + 3], dtype=np.int32).item(),
            "suggest": np.frombuffer(data[idx + 4], dtype=np.int32).item(),
            "route_completion": np.frombuffer(data[idx + 5], dtype=np.float32).item(),
            "speed_hold_frames": np.frombuffer(data[idx + 6], dtype=np.int32).item(),
            "speed_hold_target_speed": np.frombuffer(
                data[idx + 7], dtype=np.float32
            ).item(),
            "infraction_type": bytes(data[idx + 8]).decode("utf-8"),
            "reward_speed_penalty": np.frombuffer(
                data[idx + 9], dtype=np.float32
            ).item(),
            "reward_ttc_penalty": np.frombuffer(
                data[idx + 10], dtype=np.float32
            ).item(),
            "reward_comfort_penalty": np.frombuffer(
                data[idx + 11], dtype=np.float32
            ).item(),
            "reward_lane_centering": np.frombuffer(
                data[idx + 12], dtype=np.float32
            ).item(),
            "reward_outside_lanes": np.frombuffer(
                data[idx + 13], dtype=np.float32
            ).item(),
        }
        self.episode_return += float(reward)
        self.episode_length += 1
        self._ep_reward_components["speed_penalty"] += info["reward_speed_penalty"]
        self._ep_reward_components["ttc_frac"] += info["reward_ttc_penalty"]
        self._ep_reward_components["comfort_penalty"] += info["reward_comfort_penalty"]
        self._ep_reward_components["lane_centering"] += info["reward_lane_centering"]
        self._ep_reward_components["outside_lanes_frac"] += info["reward_outside_lanes"]
        if termination or truncation:
            n = float(self.episode_length)
            info["tfv6_episode"] = {
                "r": float(self.episode_return),
                "l": int(self.episode_length),
                "speed_penalty": self._ep_reward_components["speed_penalty"] / n,
                "ttc_frac": self._ep_reward_components["ttc_frac"] / n,
                "comfort_penalty": self._ep_reward_components["comfort_penalty"] / n,
                "lane_centering": self._ep_reward_components["lane_centering"] / n,
                "outside_lanes_frac": self._ep_reward_components["outside_lanes_frac"]
                / n,
            }
        num_sent = np.frombuffer(data[idx + 14], dtype=np.uint64).item()
        self._check_or_resync_counter(num_sent)

        return observation, reward, termination, truncation, info

    def close(self):
        print("Called close!")
