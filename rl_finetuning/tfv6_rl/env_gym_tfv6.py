from __future__ import annotations

import os
import pathlib

import gymnasium as gym
import numpy as np
import zmq
from gymnasium import spaces

from rl_finetuning.tfv6_rl.action_codec import ActionCodec
from rl_finetuning.tfv6_rl.obs_codec import ObsCodec
from rl_finetuning.tfv6_rl.policy_tfv6_ppo import load_training_config


class CARLAEnvTFv6(gym.Env):
    """Gym env for TFv6 PPO training via ZMQ."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, port, config, render_mode="rgb_array"):
        self.num_recv = 0
        self.port = port
        self.initialized = False
        self.rl_config = config

        tfv6_checkpoint = getattr(config, "tfv6_checkpoint", None)
        if tfv6_checkpoint is None:
            raise ValueError("tfv6_checkpoint must be set in config")
        self.training_config = load_training_config(tfv6_checkpoint)

        self.obs_codec = ObsCodec(self.training_config)
        self.action_codec = ActionCodec(self.training_config)

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

        obs_buffers = data[: len(self.obs_codec.specs)]
        observation = self.obs_codec.unpack(obs_buffers)

        idx = len(self.obs_codec.specs)
        info = {
            "n_steps": np.frombuffer(data[idx + 3], dtype=np.int32),
            "suggest": np.frombuffer(data[idx + 4], dtype=np.int32),
        }
        num_sent = np.frombuffer(data[idx + 5], dtype=np.uint64).item()

        if self.num_recv != num_sent:
            raise ValueError(
                "Communication breakdown, Leaderboard sent more frames than client consumed."
                f" num_recv: {self.num_recv}, num_sent: {num_sent}"
            )

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
        }
        num_sent = np.frombuffer(data[idx + 5], dtype=np.uint64).item()

        if self.num_recv != num_sent:
            raise ValueError(
                "Communication breakdown, Leaderboard sent more frames than client consumed."
                f" num_recv: {self.num_recv}, num_sent: {num_sent}"
            )

        return observation, reward, termination, truncation, info

    def close(self):
        print("Called close!")
