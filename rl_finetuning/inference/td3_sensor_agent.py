"""TD3 residual sensor agent for leaderboard evaluation.

Use this agent (--agent=rl_finetuning/inference/td3_sensor_agent.py) for
checkpoints produced by the TD3 residual trainer.

Use rl_finetuning/inference/residual_sensor_agent.py for PPO checkpoints.
Use lead/inference/sensor_agent.py for plain supervised checkpoints.

Checkpoint directory layout (produced by extract_trained_tfv6_model_from_policy.py):
    config.json      <- TFv6 TrainingConfig   (copied from base TFv6 checkpoint)
    model_*.pth      <- TFv6-only weights      (copied from base TFv6 checkpoint)
    td3_policy.pth   <- TD3 checkpoint         (backbone encoder + actor head)
    rl_config.json   <- residual hyperparams   (required — absence raises an error)
"""

from __future__ import annotations

import json
import logging
import os
import types
from collections import deque

import torch

from lead.common.pid_controller import LateralPIDController, PIDController
from lead.inference.closed_loop_inference import (
    ClosedLoopInference,
    ClosedLoopPrediction,
)
from lead.inference.open_loop_inference import OpenLoopPrediction
from lead.inference.sensor_agent import SensorAgent
from rl_finetuning.tfv6_rl.policy_tfv6_td3 import TFv6ResidualActorTD3
from rl_finetuning.tfv6_rl.residual_backbone import TFv6ResidualBackbone

LOG = logging.getLogger(__name__)


def get_entry_point():  # dead: disable
    return "TD3SensorAgent"


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------


class ResidualTD3ClosedLoopInference:
    """Inference wrapper that runs TFv6ResidualBackbone + TFv6ResidualActorTD3.

    Mirrors the interface of ClosedLoopInference.forward() so SensorAgent.run_step()
    works unchanged: receives a dict of sensor tensors, returns a ClosedLoopPrediction.

    The TD3 actor is deterministic — forward() returns the mean action directly
    with no noise sampling, equivalent to the base TFv6 argmax behaviour + residual.
    """

    def __init__(
        self,
        config_closed_loop,
        config_expert,
        checkpoint_dir: str,
        device: torch.device,
        rl_config,
    ):
        self.config_closed_loop = config_closed_loop
        self.config_expert = config_expert
        self.device = device

        # PID controllers — identical setup to ClosedLoopInference.__init__
        self.lateral_route_controller = LateralPIDController(config_closed_loop)
        self.longitudinal_target_speed_controller = PIDController(
            k_p=config_closed_loop.speed_kp,
            k_i=config_closed_loop.speed_ki,
            k_d=config_closed_loop.speed_kd,
            n=config_closed_loop.speed_n,
        )

        # Build TFv6ResidualBackbone.
        # Reads config.json + model_*.pth from checkpoint_dir to load frozen TFv6.
        LOG.info(
            f"[ResidualTD3ClosedLoopInference] Building TFv6ResidualBackbone from {checkpoint_dir}"
        )
        self.backbone = TFv6ResidualBackbone(
            tfv6_checkpoint=checkpoint_dir,
            tfv6_prefix="model",
            device=device,
            rl_config=rl_config,
        ).to(device)

        # Load trained backbone encoder weights (residual_cnn + residual_status_proj)
        # from td3_policy.pth. TFv6 weights are already loaded above from model_*.pth.
        policy_path = os.path.join(checkpoint_dir, "td3_policy.pth")
        if not os.path.exists(policy_path):
            raise FileNotFoundError(
                f"td3_policy.pth not found in {checkpoint_dir}. "
                "Re-run extract_trained_tfv6_model_from_policy.py to regenerate."
            )
        LOG.info(
            f"[ResidualTD3ClosedLoopInference] Loading TD3 checkpoint from {policy_path}"
        )
        td3_ckpt = torch.load(policy_path, map_location=device, weights_only=False)

        self.backbone.load_encoder_state_dict(td3_ckpt["backbone"])
        self.backbone.eval()

        # Build deterministic actor and load its output head weights.
        self.actor = TFv6ResidualActorTD3(self.backbone, rl_config).to(device)
        self.actor.residual_out.load_state_dict(td3_ckpt["actor_residual_out"])
        self.actor.eval()

        # Rolling ego-speed history fed to the residual head. Trained models with
        # speed_history_len > 0 expect obs["speed_history"]; the env builds it as a
        # zero-padded deque of the previous N raw speeds (see env_agent_tfv6.py).
        # Without it, forward_coeffs silently drops the concat and residual_out's
        # first Linear gets the wrong input width.
        self.speed_history_len = int(self.backbone.speed_history_len)
        self._speed_history = (
            deque([0.0] * self.speed_history_len, maxlen=self.speed_history_len)
            if self.speed_history_len > 0
            else None
        )

        self.step = 0

    def inject_speed_history(self, data: dict) -> None:
        """Add the current speed_history window to ``data`` and roll in this speed.

        Mirrors env_agent_tfv6: the observation carries the previous N speeds, then
        the current speed is appended for the next step. No-op when the model was
        trained without speed history.
        """
        if self._speed_history is None:
            return
        data["speed_history"] = torch.tensor(
            self._speed_history, dtype=torch.float32, device=self.device
        ).view(1, self.speed_history_len)
        self._speed_history.append(float(data["speed"].reshape(-1)[0]))

    @torch.inference_mode()
    def forward(self, data: dict) -> ClosedLoopPrediction:
        self.step += 1

        # --- Deterministic TD3 forward pass ---
        # forward_coeffs runs frozen TFv6 + residual CNN → coefficient vector.
        # coeffs_to_action combines base TFv6 action with residual correction.
        self.inject_speed_history(data)
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.backbone.autocast_dtype,
            enabled=self.backbone.autocast_enabled,
        ):
            coeff = self.actor.forward_coeffs(data)
        action = self.actor.coeffs_to_action(coeff)
        # action: (1, route_dim+1) — normalized encoded action (route_norm ++ speed_norm)

        return self.action_to_prediction(action, data)

    def action_to_prediction(self, action, data: dict) -> ClosedLoopPrediction:
        """Decode an encoded action + run the PID controller → ClosedLoopPrediction.

        Split out from forward() so the ensemble agent can reuse the identical
        decode + controller path after averaging coefficients across members.
        """
        # Decode to physical route (1, N, 2) and target speed (1, 1)
        route, _, target_speed = self.backbone.action_codec.decode_to_control_tensors(
            action, self.device
        )

        ego_speed = data["speed"].to(self.device, dtype=torch.float32).view(1, 1)

        # --- Controller: same logic as ClosedLoopInference ---
        steer, throttle, brake = self.execute_route_and_target_speed(
            route, target_speed, ego_speed
        )

        # Suppress throttle while braking (mirrors ClosedLoopInference.ensemble)
        if brake > 0.0:
            throttle = 0.0
            if ego_speed.item() < 0.01:
                steer = 0.0

        open_loop = OpenLoopPrediction(
            pred_future_waypoints=None,
            pred_future_headings=None,
            pred_target_speed_scalar=target_speed,
            pred_target_speed_distribution=None,
            pred_route=route,
            pred_semantic=None,
            pred_depth=None,
            pred_bev_semantic=None,
            pred_bounding_box_vehicle_system=None,
            pred_bounding_box_image_system=None,
            pred_radar_predictions=None,
        )
        return ClosedLoopPrediction(
            **vars(open_loop),
            steer=steer,
            throttle=throttle,
            brake=brake,
            waypoints_steer=steer,
            waypoints_throttle=throttle,
            waypoints_brake=brake,
            route_steer=steer,
            target_speed_throttle=throttle,
            target_speed_brake=brake,
        )

    execute_route_and_target_speed = ClosedLoopInference.execute_route_and_target_speed


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class TD3SensorAgent(SensorAgent):
    """SensorAgent for TD3 residual RL checkpoints (TFv6 + trained residual head).

    Requires a checkpoint directory produced by extract_trained_tfv6_model_from_policy.py
    containing config.json, model_*.pth, td3_policy.pth, and rl_config.json.

    Use rl_finetuning/inference/residual_sensor_agent.py for PPO residual checkpoints.
    Use lead/inference/sensor_agent.py for plain supervised checkpoints.
    """

    def setup(self, path_to_conf_file: str, _=None, __=None):
        clean_dir = path_to_conf_file.split("+", maxsplit=1)[0]

        rl_config = _load_rl_config(clean_dir)
        if rl_config is None:
            raise FileNotFoundError(
                f"rl_config.json not found in {clean_dir}. "
                "TD3SensorAgent requires a TD3 residual checkpoint. "
                "Use lead/inference/sensor_agent.py for supervised checkpoints."
            )

        super().setup(path_to_conf_file, _, __)

        self.closed_loop_inference = ResidualTD3ClosedLoopInference(
            config_closed_loop=self.config_closed_loop,
            config_expert=self.config_expert,
            checkpoint_dir=clean_dir,
            device=self.device,
            rl_config=rl_config,
        )
        LOG.info("[TD3SensorAgent] ResidualTD3ClosedLoopInference ready.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_rl_config(checkpoint_dir: str):
    path = os.path.join(checkpoint_dir, "rl_config.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return types.SimpleNamespace(**d)
