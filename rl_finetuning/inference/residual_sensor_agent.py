"""Residual-policy sensor agent for leaderboard evaluation.

Use this agent (--agent=rl_finetuning/inference/residual_sensor_agent.py) for
checkpoints that contain a trained residual head on top of frozen TFv6.

Use lead/inference/sensor_agent.py for plain supervised checkpoints.

Checkpoint directory layout (produced by extract_trained_tfv6_model_from_policy.py):
    config.json            <- TFv6 TrainingConfig   (read by SensorAgent.setup)
    model_rl_finetuned.pth <- TFv6-only weights     (loaded by TFv6PPOPolicy init)
    residual_policy.pth    <- full TFv6PPOPolicy sd  (loaded on top to add residual head)
    rl_config.json         <- residual hyperparams   (required — absence raises an error)
"""

from __future__ import annotations

import json
import logging
import os
import types

import torch

from lead.common.pid_controller import LateralPIDController, PIDController
from lead.inference.closed_loop_inference import (
    ClosedLoopInference,
    ClosedLoopPrediction,
)
from lead.inference.open_loop_inference import OpenLoopPrediction
from lead.inference.sensor_agent import SensorAgent
from rl_finetuning.tfv6_rl.policy_tfv6_ppo import TFv6PPOPolicy

LOG = logging.getLogger(__name__)


def get_entry_point():  # dead: disable
    return "ResidualSensorAgent"


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------


class ResidualClosedLoopInference:
    """Inference wrapper that runs TFv6PPOPolicy in deterministic residual mode.

    Mirrors the interface of ClosedLoopInference.forward() so SensorAgent.run_step()
    works unchanged: receives a dict of sensor tensors, returns a ClosedLoopPrediction.

    The residual correction is applied deterministically (sample_type="mean"):
      action = base_route + alpha * coeff_mean @ basis.T  (route axis)
              base_speed + alpha_speed * speed_coeff_mean  (speed axis)
    No noise is sampled — equivalent to the base TFv6 argmax behaviour.
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

        # Build TFv6PPOPolicy.
        # tfv6_checkpoint=checkpoint_dir means it reads config.json + model_rl_finetuned.pth
        # from the extracted dir to construct the TFv6 architecture and load base weights.
        LOG.info(
            f"[ResidualClosedLoopInference] Building TFv6PPOPolicy from {checkpoint_dir}"
        )
        self.policy = TFv6PPOPolicy(
            tfv6_checkpoint=checkpoint_dir,
            tfv6_prefix="model",
            device=device,
            rl_config=rl_config,
        ).to(device)

        # Load the full residual policy on top — this overwrites the TFv6 weights loaded
        # above with the trained ones and adds the trained residual_head weights.
        policy_path = os.path.join(checkpoint_dir, "residual_policy.pth")
        if not os.path.exists(policy_path):
            raise FileNotFoundError(
                f"residual_policy.pth not found in {checkpoint_dir}. "
                "Re-run extract_trained_tfv6_model_from_policy.py to regenerate."
            )
        LOG.info(
            f"[ResidualClosedLoopInference] Loading residual policy from {policy_path}"
        )
        state_dict = torch.load(policy_path, map_location=device, weights_only=True)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()

        self.step = 0

    @torch.inference_mode()
    def forward(self, data: dict) -> ClosedLoopPrediction:
        self.step += 1

        # --- Deterministic residual forward pass ---
        # sample_type="mean" uses predicted coefficient means — no noise sampling.
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.policy.autocast_dtype,
            enabled=self.policy.autocast_enabled,
        ):
            action_enc, *_ = self.policy.forward(data, sample_type="mean")
        # action_enc: (1, route_dim+1) — normalized encoded action (route_norm ++ speed_norm)

        # Decode to physical route (1, N, 2) and target speed (1, 1)
        route, _, target_speed = self.policy.action_codec.decode_to_control_tensors(
            action_enc, self.device
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

        # Build prediction dataclass.
        # pred_route and pred_target_speed_scalar are the residual-corrected values —
        # the visualizer and curvature calculation in SensorAgent.run_step() use these.
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
            waypoints_steer=steer,  # no waypoint head; reuse route values
            waypoints_throttle=throttle,
            waypoints_brake=brake,
            route_steer=steer,
            target_speed_throttle=throttle,
            target_speed_brake=brake,
        )

    # Borrow the beartype-decorated controller method from ClosedLoopInference.
    # Works because self has the same attributes the method reads
    # (config_closed_loop, lateral_route_controller, config_expert).
    execute_route_and_target_speed = ClosedLoopInference.execute_route_and_target_speed


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ResidualSensorAgent(SensorAgent):
    """SensorAgent for residual RL checkpoints (TFv6 + trained residual head).

    Requires a checkpoint directory produced by extract_trained_tfv6_model_from_policy.py
    containing config.json, model_rl_finetuned.pth, residual_policy.pth and rl_config.json.

    Use lead/inference/sensor_agent.py for plain supervised checkpoints.
    """

    def setup(self, path_to_conf_file: str, _=None, __=None):
        # SensorAgent.setup() handles the Bench2Drive "+route_id" suffix internally,
        # but we need the clean directory path here before calling super().
        clean_dir = path_to_conf_file.split("+", maxsplit=1)[0]

        rl_config = _load_rl_config(clean_dir)
        if rl_config is None:
            raise FileNotFoundError(
                f"rl_config.json not found in {clean_dir}. "
                "ResidualSensorAgent requires a residual checkpoint. "
                "Use lead/inference/sensor_agent.py for supervised checkpoints."
            )

        # super().setup() reads config.json (TFv6 TrainingConfig) and initialises all
        # sensor processing, GPS/routing, video recorder, and post-processors.
        # It also creates a ClosedLoopInference with the TFv6-only weights, which we
        # immediately replace with ResidualClosedLoopInference below.
        super().setup(path_to_conf_file, _, __)

        self.closed_loop_inference = ResidualClosedLoopInference(
            config_closed_loop=self.config_closed_loop,
            config_expert=self.config_expert,
            checkpoint_dir=clean_dir,
            device=self.device,
            rl_config=rl_config,
        )
        LOG.info("[ResidualSensorAgent] ResidualClosedLoopInference ready.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_rl_config(checkpoint_dir: str):
    """Load rl_config.json as a SimpleNamespace, or return None if absent.

    Returns None for supervised checkpoints (no rl_config.json present).
    Returns a SimpleNamespace for residual checkpoints so getattr() works
    the same way TFv6PPOPolicy uses it internally.
    """
    path = os.path.join(checkpoint_dir, "rl_config.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return types.SimpleNamespace(**d)
