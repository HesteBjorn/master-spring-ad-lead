"""TD3 residual *ensemble* sensor agent for leaderboard evaluation.

Use this agent (--agent=rl_finetuning/inference/td3_ensemble_sensor_agent.py)
to blend several TD3 residual checkpoints that share the same frozen TFv6 base.

The ensemble happens on the residual *coefficients*: each member runs
``actor.forward_coeffs(obs)`` and we take the pure mean before calling
``coeffs_to_action``. Because the frozen TFv6 base action, the fixed
``residual_route_basis`` and ``residual_alpha`` are identical across members,
the averaged coefficients decode to a single well-defined action.

Config string (--agent-config): a comma-separated list of checkpoint dirs,
    dirA,dirB,dirC[+save_name]
each dir produced by extract_trained_tfv6_model_from_policy.py (config.json,
model_*.pth, td3_policy.pth, rl_config.json). The optional ``+save_name``
suffix (handled by SensorAgent) is stripped before splitting on commas.

A single dir works too and is equivalent to td3_sensor_agent.py.
"""

from __future__ import annotations

import logging

import torch

from lead.inference.closed_loop_inference import ClosedLoopPrediction
from rl_finetuning.inference.td3_sensor_agent import (
    ResidualTD3ClosedLoopInference,
    TD3SensorAgent,
    _load_rl_config,
)

LOG = logging.getLogger(__name__)


def get_entry_point():  # dead: disable
    return "TD3EnsembleSensorAgent"


class EnsembleResidualTD3ClosedLoopInference:
    """Average residual coefficients across several TD3 residual members.

    Builds one ``ResidualTD3ClosedLoopInference`` per checkpoint dir (each loads
    its own frozen TFv6 + residual head). forward() averages the per-member
    coefficient vectors and reuses the primary member's decode + PID path.
    """

    def __init__(
        self,
        config_closed_loop,
        config_expert,
        checkpoint_dirs: list[str],
        device: torch.device,
        rl_configs: list,
    ):
        self.device = device
        self.members = [
            ResidualTD3ClosedLoopInference(
                config_closed_loop=config_closed_loop,
                config_expert=config_expert,
                checkpoint_dir=d,
                device=device,
                rl_config=rc,
            )
            for d, rc in zip(checkpoint_dirs, rl_configs, strict=True)
        ]
        # The primary member owns the decode + controller path and the cached
        # base action used by coeffs_to_action.
        self.primary = self.members[0]

        # Coefficient averaging only makes sense if every member shares the same
        # residual basis dimensionality. Guard against mismatched checkpoints.
        ranks = {m.actor.rank for m in self.members}
        action_dims = {m.actor.action_dim for m in self.members}
        if len(ranks) != 1 or len(action_dims) != 1:
            raise ValueError(
                "Ensemble members have incompatible residual shapes: "
                f"ranks={ranks}, action_dims={action_dims}. "
                "All checkpoints must share the same residual_route_rank and action codec."
            )
        LOG.info(
            f"[EnsembleResidualTD3ClosedLoopInference] {len(self.members)} members ready "
            f"(rank={ranks.pop()})."
        )
        self.step = 0

    @torch.inference_mode()
    def forward(self, data: dict) -> ClosedLoopPrediction:
        self.step += 1

        # Each member runs its own frozen TFv6 + residual encoder → coefficients.
        # forward_coeffs also caches that member's base action; the primary's
        # cache is what coeffs_to_action below relies on.
        coeffs = []
        for member in self.members:
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=member.backbone.autocast_dtype,
                enabled=member.backbone.autocast_enabled,
            ):
                coeffs.append(member.actor.forward_coeffs(data))

        coeff = torch.stack(coeffs, dim=0).mean(dim=0)
        action = self.primary.actor.coeffs_to_action(coeff)

        return self.primary.action_to_prediction(action, data)


class TD3EnsembleSensorAgent(TD3SensorAgent):
    """SensorAgent that blends several TD3 residual checkpoints (coefficient mean).

    --agent-config is a comma-separated list of checkpoint dirs sharing one
    frozen TFv6 base. See module docstring.
    """

    def setup(self, path_to_conf_file: str, _=None, __=None):
        # Strip the optional "+save_name" suffix, then split into member dirs.
        clean = path_to_conf_file.split("+", maxsplit=1)[0]
        checkpoint_dirs = [d for d in clean.split(",") if d]
        if not checkpoint_dirs:
            raise ValueError(
                f"No checkpoint dirs parsed from agent-config: {path_to_conf_file!r}"
            )

        rl_configs = []
        for d in checkpoint_dirs:
            rc = _load_rl_config(d)
            if rc is None:
                raise FileNotFoundError(
                    f"rl_config.json not found in {d}. "
                    "Every ensemble member must be a TD3 residual checkpoint."
                )
            rl_configs.append(rc)

        # SensorAgent.setup() loads the base TFv6 config/model from the *first*
        # member (its config/model_*.pth) — shared across members by construction.
        # Pass a single-dir config string so the parent ignores the comma list.
        save_suffix = ""
        if "+" in path_to_conf_file:
            save_suffix = "+" + path_to_conf_file.rsplit("+", maxsplit=1)[1]
        SensorAgent_setup = super(TD3SensorAgent, self).setup
        SensorAgent_setup(checkpoint_dirs[0] + save_suffix, _, __)

        self.closed_loop_inference = EnsembleResidualTD3ClosedLoopInference(
            config_closed_loop=self.config_closed_loop,
            config_expert=self.config_expert,
            checkpoint_dirs=checkpoint_dirs,
            device=self.device,
            rl_configs=rl_configs,
        )
        LOG.info(
            f"[TD3EnsembleSensorAgent] ready with {len(checkpoint_dirs)} members: {checkpoint_dirs}"
        )
