"""TFv6-owned RL config wrapper on top of CaRL defaults."""

# ruff: noqa: E402
from rl_finetuning.tfv6_rl.path_utils import ensure_carl_paths

ensure_carl_paths()

from rl_config import GlobalConfig as CaRLGlobalConfig


class GlobalConfig(CaRLGlobalConfig):
    """TFv6 finetuning defaults without modifying 3rd_party/CaRL."""

    def __init__(self):
        super().__init__()
        # Keep TFv6 default behavior independent from upstream CaRL defaults.
        self.lr_schedule = "linear"
        self.use_value_measurements = True
        # TFv6 PPO policy defaults.
        self.use_correlated_noise = True
        self.correlated_noise_rho = 0.8
        self.noise_ramp = True
        self.skip_perception_heads = True
        self.train_planning_decoder_only = True
        self.action_noise_dist = "gaussian"
        self.route_sampling_technique = (
            "legacy_noise_sampling"  # "spline_curvature_perturbation"
        )
        self.heading_amplitude1_std_init = 0.07
        self.heading_amplitude2_std_init = 0.03
        self.path_std_base_frac = 0.15
        self.lowrank_route_rank = 6
        self.lowrank_route_std_init = 0.015
        # Keep a lightweight trust region to the original TFv6 behavior during RL finetuning.
        self.use_kl_to_reference = True
        self.kl_to_reference_coef = 1e-4
        # TFv6 PPO policy noise defaults (state-dependent log_std head).
        self.log_std_init = -4.0
        self.log_std_init_route = None
        self.log_std_init_speed = None
        self.log_std_min = -5.0
        self.log_std_max = 1.0
        # Keep constant-noise behavior by default unless explicitly overridden.
        self.disable_learned_noise_head = True
        # Use a higher LR for std head so state-dependent uncertainty adapts faster.
        self.log_std_head_lr_mult = 5.0
