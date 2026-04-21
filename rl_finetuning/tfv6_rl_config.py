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
        self.use_correlated_noise = False
        self.correlated_noise_rho = 0.8
        self.noise_ramp = False
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
        self.use_kl_to_reference = False
        self.kl_to_reference_coef = 1e-4
        # TFv6 PPO policy noise defaults (state-dependent log_std head).
        self.log_std_init = -4.0
        self.log_std_init_route = None
        self.log_std_init_speed = None
        self.log_std_min = -5.0
        self.log_std_max = 1.0
        self.speed_sampling_style = "native_categorical"
        self.standstill_speed_hold_enabled = False
        self.standstill_speed_hold_frames = 3
        self.standstill_speed_hold_ego_speed_threshold = 0.1
        self.standstill_speed_hold_target_speed_threshold = 1.0 / 3.6
        # Keep constant-noise behavior by default unless explicitly overridden.
        self.disable_learned_noise_head = True
        self.speed_temperature = 1.0
        # Use a higher LR for std head so state-dependent uncertainty adapts faster.
        self.log_std_head_lr_mult = 5.0
        # Higher LR for critic (value_head + value_queries) so it can bootstrap
        # from a cold init while the actor stays at the conservative base LR.
        self.value_head_lr_mult = 1.0
        # Optional memory-saving path: keep rollout observations on CPU and
        # transfer only active PPO minibatch observation slices to GPU.
        self.stream_obs_minibatches_from_cpu_to_save_gpu_memory = False
        # Residual RL policy: freeze entire TFv6 and learn a small corrective residual.
        # When enabled, use_kl_to_reference is forced off (architectural constraint replaces it).
        self.use_residual_policy = False
        self.residual_route_rank = (
            2  # basis modes: 2=lateral only, 4=+longitudinal, 6=+higher-freq
        )
        self.residual_alpha = (
            0.15  # max route correction per mode in normalized space (~6m at rank-peak)
        )
        self.residual_alpha_speed = (
            0.15  # max speed correction in normalized space (fraction of max_speed)
        )

        # ── TD3 algorithm hyperparameters ─────────────────────────────────
        # Replay buffer capacity (transitions).
        self.buffer_size = 100_000
        # Number of environment steps collected before any learning begins.
        self.learning_starts = 5_000
        # TD3 minibatch size sampled from the replay buffer per update.
        self.td3_batch_size = 256
        # Polyak averaging coefficient for target network updates.
        self.tau = 0.005
        # Actor (and target networks) update every ``policy_delay`` critic steps.
        self.policy_delay = 2
        # Std of Gaussian exploration noise added to actor output at rollout.
        self.exploration_noise = 0.1
        # Std of target policy smoothing noise (clipped by target_noise_clip).
        self.target_policy_noise = 0.2
        # Clip bound for target policy smoothing noise.
        self.target_noise_clip = 0.5
        # Critic-only updates before actor training starts (ResFiT-style warmup).
        self.critic_warmup_steps = 10_000
        # Separate learning rates for actor and twin critics.
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3
