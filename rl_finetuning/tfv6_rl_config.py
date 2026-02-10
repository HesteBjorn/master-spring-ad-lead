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
