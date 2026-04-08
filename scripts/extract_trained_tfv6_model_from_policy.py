#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

# Keys from GlobalConfig that TFv6PPOPolicy reads via getattr(rl_config, key, default).
# Only primitive-valued keys are safe to round-trip through plain json.load().
_RL_CONFIG_KEYS_FOR_EVAL = (
    "use_residual_policy",
    "residual_route_rank",
    "residual_alpha",
    "residual_alpha_speed",
    "disable_residual_route",
    "disable_learned_noise_head",
    "log_std_init",
    "log_std_init_route",
    "log_std_init_speed",
    "log_std_min",
    "log_std_max",
    "speed_temperature",
    "train_planning_decoder_only",
    "use_correlated_noise",
    "correlated_noise_rho",
    "noise_ramp",
    "action_noise_dist",
    "speed_sampling_style",
    "route_sampling_technique",
    "lowrank_route_rank",
    "lowrank_route_std_init",
    "use_kl_to_reference",
    "kl_to_reference_coef",
    "critic_updates_shared_features",
    "use_privileged_measurements",
    "num_privileged_measurements",
    "use_lstm",
)

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch

from lead.tfv6.tfv6 import TFv6
from rl_finetuning.tfv6_rl.policy_tfv6_ppo import load_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract plain TFv6 weights from a TFv6PPOPolicy checkpoint "
            "(e.g. model_best.pth / model_latest_*.pth)."
        )
    )
    parser.add_argument(
        "policy_checkpoint",
        type=Path,
        help="Path to PPO policy checkpoint file (.pth) inside an RL run folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Destination TFv6 checkpoint folder. Defaults to sibling of the base TFv6 "
            "checkpoint named '<base>_rlfinetuned'."
        ),
    )
    parser.add_argument(
        "--base-checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Override base TFv6 checkpoint folder (must contain config.json and model*.pth). "
            "If omitted, inferred from the RL run config.json field 'tfv6_checkpoint'."
        ),
    )
    parser.add_argument(
        "--output-model-name",
        type=str,
        default="model_rl_finetuned.pth",
        help="Filename for the extracted TFv6 weights (must start with 'model').",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _infer_base_checkpoint_dir(policy_ckpt: Path, override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()

    run_dir = policy_ckpt.parent
    rl_config_path = run_dir / "config.json"
    if not rl_config_path.exists():
        raise FileNotFoundError(
            f"Could not infer base TFv6 checkpoint: missing RL run config {rl_config_path}"
        )

    rl_config = _read_json(rl_config_path)
    tfv6_checkpoint = rl_config.get("tfv6_checkpoint")
    if not isinstance(tfv6_checkpoint, str) or not tfv6_checkpoint:
        raise KeyError(
            "RL run config.json does not contain a valid 'tfv6_checkpoint' string."
        )
    return Path(tfv6_checkpoint).expanduser().resolve()


def _normalize_state_dict_for_tfv6(
    raw_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if any(k.startswith("tfv6.") for k in raw_state_dict.keys()):
        return {
            key.replace("tfv6.", "", 1): value
            for key, value in raw_state_dict.items()
            if key.startswith("tfv6.")
        }
    return raw_state_dict


def _validate_base_checkpoint_dir(base_dir: Path) -> None:
    config_path = base_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing base TFv6 config: {config_path}")
    if not any(base_dir.glob("model*.pth")):
        raise FileNotFoundError(f"Missing base TFv6 model*.pth in: {base_dir}")


def _prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --force to overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)


def _save_residual_eval_files(
    policy_ckpt: Path,
    rl_config: dict,
    output_dir: Path,
) -> None:
    """Save the extra files needed for ResidualSensorAgent eval.

    Adds two files to the already-prepared output_dir:
      - residual_policy.pth : full TFv6PPOPolicy state dict (TFv6 + residual head weights)
      - rl_config.json      : primitive-valued subset of GlobalConfig for policy construction

    The existing model_rl_finetuned.pth (TFv6-only weights) is still kept so the
    standard SensorAgent can load the dir as a plain TFv6 checkpoint if needed.
    """
    # Copy the full policy state dict — ResidualClosedLoopInference loads this on top of TFv6.
    residual_policy_path = output_dir / "residual_policy.pth"
    shutil.copy2(policy_ckpt, residual_policy_path)
    print(f"[extract-tfv6] residual_policy.pth  -> {residual_policy_path}", flush=True)

    # Save only the primitive-valued keys that TFv6PPOPolicy reads via getattr().
    rl_config_out = {
        k: rl_config[k]
        for k in _RL_CONFIG_KEYS_FOR_EVAL
        if k in rl_config and isinstance(rl_config[k], (bool, int, float, str, type(None)))
    }
    rl_config_path = output_dir / "rl_config.json"
    with rl_config_path.open("w", encoding="utf-8") as f:
        json.dump(rl_config_out, f, indent=2, sort_keys=True)
    print(f"[extract-tfv6] rl_config.json       -> {rl_config_path}", flush=True)
    print(f"[extract-tfv6] residual keys saved  : {sorted(rl_config_out.keys())}", flush=True)


def _validate_extracted_checkpoint(output_dir: Path) -> None:
    training_config = load_training_config(str(output_dir))
    model = TFv6(torch.device("cpu"), training_config).eval()
    weight_files = sorted(output_dir.glob("model*.pth"))
    if not weight_files:
        raise FileNotFoundError(f"No model*.pth found in {output_dir}")
    weights_path = weight_files[0]
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)


def main() -> None:
    args = parse_args()
    policy_ckpt = args.policy_checkpoint.expanduser().resolve()

    if not policy_ckpt.exists():
        raise FileNotFoundError(f"Policy checkpoint not found: {policy_ckpt}")
    if policy_ckpt.suffix != ".pth":
        raise ValueError(f"Expected .pth checkpoint file, got: {policy_ckpt}")
    if not args.output_model_name.startswith("model"):
        raise ValueError("--output-model-name must start with 'model' for eval script compatibility.")

    base_ckpt_dir = _infer_base_checkpoint_dir(policy_ckpt, args.base_checkpoint_dir)
    _validate_base_checkpoint_dir(base_ckpt_dir)

    if args.output_dir is None:
        output_dir = base_ckpt_dir.parent / f"{base_ckpt_dir.name}_rlfinetuned"
    else:
        output_dir = args.output_dir.expanduser().resolve()

    print(f"[extract-tfv6] policy_checkpoint={policy_ckpt}", flush=True)
    print(f"[extract-tfv6] base_checkpoint_dir={base_ckpt_dir}", flush=True)
    print(f"[extract-tfv6] output_dir={output_dir}", flush=True)

    raw = torch.load(policy_ckpt, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError(f"Unexpected checkpoint payload type: {type(raw)}")
    extracted = _normalize_state_dict_for_tfv6(raw)
    if not extracted:
        raise ValueError("No TFv6 parameters found in checkpoint (expected keys prefixed with 'tfv6.').")

    ppo_only_keys = [k for k in raw.keys() if not k.startswith("tfv6.")]
    print(
        f"[extract-tfv6] raw_keys={len(raw)} tfv6_keys={len(extracted)} "
        f"non_tfv6_keys={len(ppo_only_keys)}",
        flush=True,
    )

    _prepare_output_dir(output_dir, force=args.force)

    shutil.copy2(base_ckpt_dir / "config.json", output_dir / "config.json")
    output_model_path = output_dir / args.output_model_name
    torch.save(extracted, output_model_path)

    meta = {
        "source_policy_checkpoint": str(policy_ckpt),
        "source_rl_run_dir": str(policy_ckpt.parent),
        "base_tfv6_checkpoint_dir": str(base_ckpt_dir),
        "copied_config": str(base_ckpt_dir / "config.json"),
        "output_model_file": str(output_model_path),
        "num_raw_state_keys": len(raw),
        "num_extracted_tfv6_keys": len(extracted),
        "num_non_tfv6_keys_dropped": len(ppo_only_keys),
        "dropped_key_prefixes": sorted({k.split(".", 1)[0] for k in ppo_only_keys}),
    }
    with (output_dir / "extraction_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    _validate_extracted_checkpoint(output_dir)

    print("[extract-tfv6] Validation OK: TFv6 loads with strict=True.", flush=True)

    # --- Residual policy: save extra eval files when use_residual_policy is set ---
    run_rl_config = _read_json(policy_ckpt.parent / "config.json")
    if run_rl_config.get("use_residual_policy", False):
        print("[extract-tfv6] Residual policy detected — saving residual eval files.", flush=True)
        _save_residual_eval_files(policy_ckpt, run_rl_config, output_dir)
        print(
            "[extract-tfv6] Residual checkpoint ready for ResidualSensorAgent "
            "(contains config.json + model*.pth + residual_policy.pth + rl_config.json).",
            flush=True,
        )
    else:
        print(
            "[extract-tfv6] Compatible folder ready for lead eval scripts "
            "(contains config.json + model*.pth).",
            flush=True,
        )


if __name__ == "__main__":
    main()
