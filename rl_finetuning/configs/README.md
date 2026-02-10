`RUN_CONFIG_FILE` profiles for TFv6 finetuning launch scripts.

Usage:

```bash
RUN_CONFIG_FILE=rl_finetuning/configs/tfv6_default.env \
bash rl_finetuning/run_tfv6_watchdog_overnight.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_EXPERIMENT
```

Notes:
- Profiles are optional. If `RUN_CONFIG_FILE` is unset, script behavior stays unchanged.
- You can pass extra trainer CLI flags via `TRAINER_EXTRA_ARGS`.
