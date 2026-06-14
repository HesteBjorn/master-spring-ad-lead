# Thesis video route selection

Tooling to find which Town13 left-turn routes show each behaviour we want to film,
then re-run only those routes cleanly. The behaviours are *outcomes of a model on a
route* (stop-sign violation, gap-taking success, impatient collision, TFv6-vs-TD3
divergence), so we run deterministic closed-loop inference over the route pool,
read the leaderboard outcome JSON, and pick from a table. Deterministic inference is
reproducible, so a chosen route renders the same moment again with viz on.

Route pool: `data/rl_finetuning_training_data/NonSignalizedJunctionLeftTurnEnterFlow`,
Town13 = **53 unique routes** (the shipped `route_Town13_00/01/02.xml.gz` are
byte-identical duplicates; staging de-duplicates them).

## Models

| label | agent | checkpoint | used for |
|-------|-------|-----------|----------|
| base | `lead/inference/sensor_agent.py` | `outputs/checkpoints/tfv6_resnet34` | reference / side-by-side |
| td3 (final) | `rl_finetuning/inference/td3_sensor_agent.py` | `outputs/eval_viz/checkpoints/td3_final` | clips #4 #5 #6 |
| td3 (stopsign) | `rl_finetuning/inference/td3_sensor_agent.py` | `outputs/eval_viz/checkpoints/td3_stopsign` | clip #3 |

`td3_final` is the converged IDUN baseline `TD3_idun_baseline` (1.5M steps), the
checkpoint the rest of the thesis reports. `td3_stopsign` is the **75k-step**
checkpoint from the local baseline run `TFV6_TD3_LOCAL_RESIDUAL_8_baseline_T1213`
(same speed-only residual config), used because IDUN saved only the final checkpoint
and stop-sign violations are a transient training phenomenon: the rate peaks at ~0.82
around step 70k–85k and the converged model rarely runs a stop sign (~2%). The
spike-era checkpoint makes clip #3 easy to capture.

## Steps

```bash
# 1. Build the TD3 inference checkpoint dirs (no CARLA needed).
bash rl_finetuning/eval_viz/extract_checkpoints.sh

# 2. Scan the routes and aggregate outcomes (needs CARLA + your eval env).
#    Run as you run your other local evaluations: conda env active, CARLA_ROOT set,
#    $PWD/scripts on PATH. A quick first probe with MAX_ROUTES, then the full 53.
MAX_ROUTES=20 bash rl_finetuning/eval_viz/scan_routes.sh   # ~probe
bash rl_finetuning/eval_viz/scan_routes.sh                 # full 53 routes
```

The scan prints a per-route table and clip picks, e.g.:

```
=== clip picks from 'td3' ===
            stop_sign (#3): n=...   <route ids>
          gap success (#4): n=...   <route ids>
  impatient collision (#5): n=...   <route ids>
=== td3 vs base ... divergence ... ===
[pick] side-by-side candidates (base fails, td3 succeeds): <route ids>
```

For clip #3, point the `td3` model at `td3_stopsign` instead of `td3_final`
(edit the `MODELS` array in `scan_routes.sh`).

The full table is written to `outputs/eval_viz/outcomes_Town13.tsv`.

## What runs where

- `stage_routes.py` — gunzip + town filter + dedup + optional `--max-routes` trim
  into one plain `.xml`. The training routes already carry their `<scenario>` block,
  so they run under the Bench2Drive evaluator unchanged.
- `scan_routes.sh` — stage, then per model run the proven `scripts/eval_bench2drive.sh`
  with `NO_SAVE=1` (outcomes only) and `REPETITIONS=1`, then aggregate.
- `aggregate_outcomes.py` — parse leaderboard checkpoint JSONs into the outcome table.
- `extract_checkpoints.sh` — wrap `scripts/extract_trained_tfv6_model_from_policy.py`.

`scripts/eval_bench2drive.sh` gained two backward-compatible knobs for the scan:
`NO_SAVE=1` (unset `SAVE_PATH`, no debug frames) and `REPETITIONS` override. Defaults
are unchanged (`REPETITIONS=3`, frames on), so normal eval runs are unaffected.

## After picking routes

Re-run only the chosen routes with frames on (keep `SAVE_PATH`) to produce the clean
camera + BEV + speed-correction-bar clips. That viz step (overlay + base-route ghost)
is built separately.

## Environment

`scan_routes.sh` wraps `scripts/eval_bench2drive.sh`, so it assumes the same
environment as your normal bench2drive eval: conda env `lead` active and
`CARLA_ROOT` set (defaults to `3rd_party/CARLA_0915`). It launches CARLA itself
via `scripts/start_carla.sh` on the fixed port 2000 / TM 8000, so run it when no
other CARLA eval is using those ports.

Staging, extraction, and aggregation are validated offline; the CARLA scan uses
your proven harness.
