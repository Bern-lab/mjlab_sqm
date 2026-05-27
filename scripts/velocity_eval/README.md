# Velocity Evaluation

This folder contains fixed, offline evaluation utilities for velocity policies.
It does not register new tasks and does not modify training environments. Each
script loads an existing task config, deep-copies it through the task registry,
then applies evaluation-only overrides at runtime.

## Eval-v1

The first fixed set is intentionally small:

- `flat`
- `upstairs_10cm`
- `upstairs_15cm`
- `upstairs_20cm`

Commands are fixed by default:

- `vx = 0.4 m/s`
- `vy = 0.0 m/s`
- `wz = 0.0 rad/s`

Domain randomization, command curriculum, pushes, observation noise, and
observation delays are disabled by default for clean comparison. Observation
history length is preserved so the actor input shape remains compatible with
the checkpoint.

## Run Policy Evaluation

Example with a local checkpoint:

```bash
uv run python scripts/velocity_eval/eval_policy_on_terrains.py \
  Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1 \
  --checkpoint-file /path/to/model.pt \
  --episodes-per-terrain 50 \
  --num-envs 50
```

Example with a W&B run:

```bash
uv run python scripts/velocity_eval/eval_policy_on_terrains.py \
  Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1 \
  --wandb-run-path entity/project/run_id \
  --wandb-checkpoint-name model_40000.pt
```

The JSON output includes:

- `success_rate`
- `fall_rate`
- `mean_episode_length_s`
- `tracking_error`
- `tracking_lin_error`
- `tracking_yaw_error`
- `toe_riser_collision_count`
- `heel_riser_collision_count`
- `foot_lip_collision_count`
- `base_pitch_roll_rms`
- `action_smoothness`
- `torque_cost`
- `foot_clearance`
- collision counts by stair level

The evaluator also writes a compact PNG table next to the JSON by default:

```text
eval_outputs/velocity/g1_blind_rough_teacherkl/0526_143012/eval_eval_v1_table.png
```

If `--output-file` is omitted, each evaluation run creates a folder grouped by
the trained policy family and then by timestamp, for example:

```text
eval_outputs/velocity/g1_blind_rough_teacherkl/0526_143012/eval_eval_v1.json
```

Pass `--output-dir` to choose the timestamp folder yourself, or `--output-file`
to write to one exact file.

## Collect Latents

```bash
uv run python scripts/velocity_eval/collect_policy_latents.py \
  Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1 \
  --checkpoint-file /path/to/model.pt \
  --episodes-per-terrain 20 \
  --num-envs 20 \
  --steps-per-episode 500
```

For MLP actors, the saved latent is the hidden activation before the final actor
linear layer. For recurrent actors, it is the last recurrent hidden state.

If `--output-file` is omitted, each collection run uses the same grouped folder
layout, for example:

```text
eval_outputs/velocity/g1_blind_rough_teacherkl/0526_143012/latents_cluster_v1.npz
```

## Quick PCA

```bash
uv run python scripts/velocity_eval/analyze_latent_clusters.py \
  --input-file eval_outputs/velocity/g1_blind_rough_teacherkl/0526_143012/latents_cluster_v1.npz \
  --phase-bins 8
```

By default, analysis outputs are written next to the input `.npz`:

- `latent_pca.csv`
- `latent_pca.png`
- `phase_bins/phase_*.png` when `--phase-bins` is provided
