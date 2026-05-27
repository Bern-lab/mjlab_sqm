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

Stair terrains use 10 riser levels by default for by-level collision statistics.
With the current fixed geometry this gives a square stair tile of about
`10.4 m x 10.4 m`, `0.30 m` step run, and `3.0 m` center platform.

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
- `collision_by_stair_level_low_to_high`
- `heading_failure_rate`

For G1 blind policies, eval adds an evaluation-only `toe_terrain_contact`
MuJoCo contact sensor when needed. `toe_riser_collision_count` and
`heel_riser_collision_count` are therefore true foot-vs-terrain contact
transition counts on vertical riser-like surfaces, not actor observations and
not geometric danger-zone occupancy. They are sampled at the policy control
step. If that sensor is unavailable, the evaluator falls back to the older
geometry-based approximation and records the source in `collision_event_source`.

If the robot yaw deviates from the commanded travel direction by more than
`--heading-failure-angle-deg` (default `45`), the evaluator ends that episode as
a failure. Collision statistics from heading-failed episodes are excluded from
toe/heel collision averages and the low-to-high per-level collision table, so a
policy that turns sideways does not pollute the early stair-level counts.

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

## Goal Pyramid Evaluation

`eval_policy_goal_pyramid.py` builds a single large convex `pyramid_stairs`
terrain for target-navigation stair climbing. Robots spawn randomly on one of
the four low outer sides, within the top platform projection, facing the pyramid
center. At every control step the command points toward the center top platform:

```text
vx = goal_speed * max(cos(yaw_error), 0)
vy = 0
wz = clip(yaw_kp * yaw_error, -yaw_rate_limit, yaw_rate_limit)
```

An episode succeeds when the robot reaches the top center region. If the robot
turns too far away from the side-normal approach direction, the episode is
marked as a heading failure and its collision counts are excluded from
success-only collision statistics.

Example:

```bash
uv run python scripts/velocity_eval/eval_policy_goal_pyramid.py \
  Mjlab-Velocity-Blind-Rough-LSTM-TeacherKL-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_blind_rough_lstm_teacherkl/Mjlab-Velocity-Blind-Rough-LSTM-TeacherKL-Unitree-G1/5.25deployed/model_14800.pt \
  --episodes 50 \
  --num-envs 50 \
  --max-episode-length-s 12.0 \
  --stair-levels 10 \
  --stair-height 0.15 \
  --step-width 0.30 \
  --platform-width 3.0 \
  --flat-apron-width 3.0 \
  --terrain-border-width 12.0 \
  --goal-radius 0.75 \
  --output-file eval_outputs/velocity
```

The stair size switches are:

- `--stair-levels`: number of low-to-high riser levels, default `10`
- `--stair-height`: riser height in meters, default `0.15`
- `--step-width`: stair tread/run width in meters, default `0.30`
- `--platform-width`: square top platform width in meters, default `3.0`
- `--flat-apron-width`: flat ground connected to the pyramid bottom, default `3.0`
- `--terrain-border-width`: extra terrain-generator border around the tile, default `12.0`
- `--start-distance`: optional explicit spawn radius; otherwise computed from the stair geometry
- `--spawn-tangent-half-width`: optional side-wise spawn range; otherwise kept inside the top platform width

Outputs are timestamped by default when `--output-file` points at a directory:

```text
eval_outputs/velocity/g1_blind_rough_lstm_teacherkl/0527_165514/goal_pyramid_h15cm.json
eval_outputs/velocity/g1_blind_rough_lstm_teacherkl/0527_165514/goal_pyramid_h15cm_table.png
```

The JSON contains `success_rate`, failure rates, spawn side counts, and
`collision_by_stair_level_low_to_high_success_only`. The table image shows the
same summary plus mean toe/heel collision counts for each stair level, averaged
over successful episodes only.

To watch the same eval task in a viewer, add `--play`. Use a small `--num-envs`
when you want to inspect motion clearly:

```bash
uv run python scripts/velocity_eval/eval_policy_goal_pyramid.py \
  Mjlab-Velocity-Blind-Rough-LSTM-TeacherKL-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_blind_rough_lstm_teacherkl/Mjlab-Velocity-Blind-Rough-LSTM-TeacherKL-Unitree-G1/5.25deployed/model_14800.pt \
  --play \
  --num-envs 4 \
  --max-episode-length-s 12.0
```

The play mode automatically respawns robots on the pyramid apron after success,
fall, timeout, or heading failure. Pass `--viewer native` or `--viewer viser` to
force a backend; `auto` uses native when a display is available and Viser
otherwise.

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
