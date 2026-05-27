"""Evaluate velocity policies on fixed terrain sets."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from scripts.velocity_eval.eval_metrics import (
  EVENT_COUNT_NAMES,
  LEVEL_EVENT_NAMES,
  MEAN_METRIC_NAMES,
  StairEventDetector,
  compute_velocity_metrics,
)
from scripts.velocity_eval.eval_terrains import (
  EvalTerrainSpec,
  apply_eval_overrides,
  get_terrain_set,
)
from scripts.velocity_eval.policy_io import (
  get_policy_output_name,
  load_inference_policy,
  make_timestamped_policy_output_dir,
  resolve_checkpoint_path,
)

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.lstm import reset_policy_state_from_step
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class EvalPolicyConfig:
  """Configuration for fixed velocity policy evaluation."""

  checkpoint_file: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  terrain_set: str = "eval_v1"
  episodes_per_terrain: int = 50
  num_envs: int = 50
  max_episode_length_s: float = 20.0
  command_vx: float = 0.4
  command_vy: float = 0.0
  command_wz: float = 0.0
  seed: int = 12345
  device: str | None = None
  output_root: str = "eval_outputs/velocity"
  output_dir: str | None = None
  output_file: str | None = None
  write_table_image: bool = True
  table_image_file: str | None = None
  clean_observations: bool = True
  disable_observation_delay: bool = True
  disable_actuator_delay: bool = True
  max_stair_levels: int = 12


def _empty_batch_tensors(
  num_envs: int, max_levels: int, device: str
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
  sums = {
    name: torch.zeros(num_envs, device=device)
    for name in (*MEAN_METRIC_NAMES, *EVENT_COUNT_NAMES)
  }
  level_sums = {
    name: torch.zeros(num_envs, max_levels, device=device)
    for name in LEVEL_EVENT_NAMES
  }
  return sums, level_sums


def _run_batch(
  *,
  task_id: str,
  agent_cfg,
  checkpoint_path: Path,
  terrain: EvalTerrainSpec,
  cfg: EvalPolicyConfig,
  batch_size: int,
  batch_index: int,
  device: str,
) -> dict:
  env_cfg = load_env_cfg(task_id, play=False)
  apply_eval_overrides(
    env_cfg,
    terrain,
    num_envs=batch_size,
    seed=cfg.seed + 1009 * batch_index,
    max_episode_length_s=cfg.max_episode_length_s,
    command=(cfg.command_vx, cfg.command_vy, cfg.command_wz),
    clean_observations=cfg.clean_observations,
    disable_observation_delay=cfg.disable_observation_delay,
    disable_actuator_delay=cfg.disable_actuator_delay,
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  try:
    policy, _runner = load_inference_policy(
      env=wrapped,
      task_id=task_id,
      agent_cfg=agent_cfg,
      checkpoint_path=checkpoint_path,
      device=device,
    )
    detector = StairEventDetector(wrapped.unwrapped)
    obs = wrapped.get_observations()

    metric_sums, level_sums = _empty_batch_tensors(
      batch_size, cfg.max_stair_levels, device
    )
    step_counts = torch.zeros(batch_size, device=device)
    done_envs = torch.zeros(batch_size, dtype=torch.bool, device=device)
    success = torch.zeros(batch_size, dtype=torch.bool, device=device)
    fell = torch.zeros(batch_size, dtype=torch.bool, device=device)

    max_steps = wrapped.unwrapped.max_episode_length + 2
    for _step in range(max_steps):
      active = ~done_envs
      if not bool(active.any().item()):
        break

      with torch.no_grad():
        step_metrics, step_levels = compute_velocity_metrics(
          wrapped.unwrapped,
          detector,
          terrain_height_m=terrain.height_m,
          max_levels=cfg.max_stair_levels,
        )
      for name, value in step_metrics.items():
        metric_sums[name] += torch.where(active, value, torch.zeros_like(value))
      for name, value in step_levels.items():
        level_sums[name] += value * active[:, None].float()
      step_counts += active.float()

      with torch.no_grad():
        actions = policy(obs)
      step_result = wrapped.step(actions)
      reset_policy_state_from_step(policy, step_result)
      obs, _rewards, dones, _extras = step_result

      dones = dones.bool()
      terminated = wrapped.unwrapped.termination_manager.terminated.bool()
      truncated = wrapped.unwrapped.termination_manager.time_outs.bool()
      newly_done = dones & active
      if "fell_over" in wrapped.unwrapped.termination_manager.active_terms:
        fell_now = wrapped.unwrapped.termination_manager.get_term("fell_over").bool()
      else:
        fell_now = terminated
      success |= newly_done & truncated & ~terminated
      fell |= newly_done & fell_now
      done_envs |= newly_done

    unfinished = ~done_envs
    success |= unfinished
    safe_counts = step_counts.clamp_min(1.0)
    mean_metrics = {
      name: (metric_sums[name] / safe_counts).detach().cpu().tolist()
      for name in MEAN_METRIC_NAMES
    }
    event_counts = {
      name: metric_sums[name].detach().cpu().tolist() for name in EVENT_COUNT_NAMES
    }
    level_counts = {
      name: level_sums[name].detach().cpu().tolist() for name in LEVEL_EVENT_NAMES
    }
    return {
      "success": success.detach().cpu().tolist(),
      "fell": fell.detach().cpu().tolist(),
      "episode_length_steps": step_counts.detach().cpu().tolist(),
      "mean_metrics": mean_metrics,
      "event_counts": event_counts,
      "level_counts": level_counts,
      "step_dt": wrapped.unwrapped.step_dt,
    }
  finally:
    wrapped.close()


def _summarize_batches(terrain: EvalTerrainSpec, batches: list[dict]) -> dict:
  success = [item for batch in batches for item in batch["success"]]
  fell = [item for batch in batches for item in batch["fell"]]
  lengths = [item for batch in batches for item in batch["episode_length_steps"]]
  step_dt = batches[0]["step_dt"] if batches else 0.0
  episodes = max(1, len(success))

  summary = {
    "terrain": terrain.name,
    "terrain_label": terrain.label,
    "terrain_kind": terrain.kind,
    "height_m": terrain.height_m,
    "episodes": len(success),
    "success_rate": float(sum(success) / episodes),
    "fall_rate": float(sum(fell) / episodes),
    "mean_episode_length_s": float(sum(lengths) * step_dt / episodes),
    "mean_episode_length_steps": float(sum(lengths) / episodes),
  }

  for name in MEAN_METRIC_NAMES:
    values = [v for batch in batches for v in batch["mean_metrics"][name]]
    summary[name] = float(sum(values) / max(1, len(values)))

  for name in EVENT_COUNT_NAMES:
    values = [v for batch in batches for v in batch["event_counts"][name]]
    summary[f"{name}_count"] = float(sum(values) / max(1, len(values)))

  for name in LEVEL_EVENT_NAMES:
    total = [0.0 for _ in range(len(batches[0]["level_counts"][name][0]))]
    for batch in batches:
      for row in batch["level_counts"][name]:
        for idx, value in enumerate(row):
          total[idx] += value
    summary[name] = [value / episodes for value in total]

  return summary


def _resolve_output_path(
  *,
  cfg: EvalPolicyConfig,
  task_id: str,
  agent_cfg,
  checkpoint_path: Path,
) -> Path | None:
  if cfg.output_file is not None:
    output_path = Path(cfg.output_file)
    if output_path.is_dir() or output_path.suffix.lower() != ".json":
      output_dir = make_timestamped_policy_output_dir(
        output_root=output_path,
        task_id=task_id,
        agent_cfg=agent_cfg,
        checkpoint_path=checkpoint_path,
      )
      return output_dir / f"eval_{cfg.terrain_set}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path

  output_dir = (
    Path(cfg.output_dir)
    if cfg.output_dir is not None
    else make_timestamped_policy_output_dir(
      output_root=cfg.output_root,
      task_id=task_id,
      agent_cfg=agent_cfg,
      checkpoint_path=checkpoint_path,
    )
  )
  output_dir.mkdir(parents=True, exist_ok=True)
  return output_dir / f"eval_{cfg.terrain_set}.json"


def _resolve_table_image_path(
  *,
  cfg: EvalPolicyConfig,
  output_path: Path | None,
) -> Path | None:
  if cfg.table_image_file is not None:
    table_path = Path(cfg.table_image_file)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    return table_path
  if not cfg.write_table_image or output_path is None:
    return None
  return output_path.with_name(f"{output_path.stem}_table.png")


def _metric_cell_color(column_name: str, value: float) -> str:
  if column_name == "success %":
    if value >= 90.0:
      return "#d8f0dd"
    if value >= 70.0:
      return "#fff0c2"
    return "#f7d4d4"
  if column_name == "fall %":
    return "#f7d4d4" if value > 0.0 else "#f3f6f8"
  if column_name in {"toe", "heel", "lip"}:
    if value <= 0.0:
      return "#f3f6f8"
    if value <= 5.0:
      return "#fff0c2"
    return "#f7d4d4"
  return "#ffffff"


def _write_summary_table_image(payload: dict, output_path: Path) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  columns = [
    "terrain",
    "success %",
    "fall %",
    "len s",
    "track err",
    "toe",
    "heel",
    "lip",
    "pitch/roll",
    "torque",
  ]
  numeric_rows: list[list[float | str]] = []
  text_rows: list[list[str]] = []
  for item in payload["terrains"]:
    row_values: list[float | str] = [
      item["terrain"],
      item["success_rate"] * 100.0,
      item["fall_rate"] * 100.0,
      item["mean_episode_length_s"],
      item["tracking_error"],
      item["toe_riser_collision_count"],
      item["heel_riser_collision_count"],
      item["foot_lip_collision_count"],
      item["base_pitch_roll_rms"],
      item["torque_cost"],
    ]
    numeric_rows.append(row_values)
    text_rows.append(
      [
        str(row_values[0]),
        f"{row_values[1]:.0f}",
        f"{row_values[2]:.0f}",
        f"{row_values[3]:.2f}",
        f"{row_values[4]:.3f}",
        f"{row_values[5]:.2f}",
        f"{row_values[6]:.2f}",
        f"{row_values[7]:.2f}",
        f"{row_values[8]:.3f}",
        f"{row_values[9]:.3f}",
      ]
    )

  fig_height = max(2.8, 1.35 + 0.42 * max(1, len(text_rows)))
  fig, ax = plt.subplots(figsize=(13.0, fig_height), dpi=180)
  ax.axis("off")
  command = payload.get("command", {})
  title = (
    f"{payload.get('policy_output_name', payload.get('task_id', 'policy'))} | "
    f"{payload.get('terrain_set', 'terrain_set')} | "
    f"{payload.get('episodes_per_terrain', '?')} eps/terrain | "
    f"vx={command.get('vx', '?')}, vy={command.get('vy', '?')}, wz={command.get('wz', '?')}"
  )
  ax.text(
    0.5,
    0.96,
    title,
    ha="center",
    va="center",
    fontsize=11,
    fontweight="bold",
    transform=ax.transAxes,
  )

  table = ax.table(
    cellText=text_rows,
    colLabels=columns,
    cellLoc="center",
    loc="center",
  )
  table.auto_set_font_size(False)
  table.set_fontsize(8.5)
  table.scale(1.0, 1.35)

  for (row_idx, col_idx), cell in table.get_celld().items():
    cell.set_edgecolor("#cfd6de")
    cell.set_linewidth(0.7)
    if row_idx == 0:
      cell.set_facecolor("#243447")
      cell.set_text_props(color="white", fontweight="bold")
      continue
    if col_idx == 0:
      cell.set_facecolor("#eef2f5")
      cell.set_text_props(ha="left", fontweight="bold")
      continue
    value = float(numeric_rows[row_idx - 1][col_idx])
    cell.set_facecolor(_metric_cell_color(columns[col_idx], value))

  footer = "Collision columns are mean event counts per episode. Lower tracking, collision, pitch/roll, and torque are better."
  ax.text(
    0.5,
    0.04,
    footer,
    ha="center",
    va="center",
    fontsize=8,
    color="#4d5b68",
    transform=ax.transAxes,
  )
  fig.tight_layout()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output_path, bbox_inches="tight")
  plt.close(fig)


def run_eval_policy(task_id: str, cfg: EvalPolicyConfig) -> dict:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  terrains = get_terrain_set(cfg.terrain_set)
  agent_cfg = load_rl_cfg(task_id)
  checkpoint_path = resolve_checkpoint_path(
    task_id=task_id,
    agent_cfg=agent_cfg,
    checkpoint_file=cfg.checkpoint_file,
    wandb_run_path=cfg.wandb_run_path,
    wandb_checkpoint_name=cfg.wandb_checkpoint_name,
  )
  output_path = _resolve_output_path(
    cfg=cfg,
    task_id=task_id,
    agent_cfg=agent_cfg,
    checkpoint_path=checkpoint_path,
  )
  policy_output_name = get_policy_output_name(
    task_id=task_id,
    agent_cfg=agent_cfg,
    checkpoint_path=checkpoint_path,
  )

  payload = {
    "task_id": task_id,
    "policy_output_name": policy_output_name,
    "checkpoint": str(checkpoint_path),
    "output_dir": str(output_path.parent) if output_path is not None else None,
    "terrain_set": cfg.terrain_set,
    "episodes_per_terrain": cfg.episodes_per_terrain,
    "command": {
      "vx": cfg.command_vx,
      "vy": cfg.command_vy,
      "wz": cfg.command_wz,
    },
    "max_episode_length_s": cfg.max_episode_length_s,
    "seed": cfg.seed,
    "terrains": [],
  }

  batch_capacity = max(1, cfg.num_envs)
  batch_index = 0
  for terrain in terrains:
    remaining = cfg.episodes_per_terrain
    batches = []
    print(f"[INFO] Evaluating {terrain.name} ({cfg.episodes_per_terrain} episodes)")
    while remaining > 0:
      batch_size = min(batch_capacity, remaining)
      batches.append(
        _run_batch(
          task_id=task_id,
          agent_cfg=agent_cfg,
          checkpoint_path=checkpoint_path,
          terrain=terrain,
          cfg=cfg,
          batch_size=batch_size,
          batch_index=batch_index,
          device=device,
        )
      )
      remaining -= batch_size
      batch_index += 1
    summary = _summarize_batches(terrain, batches)
    payload["terrains"].append(summary)
    print(
      "[INFO] "
      f"{terrain.name}: success={summary['success_rate']:.3f}, "
      f"fall={summary['fall_rate']:.3f}, "
      f"toe={summary['toe_riser_collision_count']:.3f}, "
      f"heel={summary['heel_riser_collision_count']:.3f}"
    )

  if output_path is not None:
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote evaluation results to {output_path}")
  table_image_path = _resolve_table_image_path(cfg=cfg, output_path=output_path)
  if table_image_path is not None:
    _write_summary_table_image(payload, table_image_path)
    print(f"[INFO] Wrote evaluation table image to {table_image_path}")

  return payload


def main() -> None:
  import mjlab.tasks  # noqa: F401

  velocity_tasks = [task for task in list_tasks() if "Velocity" in task]
  if not velocity_tasks:
    print("No velocity tasks found.")
    sys.exit(1)

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(velocity_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  cfg = tyro.cli(
    EvalPolicyConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_eval_policy(chosen_task, cfg)


if __name__ == "__main__":
  main()
