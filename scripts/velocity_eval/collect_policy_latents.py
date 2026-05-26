"""Collect policy latents on fixed velocity evaluation terrains."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tyro
from scripts.velocity_eval.eval_terrains import apply_eval_overrides, get_terrain_set
from scripts.velocity_eval.policy_io import (
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
class CollectLatentsConfig:
  checkpoint_file: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  terrain_set: str = "cluster_v1"
  episodes_per_terrain: int = 20
  num_envs: int = 20
  steps_per_episode: int = 500
  sample_every: int = 1
  command_vx: float = 0.4
  command_vy: float = 0.0
  command_wz: float = 0.0
  seed: int = 23456
  device: str | None = None
  output_root: str = "eval_outputs/velocity"
  output_dir: str | None = None
  output_file: str | None = None
  clean_observations: bool = True
  disable_observation_delay: bool = True
  disable_actuator_delay: bool = True


def _resolve_output_path(
  *,
  cfg: CollectLatentsConfig,
  task_id: str,
  agent_cfg,
  checkpoint_path: Path,
) -> Path:
  if cfg.output_file is not None:
    output_path = Path(cfg.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path

  if cfg.output_dir is not None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
  else:
    output_dir = make_timestamped_policy_output_dir(
      output_root=cfg.output_root,
      task_id=task_id,
      agent_cfg=agent_cfg,
      checkpoint_path=checkpoint_path,
    )

  return output_dir / f"latents_{cfg.terrain_set}.npz"


def _mlp_hidden_before_final_linear(policy, obs) -> torch.Tensor:
  latent = policy.get_latent(obs)
  layers = list(policy.mlp)
  linear_indices = [
    idx for idx, layer in enumerate(layers) if isinstance(layer, nn.Linear)
  ]
  if not linear_indices:
    return latent
  final_linear_idx = linear_indices[-1]
  x = latent
  for layer in layers[:final_linear_idx]:
    x = layer(x)
  return x


def extract_policy_latent(policy, obs) -> torch.Tensor:
  """Extract a per-env policy latent without mutating recurrent state."""
  if bool(getattr(policy, "is_recurrent", False)):
    hidden = policy.get_hidden_state()
    if isinstance(hidden, tuple):
      hidden = hidden[0]
    if hidden is None:
      raise RuntimeError("Recurrent policy did not expose a hidden state.")
    return hidden[-1].detach()
  return _mlp_hidden_before_final_linear(policy, obs).detach()


def _get_gait_period(env) -> float:
  try:
    term = env.cfg.observations["actor"].terms["gait_phase"]
    return float(term.params.get("period", 0.6))
  except (AttributeError, KeyError, TypeError):
    return 0.6


def _compute_gait_phase(env) -> tuple[torch.Tensor, torch.Tensor]:
  period = _get_gait_period(env)
  phase = (env.episode_length_buf.float() * env.step_dt) % period / period
  phase_sincos = torch.stack(
    [torch.sin(phase * torch.pi * 2.0), torch.cos(phase * torch.pi * 2.0)],
    dim=-1,
  )
  return phase, phase_sincos


def _collect_for_terrain(
  *,
  task_id: str,
  agent_cfg,
  checkpoint_path: Path,
  terrain,
  cfg: CollectLatentsConfig,
  terrain_index: int,
  batch_index: int,
  batch_size: int,
  episode_offset: int,
  device: str,
) -> dict[str, list[np.ndarray] | list[str] | list[float] | list[int]]:
  env_cfg = load_env_cfg(task_id, play=False)
  apply_eval_overrides(
    env_cfg,
    terrain,
    num_envs=batch_size,
    seed=cfg.seed + 1009 * terrain_index + 9173 * batch_index,
    max_episode_length_s=max(1.0, cfg.steps_per_episode * 0.02),
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
    obs = wrapped.get_observations()
    arrays: dict[str, list] = {
      "latent": [],
      "action": [],
      "command": [],
      "base_state": [],
      "gait_phase": [],
      "gait_phase_sincos": [],
      "terrain_label": [],
      "terrain_name": [],
      "terrain_height_m": [],
      "episode_id": [],
      "time_step": [],
    }

    episode_offsets = torch.arange(batch_size, device=device) + episode_offset
    for step in range(cfg.steps_per_episode):
      with torch.no_grad():
        actions = policy(obs)
        latent = extract_policy_latent(policy, obs)

      if step % cfg.sample_every == 0:
        robot = wrapped.unwrapped.scene["robot"]
        command = wrapped.unwrapped.command_manager.get_command("twist")
        assert command is not None
        gait_phase, gait_phase_sincos = _compute_gait_phase(wrapped.unwrapped)
        base_state = torch.cat(
          [
            robot.data.root_link_pos_w,
            robot.data.root_link_quat_w,
            robot.data.root_link_lin_vel_b,
            robot.data.root_link_ang_vel_b,
          ],
          dim=-1,
        )
        arrays["latent"].append(latent.detach().cpu().numpy())
        arrays["action"].append(actions.detach().cpu().numpy())
        arrays["command"].append(command.detach().cpu().numpy())
        arrays["base_state"].append(base_state.detach().cpu().numpy())
        arrays["gait_phase"].append(gait_phase.detach().cpu().numpy())
        arrays["gait_phase_sincos"].append(
          gait_phase_sincos.detach().cpu().numpy()
        )
        arrays["terrain_label"].extend([terrain.label] * batch_size)
        arrays["terrain_name"].extend([terrain.name] * batch_size)
        arrays["terrain_height_m"].extend(
          [float(terrain.height_m or 0.0)] * batch_size
        )
        arrays["episode_id"].append(episode_offsets.detach().cpu().numpy())
        arrays["time_step"].append(
          torch.full((batch_size,), step, device=device).detach().cpu().numpy()
        )

      step_result = wrapped.step(actions)
      reset_policy_state_from_step(policy, step_result)
      obs, _rewards, _dones, _extras = step_result

    return arrays
  finally:
    wrapped.close()


def run_collect_latents(task_id: str, cfg: CollectLatentsConfig) -> dict[str, np.ndarray]:
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
  print(f"[INFO] Output directory: {output_path.parent}")

  chunks = []
  batch_index = 0
  for terrain_index, terrain in enumerate(terrains):
    remaining = cfg.episodes_per_terrain
    episode_offset = 0
    print(f"[INFO] Collecting latents on {terrain.name}")
    while remaining > 0:
      batch_size = min(max(1, cfg.num_envs), remaining)
      chunks.append(
        _collect_for_terrain(
          task_id=task_id,
          agent_cfg=agent_cfg,
          checkpoint_path=checkpoint_path,
          terrain=terrain,
          cfg=cfg,
          terrain_index=terrain_index,
          batch_index=batch_index,
          batch_size=batch_size,
          episode_offset=episode_offset,
          device=device,
        )
      )
      remaining -= batch_size
      episode_offset += batch_size
      batch_index += 1

  output: dict[str, np.ndarray] = {}
  for key in ("latent", "action", "command", "base_state", "gait_phase_sincos"):
    output[key] = np.concatenate(
      [np.concatenate(chunk[key], axis=0) for chunk in chunks], axis=0
    )
  for key in ("episode_id", "time_step", "gait_phase"):
    output[key] = np.concatenate(
      [np.concatenate(chunk[key], axis=0) for chunk in chunks], axis=0
    )
  for key in ("terrain_label", "terrain_name", "terrain_height_m"):
    values = []
    for chunk in chunks:
      values.extend(chunk[key])
    output[key] = np.asarray(values)

  np.savez_compressed(output_path, **output)
  print(f"[INFO] Wrote latent dataset to {output_path}")
  return output


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
    CollectLatentsConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_collect_latents(chosen_task, cfg)


if __name__ == "__main__":
  main()
