"""Evaluate velocity policies on a goal-directed convex pyramid stair task."""

from __future__ import annotations

import json
import math
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import torch
import tyro
from scripts.velocity_eval.eval_metrics import (
  EVENT_COUNT_NAMES,
  LEVEL_EVENT_NAMES,
  MEAN_METRIC_NAMES,
  StairEventDetector,
  compute_velocity_metrics,
)
from scripts.velocity_eval.eval_terrains import EvalTerrainSpec, apply_eval_overrides
from scripts.velocity_eval.policy_io import (
  get_policy_output_name,
  load_inference_policy,
  make_timestamped_policy_output_dir,
  resolve_checkpoint_path,
)
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, wrap_to_pi
from mjlab.utils.lstm import (
  extract_dones,
  reset_policy_state,
  reset_policy_state_from_step,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, VerbosityLevel, ViserPlayViewer

SIDE_NAMES = ("left", "right", "bottom", "top")


@dataclass(frozen=True)
class GoalPyramidEvalConfig:
  """Configuration for goal-directed convex pyramid stair evaluation."""

  checkpoint_file: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  episodes: int = 20
  num_envs: int = 20
  max_episode_length_s: float = 12.0
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
  play: bool = False
  viewer: Literal["auto", "native", "viser"] = "auto"

  stair_levels: int = 10
  stair_height: float = 0.15
  step_width: float = 0.30
  platform_width: float = 3.0
  flat_apron_width: float = 3.0
  terrain_border_width: float = 12.0

  start_distance: float | None = None
  spawn_tangent_half_width: float | None = None
  spawn_tangent_margin: float = 0.35
  start_z_offset: float = 0.03
  goal_radius: float = 0.75
  goal_height_tolerance: float = 0.20

  goal_speed: float = 0.7
  yaw_kp: float = 1.5
  yaw_rate_limit: float = 1.0
  heading_failure_angle_deg: float = 45.0
  heading_failure_grace_s: float = 0.25


@dataclass
class SpawnInfo:
  side_ids: torch.Tensor
  side_names: list[str]
  approach_yaw: torch.Tensor
  goal_xy_w: torch.Tensor
  top_z_w: torch.Tensor
  nominal_root_height: torch.Tensor


def _make_goal_terrain(cfg: GoalPyramidEvalConfig) -> EvalTerrainSpec:
  return EvalTerrainSpec(
    name="goal_pyramid_stairs",
    label="goal_pyramid_stairs",
    kind="downstairs",
    height_m=cfg.stair_height,
    step_width=cfg.step_width,
    platform_width=cfg.platform_width,
    border_width=cfg.terrain_border_width,
    stair_riser_levels=cfg.stair_levels,
    stair_border_width=cfg.flat_apron_width,
  )


def _computed_start_distance(cfg: GoalPyramidEvalConfig) -> float:
  if cfg.start_distance is not None:
    return cfg.start_distance
  stair_run = cfg.step_width * max(0, cfg.stair_levels - 1)
  return 0.5 * cfg.platform_width + stair_run + 0.55 * cfg.flat_apron_width


def _computed_spawn_tangent_half_width(cfg: GoalPyramidEvalConfig) -> float:
  if cfg.spawn_tangent_half_width is not None:
    return cfg.spawn_tangent_half_width
  return max(0.05, 0.5 * cfg.platform_width - cfg.spawn_tangent_margin)


def _spawn_on_pyramid_apron(
  env: ManagerBasedRlEnv,
  cfg: GoalPyramidEvalConfig,
  *,
  seed: int,
  env_ids: torch.Tensor | None = None,
  spawn: SpawnInfo | None = None,
) -> SpawnInfo:
  asset = env.scene["robot"]
  if env_ids is None:
    target_env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    target_env_ids = env_ids.to(device=env.device, dtype=torch.long)
  num_targets = int(target_env_ids.numel())
  if num_targets == 0:
    if spawn is None:
      raise ValueError("Cannot create SpawnInfo with zero env_ids.")
    return spawn

  device = env.device
  generator = torch.Generator(device="cpu")
  generator.manual_seed(seed)

  side_ids = torch.randint(0, 4, (num_targets,), generator=generator).to(device=device)
  tangent_half_width = _computed_spawn_tangent_half_width(cfg)
  tangent = (
    torch.rand(num_targets, generator=generator).to(device=device) * 2.0 - 1.0
  ) * tangent_half_width
  radial = _computed_start_distance(cfg)

  offsets = torch.zeros(num_targets, 2, device=device)
  approach_yaw = torch.zeros(num_targets, device=device)

  left = side_ids == 0
  right = side_ids == 1
  bottom = side_ids == 2
  top = side_ids == 3
  offsets[left, 0] = -radial
  offsets[left, 1] = tangent[left]
  approach_yaw[left] = 0.0

  offsets[right, 0] = radial
  offsets[right, 1] = tangent[right]
  approach_yaw[right] = math.pi

  offsets[bottom, 0] = tangent[bottom]
  offsets[bottom, 1] = -radial
  approach_yaw[bottom] = 0.5 * math.pi

  offsets[top, 0] = tangent[top]
  offsets[top, 1] = radial
  approach_yaw[top] = -0.5 * math.pi

  env_origins = env.scene.env_origins[target_env_ids]
  goal_xy_w = env_origins[:, :2].clone()
  top_z_w = env_origins[:, 2].clone()
  bottom_z_w = top_z_w - float(cfg.stair_levels) * cfg.stair_height

  default_root_state = asset.data.default_root_state[target_env_ids].clone()
  nominal_root_height = default_root_state[:, 2].clone()
  root_pos_w = torch.zeros(num_targets, 3, device=device)
  root_pos_w[:, :2] = goal_xy_w + offsets
  root_pos_w[:, 2] = bottom_z_w + nominal_root_height + cfg.start_z_offset

  zeros = torch.zeros(num_targets, device=device)
  yaw_delta = quat_from_euler_xyz(zeros, zeros, approach_yaw)
  root_quat_w = quat_mul(default_root_state[:, 3:7], yaw_delta)
  root_vel_w = torch.zeros(num_targets, 6, device=device)

  asset.write_root_link_pose_to_sim(
    torch.cat([root_pos_w, root_quat_w], dim=-1), env_ids=target_env_ids
  )
  asset.write_root_link_velocity_to_sim(root_vel_w, env_ids=target_env_ids)
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.sim.sense()

  sampled_side_names = [SIDE_NAMES[int(side_id)] for side_id in side_ids.detach().cpu()]
  if spawn is not None:
    spawn.side_ids[target_env_ids] = side_ids
    for env_index, side_name in zip(
      target_env_ids.detach().cpu().tolist(), sampled_side_names, strict=True
    ):
      spawn.side_names[int(env_index)] = side_name
    spawn.approach_yaw[target_env_ids] = approach_yaw
    spawn.goal_xy_w[target_env_ids] = goal_xy_w
    spawn.top_z_w[target_env_ids] = top_z_w
    spawn.nominal_root_height[target_env_ids] = nominal_root_height
    return spawn

  full_side_ids = torch.zeros(env.num_envs, dtype=torch.long, device=device)
  full_approach_yaw = torch.zeros(env.num_envs, device=device)
  full_goal_xy_w = env.scene.env_origins[:, :2].clone()
  full_top_z_w = env.scene.env_origins[:, 2].clone()
  full_nominal_root_height = asset.data.default_root_state[:, 2].clone()
  full_side_names = ["left" for _ in range(env.num_envs)]

  full_side_ids[target_env_ids] = side_ids
  full_approach_yaw[target_env_ids] = approach_yaw
  full_goal_xy_w[target_env_ids] = goal_xy_w
  full_top_z_w[target_env_ids] = top_z_w
  full_nominal_root_height[target_env_ids] = nominal_root_height
  for env_index, side_name in zip(
    target_env_ids.detach().cpu().tolist(), sampled_side_names, strict=True
  ):
    full_side_names[int(env_index)] = side_name

  return SpawnInfo(
    side_ids=full_side_ids,
    side_names=full_side_names,
    approach_yaw=full_approach_yaw,
    goal_xy_w=full_goal_xy_w,
    top_z_w=full_top_z_w,
    nominal_root_height=full_nominal_root_height,
  )


def _get_twist_term(env: ManagerBasedRlEnv) -> UniformVelocityCommand:
  term = env.command_manager.get_term("twist")
  if not isinstance(term, UniformVelocityCommand):
    raise TypeError("Goal pyramid eval expects the 'twist' command to be velocity-like.")
  return term


def _update_goal_command(
  env: ManagerBasedRlEnv,
  cfg: GoalPyramidEvalConfig,
  spawn: SpawnInfo,
  active: torch.Tensor,
) -> torch.Tensor:
  asset = env.scene["robot"]
  root_xy = asset.data.root_link_pos_w[:, :2]
  delta_xy = spawn.goal_xy_w - root_xy
  distance = torch.norm(delta_xy, dim=-1)
  target_yaw = torch.atan2(delta_xy[:, 1], delta_xy[:, 0])
  yaw_error = wrap_to_pi(target_yaw - asset.data.heading_w)

  speed = cfg.goal_speed * torch.clamp(torch.cos(yaw_error), min=0.0, max=1.0)
  speed = torch.where(distance <= cfg.goal_radius, torch.zeros_like(speed), speed)
  yaw_rate = torch.clamp(
    cfg.yaw_kp * yaw_error,
    min=-cfg.yaw_rate_limit,
    max=cfg.yaw_rate_limit,
  )

  term = _get_twist_term(env)
  active_ids = active.nonzero(as_tuple=False).flatten()
  term.vel_command_b.zero_()
  if active_ids.numel() > 0:
    term.vel_command_b[active_ids, 0] = speed[active_ids]
    term.vel_command_b[active_ids, 2] = yaw_rate[active_ids]
  term.vel_command_w.zero_()
  term.heading_target.copy_(target_yaw)
  term.is_standing_env.fill_(False)
  term.is_world_env.fill_(False)
  term.is_forward_env.fill_(False)
  term.is_heading_env.fill_(False)
  return yaw_error


def _goal_reached(
  env: ManagerBasedRlEnv,
  cfg: GoalPyramidEvalConfig,
  spawn: SpawnInfo,
) -> torch.Tensor:
  asset = env.scene["robot"]
  root_pos = asset.data.root_link_pos_w
  distance = torch.norm(root_pos[:, :2] - spawn.goal_xy_w, dim=-1)
  estimated_support_z = root_pos[:, 2] - spawn.nominal_root_height
  height_ok = estimated_support_z >= spawn.top_z_w - cfg.goal_height_tolerance
  return (distance <= cfg.goal_radius) & height_ok


def _heading_failure(
  env: ManagerBasedRlEnv,
  cfg: GoalPyramidEvalConfig,
  spawn: SpawnInfo,
  *,
  step_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  asset = env.scene["robot"]
  heading_error = torch.abs(wrap_to_pi(asset.data.heading_w - spawn.approach_yaw))
  grace_done = step_counts * env.step_dt >= cfg.heading_failure_grace_s
  failed = grace_done & (heading_error > math.radians(cfg.heading_failure_angle_deg))
  return failed, heading_error


def _fresh_obs_with_history(env: ManagerBasedRlEnv) -> TensorDict:
  env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  env.observation_manager.reset(env_ids)
  obs_dict = env.observation_manager.compute(update_history=True)
  return TensorDict(obs_dict, batch_size=[env.num_envs])


def _empty_batch_tensors(
  num_envs: int, max_levels: int, device: str
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
  metric_sums = {
    name: torch.zeros(num_envs, device=device)
    for name in (*MEAN_METRIC_NAMES, *EVENT_COUNT_NAMES)
  }
  level_sums = {
    name: torch.zeros(num_envs, max_levels, device=device)
    for name in LEVEL_EVENT_NAMES
  }
  return metric_sums, level_sums


def _run_batch(
  *,
  task_id: str,
  agent_cfg,
  checkpoint_path: Path,
  cfg: GoalPyramidEvalConfig,
  batch_size: int,
  batch_index: int,
  device: str,
) -> dict:
  terrain = _make_goal_terrain(cfg)
  env_cfg = load_env_cfg(task_id, play=False)
  apply_eval_overrides(
    env_cfg,
    terrain,
    num_envs=batch_size,
    seed=cfg.seed + 1009 * batch_index,
    max_episode_length_s=cfg.max_episode_length_s,
    command=(0.0, 0.0, 0.0),
    clean_observations=cfg.clean_observations,
    disable_observation_delay=cfg.disable_observation_delay,
    disable_actuator_delay=cfg.disable_actuator_delay,
    enable_riser_contact_sensor="g1" in task_id.lower(),
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
    spawn = _spawn_on_pyramid_apron(
      wrapped.unwrapped,
      cfg,
      seed=cfg.seed + 1543 * batch_index,
    )

    all_envs_active = torch.ones(batch_size, dtype=torch.bool, device=device)
    _update_goal_command(wrapped.unwrapped, cfg, spawn, all_envs_active)
    obs = _fresh_obs_with_history(wrapped.unwrapped)

    metric_sums, level_sums = _empty_batch_tensors(
      batch_size, cfg.stair_levels, device
    )
    step_counts = torch.zeros(batch_size, device=device)
    done_envs = torch.zeros(batch_size, dtype=torch.bool, device=device)
    success = torch.zeros(batch_size, dtype=torch.bool, device=device)
    fell = torch.zeros(batch_size, dtype=torch.bool, device=device)
    heading_failed = torch.zeros(batch_size, dtype=torch.bool, device=device)
    timeout_failed = torch.zeros(batch_size, dtype=torch.bool, device=device)
    max_heading_error = torch.zeros(batch_size, device=device)

    max_steps = wrapped.unwrapped.max_episode_length + 2
    for _step in range(max_steps):
      active = ~done_envs
      reached_now = _goal_reached(wrapped.unwrapped, cfg, spawn) & active
      if bool(reached_now.any().item()):
        success |= reached_now
        done_envs |= reached_now
        active = ~done_envs
      if not bool(active.any().item()):
        break

      _update_goal_command(wrapped.unwrapped, cfg, spawn, active)
      with torch.no_grad():
        step_metrics, step_levels = compute_velocity_metrics(
          wrapped.unwrapped,
          detector,
          terrain_height_m=cfg.stair_height,
          max_levels=cfg.stair_levels,
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

      reached_after_step = _goal_reached(wrapped.unwrapped, cfg, spawn) & active
      reached_after_step &= ~newly_done
      heading_failed_now, heading_error = _heading_failure(
        wrapped.unwrapped,
        cfg,
        spawn,
        step_counts=step_counts,
      )
      heading_failed_now &= active & ~newly_done & ~reached_after_step
      valid_heading_sample = active & ~newly_done
      max_heading_error = torch.where(
        valid_heading_sample,
        torch.maximum(max_heading_error, heading_error),
        max_heading_error,
      )

      if "fell_over" in wrapped.unwrapped.termination_manager.active_terms:
        fell_now = wrapped.unwrapped.termination_manager.get_term("fell_over").bool()
      else:
        fell_now = terminated

      success |= reached_after_step
      fell |= newly_done & fell_now
      timeout_failed |= newly_done & truncated & ~terminated
      heading_failed |= heading_failed_now
      done_envs |= newly_done | reached_after_step | heading_failed_now

    unfinished = ~done_envs
    timeout_failed |= unfinished

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
      "event_source": detector.event_source,
      "success": success.detach().cpu().tolist(),
      "fell": fell.detach().cpu().tolist(),
      "heading_failed": heading_failed.detach().cpu().tolist(),
      "timeout_failed": timeout_failed.detach().cpu().tolist(),
      "spawn_side": spawn.side_names,
      "max_heading_error_deg": torch.rad2deg(max_heading_error)
      .detach()
      .cpu()
      .tolist(),
      "episode_length_steps": step_counts.detach().cpu().tolist(),
      "mean_metrics": mean_metrics,
      "event_counts": event_counts,
      "level_counts": level_counts,
      "step_dt": wrapped.unwrapped.step_dt,
    }
  finally:
    wrapped.close()


class _GoalPyramidViewerProtocol(Protocol):
  env: RslRlVecEnvWrapper
  policy: Any
  _step_count: int
  _stats_steps: int
  _sim_budget: float
  _last_error: str | None

  def log(self, message: str, level: VerbosityLevel = VerbosityLevel.INFO) -> None: ...

  def pause(self) -> None: ...


class GoalPyramidPlayMixin:
  """Viewer mixin that injects goal-pyramid navigation before policy inference."""

  def __init__(
    self,
    *args,
    goal_cfg: GoalPyramidEvalConfig,
    spawn: SpawnInfo,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    self._goal_cfg = goal_cfg
    self._goal_spawn = spawn
    self._goal_respawn_counter = 0

  def _respawn_goal_envs(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    viewer = cast("_GoalPyramidViewerProtocol", self)
    env = viewer.env.unwrapped
    env.reset(env_ids=env_ids)
    self._goal_respawn_counter += 1
    self._goal_spawn = _spawn_on_pyramid_apron(
      env,
      self._goal_cfg,
      seed=self._goal_cfg.seed + 7919 * self._goal_respawn_counter,
      env_ids=env_ids,
      spawn=self._goal_spawn,
    )
    env.observation_manager.reset(env_ids)
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    _update_goal_command(env, self._goal_cfg, self._goal_spawn, active)

    dones = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    dones[env_ids] = True
    reset_policy_state(viewer.policy, dones)

  def _execute_step(self) -> bool:
    try:
      viewer = cast("_GoalPyramidViewerProtocol", self)
      env = viewer.env.unwrapped
      active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
      _update_goal_command(env, self._goal_cfg, self._goal_spawn, active)

      with torch.no_grad():
        obs = viewer.env.get_observations()
        actions = viewer.policy(obs)
        step_result = viewer.env.step(actions)

      built_in_dones = extract_dones(step_result)
      if built_in_dones is None:
        built_in_dones = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
      else:
        built_in_dones = built_in_dones.to(device=env.device, dtype=torch.bool)

      reached = _goal_reached(env, self._goal_cfg, self._goal_spawn)
      heading_failed, _heading_error = _heading_failure(
        env,
        self._goal_cfg,
        self._goal_spawn,
        step_counts=env.episode_length_buf.float(),
      )
      custom_dones = (reached | heading_failed) & ~built_in_dones
      reset_mask = built_in_dones | custom_dones
      if bool(reset_mask.any().item()):
        reset_env_ids = reset_mask.nonzero(as_tuple=False).flatten()
        self._respawn_goal_envs(reset_env_ids)
      else:
        reset_policy_state_from_step(viewer.policy, step_result)

      self._step_count += 1
      self._stats_steps += 1
      return True
    except Exception:
      self._last_error = traceback.format_exc()
      viewer = cast("_GoalPyramidViewerProtocol", self)
      viewer.log(
        f"[ERROR] Exception during step:\n{self._last_error}",
        VerbosityLevel.SILENT,
      )
      viewer.pause()
      return False

  def reset_environment(self) -> None:
    viewer = cast("_GoalPyramidViewerProtocol", self)
    viewer.env.reset()
    reset_policy_state(viewer.policy)
    self._goal_respawn_counter += 1
    self._goal_spawn = _spawn_on_pyramid_apron(
      viewer.env.unwrapped,
      self._goal_cfg,
      seed=self._goal_cfg.seed + 7919 * self._goal_respawn_counter,
    )
    active = torch.ones(
      viewer.env.unwrapped.num_envs,
      dtype=torch.bool,
      device=viewer.env.unwrapped.device,
    )
    _update_goal_command(viewer.env.unwrapped, self._goal_cfg, self._goal_spawn, active)
    self._step_count = 0
    self._sim_budget = 0.0
    self._last_error = None


class GoalPyramidNativeViewer(GoalPyramidPlayMixin, NativeMujocoViewer):
  pass


class GoalPyramidViserViewer(GoalPyramidPlayMixin, ViserPlayViewer):
  pass


def run_goal_pyramid_play(task_id: str, cfg: GoalPyramidEvalConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  agent_cfg = load_rl_cfg(task_id)
  checkpoint_path = resolve_checkpoint_path(
    task_id=task_id,
    agent_cfg=agent_cfg,
    checkpoint_file=cfg.checkpoint_file,
    wandb_run_path=cfg.wandb_run_path,
    wandb_checkpoint_name=cfg.wandb_checkpoint_name,
  )

  terrain = _make_goal_terrain(cfg)
  env_cfg = load_env_cfg(task_id, play=True)
  apply_eval_overrides(
    env_cfg,
    terrain,
    num_envs=cfg.num_envs,
    seed=cfg.seed,
    max_episode_length_s=cfg.max_episode_length_s,
    command=(0.0, 0.0, 0.0),
    clean_observations=cfg.clean_observations,
    disable_observation_delay=cfg.disable_observation_delay,
    disable_actuator_delay=cfg.disable_actuator_delay,
    enable_riser_contact_sensor="g1" in task_id.lower(),
  )
  env_cfg.viewer.distance = max(env_cfg.viewer.distance, 8.0)
  env_cfg.viewer.elevation = min(env_cfg.viewer.elevation, -30.0)
  env_cfg.viewer.max_extra_envs = max(env_cfg.viewer.max_extra_envs, cfg.num_envs - 1)

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
    spawn = _spawn_on_pyramid_apron(wrapped.unwrapped, cfg, seed=cfg.seed)
    active = torch.ones(cfg.num_envs, dtype=torch.bool, device=wrapped.unwrapped.device)
    _update_goal_command(wrapped.unwrapped, cfg, spawn, active)
    _fresh_obs_with_history(wrapped.unwrapped)

    if cfg.viewer == "auto":
      has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
      resolved_viewer = "native" if has_display else "viser"
    else:
      resolved_viewer = cfg.viewer

    print(
      f"[INFO] Playing goal_pyramid with {cfg.num_envs} envs "
      f"using {resolved_viewer} viewer"
    )
    if resolved_viewer == "native":
      GoalPyramidNativeViewer(
        wrapped,
        policy,
        goal_cfg=cfg,
        spawn=spawn,
      ).run()
    elif resolved_viewer == "viser":
      GoalPyramidViserViewer(
        wrapped,
        policy,
        goal_cfg=cfg,
        spawn=spawn,
      ).run()
    else:
      raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
  finally:
    wrapped.close()


def _average_values(values: list[float], mask: list[bool] | None = None) -> float:
  if mask is not None:
    values = [value for value, keep in zip(values, mask, strict=True) if keep]
  if not values:
    return 0.0
  return float(sum(values) / len(values))


def _level_collision_table(summary: dict, cfg: GoalPyramidEvalConfig) -> list[dict]:
  toe = summary["toe_riser_collision_by_level_success_only"]
  heel = summary["heel_riser_collision_by_level_success_only"]
  lip = summary["foot_lip_collision_by_level_success_only"]
  rows = []
  for index in range(cfg.stair_levels):
    toe_count = float(toe[index])
    heel_count = float(heel[index])
    lip_count = float(lip[index])
    rows.append(
      {
        "level_low_to_high": index + 1,
        "height_m": float((index + 1) * cfg.stair_height),
        "toe_riser_collision_count": toe_count,
        "heel_riser_collision_count": heel_count,
        "foot_lip_collision_count": lip_count,
        "total_riser_collision_count": toe_count + heel_count,
      }
    )
  return rows


def _summarize_batches(cfg: GoalPyramidEvalConfig, batches: list[dict]) -> dict:
  success = [item for batch in batches for item in batch["success"]]
  fell = [item for batch in batches for item in batch["fell"]]
  heading_failed = [item for batch in batches for item in batch["heading_failed"]]
  timeout_failed = [item for batch in batches for item in batch["timeout_failed"]]
  spawn_sides = [item for batch in batches for item in batch["spawn_side"]]
  lengths = [item for batch in batches for item in batch["episode_length_steps"]]
  heading_errors = [
    item for batch in batches for item in batch["max_heading_error_deg"]
  ]
  step_dt = batches[0]["step_dt"] if batches else 0.0
  episodes = max(1, len(success))
  success_count = int(sum(bool(item) for item in success))
  success_mask = [bool(item) for item in success]

  summary = {
    "episodes": len(success),
    "success_episodes": success_count,
    "collision_stat_episodes": success_count,
    "collision_stat_policy": "success_only",
    "success_rate": float(sum(success) / episodes),
    "fall_rate": float(sum(fell) / episodes),
    "heading_failure_rate": float(sum(heading_failed) / episodes),
    "timeout_failure_rate": float(sum(timeout_failed) / episodes),
    "mean_episode_length_s": float(sum(lengths) * step_dt / episodes),
    "mean_success_time_s": _average_values(
      [length * step_dt for length in lengths], success_mask
    ),
    "mean_max_heading_error_deg": _average_values(heading_errors),
    "spawn_side_counts": {
      side: int(sum(1 for item in spawn_sides if item == side)) for side in SIDE_NAMES
    },
    "collision_event_source": batches[0].get("event_source") if batches else None,
  }

  for name in MEAN_METRIC_NAMES:
    values = [v for batch in batches for v in batch["mean_metrics"][name]]
    summary[name] = _average_values(values)
    summary[f"{name}_success_only"] = _average_values(values, success_mask)

  for name in EVENT_COUNT_NAMES:
    values = [v for batch in batches for v in batch["event_counts"][name]]
    summary[f"{name}_count_success_only"] = _average_values(values, success_mask)

  for name in LEVEL_EVENT_NAMES:
    total = [0.0 for _ in range(cfg.stair_levels)]
    for batch in batches:
      for row, keep in zip(
        batch["level_counts"][name],
        batch["success"],
        strict=True,
      ):
        if not keep:
          continue
        for idx, value in enumerate(row[: cfg.stair_levels]):
          total[idx] += value
    denom = max(1, success_count)
    summary[f"{name}_success_only"] = [float(value / denom) for value in total]

  summary["collision_by_stair_level_low_to_high_success_only"] = (
    _level_collision_table(summary, cfg)
  )
  return summary


def _resolve_output_path(
  *,
  cfg: GoalPyramidEvalConfig,
  task_id: str,
  agent_cfg,
  checkpoint_path: Path,
) -> Path:
  default_name = f"goal_pyramid_h{int(round(cfg.stair_height * 100)):02d}cm.json"
  if cfg.output_file is not None:
    output_path = Path(cfg.output_file)
    if output_path.is_dir() or output_path.suffix.lower() != ".json":
      output_dir = make_timestamped_policy_output_dir(
        output_root=output_path,
        task_id=task_id,
        agent_cfg=agent_cfg,
        checkpoint_path=checkpoint_path,
      )
      return output_dir / default_name
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
  return output_dir / default_name


def _resolve_table_image_path(
  cfg: GoalPyramidEvalConfig,
  output_path: Path,
) -> Path | None:
  if cfg.table_image_file is not None:
    path = Path(cfg.table_image_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
  if not cfg.write_table_image:
    return None
  return output_path.with_name(f"{output_path.stem}_table.png")


def _write_table_image(payload: dict, output_path: Path) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  summary = payload["summary"]
  terrain = payload["terrain"]
  nav = payload["navigation"]
  summary_columns = [
    "success %",
    "fall %",
    "dir fail %",
    "timeout %",
    "time s",
    "succ time s",
    "toe",
    "heel",
    "total",
    "succ eps",
  ]
  toe = summary["toe_riser_collision_count_success_only"]
  heel = summary["heel_riser_collision_count_success_only"]
  summary_values = [
    summary["success_rate"] * 100.0,
    summary["fall_rate"] * 100.0,
    summary["heading_failure_rate"] * 100.0,
    summary["timeout_failure_rate"] * 100.0,
    summary["mean_episode_length_s"],
    summary["mean_success_time_s"],
    toe,
    heel,
    toe + heel,
    summary["success_episodes"],
  ]
  summary_text = [
    [
      f"{summary_values[0]:.0f}",
      f"{summary_values[1]:.0f}",
      f"{summary_values[2]:.0f}",
      f"{summary_values[3]:.0f}",
      f"{summary_values[4]:.2f}",
      f"{summary_values[5]:.2f}",
      f"{summary_values[6]:.2f}",
      f"{summary_values[7]:.2f}",
      f"{summary_values[8]:.2f}",
      f"{summary_values[9]:.0f}",
    ]
  ]

  level_rows = []
  for row in summary["collision_by_stair_level_low_to_high_success_only"]:
    level_rows.append(
      [
        str(row["level_low_to_high"]),
        f"{row['height_m']:.2f}",
        f"{row['toe_riser_collision_count']:.2f}",
        f"{row['heel_riser_collision_count']:.2f}",
        f"{row['total_riser_collision_count']:.2f}",
      ]
    )

  fig_height = max(5.4, 3.2 + 0.28 * len(level_rows))
  fig, axes = plt.subplots(2, 1, figsize=(12.5, fig_height), dpi=180)
  for ax in axes:
    ax.axis("off")

  title = (
    f"{payload.get('policy_output_name', payload.get('task_id', 'policy'))} | "
    f"goal pyramid | levels={terrain['stair_levels']} | "
    f"h={terrain['stair_height']}, run={terrain['step_width']}, "
    f"platform={terrain['platform_width']} | speed={nav['goal_speed']}"
  )
  axes[0].text(
    0.5,
    0.92,
    title,
    ha="center",
    va="center",
    fontsize=10.5,
    fontweight="bold",
    transform=axes[0].transAxes,
  )
  summary_table = axes[0].table(
    cellText=summary_text,
    colLabels=summary_columns,
    cellLoc="center",
    loc="center",
  )
  summary_table.auto_set_font_size(False)
  summary_table.set_fontsize(8.0)
  summary_table.scale(1.0, 1.35)

  for (row_idx, col_idx), cell in summary_table.get_celld().items():
    cell.set_edgecolor("#cfd6de")
    cell.set_linewidth(0.7)
    if row_idx == 0:
      cell.set_facecolor("#243447")
      cell.set_text_props(color="white", fontweight="bold")
    elif col_idx == 0:
      value = summary_values[0]
      cell.set_facecolor("#d8f0dd" if value >= 80.0 else "#fff0c2")
    elif col_idx in {1, 2, 3}:
      value = summary_values[col_idx]
      cell.set_facecolor("#f7d4d4" if value > 0.0 else "#f3f6f8")
    elif col_idx in {6, 7, 8}:
      value = summary_values[col_idx]
      cell.set_facecolor("#f3f6f8" if value <= 0.0 else "#fff0c2")

  level_table = axes[1].table(
    cellText=level_rows,
    colLabels=["level", "height m", "toe", "heel", "total"],
    cellLoc="center",
    loc="center",
  )
  level_table.auto_set_font_size(False)
  level_table.set_fontsize(8.0)
  level_table.scale(1.0, 1.22)
  for (row_idx, _col_idx), cell in level_table.get_celld().items():
    cell.set_edgecolor("#cfd6de")
    cell.set_linewidth(0.7)
    if row_idx == 0:
      cell.set_facecolor("#243447")
      cell.set_text_props(color="white", fontweight="bold")
    else:
      cell.set_facecolor("#ffffff")

  axes[1].text(
    0.5,
    0.04,
    "Per-level collision counts are low-to-high means over successful episodes only.",
    ha="center",
    va="center",
    fontsize=8,
    color="#4d5b68",
    transform=axes[1].transAxes,
  )
  fig.tight_layout()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output_path, bbox_inches="tight")
  plt.close(fig)


def run_goal_pyramid_eval(task_id: str, cfg: GoalPyramidEvalConfig) -> dict:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
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

  batches = []
  remaining = cfg.episodes
  batch_index = 0
  while remaining > 0:
    batch_size = min(max(1, cfg.num_envs), remaining)
    print(
      f"[INFO] Evaluating goal_pyramid batch {batch_index} "
      f"({batch_size} episodes)"
    )
    batches.append(
      _run_batch(
        task_id=task_id,
        agent_cfg=agent_cfg,
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        batch_size=batch_size,
        batch_index=batch_index,
        device=device,
      )
    )
    remaining -= batch_size
    batch_index += 1

  summary = _summarize_batches(cfg, batches)
  terrain_spec = _make_goal_terrain(cfg)
  tile_size = terrain_spec.generator_size()
  terrain = {
    "type": "pyramid_stairs",
    "stair_levels": cfg.stair_levels,
    "stair_height": cfg.stair_height,
    "step_width": cfg.step_width,
    "platform_width": cfg.platform_width,
    "flat_apron_width": cfg.flat_apron_width,
    "terrain_border_width": cfg.terrain_border_width,
    "tile_size_m": [float(tile_size[0]), float(tile_size[1])],
    "start_distance": _computed_start_distance(cfg),
    "spawn_tangent_half_width": _computed_spawn_tangent_half_width(cfg),
  }
  navigation = {
    "goal_radius": cfg.goal_radius,
    "goal_height_tolerance": cfg.goal_height_tolerance,
    "goal_speed": cfg.goal_speed,
    "yaw_kp": cfg.yaw_kp,
    "yaw_rate_limit": cfg.yaw_rate_limit,
    "heading_failure_angle_deg": cfg.heading_failure_angle_deg,
    "heading_failure_grace_s": cfg.heading_failure_grace_s,
  }
  payload = {
    "task_id": task_id,
    "policy_output_name": policy_output_name,
    "checkpoint": str(checkpoint_path),
    "output_dir": str(output_path.parent),
    "mode": "goal_pyramid",
    "config": asdict(cfg),
    "terrain": terrain,
    "navigation": navigation,
    "summary": summary,
  }

  output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
  print(f"[INFO] Wrote goal pyramid evaluation to {output_path}")
  table_path = _resolve_table_image_path(cfg, output_path)
  if table_path is not None:
    _write_table_image(payload, table_path)
    print(f"[INFO] Wrote goal pyramid table image to {table_path}")

  print(
    "[INFO] goal_pyramid: "
    f"success={summary['success_rate']:.3f}, "
    f"fall={summary['fall_rate']:.3f}, "
    f"dir_fail={summary['heading_failure_rate']:.3f}, "
    f"toe={summary['toe_riser_collision_count_success_only']:.3f}, "
    f"heel={summary['heel_riser_collision_count_success_only']:.3f}"
  )
  return payload


def _normalize_standalone_bool_flags(
  args: list[str], flag_names: tuple[str, ...]
) -> list[str]:
  """Allow selected bool options to be passed as bare flags despite TYRO_FLAGS."""
  normalized: list[str] = []
  flag_set = set(flag_names)
  index = 0
  while index < len(args):
    arg = args[index]
    normalized.append(arg)
    if arg in flag_set and (index + 1 >= len(args) or args[index + 1].startswith("-")):
      normalized.append("True")
    index += 1
  return normalized


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
    GoalPyramidEvalConfig,
    args=_normalize_standalone_bool_flags(list(remaining_args), ("--play",)),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  if cfg.play:
    run_goal_pyramid_play(chosen_task, cfg)
    return
  run_goal_pyramid_eval(chosen_task, cfg)


if __name__ == "__main__":
  main()
