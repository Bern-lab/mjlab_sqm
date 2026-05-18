from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from mjlab.utils.lab_api.math import wrap_to_pi

from .velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class TeacherTargetHeadingVelocityCommand(UniformVelocityCommand):
  cfg: TeacherTargetHeadingVelocityCommandCfg

  def __init__(
    self,
    cfg: TeacherTargetHeadingVelocityCommandCfg,
    env: ManagerBasedRlEnv,
  ):
    super().__init__(cfg, env)

    self.is_target_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_random_heading_env = torch.zeros_like(self.is_target_env)

    self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_distance = torch.zeros(self.num_envs, device=self.device)
    self.target_reached = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    self.metrics["target_distance"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["target_reached"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def target_cfg(self) -> TeacherTargetHeadingVelocityCommandCfg:
    return cast(TeacherTargetHeadingVelocityCommandCfg, self.cfg)

  def _update_metrics(self) -> None:
    super()._update_metrics()
    cfg = self.target_cfg
    max_command_time = cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    target_mask = self.is_target_env.float()
    self.metrics["target_distance"] += (
      self.target_distance * target_mask / max_command_step
    )
    self.metrics["target_reached"] += self.target_reached.float() / max_command_step

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return

    cfg = self.target_cfg
    r = torch.empty(len(env_ids), device=self.device)
    mode_sum = cfg.rel_target_envs + cfg.rel_random_heading_envs + cfg.rel_standing_envs
    mode_sample = r.uniform_(0.0, mode_sum)
    target_limit = cfg.rel_target_envs
    heading_limit = target_limit + cfg.rel_random_heading_envs

    self.is_target_env[env_ids] = mode_sample < target_limit
    self.is_random_heading_env[env_ids] = (mode_sample >= target_limit) & (
      mode_sample < heading_limit
    )
    self.is_standing_env[env_ids] = mode_sample >= heading_limit

    self.is_heading_env[env_ids] = (
      self.is_target_env[env_ids] | self.is_random_heading_env[env_ids]
    )
    self.is_world_env[env_ids] = False
    self.is_forward_env[env_ids] = False
    self.target_reached[env_ids] = False

    self.vel_command_b[env_ids, 0] = r.uniform_(*cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = r.uniform_(*cfg.ranges.ang_vel_z)
    self.vel_command_w[env_ids, :] = 0.0

    zero_y_ids = env_ids[
      self.is_target_env[env_ids]
      | self.is_random_heading_env[env_ids]
      | self.is_standing_env[env_ids]
    ]
    if cfg.zero_lateral_velocity and len(zero_y_ids) > 0:
      self.vel_command_b[zero_y_ids, 1] = 0.0

    turn_ids = env_ids[self.is_random_heading_env[env_ids]]
    if len(turn_ids) > 0:
      self.vel_command_b[turn_ids, :2] = 0.0
      assert cfg.ranges.heading is not None
      self.heading_target[turn_ids] = torch.empty(
        len(turn_ids), device=self.device
      ).uniform_(*cfg.ranges.heading)

    stand_ids = env_ids[self.is_standing_env[env_ids]]
    if len(stand_ids) > 0:
      self.vel_command_b[stand_ids, :] = 0.0

    target_ids = env_ids[self.is_target_env[env_ids]]
    if len(target_ids) > 0:
      self._sample_targets(target_ids)

  def _update_command(self) -> None:
    cfg = self.target_cfg
    self.target_reached[:] = False

    target_env_ids = self.is_target_env.nonzero(as_tuple=False).flatten()
    if len(target_env_ids) > 0:
      target_delta = (
        self.target_pos_w[target_env_ids, :2]
        - self.robot.data.root_link_pos_w[target_env_ids, :2]
      )
      self.target_distance[target_env_ids] = torch.linalg.norm(target_delta, dim=1)
      too_far_ids = target_env_ids[
        self.target_distance[target_env_ids] > cfg.target_max_distance
      ]
      if len(too_far_ids) > 0:
        self._sample_targets(too_far_ids)
        target_delta = (
          self.target_pos_w[target_env_ids, :2]
          - self.robot.data.root_link_pos_w[target_env_ids, :2]
        )
        self.target_distance[target_env_ids] = torch.linalg.norm(target_delta, dim=1)
      self.heading_target[target_env_ids] = torch.atan2(
        target_delta[:, 1], target_delta[:, 0]
      )

    self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
    heading_env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
    if len(heading_env_ids) > 0:
      self.vel_command_b[heading_env_ids, 2] = torch.clip(
        cfg.heading_control_stiffness * self.heading_error[heading_env_ids],
        min=cfg.ranges.ang_vel_z[0],
        max=cfg.ranges.ang_vel_z[1],
      )

    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    if len(standing_env_ids) > 0:
      self.vel_command_b[standing_env_ids, :] = 0.0
      self.vel_command_w[standing_env_ids, :] = 0.0

    if len(target_env_ids) > 0:
      reached_ids = target_env_ids[
        self.target_distance[target_env_ids] <= cfg.target_reached_threshold
      ]
      if len(reached_ids) > 0:
        self._resample(reached_ids)
        self.target_reached[reached_ids] = True

  def _sample_targets(self, env_ids: torch.Tensor) -> None:
    cfg = self.target_cfg
    terrain = self._env.scene.terrain
    if (
      terrain is None
      or cfg.patch_name not in terrain.flat_patches
      or terrain.terrain_origins is None
    ):
      self.target_pos_w[env_ids] = self.robot.data.root_link_pos_w[env_ids]
      self.target_distance[env_ids] = 0.0
      return

    patches = terrain.flat_patches[cfg.patch_name]
    num_rows, num_cols, num_patches, _ = patches.shape
    rows, cols = self._current_tile_indices(env_ids, num_rows, num_cols)
    offsets = self._candidate_tile_offsets()

    root_xy = self.robot.data.root_link_pos_w[env_ids, :2]
    targets = self.target_pos_w[env_ids].clone()
    valid_distance = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

    for _ in range(16):
      sample_ids = (~valid_distance).nonzero(as_tuple=False).flatten()
      if len(sample_ids) == 0:
        break

      sample_rows = rows[sample_ids]
      sample_cols = cols[sample_ids]
      target_rows, target_cols = self._sample_neighbor_tiles(
        sample_rows,
        sample_cols,
        num_rows,
        num_cols,
        offsets,
      )
      patch_ids = torch.randint(0, num_patches, (len(sample_ids),), device=self.device)
      sampled_targets = patches[target_rows, target_cols, patch_ids]
      distances = torch.linalg.norm(sampled_targets[:, :2] - root_xy[sample_ids], dim=1)
      accepted = (distances >= cfg.target_min_distance) & (
        distances <= cfg.target_max_distance
      )
      targets[sample_ids] = torch.where(
        accepted.unsqueeze(1), sampled_targets, targets[sample_ids]
      )
      valid_distance[sample_ids] = accepted

    if not valid_distance.all():
      fallback_ids = (~valid_distance).nonzero(as_tuple=False).flatten()
      target_rows, target_cols = self._sample_neighbor_tiles(
        rows[fallback_ids],
        cols[fallback_ids],
        num_rows,
        num_cols,
        offsets,
      )
      patch_ids = torch.randint(
        0, num_patches, (len(fallback_ids),), device=self.device
      )
      targets[fallback_ids] = patches[target_rows, target_cols, patch_ids]

    self.target_pos_w[env_ids] = targets
    self.target_distance[env_ids] = torch.linalg.norm(
      self.target_pos_w[env_ids, :2] - root_xy, dim=1
    )

  def _current_tile_indices(
    self,
    env_ids: torch.Tensor,
    num_rows: int,
    num_cols: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    terrain = self._env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None

    size_x, size_y = terrain_generator.size
    grid_min_x = -num_rows * size_x * 0.5
    grid_min_y = -num_cols * size_y * 0.5

    root_pos = self.robot.data.root_link_pos_w[env_ids]
    rows = torch.floor((root_pos[:, 0] - grid_min_x) / size_x).long()
    cols = torch.floor((root_pos[:, 1] - grid_min_y) / size_y).long()
    rows = torch.clamp(rows, 0, num_rows - 1)
    cols = torch.clamp(cols, 0, num_cols - 1)

    first_command = self.command_counter[env_ids] == 0
    if terrain.terrain_levels is not None and terrain.terrain_types is not None:
      rows = torch.where(first_command, terrain.terrain_levels[env_ids], rows)
      cols = torch.where(first_command, terrain.terrain_types[env_ids], cols)

    return rows, cols

  def _candidate_tile_offsets(self) -> torch.Tensor:
    cfg = self.target_cfg
    radius = cfg.target_tile_radius
    offsets = [
      (row, col)
      for row in range(-radius, radius + 1)
      for col in range(-radius, radius + 1)
      if cfg.include_current_tile or row != 0 or col != 0
    ]
    if not offsets:
      offsets = [(0, 0)]
    return torch.tensor(offsets, device=self.device, dtype=torch.long)

  def _sample_neighbor_tiles(
    self,
    rows: torch.Tensor,
    cols: torch.Tensor,
    num_rows: int,
    num_cols: int,
    offsets: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    candidate_rows = rows.unsqueeze(1) + offsets[:, 0].unsqueeze(0)
    candidate_cols = cols.unsqueeze(1) + offsets[:, 1].unsqueeze(0)
    valid = (
      (candidate_rows >= 0)
      & (candidate_rows < num_rows)
      & (candidate_cols >= 0)
      & (candidate_cols < num_cols)
    )

    if not valid.all(dim=1).all():
      no_valid = ~valid.any(dim=1)
      if no_valid.any():
        candidate_rows[no_valid, 0] = rows[no_valid]
        candidate_cols[no_valid, 0] = cols[no_valid]
        valid[no_valid, 0] = True

    random_scores = torch.rand(valid.shape, device=self.device)
    random_scores = torch.where(
      valid, random_scores, torch.full_like(random_scores, -1.0)
    )
    choice = torch.argmax(random_scores, dim=1)
    batch_ids = torch.arange(len(rows), device=self.device)
    return candidate_rows[batch_ids, choice], candidate_cols[batch_ids, choice]

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    cfg = self.target_cfg
    viz = cast(TeacherTargetHeadingVelocityCommandCfg.VizCfg, cfg.viz)
    env_indices = list(visualizer.get_env_indices(self.num_envs))
    if not env_indices:
      return

    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    target_pos_ws = self.target_pos_w.cpu().numpy()
    is_target_env = self.is_target_env.cpu().numpy()

    for batch in env_indices:
      if not is_target_env[batch]:
        continue

      base_pos_w = base_pos_ws[batch]
      target_pos_w = target_pos_ws[batch].copy()
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      target_pos_w[2] += viz.z_offset
      start = base_pos_w.copy()
      start[2] += viz.z_offset

      visualizer.add_sphere(
        center=target_pos_w,
        radius=viz.target_radius,
        color=viz.target_color,
        label=f"teacher_target_heading_target_{batch}",
      )
      visualizer.add_arrow(
        start=start,
        end=target_pos_w,
        color=viz.target_arrow_color,
        width=0.02,
        label=f"teacher_target_heading_direction_{batch}",
      )


@dataclass(kw_only=True)
class TeacherTargetHeadingVelocityCommandCfg(UniformVelocityCommandCfg):
  rel_target_envs: float = 0.7
  rel_random_heading_envs: float = 0.2
  patch_name: str = "target"
  target_reached_threshold: float = 0.5
  target_min_distance: float = 1.0
  target_max_distance: float = 12.0
  target_tile_radius: int = 1
  include_current_tile: bool = False
  zero_lateral_velocity: bool = True

  @dataclass
  class VizCfg(UniformVelocityCommandCfg.VizCfg):
    target_radius: float = 0.12
    target_color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.85)
    target_arrow_color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.75)

  viz: UniformVelocityCommandCfg.VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> TeacherTargetHeadingVelocityCommand:
    return TeacherTargetHeadingVelocityCommand(self, env)

  def __post_init__(self) -> None:
    super().__post_init__()
    if not self.heading_command:
      raise ValueError(
        "TeacherTargetHeadingVelocityCommandCfg requires heading_command=True."
      )
    if self.rel_target_envs < 0.0 or self.rel_random_heading_envs < 0.0:
      raise ValueError("Target-heading command mode ratios must be non-negative.")
    if self.rel_standing_envs < 0.0:
      raise ValueError("rel_standing_envs must be non-negative.")
    mode_sum = (
      self.rel_target_envs + self.rel_random_heading_envs + self.rel_standing_envs
    )
    if mode_sum <= 0.0:
      raise ValueError("At least one target-heading command mode must be enabled.")
    if self.target_tile_radius < 0:
      raise ValueError("target_tile_radius must be non-negative.")
