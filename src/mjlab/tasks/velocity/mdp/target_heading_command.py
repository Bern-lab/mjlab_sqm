from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class TargetHeadingVelocityCommand(UniformVelocityCommand):
  """Velocity command whose heading target comes from terrain target points.

  The public command stays identical to ``UniformVelocityCommand``:
  ``[lin_vel_x, lin_vel_y, ang_vel_z]``. Target points only update the internal
  heading target used by the existing heading controller.
  """

  cfg: TargetHeadingVelocityCommandCfg

  def __init__(
    self,
    cfg: TargetHeadingVelocityCommandCfg,
    env: ManagerBasedRlEnv,
  ):
    super().__init__(cfg, env)

    terrain = env.scene.terrain
    if terrain is None:
      raise RuntimeError("TargetHeadingVelocityCommand requires scene terrain.")
    if cfg.patch_name not in terrain.flat_patches:
      raise RuntimeError(
        f"TargetHeadingVelocityCommand requires "
        f"terrain.flat_patches['{cfg.patch_name}']. Available flat patch names: "
        f"{list(terrain.flat_patches.keys())}"
      )

    self.terrain = terrain
    self.valid_targets = terrain.flat_patches[cfg.patch_name]
    terrain_cfg = terrain.cfg.terrain_generator
    if terrain_cfg is None:
      raise RuntimeError(
        "TargetHeadingVelocityCommand requires generated terrain with tile size."
      )
    self.tile_size = terrain_cfg.size
    self.num_target_rows = self.valid_targets.shape[0]
    self.num_target_cols = self.valid_targets.shape[1]
    self.grid_min_x = -self.num_target_rows * self.tile_size[0] * 0.5
    self.grid_min_y = -self.num_target_cols * self.tile_size[1] * 0.5

    self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.has_target = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.target_reached_this_step = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_target_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_random_heading_env = torch.zeros_like(self.is_target_env)

    self.metrics["target_dist"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["num_target_envs"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["num_random_heading_envs"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["num_standing_envs"] = torch.zeros(self.num_envs, device=self.device)

  def _update_metrics(self) -> None:
    super()._update_metrics()

    target_ids = (
      (self.is_target_env & self.has_target).nonzero(as_tuple=False).flatten()
    )
    if len(target_ids) > 0:
      dist = torch.norm(
        self.target_pos_w[target_ids, :2]
        - self.robot.data.root_link_pos_w[target_ids, :2],
        dim=-1,
      )
      self.metrics["target_dist"][target_ids] += dist / self._env.max_episode_length

    self.metrics["num_target_envs"] += (
      self.is_target_env.float() / self._env.max_episode_length
    )
    self.metrics["num_random_heading_envs"] += (
      self.is_random_heading_env.float() / self._env.max_episode_length
    )
    self.metrics["num_standing_envs"] += (
      self.is_standing_env.float() / self._env.max_episode_length
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Sample a mutually exclusive command mode for each environment."""
    if len(env_ids) == 0:
      return

    self.is_target_env[env_ids] = False
    self.is_random_heading_env[env_ids] = False
    self.is_standing_env[env_ids] = False
    self.is_heading_env[env_ids] = False
    self.is_world_env[env_ids] = False
    self.is_forward_env[env_ids] = False
    self.has_target[env_ids] = False

    self.vel_command_b[env_ids, :] = 0.0
    self.vel_command_w[env_ids, :] = 0.0

    p_target = self.cfg.rel_target_envs
    p_random_heading = self.cfg.rel_random_heading_envs
    p_standing = self.cfg.rel_standing_envs
    p_total = p_target + p_random_heading + p_standing
    if p_total <= 0.0:
      raise ValueError(
        "At least one of rel_target_envs, rel_random_heading_envs, "
        "rel_standing_envs must be positive."
      )

    r = torch.rand(len(env_ids), device=self.device)
    target_cut = p_target / p_total
    random_heading_cut = (p_target + p_random_heading) / p_total

    target_ids = env_ids[r < target_cut]
    random_heading_ids = env_ids[(r >= target_cut) & (r < random_heading_cut)]
    standing_ids = env_ids[r >= random_heading_cut]

    if len(target_ids) > 0:
      self.is_target_env[target_ids] = True
      self.is_heading_env[target_ids] = True
      self._sample_target_points(target_ids)

      self.vel_command_b[target_ids, 0] = torch.empty(
        len(target_ids), device=self.device
      ).uniform_(*self.cfg.ranges.lin_vel_x)
      if self.cfg.zero_lateral_velocity:
        self.vel_command_b[target_ids, 1] = 0.0
      else:
        self.vel_command_b[target_ids, 1] = torch.empty(
          len(target_ids), device=self.device
        ).uniform_(*self.cfg.ranges.lin_vel_y)

    if len(random_heading_ids) > 0:
      self.is_random_heading_env[random_heading_ids] = True
      self.is_heading_env[random_heading_ids] = True
      assert self.cfg.ranges.heading is not None
      self.heading_target[random_heading_ids] = torch.empty(
        len(random_heading_ids), device=self.device
      ).uniform_(*self.cfg.ranges.heading)

    if len(standing_ids) > 0:
      self.is_standing_env[standing_ids] = True

  def _sample_target_points(self, env_ids: torch.Tensor) -> None:
    """Sample active target points from ``terrain.flat_patches``."""
    for env_id_tensor in env_ids:
      env_id = int(env_id_tensor.item())
      tile_indices = self._neighbor_tile_indices(env_id)
      if not tile_indices:
        self.has_target[env_id] = False
        continue

      candidates = torch.cat(
        [self.valid_targets[row, col] for row, col in tile_indices],
        dim=0,
      )
      if candidates.numel() == 0:
        self.has_target[env_id] = False
        continue

      robot_xy = self.robot.data.root_link_pos_w[env_id, :2]
      dist = torch.norm(candidates[:, :2] - robot_xy[None, :], dim=-1)
      valid_mask = dist >= self.cfg.target_min_distance
      if self.cfg.target_max_distance is not None:
        valid_mask &= dist <= self.cfg.target_max_distance
      if valid_mask.any():
        candidates = candidates[valid_mask]

      idx = int(
        torch.randint(0, candidates.shape[0], (1,), device=self.device).item()
      )
      self.target_pos_w[env_id] = candidates[idx]
      self.has_target[env_id] = True

  def _neighbor_tile_indices(self, env_id: int) -> list[tuple[int, int]]:
    """Return target tile indices around the robot's current terrain tile."""
    robot_pos = self.robot.data.root_link_pos_w[env_id]
    row = int(torch.floor((robot_pos[0] - self.grid_min_x) / self.tile_size[0]).item())
    col = int(torch.floor((robot_pos[1] - self.grid_min_y) / self.tile_size[1]).item())
    row = max(0, min(row, self.num_target_rows - 1))
    col = max(0, min(col, self.num_target_cols - 1))

    tile_indices: list[tuple[int, int]] = []
    radius = self.cfg.target_tile_radius
    for drow in range(-radius, radius + 1):
      for dcol in range(-radius, radius + 1):
        if drow == 0 and dcol == 0 and not self.cfg.include_current_tile:
          continue
        neighbor_row = row + drow
        neighbor_col = col + dcol
        if (
          0 <= neighbor_row < self.num_target_rows
          and 0 <= neighbor_col < self.num_target_cols
        ):
          tile_indices.append((neighbor_row, neighbor_col))

    if not tile_indices and not self.cfg.include_current_tile:
      tile_indices.append((row, col))
    return tile_indices

  def _update_command(self) -> None:
    self.target_reached_this_step[:] = False

    target_ids = (
      (self.is_target_env & self.has_target).nonzero(as_tuple=False).flatten()
    )
    if len(target_ids) > 0:
      dist = torch.norm(
        self.target_pos_w[target_ids, :2]
        - self.robot.data.root_link_pos_w[target_ids, :2],
        dim=-1,
      )
      resample_mask = dist < self.cfg.target_reached_threshold
      if self.cfg.target_max_distance is not None:
        resample_mask |= dist > self.cfg.target_max_distance
      reached_ids = target_ids[dist < self.cfg.target_reached_threshold]
      if len(reached_ids) > 0:
        self.target_reached_this_step[reached_ids] = True
      resample_ids = target_ids[resample_mask]
      if len(resample_ids) > 0:
        self._resample(resample_ids)

    target_ids = (
      (self.is_target_env & self.has_target).nonzero(as_tuple=False).flatten()
    )
    if len(target_ids) > 0:
      target_vec = (
        self.target_pos_w[target_ids, :2]
        - self.robot.data.root_link_pos_w[target_ids, :2]
      )
      self.heading_target[target_ids] = torch.atan2(target_vec[:, 1], target_vec[:, 0])

    super()._update_command()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    super()._debug_vis_impl(visualizer)

    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    target_pos_ws = self.target_pos_w.detach().cpu().numpy()
    has_target = self.has_target.detach().cpu().numpy()
    is_target_env = self.is_target_env.detach().cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.detach().cpu().numpy()

    for env_id in env_indices:
      if not (has_target[env_id] and is_target_env[env_id]):
        continue

      target_pos = target_pos_ws[env_id].copy()
      target_pos[2] += 0.15
      visualizer.add_sphere(
        center=target_pos,
        radius=self.cfg.target_marker_radius,
        color=(1.0, 0.0, 0.0, 1.0),
        label="active target",
      )

      start = base_pos_ws[env_id].copy()
      start[2] += self.cfg.viz.z_offset
      visualizer.add_arrow(
        start=start,
        end=target_pos,
        color=(1.0, 0.0, 0.0, 0.7),
        width=0.01,
        label="target direction",
      )


@dataclass(kw_only=True)
class TargetHeadingVelocityCommandCfg(UniformVelocityCommandCfg):
  """Config for target-point-guided heading velocity command."""

  patch_name: str = "target"
  rel_target_envs: float = 0.7
  rel_random_heading_envs: float = 0.2
  target_reached_threshold: float = 0.5
  target_min_distance: float = 1.0
  target_max_distance: float | None = None
  zero_lateral_velocity: bool = True
  target_marker_radius: float = 0.12
  target_tile_radius: int = 1
  include_current_tile: bool = False

  def build(self, env: ManagerBasedRlEnv) -> TargetHeadingVelocityCommand:
    return TargetHeadingVelocityCommand(self, env)
