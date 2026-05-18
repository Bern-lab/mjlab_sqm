from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity.mdp.terrain_utils import terrain_normal_from_sensors
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.velocity.mdp.target_heading_command import (
    TargetHeadingVelocityCommand,
  )
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_FOOT_BODY_CFG = SceneEntityCfg(
  "robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link")
)
_DEFAULT_FOOT_SITE_CFG = SceneEntityCfg(
  "robot", site_names=("left_foot", "right_foot")
)


def _make_foot_volume_points(
  device: str,
  x_range: tuple[float, float] = (-0.055, 0.132),
  y_range: tuple[float, float] = (-0.030, 0.030),
  z_range: tuple[float, float] = (-0.035, -0.015),
  grid_shape: tuple[int, int, int] = (8, 4, 2),
  heel_x_max: float = -0.020,
  front_sole_x_min: float = 0.070,
  toe_tip_x_min: float = 0.115,
  heel_weight: float = 0.5,
  front_sole_weight: float = 0.7,
  midfoot_weight: float = 1.0,
  toe_tip_weight: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
  xs = torch.linspace(x_range[0], x_range[1], grid_shape[0], device=device)
  ys = torch.linspace(y_range[0], y_range[1], grid_shape[1], device=device)
  zs = torch.linspace(z_range[0], z_range[1], grid_shape[2], device=device)
  xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
  points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

  x = points[:, 0]
  weights = torch.full_like(x, midfoot_weight)
  weights = torch.where(x < heel_x_max, heel_weight, weights)
  front_mask = (x >= front_sole_x_min) & (x < toe_tip_x_min)
  weights = torch.where(front_mask, front_sole_weight, weights)
  weights = torch.where(x >= toe_tip_x_min, toe_tip_weight, weights)
  return points, weights


def _current_step_boundaries(
  env: ManagerBasedRlEnv,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
  terrain = getattr(env.scene, "terrain", None)
  if terrain is None or not hasattr(terrain, "step_boundaries_by_tile"):
    return None, None
  if getattr(terrain, "terrain_levels", None) is None:
    return None, None

  boundaries_by_tile = terrain.step_boundaries_by_tile
  if boundaries_by_tile.shape[2] == 0:
    return None, None

  levels = terrain.terrain_levels
  terrain_types = terrain.terrain_types
  boundaries = boundaries_by_tile[levels, terrain_types]
  counts = terrain.step_boundary_counts[levels, terrain_types]
  boundary_ids = torch.arange(boundaries.shape[1], device=env.device)
  valid = boundary_ids.unsqueeze(0) < counts.unsqueeze(1)
  return boundaries, valid


def _terrain_level_active(
  env: ManagerBasedRlEnv, min_terrain_level: int | None
) -> torch.Tensor:
  if min_terrain_level is None:
    return torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

  terrain = getattr(env.scene, "terrain", None)
  levels = getattr(terrain, "terrain_levels", None)
  if levels is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  return levels >= min_terrain_level


class _StepBoundaryFootVolume:
  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    self._foot_ref_local = torch.tensor(
      params.get("foot_ref_local", (0.04, 0.0, -0.025)),
      device=env.device,
      dtype=torch.float32,
    )
    self._local_points, self._point_weights = _make_foot_volume_points(
      env.device,
      x_range=params.get("x_range", (-0.055, 0.132)),
      y_range=params.get("y_range", (-0.030, 0.030)),
      z_range=params.get("z_range", (-0.035, -0.015)),
      grid_shape=params.get("grid_shape", (8, 4, 2)),
      heel_x_max=params.get("heel_x_max", -0.020),
      front_sole_x_min=params.get("front_sole_x_min", 0.070),
      toe_tip_x_min=params.get("toe_tip_x_min", 0.115),
      heel_weight=params.get("heel_weight", 0.5),
      front_sole_weight=params.get("front_sole_weight", 0.7),
      midfoot_weight=params.get("midfoot_weight", 1.0),
      toe_tip_weight=params.get("toe_tip_weight", 0.3),
    )
    self._local_x = self._local_points[:, 0]
    self._max_point_ref_distance = torch.norm(
      self._local_points - self._foot_ref_local, dim=-1
    ).max()

  def _foot_points_w(
    self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
  ) -> tuple[torch.Tensor, torch.Tensor]:
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    foot_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    foot_lin_vel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]
    foot_ang_vel_w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]

    num_envs, num_feet = foot_pos_w.shape[:2]
    num_points = self._local_points.shape[0]
    local_points = self._local_points.view(1, 1, num_points, 3).expand(
      num_envs, num_feet, num_points, 3
    )
    foot_quat = foot_quat_w[:, :, None, :].expand(
      num_envs, num_feet, num_points, 4
    )
    point_offsets_w = quat_apply(foot_quat, local_points)
    points_w = foot_pos_w[:, :, None, :] + point_offsets_w
    point_vel_w = foot_lin_vel_w[:, :, None, :] + torch.cross(
      foot_ang_vel_w[:, :, None, :].expand_as(point_offsets_w),
      point_offsets_w,
      dim=-1,
    )
    return points_w, point_vel_w

  def _foot_ref_w(
    self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    foot_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    num_envs, num_feet = foot_pos_w.shape[:2]
    foot_ref_local = self._foot_ref_local.view(1, 1, 3).expand(
      num_envs, num_feet, 3
    )
    return foot_pos_w + quat_apply(foot_quat_w, foot_ref_local)

  def _flat_point_weights(self, num_envs: int, num_feet: int) -> torch.Tensor:
    return self._point_weights.view(1, 1, -1).expand(num_envs, num_feet, -1).reshape(
      num_envs, num_feet * self._point_weights.shape[0]
    )

  @staticmethod
  def _point_to_segment_distance(
    points: torch.Tensor,
    p0: torch.Tensor,
    p1: torch.Tensor,
  ) -> torch.Tensor:
    segment = p1 - p0
    segment_len_sq = torch.sum(torch.square(segment), dim=-1).clamp_min(1e-12)
    point_delta = points[:, :, :, None, :] - p0[:, :, None, :, :]
    t = torch.sum(point_delta * segment[:, :, None, :, :], dim=-1)
    t = t / segment_len_sq[:, :, None, :]
    t = torch.clamp(t, 0.0, 1.0)
    closest = p0[:, :, None, :, :] + t[..., None] * segment[:, :, None, :, :]
    return torch.norm(points[:, :, :, None, :] - closest, dim=-1)

  @staticmethod
  def _ref_to_segment_distance(
    refs: torch.Tensor,
    p0: torch.Tensor,
    p1: torch.Tensor,
  ) -> torch.Tensor:
    segment = p1 - p0
    segment_len_sq = torch.sum(torch.square(segment), dim=-1).clamp_min(1e-12)
    ref_delta = refs[:, :, None, :] - p0[:, None, :, :]
    t = torch.sum(ref_delta * segment[:, None, :, :], dim=-1)
    t = t / segment_len_sq[:, None, :]
    t = torch.clamp(t, 0.0, 1.0)
    closest = p0[:, None, :, :] + t[..., None] * segment[:, None, :, :]
    return torch.norm(refs[:, :, None, :] - closest, dim=-1)

  @staticmethod
  def _gather_by_foot(
    values: torch.Tensor,
    indices: torch.Tensor,
  ) -> torch.Tensor:
    num_envs, num_feet, num_selected = indices.shape
    value_dim = values.shape[-1]
    expanded = values[:, None, :, :].expand(num_envs, num_feet, -1, value_dim)
    gather_idx = indices[..., None].expand(num_envs, num_feet, num_selected, value_dim)
    return torch.gather(expanded, dim=2, index=gather_idx)

  @staticmethod
  def _gather_mask_by_foot(
    values: torch.Tensor,
    indices: torch.Tensor,
  ) -> torch.Tensor:
    num_envs, num_feet, num_selected = indices.shape
    expanded = values[:, None, :].expand(num_envs, num_feet, -1)
    return torch.gather(expanded, dim=2, index=indices)

  def _nearest_boundary_indices(
    self,
    ref_distance: torch.Tensor,
    valid_boundaries: torch.Tensor,
    influence_radius: torch.Tensor | float,
    nearest_boundaries: int | None,
  ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    num_boundaries = ref_distance.shape[-1]
    if (
      nearest_boundaries is None
      or nearest_boundaries <= 0
      or nearest_boundaries >= num_boundaries
    ):
      return None, None

    candidate_count = torch.sum(
      (ref_distance <= influence_radius) & valid_boundaries[:, None, :], dim=-1
    )
    fallback = candidate_count > nearest_boundaries
    masked_distance = torch.where(
      valid_boundaries[:, None, :],
      ref_distance,
      torch.full_like(ref_distance, torch.inf),
    )
    indices = torch.topk(
      masked_distance, k=nearest_boundaries, dim=-1, largest=False
    ).indices
    return indices, fallback

  def _lip_min_dist(
    self,
    points: torch.Tensor,
    boundaries: torch.Tensor,
    valid_boundaries: torch.Tensor,
    edge_height_band: float | None,
  ) -> torch.Tensor:
    p0 = boundaries[..., 0:3]
    p1 = boundaries[..., 3:6]
    z_high = boundaries[..., 10]

    distances = self._point_to_segment_distance(points, p0, p1)
    valid = valid_boundaries[:, :, None, :]
    if edge_height_band is not None and edge_height_band > 0.0:
      height_ok = points[:, :, :, None, 2] >= z_high[:, :, None, :] - edge_height_band
      valid = valid & height_ok
    distances = torch.where(valid, distances, torch.full_like(distances, torch.inf))
    return torch.min(distances, dim=-1).values

  def _riser_slab_ref_distance(
    self,
    refs: torch.Tensor,
    boundaries: torch.Tensor,
    slab_depth: float,
    u_margin: float,
    v_margin: float,
    surface_tol: float,
  ) -> torch.Tensor:
    p0 = boundaries[:, :, 0:3]
    p1 = boundaries[:, :, 3:6]
    normal_to_low = boundaries[:, :, 6:9]
    z_low = boundaries[:, :, 9]
    z_high = boundaries[:, :, 10]

    tangent_u = p1 - p0
    edge_len = torch.norm(tangent_u, dim=-1).clamp_min(1e-12)
    tangent_u = tangent_u / edge_len[..., None]
    center = 0.5 * (p0 + p1)
    center = center.clone()
    center[:, :, 2] = 0.5 * (z_low + z_high)
    half_u = 0.5 * edge_len
    half_v = 0.5 * (z_high - z_low)

    rel = refs[:, :, None, :] - center[:, None, :, :]
    s = torch.sum(rel * normal_to_low[:, None, :, :], dim=-1)
    u = torch.sum(rel * tangent_u[:, None, :, :], dim=-1)
    v = rel[..., 2]

    du = torch.relu(torch.abs(u) - (half_u[:, None, :] + u_margin))
    ds_low = torch.relu(-surface_tol - s)
    ds_high = torch.relu(s - slab_depth)
    ds = torch.maximum(ds_low, ds_high)
    dv = torch.relu(torch.abs(v) - (half_v[:, None, :] + v_margin))
    return torch.sqrt(torch.square(du) + torch.square(ds) + torch.square(dv))

  def _riser_slab_point_penalty(
    self,
    toe_points: torch.Tensor,
    toe_vel: torch.Tensor,
    boundaries: torch.Tensor,
    valid_boundaries: torch.Tensor,
    slab_depth: float,
    u_margin: float,
    v_margin: float,
    toe_v_threshold: float,
    surface_tol: float,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p0 = boundaries[..., 0:3]
    p1 = boundaries[..., 3:6]
    normal_to_low = boundaries[..., 6:9]
    z_low = boundaries[..., 9]
    z_high = boundaries[..., 10]

    tangent_u = p1 - p0
    edge_len = torch.norm(tangent_u, dim=-1).clamp_min(1e-12)
    tangent_u = tangent_u / edge_len[..., None]
    center = 0.5 * (p0 + p1)
    center = center.clone()
    center[..., 2] = 0.5 * (z_low + z_high)
    half_u = 0.5 * edge_len
    half_v = 0.5 * (z_high - z_low)

    rel = toe_points[:, :, :, None, :] - center[:, :, None, :, :]
    s = torch.sum(rel * normal_to_low[:, :, None, :, :], dim=-1)
    u = torch.sum(rel * tangent_u[:, :, None, :, :], dim=-1)
    v = rel[..., 2]
    inside_face = (torch.abs(u) <= half_u[:, :, None, :] + u_margin) & (
      torch.abs(v) <= half_v[:, :, None, :] + v_margin
    )
    inside_slab = (s >= -surface_tol) & (s <= slab_depth)
    toe_approach_speed = torch.relu(
      -torch.sum(toe_vel[:, :, :, None, :] * normal_to_low[:, :, None, :, :], dim=-1)
      - toe_v_threshold
    )
    penetration = torch.relu(slab_depth - s)
    valid = valid_boundaries[:, :, None, :] & inside_face & inside_slab
    per_face_penalty = torch.where(
      valid, penetration * toe_approach_speed, torch.zeros_like(penetration)
    )
    point_penalty = torch.max(per_face_penalty, dim=-1).values
    impact_speed_per_point = torch.max(
      torch.where(valid, toe_approach_speed, torch.zeros_like(toe_approach_speed)),
      dim=-1,
    ).values
    return point_penalty, point_penalty > 0.0, impact_speed_per_point


class foot_step_lip_volume_penalty(_StepBoundaryFootVolume):
  """Hiking-style foot-volume penalty around high-side step lips."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    edge_radius: float = 0.05,
    edge_height_band: float | None = 0.06,
    nearest_boundaries: int | None = None,
    log_only: bool = False,
    min_terrain_level: int | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_BODY_CFG,
    **_: object,
  ) -> torch.Tensor:
    boundaries, valid_boundaries = _current_step_boundaries(env)
    if boundaries is None or valid_boundaries is None:
      return torch.zeros(env.num_envs, device=env.device)

    points_w, point_vel_w = self._foot_points_w(env, asset_cfg)
    num_envs, num_feet, num_points = points_w.shape[:3]

    level_active = _terrain_level_active(env, min_terrain_level)
    base_valid = valid_boundaries & level_active[:, None]

    p0 = boundaries[:, :, 0:3]
    p1 = boundaries[:, :, 3:6]
    foot_ref_w = self._foot_ref_w(env, asset_cfg)
    ref_dist = self._ref_to_segment_distance(foot_ref_w, p0, p1)
    influence_radius = edge_radius + self._max_point_ref_distance
    selected_idx, fallback = self._nearest_boundary_indices(
      ref_dist, base_valid, influence_radius, nearest_boundaries
    )

    if selected_idx is None:
      expanded_boundaries = boundaries[:, None, :, :].expand(
        num_envs, num_feet, -1, -1
      )
      expanded_valid = base_valid[:, None, :].expand(num_envs, num_feet, -1)
      min_dist = self._lip_min_dist(
        points_w, expanded_boundaries, expanded_valid, edge_height_band
      )
      fallback_ratio = torch.zeros((), device=env.device)
    else:
      selected_boundaries = self._gather_by_foot(boundaries, selected_idx)
      selected_valid = self._gather_mask_by_foot(base_valid, selected_idx)
      min_dist = self._lip_min_dist(
        points_w, selected_boundaries, selected_valid, edge_height_band
      )
      fallback_ratio = fallback.float().mean()
      if bool(torch.any(fallback).item()):
        expanded_boundaries = boundaries[:, None, :, :].expand(
          num_envs, num_feet, -1, -1
        )
        expanded_valid = base_valid[:, None, :].expand(num_envs, num_feet, -1)
        full_min_dist = self._lip_min_dist(
          points_w, expanded_boundaries, expanded_valid, edge_height_band
        )
        min_dist = torch.where(fallback[:, :, None], full_min_dist, min_dist)

    penetration = torch.relu(edge_radius - min_dist)
    point_speed = torch.norm(point_vel_w, dim=-1)
    weights = self._point_weights.view(1, 1, num_points)
    penalty = torch.sum(weights * penetration * (point_speed + 1e-6), dim=(1, 2))

    finite = torch.isfinite(min_dist)
    finite_count = finite.float().sum().clamp_min(1.0)
    min_dist_mean = (
      torch.where(finite, min_dist, torch.zeros_like(min_dist)).sum() / finite_count
    )
    env.extras["log"]["Metrics/step_lip_penalty_mean"] = penalty.mean()
    env.extras["log"]["Metrics/step_lip_penetration_ratio"] = (
      penetration > 0.0
    ).float().mean()
    env.extras["log"]["Metrics/step_lip_min_dist_mean"] = min_dist_mean
    env.extras["log"]["Metrics/step_lip_nearest_fallback_ratio"] = fallback_ratio

    if log_only:
      return torch.zeros_like(penalty)
    return penalty


class toe_step_riser_slab_penalty(_StepBoundaryFootVolume):
  """Penalize toe points entering the low-side danger slab of a riser."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    slab_depth: float = 0.04,
    u_margin: float = 0.02,
    v_margin: float = 0.02,
    toe_x_min: float = 0.09,
    toe_v_threshold: float = 0.05,
    surface_tol: float = 0.005,
    nearest_boundaries: int | None = None,
    log_only: bool = False,
    min_terrain_level: int | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_BODY_CFG,
    **_: object,
  ) -> torch.Tensor:
    boundaries, valid_boundaries = _current_step_boundaries(env)
    if boundaries is None or valid_boundaries is None:
      return torch.zeros(env.num_envs, device=env.device)

    toe_mask = self._local_x >= toe_x_min
    if toe_mask.sum().item() == 0:
      return torch.zeros(env.num_envs, device=env.device)

    points_w, point_vel_w = self._foot_points_w(env, asset_cfg)
    toe_points = points_w[:, :, toe_mask, :]
    toe_vel = point_vel_w[:, :, toe_mask, :]
    num_envs, num_feet = toe_points.shape[:2]

    level_active = _terrain_level_active(env, min_terrain_level)
    base_valid = valid_boundaries & level_active[:, None]

    foot_ref_w = self._foot_ref_w(env, asset_cfg)
    ref_dist = self._riser_slab_ref_distance(
      foot_ref_w,
      boundaries,
      slab_depth,
      u_margin,
      v_margin,
      surface_tol,
    )
    toe_ref_radius = torch.norm(
      self._local_points[toe_mask] - self._foot_ref_local, dim=-1
    ).max()
    selected_idx, fallback = self._nearest_boundary_indices(
      ref_dist, base_valid, toe_ref_radius, nearest_boundaries
    )

    if selected_idx is None:
      expanded_boundaries = boundaries[:, None, :, :].expand(
        num_envs, num_feet, -1, -1
      )
      expanded_valid = base_valid[:, None, :].expand(num_envs, num_feet, -1)
      point_penalty, active, impact_speed_per_point = self._riser_slab_point_penalty(
        toe_points,
        toe_vel,
        expanded_boundaries,
        expanded_valid,
        slab_depth,
        u_margin,
        v_margin,
        toe_v_threshold,
        surface_tol,
      )
      fallback_ratio = torch.zeros((), device=env.device)
    else:
      selected_boundaries = self._gather_by_foot(boundaries, selected_idx)
      selected_valid = self._gather_mask_by_foot(base_valid, selected_idx)
      point_penalty, active, impact_speed_per_point = self._riser_slab_point_penalty(
        toe_points,
        toe_vel,
        selected_boundaries,
        selected_valid,
        slab_depth,
        u_margin,
        v_margin,
        toe_v_threshold,
        surface_tol,
      )
      fallback_ratio = fallback.float().mean()
      if bool(torch.any(fallback).item()):
        expanded_boundaries = boundaries[:, None, :, :].expand(
          num_envs, num_feet, -1, -1
        )
        expanded_valid = base_valid[:, None, :].expand(num_envs, num_feet, -1)
        full_penalty, full_active, full_impact = self._riser_slab_point_penalty(
          toe_points,
          toe_vel,
          expanded_boundaries,
          expanded_valid,
          slab_depth,
          u_margin,
          v_margin,
          toe_v_threshold,
          surface_tol,
        )
        fallback_mask = fallback[:, :, None]
        point_penalty = torch.where(fallback_mask, full_penalty, point_penalty)
        active = torch.where(fallback_mask, full_active, active)
        impact_speed_per_point = torch.where(
          fallback_mask, full_impact, impact_speed_per_point
        )

    penalty = torch.sum(point_penalty, dim=(1, 2))

    active_count = active.float().sum().clamp_min(1.0)
    impact_speed_mean = torch.sum(impact_speed_per_point * active.float())
    impact_speed_mean = impact_speed_mean / active_count
    env.extras["log"]["Metrics/toe_riser_slab_penalty_mean"] = penalty.mean()
    env.extras["log"]["Metrics/toe_riser_slab_active_ratio"] = active.float().mean()
    env.extras["log"]["Metrics/toe_riser_slab_impact_speed_mean"] = impact_speed_mean
    env.extras["log"]["Metrics/toe_riser_slab_nearest_fallback_ratio"] = (
      fallback_ratio
    )

    if log_only:
      return torch.zeros_like(penalty)
    return penalty


class toe_step_riser_approach_penalty(toe_step_riser_slab_penalty):
  """Backward-compatible alias for the toe riser slab penalty."""


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + xy_error
  return torch.exp(-ang_vel_error / std**2)


class upright:
  """Reward for keeping the base upright.

  Without ``terrain_sensor_names``, penalizes tilt relative to world up (correct for
  flat ground).

  With ``terrain_sensor_names``, penalizes tilt relative to the terrain surface normal.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._terrain_sensor_names: tuple[str, ...] | None = cfg.params.get(
      "terrain_sensor_names"
    )
    self._debug_vis_enabled = True
    self._env = env
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    terrain_sensor_names: tuple[str, ...] | None = None,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    if asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
      body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    else:
      body_quat_w = asset.data.root_link_quat_w  # [B, 4]

    if terrain_sensor_names is not None:
      terrain_normal = terrain_normal_from_sensors(env, terrain_sensor_names)  # [B, 3]
      # Project terrain normal into body frame. When aligned with the terrain surface
      # this should be (0, 0, 1); XY measures tilt.
      target_b = quat_apply_inverse(body_quat_w, terrain_normal)  # [B, 3]
      xy_squared = torch.sum(torch.square(target_b[:, :2]), dim=1)
    else:
      gravity_w = asset.data.gravity_vec_w  # [3]
      projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
      xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

    return torch.exp(-xy_squared / std**2)

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids  # Unused.

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._debug_vis_enabled or self._terrain_sensor_names is None:
      return

    env = self._env
    asset: Entity = env.scene[self._asset_cfg.name]

    env_indices = list(visualizer.get_env_indices(env.num_envs))
    if not env_indices:
      return

    terrain_normal = terrain_normal_from_sensors(env, self._terrain_sensor_names)
    if self._asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, self._asset_cfg.body_ids, :].squeeze(
        1
      )
    else:
      body_quat_w = asset.data.root_link_quat_w
    up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(
      body_quat_w[:, :3]
    )
    body_up_w = quat_apply(body_quat_w, up_local)

    positions = asset.data.root_link_pos_w.cpu().numpy()
    offset = np.array([0.0, 0.3, 0.0])
    terrain_normal_np = terrain_normal.cpu().numpy()
    body_up_np = body_up_w.cpu().numpy()
    scale = 0.25

    for i in env_indices:
      origin = positions[i] + offset
      # Terrain normal (magenta).
      visualizer.add_arrow(
        start=origin,
        end=origin + terrain_normal_np[i] * scale,
        color=(0.8, 0.2, 0.8, 0.8),
        width=0.01,
      )
      # Body up (orange).
      visualizer.add_arrow(
        start=origin,
        end=origin + body_up_np[i] * scale,
        color=(1.0, 0.5, 0.0, 0.8),
        width=0.01,
      )


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.sum(dim=-1).float()


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  height_sensor_name: str,
  target_height: float | None = None,
  min_height: float | None = None,
  max_height: float | None = None,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot clearance outside a target height or height range."""
  asset: Entity = env.scene[asset_cfg.name]
  height_sensor = env.scene[height_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor), (
    f"feet_clearance requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
  )
  foot_height = height_sensor.data.heights  # [B, F]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, F]
  if min_height is not None or max_height is not None:
    if min_height is None or max_height is None:
      raise ValueError("feet_clearance requires both min_height and max_height.")
    if min_height > max_height:
      raise ValueError("feet_clearance min_height must be <= max_height.")
    delta = torch.relu(min_height - foot_height) + torch.relu(foot_height - max_height)
  else:
    if target_height is None:
      raise ValueError("feet_clearance requires target_height or min/max height.")
    delta = torch.abs(foot_height - target_height)  # [B, F]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    height_sensor = env.scene[cfg.params["height_sensor_name"]]
    assert isinstance(height_sensor, TerrainHeightSensor), (
      f"feet_swing_height requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
    )
    num_feet = height_sensor.num_frames
    self.peak_heights = torch.zeros(
      (env.num_envs, num_feet), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    foot_heights = height_sensor.data.heights
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))

def idle_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.2,
  velocity_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize standing nearly still when a clear velocity command is given."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None

  commanded_speed = torch.norm(command[:, :2], dim=1)
  actual_speed = torch.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)

  penalty = (
    (commanded_speed > command_threshold) &
    (actual_speed < velocity_threshold)
  ).float()

  env.extras["log"]["Metrics/idle_penalty_ratio"] = torch.mean(penalty)
  return penalty

def feet_gait(
        env: ManagerBasedRlEnv,
        period: float,
        offset: list[float],
        threshold: float,
        command_threshold: float,
        command_name: str,
        sensor_name: str,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    current_contact_time = sensor.data.current_contact_time
    assert current_contact_time is not None, "Enable track_air_time=True for this contact sensor."
    is_contact = current_contact_time > 0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = (leg_phase < threshold)
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward

def target_progress(
  env: ManagerBasedRlEnv,
  command_name: str,
  min_distance: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward world-frame velocity projected toward the active target point."""
  command_term = env.command_manager.get_term(command_name)
  required_attrs = ("target_pos_w", "has_target", "is_target_env")
  if not all(hasattr(command_term, name) for name in required_attrs):
    return torch.zeros(env.num_envs, device=env.device)
  command_term = cast("TargetHeadingVelocityCommand", command_term)

  asset: Entity = env.scene[asset_cfg.name]
  target_vec_xy = command_term.target_pos_w[:, :2] - asset.data.root_link_pos_w[:, :2]
  target_dist = torch.norm(target_vec_xy, dim=-1)
  target_dir_xy = target_vec_xy / torch.clamp(
    target_dist.unsqueeze(-1), min=min_distance
  )
  lin_vel_w_xy = asset.data.root_link_lin_vel_w[:, :2]
  progress_speed = torch.sum(lin_vel_w_xy * target_dir_xy, dim=-1)
  active = command_term.has_target & command_term.is_target_env

  return torch.clamp(progress_speed, min=0.0) * active.float()


def target_reached_bonus(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Sparse bonus emitted after the command detects target arrival."""
  command_term = env.command_manager.get_term(command_name)
  if not hasattr(command_term, "target_reached_this_step"):
    return torch.zeros(env.num_envs, device=env.device)
  command_term = cast("TargetHeadingVelocityCommand", command_term)
  return command_term.target_reached_this_step.float()

def base_height_above_support_value(
  env: ManagerBasedRlEnv,
  height_sensor_name: str,
  contact_sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_SITE_CFG,
) -> torch.Tensor:
  """Base height relative to the terrain under the support foot/feet."""
  asset: Entity = env.scene[asset_cfg.name]
  height_sensor = env.scene[height_sensor_name]
  contact_sensor = env.scene[contact_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor), (
    "base_height_above_support_value requires a TerrainHeightSensor, got "
    f"{type(height_sensor).__name__}"
  )
  assert isinstance(contact_sensor, ContactSensor), (
    "base_height_above_support_value requires a ContactSensor, got "
    f"{type(contact_sensor).__name__}"
  )
  assert contact_sensor.data.found is not None

  base_z = asset.data.root_link_pos_w[:, 2]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  terrain_z_under_feet = foot_z - height_sensor.data.heights

  contact = (contact_sensor.data.found > 0).float()
  contact_sum = contact.sum(dim=1).clamp_min(1.0)
  support_terrain_z = (terrain_z_under_feet * contact).sum(dim=1) / contact_sum
  fallback_terrain_z = terrain_z_under_feet.max(dim=1).values
  has_contact = contact.sum(dim=1) > 0
  terrain_z = torch.where(has_contact, support_terrain_z, fallback_terrain_z)

  return base_z - terrain_z


def base_height_above_support(
    env,
    height_sensor_name: str,
    contact_sensor_name: str,
    min_height: float = 0.74,
    error_scale: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_SITE_CFG,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    height_sensor = env.scene[height_sensor_name]
    contact_sensor = env.scene[contact_sensor_name]

    base_z = asset.data.root_link_pos_w[:, 2]

    foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    foot_height_above_terrain = height_sensor.data.heights
    terrain_z_under_feet = foot_z - foot_height_above_terrain

    contact = (contact_sensor.data.found > 0).float()
    contact_sum = contact.sum(dim=1).clamp_min(1.0)

    support_terrain_z = (terrain_z_under_feet * contact).sum(dim=1) / contact_sum
    fallback_terrain_z = terrain_z_under_feet.max(dim=1).values
    has_contact = contact.sum(dim=1) > 0

    terrain_z = torch.where(has_contact, support_terrain_z, fallback_terrain_z)
    base_height_rel = base_z - terrain_z

    height_error = torch.relu(min_height - base_height_rel) * error_scale
    return torch.square(height_error)
