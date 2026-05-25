from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_FOOT_BODY_CFG = SceneEntityCfg(
  "robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link")
)
_DEFAULT_SHANK_BODY_CFG = SceneEntityCfg(
  "robot", body_names=("left_knee_link", "right_knee_link")
)


def _make_foot_volume_points(#体积点建模
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


def _make_shank_volume_points(
  device: str,
  x_range: tuple[float, float] = (0.045, 0.045),
  y_range: tuple[float, float] = (-0.035, 0.035),
  z_range: tuple[float, float] = (-0.23, -0.10),
  grid_shape: tuple[int, int, int] = (1, 3, 5),
) -> torch.Tensor:
  xs = torch.linspace(x_range[0], x_range[1], grid_shape[0], device=device)
  ys = torch.linspace(y_range[0], y_range[1], grid_shape[1], device=device)
  zs = torch.linspace(z_range[0], z_range[1], grid_shape[2], device=device)
  xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
  return torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)


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
    self._env = env
    self._debug_vis_available = bool(params.get("debug_vis_foot_points", False))
    self._debug_vis_enabled = self._debug_vis_available
    self._debug_vis_asset_cfg = params.get("asset_cfg", _DEFAULT_FOOT_BODY_CFG)
    self._debug_vis_foot_point_radius = float(
      params.get("debug_vis_foot_point_radius", 0.008)
    )
    foot_point_color = params.get("debug_vis_foot_point_color", (0.0, 1.0, 0.15, 0.9))
    self._debug_vis_foot_point_color = cast(
      tuple[float, float, float, float],
      tuple(float(c) for c in foot_point_color),
    )
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

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._debug_vis_available or not self._debug_vis_enabled:
      return

    env = self._env
    env_indices = list(visualizer.get_env_indices(env.num_envs))
    if not env_indices:
      return

    points_w, _ = self._foot_points_w(env, self._debug_vis_asset_cfg)
    points_np = points_w.detach().cpu().numpy()
    for env_idx in env_indices:
      for point in points_np[env_idx].reshape(-1, 3):
        visualizer.add_sphere(
          center=point,
          radius=self._debug_vis_foot_point_radius,
          color=self._debug_vis_foot_point_color,
        )

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
    num_envs, num_feet, _ = indices.shape
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
    approach_speed_floor: float,
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
    toe_speed_to_riser = -torch.sum(
      toe_vel[:, :, :, None, :] * normal_to_low[:, :, None, :, :], dim=-1
    )
    toe_approach_speed = torch.relu(toe_speed_to_riser - toe_v_threshold)
    approach_gate = toe_speed_to_riser > 0.0
    speed_factor = toe_approach_speed + approach_gate.float() * approach_speed_floor
    penetration = torch.relu(slab_depth - s)
    valid = valid_boundaries[:, :, None, :] & inside_face & inside_slab
    per_face_penalty = torch.where(
      valid, penetration * speed_factor, torch.zeros_like(penetration)
    )
    point_penalty = torch.max(per_face_penalty, dim=-1).values
    impact_speed_per_point = torch.max(
      torch.where(valid, speed_factor, torch.zeros_like(speed_factor)),
      dim=-1,
    ).values
    return point_penalty, point_penalty > 0.0, impact_speed_per_point

  def _riser_slab_static_penalty(
    self,
    points: torch.Tensor,
    boundaries: torch.Tensor,
    valid_boundaries: torch.Tensor,
    slab_depth: float,
    u_margin: float,
    v_margin: float,
    surface_tol: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
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

    rel = points[:, :, :, None, :] - center[:, :, None, :, :]
    s = torch.sum(rel * normal_to_low[:, :, None, :, :], dim=-1)
    u = torch.sum(rel * tangent_u[:, :, None, :, :], dim=-1)
    v = rel[..., 2]
    inside_face = (torch.abs(u) <= half_u[:, :, None, :] + u_margin) & (
      torch.abs(v) <= half_v[:, :, None, :] + v_margin
    )
    inside_slab = (s >= -surface_tol) & (s <= slab_depth)
    penetration = torch.relu(slab_depth - s)
    valid = valid_boundaries[:, :, None, :] & inside_face & inside_slab
    per_face_penalty = torch.where(
      valid, penetration, torch.zeros_like(penetration)
    )
    point_penalty = torch.max(per_face_penalty, dim=-1).values
    return point_penalty, point_penalty > 0.0

  @staticmethod
  def _foot_contact_gate(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    num_feet: int,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = contact_sensor.data
    if sensor_data.found is not None:
      gate = sensor_data.found > 0
    else:
      assert sensor_data.force is not None
      gate = torch.norm(sensor_data.force, dim=-1) > 1e-6

    if gate.shape[1] == num_feet:
      return gate.float()
    if gate.shape[1] == 1:
      return gate.expand(-1, num_feet).float()
    return gate[:, :num_feet].float()


class foot_step_lip_volume_penalty(_StepBoundaryFootVolume):
  """Hiking-style foot-volume penalty around high-side step lips."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    edge_radius: float = 0.05,
    edge_height_band: float | None = 0.06,
    support_speed_floor: float = 0.0,
    nearest_boundaries: int | None = None,
    contact_sensor_name: str = "feet_ground_contact",
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
      assert fallback is not None
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
    if support_speed_floor > 0.0:
      contact_gate = self._foot_contact_gate(env, contact_sensor_name, num_feet)
      point_speed = point_speed + contact_gate[:, :, None] * support_speed_floor
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
    approach_speed_floor: float = 0.0,
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
        approach_speed_floor,
      )
      fallback_ratio = torch.zeros((), device=env.device)
    else:
      assert fallback is not None
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
        approach_speed_floor,
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
          approach_speed_floor,
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


class heel_step_riser_clearance_penalty(_StepBoundaryFootVolume):
  """Penalize contacted heel points that sit too close to a low-side riser."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    heel_clearance: float = 0.04,
    u_margin: float = 0.02,
    v_margin: float = 0.02,
    heel_x_max: float = -0.02,
    surface_tol: float = 0.005,
    nearest_boundaries: int | None = None,
    contact_sensor_name: str = "feet_ground_contact",
    log_only: bool = False,
    min_terrain_level: int | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_BODY_CFG,
    **_: object,
  ) -> torch.Tensor:
    boundaries, valid_boundaries = _current_step_boundaries(env)
    if boundaries is None or valid_boundaries is None:
      return torch.zeros(env.num_envs, device=env.device)

    heel_mask = self._local_x <= heel_x_max
    num_heel_points = int(heel_mask.sum().item())
    if num_heel_points == 0:
      return torch.zeros(env.num_envs, device=env.device)

    points_w, _ = self._foot_points_w(env, asset_cfg)
    heel_points = points_w[:, :, heel_mask, :]
    num_envs, num_feet = heel_points.shape[:2]

    level_active = _terrain_level_active(env, min_terrain_level)
    base_valid = valid_boundaries & level_active[:, None]

    foot_ref_w = self._foot_ref_w(env, asset_cfg)
    ref_dist = self._riser_slab_ref_distance(
      foot_ref_w,
      boundaries,
      heel_clearance,
      u_margin,
      v_margin,
      surface_tol,
    )
    heel_ref_radius = torch.norm(
      self._local_points[heel_mask] - self._foot_ref_local, dim=-1
    ).max()
    selected_idx, fallback = self._nearest_boundary_indices(
      ref_dist, base_valid, heel_ref_radius, nearest_boundaries
    )

    if selected_idx is None:
      expanded_boundaries = boundaries[:, None, :, :].expand(
        num_envs, num_feet, -1, -1
      )
      expanded_valid = base_valid[:, None, :].expand(num_envs, num_feet, -1)
      point_penalty, active = self._riser_slab_static_penalty(
        heel_points,
        expanded_boundaries,
        expanded_valid,
        heel_clearance,
        u_margin,
        v_margin,
        surface_tol,
      )
      fallback_ratio = torch.zeros((), device=env.device)
    else:
      assert fallback is not None
      selected_boundaries = self._gather_by_foot(boundaries, selected_idx)
      selected_valid = self._gather_mask_by_foot(base_valid, selected_idx)
      point_penalty, active = self._riser_slab_static_penalty(
        heel_points,
        selected_boundaries,
        selected_valid,
        heel_clearance,
        u_margin,
        v_margin,
        surface_tol,
      )
      fallback_ratio = fallback.float().mean()
      if bool(torch.any(fallback).item()):
        expanded_boundaries = boundaries[:, None, :, :].expand(
          num_envs, num_feet, -1, -1
        )
        expanded_valid = base_valid[:, None, :].expand(num_envs, num_feet, -1)
        full_penalty, full_active = self._riser_slab_static_penalty(
          heel_points,
          expanded_boundaries,
          expanded_valid,
          heel_clearance,
          u_margin,
          v_margin,
          surface_tol,
        )
        fallback_mask = fallback[:, :, None]
        point_penalty = torch.where(fallback_mask, full_penalty, point_penalty)
        active = torch.where(fallback_mask, full_active, active)

    contact_gate = self._foot_contact_gate(env, contact_sensor_name, num_feet)
    foot_penalty = torch.max(point_penalty, dim=2).values
    penalty = torch.sum(foot_penalty * contact_gate, dim=1)

    gated_active = active & (contact_gate[:, :, None] > 0.0)
    active_ratio = gated_active.float().mean()
    contact_ratio = contact_gate.float().mean()
    env.extras["log"]["Metrics/heel_riser_clearance_penalty_mean"] = penalty.mean()
    env.extras["log"]["Metrics/heel_riser_clearance_active_ratio"] = active_ratio
    env.extras["log"]["Metrics/heel_riser_clearance_contact_ratio"] = contact_ratio
    env.extras["log"]["Metrics/heel_riser_clearance_nearest_fallback_ratio"] = (
      fallback_ratio
    )

    if log_only:
      return torch.zeros_like(penalty)
    return penalty


class foot_landing_flatness_penalty(_StepBoundaryFootVolume):
  """Penalize non-level feet while landing on stair treads."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    near_height: float = 0.10,
    max_tilt_deg: float = 18.0,
    max_upward_speed: float = 0.10,
    height_sensor_name: str = "foot_height_scan",
    contact_sensor_name: str = "feet_ground_contact",
    log_only: bool = False,
    min_terrain_level: int | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_FOOT_BODY_CFG,
    **_: object,
  ) -> torch.Tensor:
    boundaries, valid_boundaries = _current_step_boundaries(env)
    if boundaries is None or valid_boundaries is None:
      return torch.zeros(env.num_envs, device=env.device)

    asset: Entity = env.scene[asset_cfg.name]
    foot_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    foot_lin_vel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]
    num_envs, num_feet = foot_quat_w.shape[:2]

    height_sensor = env.scene[height_sensor_name]
    assert isinstance(height_sensor, TerrainHeightSensor), (
      f"foot_landing_flatness_penalty requires a TerrainHeightSensor, got "
      f"{type(height_sensor).__name__}"
    )
    foot_clearance = height_sensor.data.heights
    if foot_clearance.ndim == 3:
      foot_clearance = foot_clearance.min(dim=-1).values
    foot_clearance = foot_clearance[:, :num_feet]

    level_active = _terrain_level_active(env, min_terrain_level)
    stair_gate = torch.any(valid_boundaries & level_active[:, None], dim=1).float()

    contact_gate = self._foot_contact_gate(env, contact_sensor_name, num_feet)
    near_height_safe = max(near_height, 1e-6)
    near_ground_gate = torch.clamp(
      (near_height - foot_clearance) / near_height_safe, min=0.0, max=1.0
    )
    descending_or_slow_gate = (foot_lin_vel_w[..., 2] <= max_upward_speed).float()
    active_gate = torch.maximum(
      contact_gate, near_ground_gate * descending_or_slow_gate
    )
    active_gate = active_gate * stair_gate[:, None]

    local_up = torch.tensor((0.0, 0.0, 1.0), device=env.device, dtype=torch.float32)
    local_up = local_up.view(1, 1, 3).expand(num_envs, num_feet, 3)
    foot_up_w = quat_apply(foot_quat_w, local_up)
    tilt_sin = torch.norm(foot_up_w[..., :2], dim=-1)
    tilt_threshold = math.sin(math.radians(max_tilt_deg))
    tilt_excess = torch.relu(tilt_sin - tilt_threshold)
    tilt_scale = max(1.0 - tilt_threshold, 1e-6)
    tilt_penalty = torch.square(tilt_excess / tilt_scale)
    penalty = torch.sum(tilt_penalty * active_gate, dim=1)

    env.extras["log"]["Metrics/foot_landing_flatness_penalty_mean"] = (
      penalty.mean()
    )
    env.extras["log"]["Metrics/foot_landing_flatness_active_ratio"] = (
      active_gate > 0.0
    ).float().mean()
    env.extras["log"]["Metrics/foot_landing_flatness_near_ground_ratio"] = (
      near_ground_gate > 0.0
    ).float().mean()
    env.extras["log"]["Metrics/foot_landing_flatness_contact_ratio"] = (
      contact_gate > 0.0
    ).float().mean()
    env.extras["log"]["Metrics/foot_landing_flatness_stair_gate_ratio"] = (
      stair_gate.mean()
    )
    env.extras["log"]["Metrics/foot_landing_flatness_tilt_sin_mean"] = (
      tilt_sin.mean()
    )
    env.extras["log"]["Metrics/foot_landing_flatness_tilt_excess_mean"] = (
      tilt_excess.mean()
    )

    if log_only:
      return torch.zeros_like(penalty)
    return penalty


class shank_step_lip_proximity_penalty(_StepBoundaryFootVolume):
  """Penalize shank volume points that get too close to high-side step lips."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    self._env = env
    self._debug_vis_available = bool(params.get("debug_vis_shank_points", False))
    self._debug_vis_enabled = self._debug_vis_available
    self._debug_vis_asset_cfg = params.get("asset_cfg", _DEFAULT_SHANK_BODY_CFG)
    self._debug_vis_shank_point_radius = float(
      params.get("debug_vis_shank_point_radius", 0.012)
    )
    shank_point_color = params.get(
      "debug_vis_shank_point_color", (0.1, 0.8, 1.0, 0.9)
    )
    self._debug_vis_shank_point_color = cast(
      tuple[float, float, float, float],
      tuple(float(c) for c in shank_point_color),
    )
    self._shank_ref_local = torch.tensor(
      params.get("shank_ref_local", (0.045, 0.0, -0.165)),
      device=env.device,
      dtype=torch.float32,
    )
    self._shank_local_points = _make_shank_volume_points(
      env.device,
      x_range=params.get("shank_x_range", (0.045, 0.045)),
      y_range=params.get("shank_y_range", (-0.035, 0.035)),
      z_range=params.get("shank_z_range", (-0.23, -0.10)),
      grid_shape=params.get("shank_grid_shape", (1, 3, 5)),
    )
    self._max_shank_point_ref_distance = torch.norm(
      self._shank_local_points - self._shank_ref_local, dim=-1
    ).max()
    self._height_history_len = max(2, int(params.get("height_history_len", 6)))
    self._base_z_history = torch.zeros(
      env.num_envs, self._height_history_len, device=env.device
    )
    self._base_z_initialized = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )
    self._ascent_hold_counter = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._base_z_initialized[env_ids] = False
    self._ascent_hold_counter[env_ids] = 0

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._debug_vis_available or not self._debug_vis_enabled:
      return

    env = self._env
    env_indices = list(visualizer.get_env_indices(env.num_envs))
    if not env_indices:
      return

    points_w = self._shank_points_w(env, self._debug_vis_asset_cfg)
    points_np = points_w.detach().cpu().numpy()
    for env_idx in env_indices:
      for point in points_np[env_idx].reshape(-1, 3):
        visualizer.add_sphere(
          center=point,
          radius=self._debug_vis_shank_point_radius,
          color=self._debug_vis_shank_point_color,
        )

  def _shank_points_w(
    self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    shank_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    shank_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]

    num_envs, num_shanks = shank_pos_w.shape[:2]
    num_points = self._shank_local_points.shape[0]
    local_points = self._shank_local_points.view(1, 1, num_points, 3).expand(
      num_envs, num_shanks, num_points, 3
    )
    shank_quat = shank_quat_w[:, :, None, :].expand(
      num_envs, num_shanks, num_points, 4
    )
    return shank_pos_w[:, :, None, :] + quat_apply(shank_quat, local_points)

  def _shank_ref_w(
    self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    shank_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    shank_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    num_envs, num_shanks = shank_pos_w.shape[:2]
    shank_ref_local = self._shank_ref_local.view(1, 1, 3).expand(
      num_envs, num_shanks, 3
    )
    return shank_pos_w + quat_apply(shank_quat_w, shank_ref_local)

  def _base_ascent_gate(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    height_gain_threshold: float,
    ascent_hold_steps: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    asset: Entity = env.scene[asset_cfg.name]
    base_z = asset.data.root_link_pos_w[:, 2]

    not_initialized = ~self._base_z_initialized
    if bool(torch.any(not_initialized).item()):
      self._base_z_history[not_initialized] = base_z[not_initialized, None]
      self._base_z_initialized[not_initialized] = True

    oldest_base_z = self._base_z_history[:, 0].clone()
    self._base_z_history = torch.roll(self._base_z_history, shifts=-1, dims=1)
    self._base_z_history[:, -1] = base_z

    height_gain = base_z - oldest_base_z
    ascending_now = height_gain > height_gain_threshold
    if ascent_hold_steps > 0:
      hold_value = torch.full_like(self._ascent_hold_counter, ascent_hold_steps)
      decayed_hold = torch.clamp(self._ascent_hold_counter - 1, min=0)
      self._ascent_hold_counter = torch.where(
        ascending_now, hold_value, decayed_hold
      )
      ascent_gate = self._ascent_hold_counter > 0
    else:
      ascent_gate = ascending_now
    return ascent_gate.float(), height_gain

  @staticmethod
  def _shank_tilt_gate(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    shank_tilt_threshold_deg: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    asset: Entity = env.scene[asset_cfg.name]
    shank_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    num_envs, num_shanks = shank_quat_w.shape[:2]
    local_down = torch.tensor(
      (0.0, 0.0, -1.0), device=env.device, dtype=torch.float32
    )
    local_down = local_down.view(1, 1, 3).expand(num_envs, num_shanks, 3)
    shank_axis_w = quat_apply(shank_quat_w, local_down)
    tilt_sin = torch.norm(shank_axis_w[..., :2], dim=-1)
    threshold = math.sin(math.radians(shank_tilt_threshold_deg))
    return (tilt_sin > threshold).float(), tilt_sin

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    clearance_radius: float = 0.12,
    collision_radius: float = 0.05,
    collision_weight: float = 4.0,
    height_gain_threshold: float = 0.03,
    ascent_hold_steps: int = 4,
    shank_tilt_threshold_deg: float = 20.0,
    nearest_boundaries: int | None = None,
    log_only: bool = False,
    min_terrain_level: int | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_SHANK_BODY_CFG,
    **_: object,
  ) -> torch.Tensor:
    boundaries, valid_boundaries = _current_step_boundaries(env)
    if boundaries is None or valid_boundaries is None:
      return torch.zeros(env.num_envs, device=env.device)

    points_w = self._shank_points_w(env, asset_cfg)
    num_envs, num_shanks, num_points = points_w.shape[:3]

    level_active = _terrain_level_active(env, min_terrain_level)
    base_valid = valid_boundaries & level_active[:, None]

    p0 = boundaries[:, :, 0:3]
    p1 = boundaries[:, :, 3:6]
    shank_ref_w = self._shank_ref_w(env, asset_cfg)
    ref_dist = self._ref_to_segment_distance(shank_ref_w, p0, p1)
    influence_radius = clearance_radius + self._max_shank_point_ref_distance
    selected_idx, fallback = self._nearest_boundary_indices(
      ref_dist, base_valid, influence_radius, nearest_boundaries
    )

    if selected_idx is None:
      expanded_boundaries = boundaries[:, None, :, :].expand(
        num_envs, num_shanks, -1, -1
      )
      expanded_valid = base_valid[:, None, :].expand(num_envs, num_shanks, -1)
      min_dist = self._lip_min_dist(points_w, expanded_boundaries, expanded_valid, None)
      fallback_ratio = torch.zeros((), device=env.device)
    else:
      assert fallback is not None
      selected_boundaries = self._gather_by_foot(boundaries, selected_idx)
      selected_valid = self._gather_mask_by_foot(base_valid, selected_idx)
      min_dist = self._lip_min_dist(points_w, selected_boundaries, selected_valid, None)
      fallback_ratio = fallback.float().mean()
      if bool(torch.any(fallback).item()):
        expanded_boundaries = boundaries[:, None, :, :].expand(
          num_envs, num_shanks, -1, -1
        )
        expanded_valid = base_valid[:, None, :].expand(num_envs, num_shanks, -1)
        full_min_dist = self._lip_min_dist(
          points_w, expanded_boundaries, expanded_valid, None
        )
        min_dist = torch.where(fallback[:, :, None], full_min_dist, min_dist)

    safe_min_dist = torch.where(
      torch.isfinite(min_dist),
      min_dist,
      torch.full_like(min_dist, clearance_radius),
    )
    clearance = max(clearance_radius, 1e-6)
    collision = max(collision_radius, 1e-6)
    proximity = torch.relu(clearance_radius - safe_min_dist) / clearance
    collision_depth = torch.relu(collision_radius - safe_min_dist) / collision
    point_penalty = proximity + collision_weight * torch.square(collision_depth)
    leg_penalty = torch.mean(point_penalty, dim=2)
    ascent_gate, height_gain = self._base_ascent_gate(
      env, asset_cfg, height_gain_threshold, ascent_hold_steps
    )
    tilt_gate, tilt_sin = self._shank_tilt_gate(
      env, asset_cfg, shank_tilt_threshold_deg
    )
    gated_leg_penalty = leg_penalty * ascent_gate[:, None] * tilt_gate
    penalty = torch.sum(gated_leg_penalty, dim=1)

    finite = torch.isfinite(min_dist)
    finite_count = finite.float().sum().clamp_min(1.0)
    min_dist_mean = (
      torch.where(finite, min_dist, torch.zeros_like(min_dist)).sum() / finite_count
    )
    env.extras["log"]["Metrics/shank_lip_proximity_penalty_mean"] = penalty.mean()
    env.extras["log"]["Metrics/shank_lip_raw_penalty_mean"] = torch.sum(
      leg_penalty, dim=1
    ).mean()
    env.extras["log"]["Metrics/shank_lip_min_dist_mean"] = min_dist_mean
    env.extras["log"]["Metrics/shank_lip_close_ratio"] = (
      safe_min_dist < clearance_radius
    ).float().mean()
    env.extras["log"]["Metrics/shank_lip_collision_ratio"] = (
      safe_min_dist < collision_radius
    ).float().mean()
    env.extras["log"]["Metrics/shank_lip_nearest_fallback_ratio"] = fallback_ratio
    env.extras["log"]["Metrics/shank_lip_ascent_gate_ratio"] = ascent_gate.mean()
    env.extras["log"]["Metrics/shank_lip_height_gain_mean"] = height_gain.mean()
    env.extras["log"]["Metrics/shank_lip_tilt_gate_ratio"] = tilt_gate.mean()
    env.extras["log"]["Metrics/shank_lip_tilt_sin_mean"] = tilt_sin.mean()
    env.extras["log"]["Metrics/shank_lip_points_per_leg"] = torch.tensor(
      float(num_points), device=env.device
    )

    if log_only:
      return torch.zeros_like(penalty)
    return penalty
