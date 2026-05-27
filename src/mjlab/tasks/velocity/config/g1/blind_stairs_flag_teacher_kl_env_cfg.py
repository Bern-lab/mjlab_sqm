"""Teacher-KL Unitree G1 blind stair-flag velocity environment configuration."""

import math
from copy import deepcopy

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp.teacher_target_heading_command import (
  TeacherTargetHeadingVelocityCommandCfg,
)
from mjlab.tasks.velocity.mdp.teacher_target_heading_rewards import (
  teacher_target_progress,
  teacher_target_reached_bonus,
)
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from mjlab.terrains import FlatPatchSamplingCfg
from mjlab.terrains.config import flat, pyramid_stairs, pyramid_stairs_inv
from mjlab.terrains.primitive_terrains import (
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from .blind_rough_teacher_kl_env_cfg import unitree_g1_blind_rough_teacherkl_env_cfg
from .blind_rough_toe_contact_cfg import (
  configure_g1_toe_riser_contact_memory_penalty,
)

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_STAIR_TERRAIN_NAMES = ("pyramid_stairs", "pyramid_stairs_inv")
STAIRS_FLAG_ACTOR_HISTORY_LENGTH = 5  # 历史帧
STAIRS_FLAG_CRITIC_HISTORY_LENGTH = 3


STAIRS_FLAG_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=10,
  curriculum=True,
  sub_terrains={
    "flat": flat(proportion=0.4),
    "pyramid_stairs": pyramid_stairs(
      proportion=0.3,
      step_height_range=(0.04, 0.2),
      step_width=0.30,
      platform_width=3.0,
      border_width=1.0,
    ),
    "pyramid_stairs_inv": pyramid_stairs_inv(
      proportion=0.3,
      step_height_range=(0.04, 0.2),
      step_width=0.30,
      platform_width=3.0,
      border_width=1.0,
    ),
  },
  add_lights=True,
)


def _stairs_flag_play_terrain_cfg() -> TerrainGeneratorCfg:
  terrain_cfg = deepcopy(STAIRS_FLAG_TERRAINS_CFG)
  terrain_cfg.curriculum = False
  terrain_cfg.num_rows = 5
  terrain_cfg.num_cols = len(terrain_cfg.sub_terrains)
  terrain_cfg.border_width = 10.0

  for terrain_name in _STAIR_TERRAIN_NAMES:
    sub_terrain = terrain_cfg.sub_terrains[terrain_name]
    assert isinstance(
      sub_terrain,
      BoxPyramidStairsTerrainCfg | BoxInvertedPyramidStairsTerrainCfg,
    )
    sub_terrain.step_height_range = (0.14, 0.14)

  return terrain_cfg


def _add_target_flat_patch_sampling(
  terrain_generator: TerrainGeneratorCfg,
) -> TerrainGeneratorCfg:
  """Add target-point patches to the stair-flag terrains."""
  terrain_cfg = deepcopy(terrain_generator)
  for name, sub_cfg in terrain_cfg.sub_terrains.items():
    if "stairs" in name:
      target_sampling = FlatPatchSamplingCfg(
        num_patches=128,
        patch_radius=0.20,
        max_height_diff=0.02,
        x_range=(3.0, 5.0),
        y_range=(3.0, 5.0),
        grid_resolution=0.05,
      )
    else:
      target_sampling = FlatPatchSamplingCfg(
        num_patches=128,
        patch_radius=0.20,
        max_height_diff=0.02,
        x_range=(0.5, 7.5),
        y_range=(0.5, 7.5),
        grid_resolution=0.05,
      )

    sub_cfg.flat_patch_sampling = dict(sub_cfg.flat_patch_sampling or {})
    sub_cfg.flat_patch_sampling["target"] = target_sampling
  return terrain_cfg


def terrain_is_stairs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return 1 on stair terrain columns and 0 on flat terrain columns."""
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    return torch.zeros(env.num_envs, 1, device=env.device)

  terrain_generator = terrain.cfg.terrain_generator
  if terrain_generator is None:
    return torch.zeros(env.num_envs, 1, device=env.device)

  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  stair_cols = [
    i for i, name in enumerate(sub_terrain_names) if name in _STAIR_TERRAIN_NAMES
  ]
  if not stair_cols:
    return torch.zeros(env.num_envs, 1, device=env.device)

  num_cols = terrain.terrain_origins.shape[1]
  if num_cols == len(sub_terrain_names):
    asset = env.scene[asset_cfg.name]
    root_y = asset.data.root_link_pos_w[:, 1]
    tile_width_y = float(terrain_generator.size[1])
    grid_min_y = -0.5 * num_cols * tile_width_y
    terrain_cols = torch.floor((root_y - grid_min_y) / tile_width_y).long()
    terrain_cols = terrain_cols.clamp(0, num_cols - 1)
  else:
    terrain_cols = terrain.terrain_types

  is_stairs = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for col in stair_cols:
    is_stairs |= terrain_cols == col

  return is_stairs.float().unsqueeze(-1)


def terrain_is_stairs_metric(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Scalar metric companion for play-time terrain display."""
  return terrain_is_stairs(env, asset_cfg=asset_cfg).squeeze(-1)


def _add_depth_teacher_camera(cfg: ManagerBasedRlEnvCfg) -> None:
  front_camera_cfg = CameraSensorCfg(
    name="front_depth",
    parent_body="robot/torso_link",
    pos=(0.10, 0.0, 0.45),
    quat=(0.95371695, 0.0, -0.30070580, 0.0),
    fovy=80.0,
    width=64,
    height=64,
    data_types=("depth",),
    enabled_geom_groups=(0, 2, 3),
    use_shadows=False,
    use_textures=True,
  )
  sensor_names = {sensor.name for sensor in cfg.scene.sensors or ()}
  if front_camera_cfg.name not in sensor_names:
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (front_camera_cfg,)

  cfg.observations["camera"] = ObservationGroupCfg(
    terms={
      "front_depth": ObservationTermCfg(
        func=mdp.camera_depth,
        params={"sensor_name": "front_depth", "cutoff_distance": 5.0},
      ),
    },
    concatenate_terms=True,
    concatenate_dim=0,
    enable_corruption=False,
  )


def _configure_target_heading_command(
  cfg: ManagerBasedRlEnvCfg,
  play: bool,
) -> None:
  base_twist_cmd = cfg.commands["twist"]
  assert isinstance(base_twist_cmd, UniformVelocityCommandCfg)
  base_twist_cmd.viz.z_offset = 1.15

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_generator is not None
  cfg.scene.terrain.terrain_generator = _add_target_flat_patch_sampling(
    cfg.scene.terrain.terrain_generator
  )

  twist_cmd = TeacherTargetHeadingVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(7.0, 12.0),
    heading_command=True,
    heading_control_stiffness=0.5,
    rel_target_envs=0.7,
    rel_random_heading_envs=0.2,
    rel_standing_envs=0.1,
    patch_name="target",
    target_reached_threshold=0.5,
    target_min_distance=1.0,
    target_max_distance=12.0,
    target_tile_radius=1,
    include_current_tile=False,
    zero_lateral_velocity=True,
    debug_vis=True,
    ranges=TeacherTargetHeadingVelocityCommandCfg.Ranges(
      lin_vel_x=(0.0, 1.2),
      lin_vel_y=(0.0, 0.0),
      ang_vel_z=(-0.8, 0.8),
      heading=(-math.pi, math.pi),
    ),
  )
  twist_cmd.viz.z_offset = 1.15
  if play:
    twist_cmd.rel_target_envs = 1.0
    twist_cmd.rel_random_heading_envs = 0.0
    twist_cmd.rel_standing_envs = 0.0
    twist_cmd.heading_control_stiffness = 0.8
    twist_cmd.ranges.lin_vel_x = (0.3, 1.0)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)
    twist_cmd.target_min_distance = 1.0
    twist_cmd.target_max_distance = 10.0
    twist_cmd.target_tile_radius = 1

  cfg.commands["twist"] = twist_cmd

  cfg.rewards["target_progress"] = RewardTermCfg(
    func=teacher_target_progress,
    weight=0.8,
    params={"command_name": "twist", "min_distance": 0.05},
  )
  cfg.rewards["target_reached_bonus"] = RewardTermCfg(
    func=teacher_target_reached_bonus,
    weight=0.4,
    params={"command_name": "twist"},
  )


def unitree_g1_blind_stairs_flag_teacherkl_env_cfg(
  play: bool = False,
  actor_history_length: int = STAIRS_FLAG_ACTOR_HISTORY_LENGTH,
  critic_history_length: int = STAIRS_FLAG_CRITIC_HISTORY_LENGTH,
) -> ManagerBasedRlEnvCfg:
  """Create blind Teacher-KL training with an actor stair/flat mode flag."""
  cfg = unitree_g1_blind_rough_teacherkl_env_cfg(play=play)

  cfg.observations["actor"].history_length = actor_history_length
  cfg.observations["critic"].history_length = critic_history_length
  configure_g1_toe_riser_contact_memory_penalty(cfg)

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = (
    _stairs_flag_play_terrain_cfg() if play else deepcopy(STAIRS_FLAG_TERRAINS_CFG)
  )
  cfg.scene.terrain.max_init_terrain_level = 2
  _configure_target_heading_command(cfg, play=play)

  cfg.observations["actor"].terms["terrain_is_stairs"] = ObservationTermCfg(
    func=terrain_is_stairs,
  )
  _add_depth_teacher_camera(cfg)
  cfg.metrics["terrain_is_stairs"] = MetricsTermCfg(func=terrain_is_stairs_metric)

  return cfg
