"""Teacher-KL Unitree G1 blind stair-flag velocity environment configuration."""

from copy import deepcopy

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains.config import flat, pyramid_stairs, pyramid_stairs_inv
from mjlab.terrains.primitive_terrains import (
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from .blind_rough_teacher_kl_env_cfg import unitree_g1_blind_rough_teacherkl_env_cfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_STAIR_TERRAIN_NAMES = ("pyramid_stairs", "pyramid_stairs_inv")
STAIRS_FLAG_ACTOR_HISTORY_LENGTH = 5
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


def unitree_g1_blind_stairs_flag_teacherkl_env_cfg(
  play: bool = False,
  actor_history_length: int = STAIRS_FLAG_ACTOR_HISTORY_LENGTH,
  critic_history_length: int = STAIRS_FLAG_CRITIC_HISTORY_LENGTH,
) -> ManagerBasedRlEnvCfg:
  """Create blind Teacher-KL training with an actor stair/flat mode flag."""
  cfg = unitree_g1_blind_rough_teacherkl_env_cfg(play=play)

  cfg.observations["actor"].history_length = actor_history_length
  cfg.observations["critic"].history_length = critic_history_length

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = (
    _stairs_flag_play_terrain_cfg() if play else deepcopy(STAIRS_FLAG_TERRAINS_CFG)
  )
  cfg.scene.terrain.max_init_terrain_level = 2

  cfg.observations["actor"].terms["terrain_is_stairs"] = ObservationTermCfg(
    func=terrain_is_stairs,
  )
  _add_depth_teacher_camera(cfg)
  cfg.metrics["terrain_is_stairs"] = MetricsTermCfg(func=terrain_is_stairs_metric)

  return cfg
