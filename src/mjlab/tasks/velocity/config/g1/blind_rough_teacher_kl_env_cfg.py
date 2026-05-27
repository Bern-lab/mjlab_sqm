"""Teacher-KL Unitree G1 blind-rough velocity environment configuration."""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp.teacher_target_heading_command import (
  TeacherTargetHeadingVelocityCommandCfg,
)
from mjlab.tasks.velocity.mdp.teacher_target_heading_rewards import (
  teacher_target_progress,
  teacher_target_reached_bonus,
)
from mjlab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg
from mjlab.terrains.config import BLIND_HIGH_STAIRS_TERRAINS_CFG
from mjlab.terrains.primitive_terrains import (
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from .blind_rough_toe_contact_cfg import (
  configure_g1_toe_riser_contact_memory_penalty,
)
from .env_cfgs import (
  UniformVelocityCommandCfg,
  unitree_g1_rough_env_cfg,
)

TEACHER_OBSERVATION_ORDER = (
  "base_ang_vel",
  "projected_gravity",
  "velocity_commands",
  "gait_phase",
  "joint_pos_rel",
  "joint_vel_rel",
  "last_action",
  "height_scan",
  "base_lin_vel",
  "foot_height",
)

def _add_target_flat_patch_sampling(
  terrain_generator: TerrainGeneratorCfg,
) -> TerrainGeneratorCfg:
  terrain_cfg = deepcopy(terrain_generator)
  for name, sub_cfg in terrain_cfg.sub_terrains.items():
    if "stairs" in name or "slope" in name:
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


def _teacherkl_play_terrain_cfg():
  # 5.9: high-stairs terrain; previous: rough terrain
  terrain_cfg = deepcopy(BLIND_HIGH_STAIRS_TERRAINS_CFG)
  terrain_cfg.curriculum = False
  terrain_cfg.num_rows = 5
  terrain_cfg.num_cols = 5
  terrain_cfg.border_width = 10.0

  for terrain_name in ("high_stairs", "high_stairs_inv"):
    sub_terrain = terrain_cfg.sub_terrains[terrain_name]
    assert isinstance(
      sub_terrain,
      BoxPyramidStairsTerrainCfg | BoxInvertedPyramidStairsTerrainCfg,
    )
    sub_terrain.step_height_range = (0.14, 0.14)

  return terrain_cfg


def _configure_teacherkl_student_env(
  cfg: ManagerBasedRlEnvCfg, play: bool
) -> None:
  robot_cfg = deepcopy(cfg.scene.entities["robot"])
  assert robot_cfg.articulation is not None
  for actuator_cfg in robot_cfg.articulation.actuators:
    actuator_cfg.delay_min_lag = 0
    actuator_cfg.delay_max_lag = 2  # 5.9: 0; previous: 1/2
    actuator_cfg.delay_hold_prob = 0.8  # 5.9: 0.0; previous: 0.5/0.8
    actuator_cfg.delay_update_period = 5  # 5.9: 0; previous: 5
  cfg.scene.entities["robot"] = robot_cfg

  cfg.sim.nconmax = 256  # 5.9: 256
  cfg.sim.njmax = 4096  # 5.9: 4096; previous: 2048
  cfg.sim.mujoco.ccd_iterations = 50  # 5.9: 50; previous: 400

  # Keep terrain_scan available for critic/teacher, but make the deployed
  # student actor blind.
  del cfg.observations["actor"].terms["height_scan"]
  configure_g1_toe_riser_contact_memory_penalty(cfg)

  cfg.observations["actor"].history_length = 5  # 5.9: 3; previous: 5
  cfg.observations["critic"].history_length = 3  # 5.9: 3; previous: 1

  actor_terms = cfg.observations["actor"].terms
  for term_name in (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos_rel",
    "joint_vel_rel",
  ):
    actor_terms[term_name].delay_min_lag = 0
    actor_terms[term_name].delay_max_lag = 2  # 5.9: 1; previous: 2
    actor_terms[term_name].delay_hold_prob = 0.8  # 5.9: 0.8; previous: 0.5
    actor_terms[term_name].delay_update_period = 5  # 5.9: 5

  actor_terms["base_ang_vel"].noise = Unoise(n_min=-0.3, n_max=0.3)  # 5.9: +/-0.3; previous: +/-0.2
  actor_terms["projected_gravity"].noise = Unoise(n_min=-0.07, n_max=0.07)  # 5.9: +/-0.07; previous: +/-0.05
  actor_terms["joint_pos_rel"].noise = Unoise(n_min=-0.015, n_max=0.015)  # 5.9: +/-0.015; previous: +/-0.01
  actor_terms["joint_vel_rel"].noise = Unoise(n_min=-2.0, n_max=2.0)  # 5.9: +/-2.0; previous: +/-1.0
  
  if cfg.scene.terrain is not None:
    # 5.9: BLIND_HIGH_STAIRS_TERRAINS_CFG; previous: ROUGH_TERRAINS_CFG
    cfg.scene.terrain.terrain_generator = (
      _teacherkl_play_terrain_cfg() if play else BLIND_HIGH_STAIRS_TERRAINS_CFG
    )
    cfg.scene.terrain.max_init_terrain_level = 2

  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (0.0, 0.8),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-0.5, 0.5),
      },
      {
        "step": 3000 * 24,
        "lin_vel_x": (0.0, 1.0),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-0.8, 0.8),
      },
    ]#方便教师蒸馏，速度范围一样，从3000步开始

  cfg.rewards["joint_acc_l2"] = RewardTermCfg(
    func=mdp.joint_acc_l2,
    weight=-2.5e-7,
  )
  cfg.rewards["action_acc_l2"] = RewardTermCfg(
    func=mdp.action_acc_l2,
    weight=-0.05,
  )
  cfg.rewards["body_ang_vel"].weight = -0.08
  cfg.rewards["angular_momentum"].weight = -0.03


def _configure_teacherkl_target_navigation(
  cfg: ManagerBasedRlEnvCfg, play: bool
) -> None:
  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_generator is not None
  cfg.scene.terrain.terrain_generator = _add_target_flat_patch_sampling(
    cfg.scene.terrain.terrain_generator
  )

  twist_cmd = TeacherTargetHeadingVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(8.0, 13.0),
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
      lin_vel_x=(-0.5, 0.8),
      lin_vel_y=(0.0, 0.0),
      ang_vel_z=(-0.5, 0.5),
      heading=(-math.pi, math.pi),
    ),
  )
  twist_cmd.viz.z_offset = 1.15
  if play:
    twist_cmd.rel_target_envs = 1.0
    twist_cmd.rel_random_heading_envs = 0.0
    twist_cmd.rel_standing_envs = 0.0
    twist_cmd.heading_control_stiffness = 0.8
    twist_cmd.ranges.lin_vel_x = (0.3, 0.9)
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


def _reset_teacher_term_temporal_state(term: ObservationTermCfg) -> None:
  """Match the frozen teacher's feedforward observation pipeline."""
  term.delay_min_lag = 0
  term.delay_max_lag = 0
  term.delay_per_env = True
  term.delay_hold_prob = 0.0
  term.delay_update_period = 0
  term.delay_per_env_phase = True
  term.history_length = 0
  term.flatten_history_dim = True


def _make_teacher_terms(cfg: ManagerBasedRlEnvCfg) -> dict[str, ObservationTermCfg]:
  """Build teacher observations in the exact order used by the frozen teacher."""
  actor_terms = cfg.observations["actor"].terms
  critic_terms = cfg.observations["critic"].terms

  terms: dict[str, ObservationTermCfg] = {}
  for term_name in TEACHER_OBSERVATION_ORDER:
    source_terms = actor_terms if term_name in actor_terms else critic_terms
    term = deepcopy(source_terms[term_name])
    _reset_teacher_term_temporal_state(term)
    term.noise = None
    terms[term_name] = term

  return terms


def _add_teacher_depth_camera(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add the depth camera observation expected by the frozen teacher."""
  sensors = tuple(cfg.scene.sensors or ())
  if not any(getattr(sensor, "name", None) == "front_depth" for sensor in sensors):
    cfg.scene.sensors = sensors + (
      CameraSensorCfg(
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
      ),
    )

  cfg.observations["camera"] = ObservationGroupCfg(
    terms={
      "front_depth": ObservationTermCfg(
        func=mdp.camera_depth,
        params={"sensor_name": "front_depth", "cutoff_distance": 5.0},
      ),
    },
    enable_corruption=False,
    concatenate_terms=True,
    concatenate_dim=0,
  )


def unitree_g1_blind_rough_teacherkl_env_cfg(
  play: bool = False,
  use_target_navigation: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a blind-rough student env with privileged frozen-teacher observations.

  The teacher observation group is aligned to the policy exported at:
  ``logs/rsl_rl/g1_velocity_teacher/2026-04-29_19-29-39``.
  Its ONNX metadata reports input order:
  base_ang_vel, projected_gravity, velocity_commands, gait_phase, joint_pos_rel,
  joint_vel_rel, last_action, height_scan, base_lin_vel, foot_height.
  """
  cfg = unitree_g1_rough_env_cfg(play=play)
  _configure_teacherkl_student_env(cfg, play=play)
  _add_teacher_depth_camera(cfg)
  if use_target_navigation:
    _configure_teacherkl_target_navigation(cfg, play=play)

  cfg.observations["teacher"] = ObservationGroupCfg(
    terms=_make_teacher_terms(cfg),
    concatenate_terms=True,
    enable_corruption=False,
    history_length=None,
    flatten_history_dim=True,
  )

  if play and not use_target_navigation:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
