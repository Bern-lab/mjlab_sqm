"""Unitree G1 velocity environment configurations."""

import math
from copy import deepcopy

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import (
  TargetHeadingVelocityCommandCfg,
  UniformVelocityCommandCfg,
)
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains import FlatPatchSamplingCfg
from mjlab.terrains.config import BLIND_HIGH_STAIRS_TERRAINS_CFG
from mjlab.terrains.primitive_terrains import (
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from .blind_rough_toe_contact_cfg import (
  configure_g1_toe_riser_contact_memory_penalty,
)


def _blind_rough_play_terrain_cfg():
  """Return a tougher blind-rough terrain set for visual play testing."""
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


def unitree_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None:
    cfg.scene.terrain.terrain_generator = (
      _blind_rough_play_terrain_cfg() if play else BLIND_HIGH_STAIRS_TERRAINS_CFG
    )
    cfg.scene.terrain.max_init_terrain_level = 2

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.metrics["left_knee_deg"] = MetricsTermCfg(
    func=mdp.joint_pos_deg,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=("left_knee_joint",)),
    },
  )
  cfg.metrics["right_knee_deg"] = MetricsTermCfg(
    func=mdp.joint_pos_deg,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=("right_knee_joint",)),
    },
  )
  cfg.metrics["knee_max_deg"] = MetricsTermCfg(
    func=mdp.joint_pos_deg,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*knee_joint",)),
      "mode": "max",
    },
  )
  cfg.metrics["left_ankle_pitch_deg"] = MetricsTermCfg(
    func=mdp.joint_pos_deg,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=("left_ankle_pitch_joint",)),
    },
  )
  cfg.metrics["right_ankle_pitch_deg"] = MetricsTermCfg(
    func=mdp.joint_pos_deg,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=("right_ankle_pitch_joint",)),
    },
  )
  cfg.metrics["ankle_pitch_min_deg"] = MetricsTermCfg(
    func=mdp.joint_pos_deg,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*ankle_pitch_joint",)),
      "mode": "min",
    },
  )
  cfg.metrics["ankle_pitch_max_deg"] = MetricsTermCfg(
    func=mdp.joint_pos_deg,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*ankle_pitch_joint",)),
      "mode": "max",
    },
  )
  cfg.metrics["base_height_rel"] = MetricsTermCfg(
    func=mdp.base_height_above_support_value,
    params={
      "height_sensor_name": "foot_height_scan",
      "contact_sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
    },
  )

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.4,#0.3
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.45,#35
    r".*ankle_pitch.*": 0.20,#25
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    # Waist.
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    # Arms.
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
    r".*wrist.*": 0.3,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (-0.5, 0.8),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-0.5, 0.5),
      },
      {
        "step": 3000 * 24,
        "lin_vel_x": (-0.7, 1.0),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-0.6, 0.6),
      },
      {
        "step": 10000 * 24,
        "lin_vel_x": (0.0, 1.2),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-0.8, 0.8),
      },
    ]

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

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def _add_target_flat_patch_sampling(terrain_generator):
  """Add target flat-patch sampling to a copied terrain generator.

  The target patches are used only as high-level heading targets later.
  They do not change the policy observation or command interface by themselves.
  """
  terrain_generator = deepcopy(terrain_generator)

  # General flat patch sampling for non-directional terrains.
  full_tile_target = FlatPatchSamplingCfg(
    num_patches=128,
    patch_radius=0.20,
    max_height_diff=0.02,
    x_range=(0.5, 7.5),
    y_range=(0.5, 7.5),
    grid_resolution=0.05,
  )

  # For pyramid stairs / pyramid slopes in mjlab, the meaningful top/bottom
  # platform is usually around the center of the 8x8 tile.
  center_platform_target = FlatPatchSamplingCfg(
    num_patches=128,
    patch_radius=0.20,
    max_height_diff=0.02,
    x_range=(3.0, 5.0),
    y_range=(3.0, 5.0),
    grid_resolution=0.05,
  )

  for name, sub_cfg in terrain_generator.sub_terrains.items():
    sub_cfg.flat_patch_sampling = dict(sub_cfg.flat_patch_sampling or {})

    if name in (
      "pyramid_stairs",
      "pyramid_stairs_inv",
      "hf_pyramid_slope",
      "hf_pyramid_slope_inv",
    ):
      sub_cfg.flat_patch_sampling["target"] = center_platform_target
    else:
      sub_cfg.flat_patch_sampling["target"] = full_tile_target

  return terrain_generator


def unitree_g1_blind_target_heading_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create G1 blind rough locomotion config with target-heading commands."""
  cfg = unitree_g1_blind_rough_env_cfg(play=play)

  # Make actor blind: remove exteroceptive height scan from policy input.
  # Keep critic height_scan for privileged learning.
  cfg.observations["actor"].terms.pop("height_scan", None)

  # Add target flat patches to this new task only.
  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_generator is not None
  cfg.scene.terrain.terrain_generator = _add_target_flat_patch_sampling(
    cfg.scene.terrain.terrain_generator
  )

  cfg.commands["twist"] = TargetHeadingVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
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
    ranges=TargetHeadingVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.2),
      lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-0.7, 0.7),
      heading=(-math.pi, math.pi),
    ),
  )
  cfg.commands["twist"].viz.z_offset = 1.15

  cfg.curriculum.pop("command_vel", None)

  cfg.rewards["target_progress"] = RewardTermCfg(
    func=mdp.target_progress,
    weight=0.4,
    params={
      "command_name": "twist",
      "min_distance": 0.05,
    },
  )
  cfg.rewards["target_reached_bonus"] = RewardTermCfg(
    func=mdp.target_reached_bonus,
    weight=0.15,
    params={
      "command_name": "twist",
    },
  )

  return cfg


def unitree_g1_blind_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity config without actor height scan."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  cfg.sim.nconmax = 256
  cfg.sim.njmax = 4096
  cfg.sim.mujoco.ccd_iterations = 50

  # Keep terrain_scan in the scene for the privileged critic, but remove it
  # from the deployable actor so the policy does not require exteroception.
  del cfg.observations["actor"].terms["height_scan"]
  configure_g1_toe_riser_contact_memory_penalty(cfg)

  cfg.observations["actor"].history_length = 5
  cfg.observations["critic"].history_length = 3

  actor_terms = cfg.observations["actor"].terms
  for term_name in (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos_rel",
    "joint_vel_rel",
  ):
    actor_terms[term_name].delay_min_lag = 0
    actor_terms[term_name].delay_max_lag = 2
    actor_terms[term_name].delay_hold_prob = 0.8
    actor_terms[term_name].delay_update_period = 5

  actor_terms["base_ang_vel"].noise = Unoise(n_min=-0.3, n_max=0.3)
  actor_terms["projected_gravity"].noise = Unoise(n_min=-0.07, n_max=0.07)
  actor_terms["joint_pos_rel"].noise = Unoise(n_min=-0.015, n_max=0.015)
  actor_terms["joint_vel_rel"].noise = Unoise(n_min=-2.0, n_max=2.0)

  # Blind rough starts from easier terrain and conservative commands. The
  # terrain curriculum can still promote successful envs to harder rows.
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.terrain_generator = (
      _blind_rough_play_terrain_cfg() if play else BLIND_HIGH_STAIRS_TERRAINS_CFG
    )
    cfg.scene.terrain.max_init_terrain_level = 2

  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (-0.5, 0.8),
        "lin_vel_y": (-0.4, 0.4),
        "ang_vel_z": (-0.5, 0.5),
      },
      {
        "step": 3000 * 24,
        "lin_vel_x": (-0.8, 1.2),
        "lin_vel_y": (-0.8, 0.8),
        "ang_vel_z": (-0.8, 0.8),
      },
    ]

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  # if play:
  #   twist_cmd.ranges.lin_vel_x = (-1.0, 1.3)
  #   twist_cmd.ranges.lin_vel_y = (-0.7, 0.7)
  #   twist_cmd.ranges.ang_vel_z = (-0.4, 0.4)
  # else:
  #   twist_cmd.ranges.lin_vel_x = (0.0, 0.5)
  #   twist_cmd.ranges.lin_vel_y = (-0.2, 0.2)
  #   twist_cmd.ranges.ang_vel_z = (-0.3, 0.3)

  # cfg.rewards["foot_clearance"].params["target_height"] = 0.18
  # cfg.rewards["foot_swing_height"].params["target_height"] = 0.18
  # cfg.rewards["foot_slip"].weight = -0.2
  # cfg.rewards["soft_landing"].weight = -2e-5
  # cfg.rewards["action_rate_l2"].weight = -0.15
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
  # cfg.rewards["track_linear_velocity"].weight = 1.5
  # cfg.rewards["track_linear_velocity"].params["std"] = 0.6

  return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  cfg.rewards.pop("joint_acc_l2", None)
  cfg.rewards.pop("action_acc_l2", None)
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02

  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
      {"step": 3000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
      {"step": 6000 * 24, "lin_vel_x": (-2.0, 3.0)},
    ]

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
