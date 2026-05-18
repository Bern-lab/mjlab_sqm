"""Unitree G1 target-heading teacher environment configuration."""

import math
from copy import deepcopy
from dataclasses import replace

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import (
  CameraSensorCfg,
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp.teacher_target_heading_command import (
  TeacherTargetHeadingVelocityCommandCfg,
)
from mjlab.tasks.velocity.mdp.teacher_target_heading_rewards import (
  teacher_target_progress,
  teacher_target_reached_bonus,
)
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg


def _apply_teacher_actor(cfg: ManagerBasedRlEnvCfg) -> None:
  """Expose privileged terrain and velocity terms to the actor."""
  actor_terms = cfg.observations["actor"].terms
  critic_terms = cfg.observations["critic"].terms

  for term_name in ("base_lin_vel", "height_scan", "foot_height"):
    if term_name in critic_terms:
      actor_terms[term_name] = critic_terms[term_name]


def _add_target_flat_patch_sampling(
  terrain_generator: TerrainGeneratorCfg,
) -> TerrainGeneratorCfg:
  """Return a copied terrain generator with legacy target flat-patch sampling."""
  cfg = deepcopy(terrain_generator)
  for name, sub_cfg in cfg.sub_terrains.items():
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
  return cfg


def _disable_observation_noise_and_delay(cfg: ManagerBasedRlEnvCfg) -> None:
  """Disable observation noise and delay for all observation groups."""
  for group_cfg in cfg.observations.values():
    for term in group_cfg.terms.values():
      term.noise = None
      term.delay_min_lag = 0
      term.delay_max_lag = 0
      term.delay_hold_prob = 0.0
      term.delay_update_period = 0
      term.delay_per_env = True
      term.delay_per_env_phase = True


def unitree_g1_target_heading_teacher_env_cfg(
  play: bool = False,
  actor_obs_delay_min_lag: int = 0,
  actor_obs_delay_max_lag: int = 0,
  actor_obs_delay_hold_prob: float = 0.0,
  blind_obs_history_length: int = 1,
  current_obs_history_length: int = 1,
) -> ManagerBasedRlEnvCfg:
  """Create a rough-terrain privileged teacher task with target-heading commands."""
  cfg = make_velocity_env_cfg()
  _apply_teacher_actor(cfg)

  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 400
  cfg.sim.nconmax = 256
  cfg.sim.nccdmax = 64
  cfg.sim.use_cuda_graph = False

  robot_cfg = get_g1_robot_cfg()
  articulation = robot_cfg.articulation
  assert articulation is not None
  robot_cfg.articulation = replace(
    articulation,
    actuators=tuple(
      replace(
        actuator_cfg,
        delay_min_lag=0,
        delay_max_lag=0,
        delay_hold_prob=0.0,
        delay_update_period=0,
        delay_per_env_phase=True,
      )
      for actuator_cfg in articulation.actuators
    ),
  )
  cfg.scene.entities = {"robot": robot_cfg}

  actor_terms = cfg.observations["actor"].terms
  for term_name in (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos_rel",
    "joint_vel_rel",
    "base_lin_vel",
    "height_scan",
    "foot_height",
  ):
    term = deepcopy(actor_terms[term_name])
    assert isinstance(term, ObservationTermCfg)
    term.delay_min_lag = actor_obs_delay_min_lag
    term.delay_max_lag = actor_obs_delay_max_lag
    term.delay_hold_prob = actor_obs_delay_hold_prob
    actor_terms[term_name] = term

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

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
  front_camera_cfg = CameraSensorCfg(  # 深度相机挂载
    name="front_depth",
    parent_body="robot/torso_link",
    pos=(0.10, 0.0, 0.45),
    quat=(0.95371695, 0.0, -0.30070580, 0.0),
    fovy=80.0,#垂直仰角
    width=64,
    height=64,
    data_types=("depth",),
    enabled_geom_groups=(0, 2, 3),
    use_shadows=False,
    use_textures=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
    front_camera_cfg,
  )
  cfg.observations["camera"] = ObservationGroupCfg(  # 相机观测
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
  _disable_observation_noise_and_delay(cfg)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  base_twist_cmd = cfg.commands["twist"]
  assert isinstance(base_twist_cmd, UniformVelocityCommandCfg)
  base_twist_cmd.viz.z_offset = 1.15
  velocity_stages = [
    {
      "step": 0,
      "lin_vel_x": (0.0, 0.45),
      "lin_vel_y": (0.0, 0.0),
      "ang_vel_z": (-0.4, 0.4),
    },
    {
      "step": 3000 * 24,
        "lin_vel_x": (0.0, 0.7),
      "lin_vel_y": (0.0, 0.0),
      "ang_vel_z": (-0.6, 0.6),
    },
    {
      "step": 8000 * 24,
      "lin_vel_x": (0.0, 1.0),
      "lin_vel_y": (0.0, 0.0),
      "ang_vel_z": (-0.8, 0.8),
    },
    {
      "step": 15000 * 24,
      "lin_vel_x": (0.0, 1.2),
      "lin_vel_y": (0.0, 0.0),
      "ang_vel_z": (-0.8, 0.8),
    },
  ]

  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = velocity_stages

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

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
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
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

  cfg.rewards["body_ang_vel"].weight = -0.08
  cfg.rewards["angular_momentum"].weight = -0.03

  cfg.rewards["action_rate_l2"].weight = -0.15


  cfg.rewards["joint_acc_l2"] = RewardTermCfg(
    func=mdp.joint_acc_l2,
    weight=-2.5e-7,
  )
  cfg.rewards["action_acc_l2"] = RewardTermCfg(
    func=mdp.action_acc_l2,
    weight=-0.05,
  )

  cfg.rewards["is_terminated"] = RewardTermCfg(func=mdp.is_terminated, weight=-200.0)

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  if play:
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

    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0

  history_terms = (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
  )
  actor_current_terms = (
    "velocity_commands",
    "gait_phase",
    "height_scan",
    "base_lin_vel",
    "foot_height",
  )
  critic_current_terms = actor_current_terms + (
    "foot_air_time",
    "foot_contact",
    "foot_contact_forces",
  )

  cfg.observations["actor"].history_length = None
  cfg.observations["critic"].history_length = None

  for group_name, current_terms in (
    ("actor", actor_current_terms),
    ("critic", critic_current_terms),
  ):
    terms = cfg.observations[group_name].terms
    for term_name in history_terms:
      term = deepcopy(terms[term_name])
      assert isinstance(term, ObservationTermCfg)
      term.history_length = blind_obs_history_length
      terms[term_name] = term
    for term_name in current_terms:
      term = deepcopy(terms[term_name])
      assert isinstance(term, ObservationTermCfg)
      term.history_length = current_obs_history_length
      terms[term_name] = term

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
      lin_vel_x=(-1.0, 1.7),
      lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-0.7, 0.7),
      heading=(-math.pi, math.pi),
    ),
  )
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
  cfg.curriculum.pop("command_vel", None)

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

  return cfg
