"""Booster K1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots.noetix_e1.e1_constants import (
  E1_ACTION_SCALE,
  get_e1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def noetix_e1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Noetix E1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()
  cfg.sim.nconmax = 256
  cfg.sim.njmax = 512
  cfg.scene.entities = {"robot": get_e1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", ),
    reduce="none",
    num_slots=1,

  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = E1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "base_link"
  motion_cmd.body_names = (
    'base_link',
    'l_leg_hip_yaw_link',
    'l_leg_hip_roll_link',
    'l_leg_hip_pitch_link',
    'l_leg_knee_link',
    'l_leg_ankle_pitch_link',
    'l_leg_ankle_roll_link',
    'r_leg_hip_yaw_link',
    'r_leg_hip_roll_link',
    'r_leg_hip_pitch_link',
    'r_leg_knee_link',
    'r_leg_ankle_pitch_link',
    'r_leg_ankle_roll_link',
    'waist_yaw_link',
    'waist_roll_link',
    'l_arm_shoulder_pitch_link',
    'l_arm_shoulder_roll_link',
    'l_arm_shoulder_yaw_link',
    'l_arm_elbow_pitch_link',
    'l_arm_elbow_yaw_link',
    'r_arm_shoulder_pitch_link',
    'r_arm_shoulder_roll_link',
    'r_arm_shoulder_yaw_link',
    'r_arm_elbow_pitch_link',
    'r_arm_elbow_yaw_link'
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = []#r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "l_leg_ankle_roll_link", # 左脚末端
    "r_leg_ankle_roll_link", # 右脚末端
    "l_arm_elbow_yaw_link",  # 左手末端
    "r_arm_elbow_yaw_link",  # 右手末端
  )

  cfg.viewer.body_name = "base_link"

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg
