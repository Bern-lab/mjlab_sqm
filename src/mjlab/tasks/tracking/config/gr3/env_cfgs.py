"""Fourier GR3 flat tracking environment configurations."""

from mjlab.asset_zoo.robots.fourier_gr3.gr3_constants import (
  GR3_ACTION_SCALE,
  get_gr3_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def fourier_gr3_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create fourier gr3 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()
  cfg.sim.nconmax = 256
  cfg.sim.njmax = 512
  cfg.scene.entities = {"robot": get_gr3_robot_cfg()}

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
  joint_pos_action.scale = GR3_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "base_link"
  motion_cmd.body_names = (
    'base_link',
    'left_thigh_roll_link',
    'left_shank_pitch_link',
    'left_foot_roll_link',
    'right_thigh_roll_link',
    'right_shank_pitch_link',
    'right_foot_roll_link',
    'torso_link',
    'left_upper_arm_roll_link',
    'left_lower_arm_pitch_link',
    'left_end_effector_link',
    'right_upper_arm_roll_link',
    'right_lower_arm_pitch_link',
    'right_end_effector_link'
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = []
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    'left_end_effector_link',
    'right_end_effector_link',
    'left_angle_roll_link',
    'right_angle_roll_link',
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
