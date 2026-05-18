"""Booster K1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots.booster_k1.k1_constants import (
  K1_ACTION_SCALE,
  get_k1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def booster_k1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Booster K1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()
  cfg.sim.nconmax = 256
  cfg.sim.njmax = 512
  cfg.scene.entities = {"robot": get_k1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    fields=("found", ),
    reduce="none",
    num_slots=1,

  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = K1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "Trunk"
  motion_cmd.body_names = (
    'Trunk',
    'Left_Hip_Roll',
    'Left_Shank',
    'left_foot_link',
    'Right_Hip_Roll',
    'Right_Shank',
    'right_foot_link',
    'Left_Arm_2',
    'Left_Arm_3',
    'left_hand_link',
    'Right_Arm_2',
    'Right_Arm_3',
    'right_hand_link',
    'Head_2',
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("Trunk",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    'left_hand_link',
    'right_hand_link',
    "left_foot_link",
    "right_foot_link",
  )

  cfg.viewer.body_name = "Trunk"

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
