"""Booster K1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots.engineai_pm01.pm01_constants import (
  PM01_ACTION_SCALE,
  get_pm01_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def engineai_pm01_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Engineai PM01 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()
  # Match the PM01 dance setup used in mjlab_old / rl_dance_example:
  # 5 ms physics steps and 50 Hz control with larger contact buffers.
  cfg.sim.mujoco.timestep = 0.005
  cfg.decimation = 4
  cfg.sim.nconmax = 300_000
  cfg.sim.njmax = 2048
  cfg.scene.entities = {"robot": get_pm01_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="LINK_BASE", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="LINK_BASE", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,

  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PM01_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "LINK_BASE"
  motion_cmd.body_names = (
    'LINK_BASE',
    'LINK_HIP_ROLL_L',
    'LINK_KNEE_PITCH_L',
    'LINK_ANKLE_ROLL_L',
    'LINK_HIP_ROLL_R',
    'LINK_KNEE_PITCH_R',
    'LINK_ANKLE_ROLL_R',
    'LINK_TORSO_YAW',
    'LINK_SHOULDER_ROLL_L',
    'LINK_ELBOW_PITCH_L',
    'LINK_ELBOW_END_L',
    'LINK_SHOULDER_ROLL_R',
    'LINK_ELBOW_PITCH_R',
    'LINK_ELBOW_END_R',
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = [] #TODO 改正则与脚底碰撞匹配
  cfg.events["base_com"].params["asset_cfg"].body_names = ("LINK_BASE",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    'LINK_ELBOW_END_L',
    'LINK_ELBOW_END_R',
    "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_R",
  )

  cfg.viewer.body_name = "LINK_BASE"

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
