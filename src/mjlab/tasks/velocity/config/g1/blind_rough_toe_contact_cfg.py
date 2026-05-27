"""Shared toe-riser contact penalty config for G1 blind-rough tasks."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp

TOE_TERRAIN_CONTACT_SENSOR = "toe_terrain_contact"
G1_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")


def g1_foot_body_cfg() -> SceneEntityCfg:
  return SceneEntityCfg(
    "robot",
    body_names=G1_FOOT_BODY_NAMES,
    preserve_order=True,
  )


def add_g1_toe_terrain_contact_sensor(cfg: ManagerBasedRlEnvCfg) -> None:
  sensor_names = {sensor.name for sensor in cfg.scene.sensors or ()}
  if TOE_TERRAIN_CONTACT_SENSOR in sensor_names:
    return

  toe_contact_cfg = ContactSensorCfg(
    name=TOE_TERRAIN_CONTACT_SENSOR,
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force", "pos", "normal", "tangent"),
    reduce="maxforce",
    num_slots=4,
    global_frame=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (toe_contact_cfg,)


def configure_g1_toe_riser_contact_memory_penalty(
  cfg: ManagerBasedRlEnvCfg,
) -> None:
  cfg.sim.contact_sensor_maxmatch = 256
  add_g1_toe_terrain_contact_sensor(cfg)

  cfg.observations["critic"].terms[TOE_TERRAIN_CONTACT_SENSOR] = ObservationTermCfg(
    func=mdp.foot_contact,
    params={"sensor_name": TOE_TERRAIN_CONTACT_SENSOR},
  )
  cfg.observations["critic"].terms["toe_terrain_contact_forces"] = ObservationTermCfg(
    func=mdp.foot_contact_forces,
    params={"sensor_name": TOE_TERRAIN_CONTACT_SENSOR},
  )

  cfg.rewards["toe_riser_contact_memory_penalty"] = RewardTermCfg(
    func=mdp.toe_riser_contact_memory_penalty,
    weight=-1.5,
    params={
      "sensor_name": TOE_TERRAIN_CONTACT_SENSOR,
      "free_hits": 1,
      "cooldown_time": 0.20,
      "min_terrain_level": 3,
      "min_ascent_height": 0.03,
      "ascent_velocity_threshold": 0.03,
      "toe_x_min": 0.08,
      "vertical_normal_z_max": 0.4,
      "forward_velocity_threshold": 0.05,
      "force_threshold": 15.0,
      "force_scale": 60.0,
      "asset_cfg": g1_foot_body_cfg(),
    },
  )
