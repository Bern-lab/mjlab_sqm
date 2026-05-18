"""constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

E1_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "noetix_e1" / "xmls" / "e1.xml"

)
assert E1_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, E1_XML.parent / "meshes", meshdir)
  return assets

def get_spec() -> mujoco.MjSpec:
  abs_xml_path = E1_XML.absolute()
  spec = mujoco.MjSpec.from_file(str(abs_xml_path))
  spec.meshdir = str(abs_xml_path.parent / "meshes")
  spec.assets = get_assets(spec.meshdir)
  return spec


ARMATURE_4315 = 0.033048
ARMATURE_4340 = 0.032
ARMATURE_8112 = 0.04752756
ARMATURE_10020 = 0.068575968

NATURAL_FREQ = 4 * 2.0 * 3.1415926535  # 4Hz


STIFFNESS_4315 = ARMATURE_4315 * NATURAL_FREQ**2
STIFFNESS_4340 = ARMATURE_4340 * NATURAL_FREQ**2
STIFFNESS_8112 = ARMATURE_8112 * NATURAL_FREQ**2
STIFFNESS_10020 = ARMATURE_10020 * NATURAL_FREQ**2

DAMPING_4315 = 2.0 * ARMATURE_4315 * NATURAL_FREQ
DAMPING_4340 = 2.0 * ARMATURE_4340 * NATURAL_FREQ
DAMPING_8112 = 2.0 * ARMATURE_8112 * NATURAL_FREQ
DAMPING_10020 = 2.0 * ARMATURE_10020 * NATURAL_FREQ

# E1 Leg actuators - Hip Pitch (6408 motor)
E1_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint",),
  effort_limit=100.0,
  armature=ARMATURE_10020,
  stiffness=STIFFNESS_10020 * 8,
  damping=DAMPING_10020 * 4.5,
)

# E1 Leg actuators - Hip Roll (4315 motor)
E1_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint",),
  effort_limit=80.0,
  armature=ARMATURE_8112,
  stiffness=STIFFNESS_8112 *6,
  damping=DAMPING_8112 * 4,
)

# E1 Leg actuators - Hip Yaw (4310 motor)
E1_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint",),
  effort_limit=80.0,
  armature=ARMATURE_8112,
  stiffness=STIFFNESS_8112 * 6,
  damping=DAMPING_8112 * 4,
)

# E1 Leg actuators - Knee Pitch (6416 motor)
E1_ACTUATOR_KNEE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  effort_limit=100.0,
  armature=ARMATURE_10020,
  stiffness=STIFFNESS_10020 * 8,
  damping=DAMPING_10020 * 3,
)

# E1 Feet actuators
E1_ACTUATOR_FEET = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint",".*_ankle_roll_joint"),
  effort_limit=70.0,
  armature=ARMATURE_4315,
  stiffness=STIFFNESS_4315 * 10,
  damping=DAMPING_4315 * 4,
)

# E1 Arm actuators
E1_ACTUATOR_SHOULDERS = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
  ),
  effort_limit=20.0,
  armature=ARMATURE_4340,
  stiffness=STIFFNESS_4340 * 2,
  damping=DAMPING_4340 * 2,
)
E1_ACTUATOR_ELBOWS = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_elbow_pitch_joint",
    ".*_elbow_yaw_joint"
  ),
  effort_limit=20.0,
  armature=ARMATURE_4340,
  stiffness=STIFFNESS_4340 * 1.2,
  damping=DAMPING_4340 * 1.2,
)

E1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_.*_joint",),
  effort_limit=60.0,
  armature=ARMATURE_4315,
  stiffness=STIFFNESS_4315 * 2,
  damping=DAMPING_4315 * 2,
)




##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 1.0),
  joint_pos={
    "l_leg_hip_yaw_joint": 0.0,
    "l_leg_hip_roll_joint": 0.0,
    "l_leg_hip_pitch_joint": -0.1495,
    "l_leg_knee_joint": 0.3215,
    "l_leg_ankle_pitch_joint": -0.1720,
    "l_leg_ankle_roll_joint": 0.0,
    "r_leg_hip_yaw_joint": 0.0,
    "r_leg_hip_roll_joint": 0.0,
    "r_leg_hip_pitch_joint": -0.1495,
    "r_leg_knee_joint": 0.3215,
    "r_leg_ankle_pitch_joint": -0.1720,
    "r_leg_ankle_roll_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "l_arm_shoulder_pitch_joint": 0.0,
    "l_arm_shoulder_roll_joint": 0.2618,
    "l_arm_shoulder_yaw_joint": 0.0,
    "l_arm_elbow_pitch_joint": 0.0,
    "l_arm_elbow_yaw_joint": 0.0,
    "r_arm_shoulder_pitch_joint": 0.0,
    "r_arm_shoulder_roll_joint": -0.2618,
    "r_arm_shoulder_yaw_joint": 0.0,
    "r_arm_elbow_pitch_joint": 0.0,
    "r_arm_elbow_yaw_joint": 0.0,
  },
  joint_vel={".*": 0.0},
 
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.838),
  joint_pos={
    "l_leg_hip_yaw_joint": 0.0,
    "l_leg_hip_roll_joint": 0.0,
    "l_leg_hip_pitch_joint": -0.312,
    "l_leg_knee_joint": 0.669,
    "l_leg_ankle_pitch_joint": -0.363,
    "l_leg_ankle_roll_joint": 0.0,
    "r_leg_hip_yaw_joint": 0.0,
    "r_leg_hip_roll_joint": 0.0,
    "r_leg_hip_pitch_joint": -0.312,
    "r_leg_knee_joint": 0.669,
    "r_leg_ankle_pitch_joint": -0.363,
    "r_leg_ankle_roll_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "l_arm_shoulder_pitch_joint": 0.0,
    "l_arm_shoulder_roll_joint": 0.2618,
    "l_arm_shoulder_yaw_joint": 0.0,
    "l_arm_elbow_pitch_joint": 0.0,
    "l_arm_elbow_yaw_joint": 0.0,
    "r_arm_shoulder_pitch_joint": 0.0,
    "r_arm_shoulder_roll_joint": -0.2618,
    "r_arm_shoulder_yaw_joint": 0.0,
    "r_arm_elbow_pitch_joint": 0.0,
    "r_arm_elbow_yaw_joint": 0.0,


  },
  joint_vel={".*": 0.0},
  
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Foot collisions are given condim=3 while body collisions are also condim=3 for ground contact.
# Body parts: condim=3 for stable ground contact.
# Self-collisions between body parts: condim=1.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={
    r"^(left|right|l_leg_|r_leg_).*foot.*_collision$": 3,  # Foot collisions
    r"^(base_link|torso|waist|l_leg_hip|l_leg_knee|r_leg_hip|r_leg_knee)_collision$": 3,  # Body collisions
    ".*_collision": 1  # Default self-collisions
  },
  priority={
    r"^(left|right|l_leg_|r_leg_).*foot.*_collision$": 1,
    r"^(base_link|torso|waist|l_leg_hip|l_leg_knee|r_leg_hip|r_leg_knee)_collision$": 0,
  },
  friction={
    r"^(left|right|l_leg_).*foot.*_collision$": (0.6,),
    r"^(right_).*foot.*_collision$": (0.6,),
  },
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={
    r"^(left|right|l_leg_|r_leg_).*foot.*_collision$": 3,
    r"^(base_link|torso|waist|l_leg_hip|l_leg_knee|r_leg_hip|r_leg_knee)_collision$": 3,
    ".*_collision": 1
  },
  priority={
    r"^(left|right|l_leg_|r_leg_).*foot.*_collision$": 1,
    r"^(base_link|torso|waist|l_leg_hip|l_leg_knee|r_leg_hip|r_leg_knee)_collision$": 0,
  },
  friction={
    r"^(left|right|l_leg_|r_leg_).*foot.*_collision$": (0.6,),
  },
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_Ankle_.*_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

E1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    E1_ACTUATOR_HIP_PITCH,
    E1_ACTUATOR_HIP_ROLL,
    E1_ACTUATOR_HIP_YAW,
    E1_ACTUATOR_KNEE_PITCH,
    E1_ACTUATOR_FEET,
    E1_ACTUATOR_SHOULDERS,
    E1_ACTUATOR_ELBOWS,
    E1_ACTUATOR_WAIST,
  ),
  soft_joint_pos_limit_factor=0.9,
)

def get_e1_robot_cfg() -> EntityCfg:
  """Get a fresh G1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=E1_ARTICULATION,
  )

E1_ACTION_SCALE: dict[str, float] = {}
for a in E1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    E1_ACTION_SCALE[n] = 0.05 * e / s  #原0.25



if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_e1_robot_cfg())

  viewer.launch(robot.spec.compile())
