"""Booster K1 constants."""

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

K1_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "booster_k1" / "xmls" / "K1_serial_armature.xml"
  # MJLAB_SRC_PATH / "asset_zoo" / "robots" / "booster_k1" / "xmls" / "K1_serial_moonwalk.xml"
)
assert K1_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, K1_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(K1_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec

##
# Actuator config.
##

# Motor specs (from Booster K1).
# Based on the configuration provided:
# - 6416 motor: Hip Pitch, Knee Pitch
# - 4310 motor: Hip Yaw
# - 6408 motor: Hip Pitch joints  
# - 4315 motor: Hip Roll
# - ROB-14 motor: Arms and head

ARMATURE_6416 = 0.095625
ARMATURE_4310 = 0.0282528
ARMATURE_6408 = 0.071719   #原始0.0478125   0.071719
ARMATURE_4315 = 0.0353161 #原始0.0339552    0.0353161
ARMATURE_ROB_14 = 0.001    #原始0.001    0.01

ACTUATOR_6416 = ElectricActuator(
  reflected_inertia=ARMATURE_6416,
  velocity_limit=20.0,
  effort_limit=40.0,
)
ACTUATOR_4310 = ElectricActuator(
  reflected_inertia=ARMATURE_4310,
  velocity_limit=32.0,
  effort_limit=20.0,
)
ACTUATOR_6408 = ElectricActuator(
  reflected_inertia=ARMATURE_6408,
  velocity_limit=32.0,  #原32
  effort_limit=30.0,
)
ACTUATOR_4315 = ElectricActuator(
  reflected_inertia=ARMATURE_4315,
  velocity_limit=20.0,
  effort_limit=20.0,
)
ACTUATOR_ROB_14 = ElectricActuator(
  reflected_inertia=ARMATURE_ROB_14,
  velocity_limit=37.0,
  effort_limit=14.0,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_6416 = ARMATURE_6416 * NATURAL_FREQ**2
STIFFNESS_4310 = ARMATURE_4310 * NATURAL_FREQ**2
STIFFNESS_6408 = ARMATURE_6408 * NATURAL_FREQ**2
STIFFNESS_4315 = ARMATURE_4315 * NATURAL_FREQ**2
STIFFNESS_ROB_14 = ARMATURE_ROB_14 * NATURAL_FREQ**2

DAMPING_6416 = 2.0 * DAMPING_RATIO * ARMATURE_6416 * NATURAL_FREQ
DAMPING_4310 = 2.0 * DAMPING_RATIO * ARMATURE_4310 * NATURAL_FREQ
DAMPING_6408 = 2.0 * DAMPING_RATIO * ARMATURE_6408 * NATURAL_FREQ
DAMPING_4315 = 2.0 * DAMPING_RATIO * ARMATURE_4315 * NATURAL_FREQ
DAMPING_ROB_14 = 2.0 * DAMPING_RATIO * ARMATURE_ROB_14 * NATURAL_FREQ

# K1 Leg actuators - Hip Pitch (6408 motor)
K1_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Hip_Pitch",),
  effort_limit=30.0,
  armature=ARMATURE_6408,
  stiffness=80.0,
  damping=2.0,
)

# K1 Leg actuators - Hip Roll (4315 motor)
K1_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Hip_Roll",),
  effort_limit=20.0,
  armature=ARMATURE_4315,
  stiffness=80.0,
  damping=2.0,
)

# K1 Leg actuators - Hip Yaw (4310 motor)
K1_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Hip_Yaw",),
  effort_limit=20.0,
  armature=ARMATURE_4310,
  stiffness=80.0,
  damping=2.0,
)

# K1 Leg actuators - Knee Pitch (6416 motor)
K1_ACTUATOR_KNEE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Knee_Pitch",),
  effort_limit=40.0,
  armature=ARMATURE_6416,
  stiffness=80.0,
  damping=2.0,
)

# K1 Feet actuators
K1_ACTUATOR_FEET = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Ankle_Pitch", ".*_Ankle_Roll",),
  effort_limit=20.0,
  armature=2.0 * ARMATURE_4310,
  stiffness=30.0,
  damping=2.0,
)

# K1 Arm actuators
K1_ACTUATOR_ARMS = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_Shoulder_Pitch",
    ".*_Shoulder_Roll",
    ".*_Elbow_Pitch",
    ".*_Elbow_Yaw",
  ),
  effort_limit=14.0,
  armature=ARMATURE_ROB_14,
  stiffness=3.95,
  damping=0.3,
)

# K1 Head actuators
K1_ACTUATOR_HEAD = BuiltinPositionActuatorCfg(
  target_names_expr=(".*Head.*",),
  effort_limit=4.0,
  armature=0.001,
  stiffness=10.0,
  damping=2.0,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.50),
  joint_pos={
    "AAHead_yaw": 0.0,
    "Head_pitch": 0.0,
    "ALeft_Shoulder_Pitch": 0.0,
    "Left_Shoulder_Roll": -1.3,
    "Left_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": 0.0,
    "ARight_Shoulder_Pitch": 0.0,
    "Right_Shoulder_Roll": 1.3,
    "Right_Elbow_Pitch": 0.0,
    "Right_Elbow_Yaw": 0.0,
    "Left_Hip_Pitch": 0.0,
    "Left_Hip_Roll": 0.0,
    "Left_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.0,
    "Left_Ankle_Pitch": 0.0,
    "Left_Ankle_Roll": 0.0,
    "Right_Hip_Pitch": 0.0,
    "Right_Hip_Roll": 0.0,
    "Right_Hip_Yaw": 0.0,
    "Right_Knee_Pitch": 0.0,
    "Right_Ankle_Pitch": 0.0,
    "Right_Ankle_Roll": 0.0,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.50),
  joint_pos={
    "AAHead_yaw": 0.0,
    "Head_pitch": 0.0,
    "ALeft_Shoulder_Pitch": 0.0,
    "Left_Shoulder_Roll": -1.3,
    "Left_Elbow_Pitch": 0.6,
    "Left_Elbow_Yaw": 0.0,
    "ARight_Shoulder_Pitch": 0.0,
    "Right_Shoulder_Roll": 1.3,
    "Right_Elbow_Pitch": 0.6,
    "Right_Elbow_Yaw": 0.0,
    "Left_Hip_Pitch": -0.312,
    "Left_Hip_Roll": 0.0,
    "Left_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.669,
    "Left_Ankle_Pitch": -0.363,
    "Left_Ankle_Roll": 0.0,
    "Right_Hip_Pitch": -0.312,
    "Right_Hip_Roll": 0.0,
    "Right_Hip_Yaw": 0.0,
    "Right_Knee_Pitch": 0.669,
    "Right_Ankle_Pitch": -0.363,
    "Right_Ankle_Roll": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3 and custom friction and solimp.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(Left|Right)_.*_collision$": 3, ".*_collision": 1},
  priority={r"^(Left|Right)_.*_collision$": 1},
  friction={r"^(Left|Right)_.*_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(Left|Right)_.*_collision$": 3, ".*_collision": 1},
  priority={r"^(Left|Right)_.*_collision$": 1},
  friction={r"^(Left|Right)_.*_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(Left|Right)_Ankle_.*_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##


K1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    K1_ACTUATOR_HIP_PITCH,
    K1_ACTUATOR_HIP_ROLL,
    K1_ACTUATOR_HIP_YAW,
    K1_ACTUATOR_KNEE_PITCH,
    K1_ACTUATOR_FEET,
    K1_ACTUATOR_ARMS,
    K1_ACTUATOR_HEAD,
  ),
  soft_joint_pos_limit_factor=0.9,
)

def get_k1_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=K1_ARTICULATION,
  )

K1_ACTION_SCALE: dict[str, float] = {}
for a in K1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    K1_ACTION_SCALE[n] = 0.25 * e / s

# K1 Joint names (for reference)
K1_JOINTS = [
  'AAHead_yaw',
  'Head_pitch',
  'ALeft_Shoulder_Pitch',
  'Left_Shoulder_Roll',
  'Left_Elbow_Pitch',
  'Left_Elbow_Yaw',
  'ARight_Shoulder_Pitch',
  'Right_Shoulder_Roll',
  'Right_Elbow_Pitch',
  'Right_Elbow_Yaw',
  'Left_Hip_Pitch',
  'Left_Hip_Roll',
  'Left_Hip_Yaw',
  'Left_Knee_Pitch',
  'Left_Ankle_Pitch',
  'Left_Ankle_Roll',
  'Right_Hip_Pitch',
  'Right_Hip_Roll',
  'Right_Hip_Yaw',
  'Right_Knee_Pitch',
  'Right_Ankle_Pitch',
  'Right_Ankle_Roll',
]

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_K1_robot_cfg())

  viewer.launch(robot.spec.compile())
