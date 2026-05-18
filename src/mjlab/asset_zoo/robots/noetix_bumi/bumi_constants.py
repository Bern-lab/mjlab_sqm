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

BUMI_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "noetix_bumi" / "xmls" / "bumi2.xml"
  # MJLAB_SRC_PATH / "asset_zoo" / "robots" / "booster_k1" / "xmls" / "K1_serial_moonwalk.xml"
)
assert BUMI_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, BUMI_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(BUMI_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec

##
# Actuator config.  电机参数
##

ARMATURE_431025 =  0.0018606249999999999 #hip yaw
ARMATURE_431536 =  0.007642512   #hip row,hip pitch,knee
ARMATURE_431040 =  0.0045024  #waist
ARMATURE_LZ05 =  0.001    #arm

ARMATURE_DM4340 = 0.032   # 未调用
ARMATURE_YKS4315 = 0.033048  #未调用
ARMATURE_LZ00 =  0.0007   # 未调用

#############  BUMI  inertia 电子转动惯量

inertia_arm = ARMATURE_LZ05 # arm
inertia_waist_yaw = ARMATURE_431040 # waist
inertia_hip_yaw = 0.000107 +  ARMATURE_431025 # hip yaw  拟合优度 (R²) : 0.9935  巨小
inertia_hip_roll = 0.209604  + ARMATURE_431536 # hip roll    拟合优度 (R²) : 0.60
inertia_hip_pitch = 0.319813  + ARMATURE_431536  # hip pitch    拟合优度 (R²) : 0.9884
inertia_knee = 0.028678  + ARMATURE_431536 # knee    拟合优度 (R²) : 0.8447
inertia_ankle_pitch_real = 0.021115 # ankle_pitch    拟合优度 (R²) : 0.8645
inertia_ankle_roll_real = 0.002829 # ankle_roll    拟合优度 (R²) : 0.6867




#############################
ACTUATOR_431025 = ElectricActuator(
  reflected_inertia=ARMATURE_431025,
  velocity_limit=9.0,
  effort_limit=27.0,
)
ACTUATOR_431536 = ElectricActuator(
  reflected_inertia=ARMATURE_431536,
  velocity_limit=60.0,
  effort_limit=12.0,
)
ACTUATOR_431040 = ElectricActuator(
  reflected_inertia=ARMATURE_431040,
  velocity_limit=9.0,
  effort_limit=27.0,
)

# ACTUATOR_DM4340 = ElectricActuator(
#   reflected_inertia=ARMATURE_DM4340,
#   velocity_limit=,
#   effort_limit=,
# )
# ACTUATOR_YKS4315 = ElectricActuator(
#   reflected_inertia=ARMATURE_YKS4315,
#   velocity_limit=,
#   effort_limit=,
# )


NATURAL_FREQ_1 = 3 * 2.0 * 3.1415926535  #  for 3Hz
NATURAL_FREQ_2 = 4 * 2.0 * 3.1415926535  #  for 4Hz
NATURAL_FREQ_3 = 5 * 2.0 * 3.1415926535  #  for 5Hz

NATURAL_FREQ_4 = 10 * 2.0 * 3.1415926535  #  for 5Hz  用这个试试，这个应该是10Hz吧


#为什么身体各个部位用的Hz大小不一样？
STIFFNESS_arm = inertia_arm * NATURAL_FREQ_4**2 * 3
STIFFNESS_waist = inertia_waist_yaw * NATURAL_FREQ_4**2 * 3
STIFFNESS_hip_yaw = inertia_hip_yaw * NATURAL_FREQ_4**2 * 2

STIFFNESS_hip_roll = inertia_hip_roll * NATURAL_FREQ_1**2
STIFFNESS_hip_pitch = inertia_hip_pitch * NATURAL_FREQ_1**2

STIFFNESS_knee = inertia_knee * NATURAL_FREQ_3**2 * 2

STIFFNESS_ankle_pitch = inertia_ankle_pitch_real * NATURAL_FREQ_1**2
STIFFNESS_ankle_roll = inertia_ankle_roll_real * NATURAL_FREQ_1**2



DAMPING_arm = 2 * inertia_arm * NATURAL_FREQ_4 * 3
DAMPING_waist = 2 * inertia_waist_yaw * NATURAL_FREQ_4 * 6
DAMPING_hip_yaw = 2 * inertia_hip_yaw * NATURAL_FREQ_4 * 8

DAMPING_hip_roll = 0.7 * inertia_hip_roll * NATURAL_FREQ_1
DAMPING_hip_pitch = 0.7 * inertia_hip_pitch * NATURAL_FREQ_1

DAMPING_knee = 0.9 * inertia_knee * NATURAL_FREQ_3 * 2

DAMPING_ankle_pitch = 0.9 * inertia_ankle_pitch_real * NATURAL_FREQ_1
DAMPING_ankle_roll = 0.9 * inertia_ankle_roll_real * NATURAL_FREQ_1



# K1 Leg actuators - Hip Pitch (6408 motor)
BUMI_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_leg_pitch_joint",),
  effort_limit=60.0,
  armature=inertia_hip_pitch,
  stiffness=STIFFNESS_hip_pitch,
  damping=DAMPING_hip_pitch,
)

# K1 Leg actuators - Hip Roll (4315 motor)
BUMI_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_leg_roll_joint",),
  effort_limit=27.0,
  armature=inertia_hip_roll,
  stiffness=STIFFNESS_hip_roll,
  damping=DAMPING_hip_roll,
)

# K1 Leg actuators - Hip Yaw (4310 motor)
BUMI_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_leg_yaw_joint",),
  effort_limit=27.0,
  armature=inertia_hip_yaw,
  stiffness=STIFFNESS_hip_yaw,
  damping=DAMPING_hip_yaw,
)

# K1 Leg actuators - Knee Pitch (6416 motor)
BUMI_ACTUATOR_KNEE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_pitch_joint",),
  effort_limit=60.0,
  armature=inertia_knee,
  stiffness=STIFFNESS_knee,
  damping=DAMPING_knee,
)

# bumi_ankle_pitch
BUMI_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint",),
  effort_limit=15.0,
  armature=inertia_ankle_pitch_real,
  stiffness=STIFFNESS_ankle_pitch,
  damping=DAMPING_ankle_pitch,
)

# bumi_ankle_roll
BUMI_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_roll_joint",),
  effort_limit=15.0,
  armature=inertia_ankle_roll_real,
  stiffness=STIFFNESS_ankle_roll,
  damping=DAMPING_ankle_roll,
)

# BUMI Arm actuators
BUMI_ACTUATOR_ARMS = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_arm_pitch_joint",
    ".*_arm_roll_joint",
    ".*_arm_yaw_joint",
    ".*_elbow_pitch_joint",
  ),
  effort_limit=5.5,
  armature=ARMATURE_LZ05,
  stiffness=STIFFNESS_arm,
  damping=DAMPING_arm,
)

# BUMI WAIST actuators
BUMI_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw_joint",),
  effort_limit=27.0,
  armature=inertia_waist_yaw,
  stiffness=STIFFNESS_waist,
  damping=DAMPING_waist,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.65),
  joint_pos={
    "l_leg_yaw_joint": 0.0,
    "l_leg_roll_joint": 0.0,
    "l_leg_pitch_joint": -0.1495,
    "l_knee_pitch_joint": 0.3215,
    "l_ankle_pitch_joint": -0.1720,
    "l_ankle_roll_joint": 0.0,
    "r_leg_yaw_joint": 0.0,
    "r_leg_roll_joint": 0.0,
    "r_leg_pitch_joint": -0.1495,
    "r_knee_pitch_joint": 0.3215,
    "r_ankle_pitch_joint": -0.1720,
    "r_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "l_arm_pitch_joint": 0.0,
    "l_arm_roll_joint": 0.3,
    "l_arm_yaw_joint": 0.0,
    "l_elbow_pitch_joint": 0.0,
    "r_arm_pitch_joint": 0.0,
    "r_arm_roll_joint": -0.3,
    "r_arm_yaw_joint": 0.0,
    "r_elbow_pitch_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

#原始bumi.py没有该套参数
KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.65),
  joint_pos={
    "l_leg_yaw_joint": 0.0,
    "l_leg_roll_joint": 0.0,
    "l_leg_pitch_joint": -0.1495,
    "l_knee_pitch_joint": 0.3215,
    "l_ankle_pitch_joint": -0.1720,
    "l_ankle_roll_joint": 0.0,
    "r_leg_yaw_joint": 0.0,
    "r_leg_roll_joint": 0.0,
    "r_leg_pitch_joint": -0.1495,
    "r_knee_pitch_joint": 0.3215,
    "r_ankle_pitch_joint": -0.1720,
    "r_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "l_arm_pitch_joint": 0.0,
    "l_arm_roll_joint": 0.3,
    "l_arm_yaw_joint": 0.0,
    "l_elbow_pitch_joint": 0.0,
    "r_arm_pitch_joint": 0.0,
    "r_arm_roll_joint": -0.3,
    "r_arm_yaw_joint": 0.0,
    "r_elbow_pitch_joint": 0.0,
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
    geom_names_expr=(r"collision_.*",),
    condim={
        r"^collision_(left|right)_ankle$": 3,
        r"^collision_.*": 1
    },
    priority={
        r"^collision_(left|right)_ankle$": 1
    },
    friction={
        # 匹配collision_left_ankle和collision_right_ankle（对应脚踝），设防滑摩擦
        r"^collision_(left|right)_ankle$": (0.6,)
    },
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

BUMI_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    BUMI_ACTUATOR_HIP_PITCH,
    BUMI_ACTUATOR_HIP_ROLL,
    BUMI_ACTUATOR_HIP_YAW,
    BUMI_ACTUATOR_KNEE_PITCH,
    BUMI_ACTUATOR_ANKLE_PITCH,
    BUMI_ACTUATOR_ANKLE_ROLL,
    BUMI_ACTUATOR_ARMS,
    BUMI_ACTUATOR_WAIST,
  ),
  soft_joint_pos_limit_factor=0.9,
)

def get_bumi_robot_cfg() -> EntityCfg:
  """Get a fresh G1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=BUMI_ARTICULATION,
  )

BUMI_ACTION_SCALE: dict[str, float] = {}
for a in BUMI_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    BUMI_ACTION_SCALE[n] = 0.25 * e / s

# BUMI Joint names (for reference)
BUMI_JOINTS = [
     "waist_yaw_joint",
      "l_arm_pitch_joint",
      "l_arm_roll_joint",
      "l_arm_yaw_joint",
      "l_elbow_pitch_joint",
      "r_arm_pitch_joint",
      "r_arm_roll_joint",
      "r_arm_yaw_joint",
      "r_elbow_pitch_joint",
      "l_leg_pitch_joint",
      "l_leg_roll_joint",
      "l_leg_yaw_joint",
      "l_knee_pitch_joint",
      "l_ankle_pitch_joint",
      "l_ankle_roll_joint",
      "r_leg_pitch_joint",
      "r_leg_roll_joint",
      "r_leg_yaw_joint",
      "r_knee_pitch_joint",
      "r_ankle_pitch_joint",
      "r_ankle_roll_joint"
     ]

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_bumi_robot_cfg())

  viewer.launch(robot.spec.compile())
