"""EngineAI PM01 constants aligned with dance tracking defaults."""

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

PM01_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "engineai_pm01" / "pm_v2.xml"
  # MJLAB_SRC_PATH / "asset_zoo" / "robots" / "booster_k1" / "xmls" / "K1_serial_moonwalk.xml"
)
assert PM01_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, PM01_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(PM01_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec

#######上部分为k1相关参数，以下为pm01参数
joint_armature_hip_pitch = 0.0453
joint_armature_hip_roll = 0.0453
joint_armature_hip_yaw = 0.0067
joint_armature_knee = 0.0453
joint_armature_ankle = 0.0067
joint_armature_waist = 0.0067
joint_armature_shoulder = 0.0067
joint_armature_elbow = 0.0067
joint_armature_head = 0.0067

# Reference:
# - /home/ubt2204/work_ljh/pm01-engineai_robotics_native_sdk-main/assets/config/pm01_edu/rl_dance_example/default.yaml
# - /home/ubt2204/work_ljh/mjlab_old/src/mjlab/asset_zoo/robots/pm01/pm01_constants.py
LOWER_BODY_STIFFNESS = 178.8372344970703
LOWER_BODY_DAMPING = 11.3851318359375
UPPER_BODY_STIFFNESS = 26.45054054260254
UPPER_BODY_DAMPING = 1.6838936805725098
ANKLE_DAMPING = 0.5
LOWER_BODY_EFFORT_LIMIT = 164.0
UPPER_BODY_EFFORT_LIMIT = 52.0

# PM01 Leg actuators - Hip Pitch (6408 motor)
PM01_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_HIP_PITCH_.*",),
  effort_limit=LOWER_BODY_EFFORT_LIMIT,
  armature=joint_armature_hip_pitch,
  stiffness=LOWER_BODY_STIFFNESS,
  damping=LOWER_BODY_DAMPING,
)

# K1 Leg actuators - Hip Roll (4315 motor)
PM01_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_HIP_ROLL_.*",),
  effort_limit=LOWER_BODY_EFFORT_LIMIT,
  armature=joint_armature_hip_roll,
  stiffness=LOWER_BODY_STIFFNESS,
  damping=LOWER_BODY_DAMPING,
)

# K1 Leg actuators - Hip Yaw (4310 motor)
PM01_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_HIP_YAW_.*",),
  effort_limit=UPPER_BODY_EFFORT_LIMIT,
  armature=joint_armature_hip_yaw,
  stiffness=UPPER_BODY_STIFFNESS,
  damping=UPPER_BODY_DAMPING,
)

# K1 Leg actuators - Knee Pitch (6416 motor)
PM01_ACTUATOR_KNEE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_KNEE_PITCH_.*",),
  effort_limit=LOWER_BODY_EFFORT_LIMIT,
  armature=joint_armature_knee,
  stiffness=LOWER_BODY_STIFFNESS,
  damping=LOWER_BODY_DAMPING,
)

# PM01 Feet actuators
PM01_ACTUATOR_FEET = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ANKLE_PITCH_.*", ".*_ANKLE_ROLL_.*",),
  effort_limit=UPPER_BODY_EFFORT_LIMIT,
  armature=joint_armature_ankle,
  stiffness=UPPER_BODY_STIFFNESS,
  damping=ANKLE_DAMPING,
)

# PM01 Arm actuators
PM01_ACTUATOR_ARMS = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*SHOULDER.*",
    ".*ELBOW.*",
    #".*_ELBOW_YAW_.*",
  ),
  effort_limit=UPPER_BODY_EFFORT_LIMIT,
  armature=joint_armature_shoulder,
  stiffness=UPPER_BODY_STIFFNESS,
  damping=UPPER_BODY_DAMPING,
)

# PM01 waist actuators
PM01_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=(".*WAIST.*",),
  effort_limit=UPPER_BODY_EFFORT_LIMIT,
  armature=joint_armature_waist,
  stiffness=UPPER_BODY_STIFFNESS,
  damping=UPPER_BODY_DAMPING,
)

# PM01 Head actuators
PM01_ACTUATOR_HEAD = BuiltinPositionActuatorCfg(
  target_names_expr=(".*HEAD.*",),
  effort_limit=UPPER_BODY_EFFORT_LIMIT,
  armature=joint_armature_head,
  stiffness=UPPER_BODY_STIFFNESS,
  damping=UPPER_BODY_DAMPING,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.9),
  joint_pos={
    "J00_HIP_PITCH_L":-0.06,
    "J01_HIP_ROLL_L":0.0,
    "J02_HIP_YAW_L":0.0,
    "J03_KNEE_PITCH_L":0.12,
    "J04_ANKLE_PITCH_L":-0.06,
    "J05_ANKLE_ROLL_L":0.0,
    "J06_HIP_PITCH_R":-0.06,
    "J07_HIP_ROLL_R":0.0,
    "J08_HIP_YAW_R":0.0,
    "J09_KNEE_PITCH_R":0.12,
    "J10_ANKLE_PITCH_R":-0.06,
    "J11_ANKLE_ROLL_R":0.0,
    "J12_WAIST_YAW":0.0,
    "J13_SHOULDER_PITCH_L":0.0,
    "J14_SHOULDER_ROLL_L":0.15,
    "J15_SHOULDER_YAW_L":0.0,
    "J16_ELBOW_PITCH_L":-0.25,
    "J17_ELBOW_YAW_L":0.0,
    "J18_SHOULDER_PITCH_R":0.0,
    "J19_SHOULDER_ROLL_R":-0.15,
    "J20_SHOULDER_YAW_R":0.0,
    "J21_ELBOW_PITCH_R":-0.25,
    "J22_ELBOW_YAW_R":0.0,
    "J23_HEAD_YAW":0.0
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.68),
  joint_pos={
    "J00_HIP_PITCH_L":-0.312,
    "J01_HIP_ROLL_L":0.0,
    "J02_HIP_YAW_L":0.0,
    "J03_KNEE_PITCH_L":0.669,
    "J04_ANKLE_PITCH_L":-0.363,
    "J05_ANKLE_ROLL_L":0.0,
    "J06_HIP_PITCH_R":-0.24,
    "J07_HIP_ROLL_R":0.0,
    "J08_HIP_YAW_R":0.0,
    "J09_KNEE_PITCH_R":0.669,
    "J10_ANKLE_PITCH_R":-0.363,
    "J11_ANKLE_ROLL_R":0.0,
    "J12_WAIST_YAW":0.0,
    "J13_SHOULDER_PITCH_L":0.0,
    "J14_SHOULDER_ROLL_L":0.0,
    "J15_SHOULDER_YAW_L":0.0,
    "J16_ELBOW_PITCH_L":0.0,
    "J17_ELBOW_YAW_L":0.0,
    "J18_SHOULDER_PITCH_R":0.0,
    "J19_SHOULDER_ROLL_R":0.0,
    "J20_SHOULDER_YAW_R":0.0,
    "J21_ELBOW_PITCH_R":0.0,
    "J22_ELBOW_YAW_R":0.0,
    "J23_HEAD_YAW":0.0
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3 and custom friction and solimp.

# FULL_COLLISION = CollisionCfg(
#   geom_names_expr=[".*_collision"],
#   condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
#   priority={r"^(left|right)_foot[1-7]_collision$": 1},
#   friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
# )
FULL_COLLISION = CollisionCfg(
    # 白名单：匹配所有以collision_开头的碰撞几何（精准贴合XML的class命名）
    geom_names_expr=(r".*",),
    # geom_names_expr=[r"collision_.*"],
    # 碰撞维度：脚底/脚趾3维精准计算（行走核心），其他所有碰撞部位1维简化
    condim={
        r"^collision_(left|right)_foot(_toe)?$": 3,  # 匹配左脚/右脚/左脚趾/右脚趾
        r"^collision_.*": 1                          # 其他collision_xxx全部兜底
    },
    # 碰撞优先级：脚底/脚趾设为最高优先级1（优先计算，避免穿模）
    priority={
        r"^collision_(left|right)_foot(_toe)?$": 1
    },
    # 摩擦系数：脚底/脚趾设0.6防滑，其他用框架默认值
    friction={
        r"^collision_(left|right)_foot(_toe)?$": (0.6,)
    },
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)


##
# Final config.
##

PM01_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    PM01_ACTUATOR_HIP_PITCH,
    PM01_ACTUATOR_HIP_ROLL,
    PM01_ACTUATOR_HIP_YAW,
    PM01_ACTUATOR_KNEE_PITCH,
    PM01_ACTUATOR_FEET,
    PM01_ACTUATOR_ARMS,
    PM01_ACTUATOR_HEAD,
    PM01_ACTUATOR_WAIST
  ),
  soft_joint_pos_limit_factor=0.9,
)

def get_pm01_robot_cfg() -> EntityCfg:
  """Get a fresh G1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=PM01_ARTICULATION,
  )

PM01_ACTION_SCALE: dict[str, float] = {}
for a in PM01_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    PM01_ACTION_SCALE[n] = 0.25 * e / s

# PM01 Joint names (for reference)
PM01_JOINTS = [
  "J00_HIP_PITCH_L",
  "J01_HIP_ROLL_L",
  "J02_HIP_YAW_L",
  "J03_KNEE_PITCH_L",
  "J04_ANKLE_PITCH_L",
  "J05_ANKLE_ROLL_L",
  "J06_HIP_PITCH_R",
  "J07_HIP_ROLL_R",
  "J08_HIP_YAW_R",
  "J09_KNEE_PITCH_R",
  "J10_ANKLE_PITCH_R",
  "J11_ANKLE_ROLL_R",
  "J12_WAIST_YAW",
  "J13_SHOULDER_PITCH_L",
  "J14_SHOULDER_ROLL_L",
  "J15_SHOULDER_YAW_L",
  "J16_ELBOW_PITCH_L",
  "J17_ELBOW_YAW_L",
  "J18_SHOULDER_PITCH_R",
  "J19_SHOULDER_ROLL_R",
  "J20_SHOULDER_YAW_R",
  "J21_ELBOW_PITCH_R",
  "J22_ELBOW_YAW_R",
  "J23_HEAD_YAW"
]

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_pm01_robot_cfg())

  viewer.launch(robot.spec.compile())
