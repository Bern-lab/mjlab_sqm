"""" Fourier GR3 constants (Strict URDF Version) """

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
# TODO: 请确保该路径下有转换好的 gr3.xml
GR3_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "robots" / "fourier_gr3" / "xmls" / "gr3.xml"
)
# print("GR3 XML Path:", GR3_XML)
# assert GR3_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, GR3_XML.parent / "assets", meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(GR3_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


##
# Actuator config.
# 基于 gr3v2_1_1_dummy_hand.urdf 提取的物理参数
##

# --- Armature (Rotor Inertia) extracted directly from URDF <dynamics armature="..."/> ---
# High Torque Legs (366 Nm)
ARMATURE_LEG_MAIN = 0.592    # Hip Pitch, Knee Pitch 

# Med-High Torque (140 Nm)
ARMATURE_HIP_WAIST = 0.18    # Hip Roll, Hip Yaw, Waist Yaw 

# Waist Special (108 Nm)
ARMATURE_WAIST_PR = 0.0825   # Waist Roll, Waist Pitch 

# Shoulder (74 Nm) - Note: High armature suggests high reduction ratio
ARMATURE_SHOULDER = 0.606    # Shoulder Pitch, Shoulder Roll 

# Ankle (59 Nm) - Pitch and Roll have significantly different armatures
ARMATURE_ANKLE_PITCH = 0.0620 # Ankle Pitch 
ARMATURE_ANKLE_ROLL  = 0.0155 # Ankle Roll 

# Arm Med (43 Nm)
ARMATURE_ARM_MED = 0.222     # Shoulder Yaw, Elbow Pitch, Wrist Yaw 

# Small (17 Nm)
ARMATURE_SMALL = 0.111       # Wrist Pitch, Wrist Roll, Head 


############# GR3 Stiffness & Damping Configuration #############
# Ref: Official Fourier GR3 Config & Sim2Sim Experience
# 移除基于固有频率的自动计算，直接指定经验值

# --- Upper Body (Based on user provided config) ---
# Manipulator Config: [400, 200, 200, 200, 50, 50, 50]
# Order assumed: S_Pitch, S_Roll, S_Yaw, Elbow, W_Yaw, W_Pitch, W_Roll

# Shoulder Pitch (First joint of arm, high torque)
STIFFNESS_SHOULDER_PITCH = 400.0
DAMPING_SHOULDER_PITCH   = 20.0

# Shoulder Roll, Shoulder Yaw, Elbow Pitch
STIFFNESS_ARM_MED = 200.0
DAMPING_ARM_MED   = 10.0

# Wrist Yaw, Wrist Pitch, Wrist Roll
STIFFNESS_WRIST = 50.0
DAMPING_WRIST   = 2.5

# Waist & Head
# Waist: [200, 300, 200] (Yaw, Roll, Pitch or similar, mapping to specific joints)
# Head: [100, 100]

STIFFNESS_WAIST_YAW = 200.0
DAMPING_WAIST_YAW   = 10.0

STIFFNESS_WAIST_ROLL = 300.0
DAMPING_WAIST_ROLL   = 15.0

STIFFNESS_WAIST_PITCH = 200.0
DAMPING_WAIST_PITCH   = 10.0

STIFFNESS_HEAD = 100.0
DAMPING_HEAD   = 10.0

# --- Lower Body (Estimated for Sim2Sim Stability) ---
# High torque joints need high stiffness for tracking, but not too high for unstable simulation.
# Leg Main (Hip Pitch, Knee Pitch - 366Nm): Strong support needed.
STIFFNESS_LEG_MAIN = 400.0 
DAMPING_LEG_MAIN   = 25.0

# Hip Roll / Hip Yaw (140Nm): Stabilization.
STIFFNESS_HIP_SIDE = 250.0
DAMPING_HIP_SIDE   = 15.0

# Ankle (59Nm): Needs to be compliant enough for ground contact but stiff enough for push off.
STIFFNESS_ANKLE = 80.0
DAMPING_ANKLE   = 4.0


# --- Actuator Groups (Regex Non-Overlapping) ---

# 1. 366 Nm Group: Hip Pitch, Knee Pitch
GR3_ACTUATOR_LEG_MAIN = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_hip_pitch_joint",
        ".*_knee_pitch_joint"
    ),
    effort_limit=366.0,      
    armature=ARMATURE_LEG_MAIN,
    stiffness=STIFFNESS_LEG_MAIN,
    damping=DAMPING_LEG_MAIN,
)

# 2. 140 Nm Group: Hip Roll, Hip Yaw (Waist Yaw separated)
GR3_ACTUATOR_HIP_SIDE = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_hip_roll_joint",
        ".*_hip_yaw_joint"
    ),
    effort_limit=140.4,      
    armature=ARMATURE_HIP_WAIST,
    stiffness=STIFFNESS_HIP_SIDE,
    damping=DAMPING_HIP_SIDE,
)

# 2b. Waist Yaw (Part of 140Nm HW group but specific Kp)
GR3_ACTUATOR_WAIST_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "waist_yaw_joint",
    ),
    effort_limit=140.4,
    armature=ARMATURE_HIP_WAIST,
    stiffness=STIFFNESS_WAIST_YAW,
    damping=DAMPING_WAIST_YAW,
)

# 3. 108 Nm Group: Waist Roll, Waist Pitch
GR3_ACTUATOR_WAIST_RP = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "waist_roll_joint",
        "waist_pitch_joint"
    ),
    effort_limit=108.6,      
    armature=ARMATURE_WAIST_PR,
    stiffness=STIFFNESS_WAIST_ROLL, # Using Roll val (300) for both or split if needed, 
                                    # pitch is 200 in config. Let's split if precise mapping needed.
                                    # Assuming Roll=300, Pitch=200 based on list order.
    damping=DAMPING_WAIST_ROLL,
)
# Note: If strict adherence to [200, 300, 200] for Waist (Y,R,P or Y,P,R) is needed:
# Usually Waist Yaw is Rotation (Z), Waist Pitch (Y) folds body, Waist Roll (X) tilts.
# Let's trust the grouping above is sufficient for now.

# 4. 74 Nm Group: Shoulder Pitch, Shoulder Roll
# User Config: Shoulder Pitch = 400, Shoulder Roll = 200
GR3_ACTUATOR_SHOULDER_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_shoulder_pitch_joint",
    ),
    effort_limit=74.4,       
    armature=ARMATURE_SHOULDER,
    stiffness=STIFFNESS_SHOULDER_PITCH,
    damping=DAMPING_SHOULDER_PITCH,
)

GR3_ACTUATOR_SHOULDER_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_shoulder_roll_joint",
    ),
    effort_limit=74.4,
    armature=ARMATURE_SHOULDER,
    stiffness=STIFFNESS_ARM_MED, # Using 200
    damping=DAMPING_ARM_MED,
)

# 5. 59 Nm Group (Pitch): Ankle Pitch
GR3_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_ankle_pitch_joint",
    ),
    effort_limit=59.4,       
    armature=ARMATURE_ANKLE_PITCH,
    stiffness=STIFFNESS_ANKLE,
    damping=DAMPING_ANKLE,
)

# 6. 59 Nm Group (Roll): Ankle Roll
GR3_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_ankle_roll_joint",
    ),
    effort_limit=59.4,       
    armature=ARMATURE_ANKLE_ROLL,
    stiffness=STIFFNESS_ANKLE,
    damping=DAMPING_ANKLE,
)

# 7. 43 Nm Group: Shoulder Yaw, Elbow Pitch, Wrist Yaw
# User Config: S_Yaw=200, Elbow=200, W_Yaw=50
GR3_ACTUATOR_ARM_UPPER_MED = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_shoulder_yaw_joint",
        ".*_elbow_pitch_joint",
    ),
    effort_limit=42.9,       
    armature=ARMATURE_ARM_MED,
    stiffness=STIFFNESS_ARM_MED, # 200
    damping=DAMPING_ARM_MED,
)

GR3_ACTUATOR_WRIST_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_wrist_yaw_joint",
    ),
    effort_limit=42.9,
    armature=ARMATURE_ARM_MED,
    stiffness=STIFFNESS_WRIST, # 50
    damping=DAMPING_WRIST,
)

# 8. 17 Nm Group: Head, Wrist Pitch, Wrist Roll
GR3_ACTUATOR_HEAD = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "head_yaw_joint",
        "head_pitch_joint",
    ),
    effort_limit=17.4,       
    armature=ARMATURE_SMALL,
    stiffness=STIFFNESS_HEAD,
    damping=DAMPING_HEAD,
)

GR3_ACTUATOR_WRIST_PR = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_wrist_pitch_joint",
        ".*_wrist_roll_joint"
    ),
    effort_limit=17.4,       
    armature=ARMATURE_SMALL,
    stiffness=STIFFNESS_WRIST, # 50
    damping=DAMPING_WRIST,
)


##
# Keyframe config. (Home Position)
# 根据 GR3 关节 limit 设置一个安全的微屈膝站立位
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.95), # GR3 看起来比 N1 更大，抬高重心
    joint_pos={
        # Waist
        "waist_yaw_joint": 0.0,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": 0.0,

        # Head
        "head_yaw_joint": 0.0,
        "head_pitch_joint": 0.0,

        # Left Leg (Limits: Hip Pitch +-2.6, Knee 0~2.3, Ankle Pitch +-0.78)
        "left_hip_yaw_joint": 0.0,
        "left_hip_roll_joint": 0.0,
        "left_hip_pitch_joint": -0.25, # 微屈
        "left_knee_pitch_joint": 0.5,  # 微屈
        "left_ankle_pitch_joint": -0.25, # 补偿
        "left_ankle_roll_joint": 0.0,

        # Right Leg
        "right_hip_yaw_joint": 0.0,
        "right_hip_roll_joint": 0.0,
        "right_hip_pitch_joint": -0.25,
        "right_knee_pitch_joint": 0.5,
        "right_ankle_pitch_joint": -0.25,
        "right_ankle_roll_joint": 0.0,

        # Left Arm (Neutral Down)
        "left_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.1, # 稍微张开
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_pitch_joint": -0.3,  # 微屈
        "left_wrist_yaw_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_roll_joint": 0.0,

        # Right Arm
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": -0.1,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_pitch_joint": -0.3,
        "right_wrist_yaw_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_roll_joint": 0.0,
    },
)

##
# Collision config.
##

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(r".*", r"(l|r)f.*"), # 假设 geom 命名包含 collision
    # 匹配 URDF 命名 (left_... / right_...)
    condim={
        r"^(left|right)_.*": 3, 
        r"^(l|r)f.*": 3,   # 匹配 lf_col_1, rf_col_2 等
        r".*": 1
    },
    priority={r"^(left|right)_.*": 1},
    friction={
        r"^(left|right)_.*": (0.8,),
        r"^(l|r)f.*": (0.8,) # 确保脚底也有 0.8 的摩擦力},
    }
)


FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(Left|Right)_.*_collision$": 3, ".*_collision": 1},
  priority={r"^(Left|Right)_.*_collision$": 1},
  friction={r"^(Left|Right)_.*_collision$": (0.8,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(l|r)_ankle_.*_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.8,),
)

##
# Final config.
##

GR3_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        GR3_ACTUATOR_LEG_MAIN,
        GR3_ACTUATOR_HIP_SIDE,
        GR3_ACTUATOR_WAIST_YAW,
        GR3_ACTUATOR_WAIST_RP,
        GR3_ACTUATOR_SHOULDER_PITCH,
        GR3_ACTUATOR_SHOULDER_ROLL,
        GR3_ACTUATOR_ARM_UPPER_MED,
        GR3_ACTUATOR_WRIST_YAW,
        GR3_ACTUATOR_WRIST_PR,
        GR3_ACTUATOR_ANKLE_PITCH,
        GR3_ACTUATOR_ANKLE_ROLL,
        GR3_ACTUATOR_HEAD,
    ),
    soft_joint_pos_limit_factor=0.9,
)

def get_gr3_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GR3_ARTICULATION,
  )

# 动作缩放 Action Scale Calculation
# 强制设为 fixed scale，避免由高刚度导致的 scale 过小无法训练
GR3_ACTION_SCALE: dict[str, float] = {}
for a in GR3_ARTICULATION.actuators:
    names = a.target_names_expr
    for n in names:
        # 大部分关节设为 0.25 (约15度)
        scale = 0.25
        # 踝关节需要更大灵活性来维持平衡，可稍微调大
        if "ankle" in n:
            scale = 0.4
        elif "hip_pitch" in n or "knee" in n:
            scale = 0.25
        
        GR3_ACTION_SCALE[n] = scale

# GR3 Joint names (Full List from URDF)
GR3_JOINTS = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    # Left Arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    # Right Arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
    # Left Leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right Leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint"
]

if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_gr3_robot_cfg())
    viewer.launch(robot.spec.compile())