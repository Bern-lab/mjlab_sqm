"""OLI constants.

This configuration is tailored for OLI whole-body tracking / dance training,
with actuator groups and gains chosen to stay close to the official deployment
controller conventions.
"""

from __future__ import annotations

import re
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

OLI_XML: Path = (
	MJLAB_SRC_PATH / "asset_zoo" / "robots" / "limx_oli" / "xmls" / "oli.xml"
)
assert OLI_XML.exists()


def _sanitize_xml_text(xml_text: str) -> str:
	"""Sanitize XML text for robust MuJoCo parsing.

	- Fix comma-separated quaternion attributes.
	- Remove XML-native actuator block to avoid duplicate actuators when using
		`EntityArticulationInfoCfg` (we inject our own actuators below).
	"""

	def _fix_quat(match: re.Match[str]) -> str:
		quat_value = match.group(1).replace(",", " ")
		quat_value = " ".join(quat_value.split())
		return f'quat="{quat_value}"'

	xml_text = re.sub(r'quat="([^"]+)"', _fix_quat, xml_text)
	xml_text = re.sub(r"<actuator>.*?</actuator>", "", xml_text, flags=re.S)
	return xml_text


def get_assets(meshdir: str) -> dict[str, bytes]:
	assets: dict[str, bytes] = {}
	assets_root = OLI_XML.parent / "assets" / "robot"
	prefix = meshdir.rstrip("/")

	# Keep subdirectory structure in asset keys (e.g. robot/base_link.STL)
	# because the XML references meshes with nested relative paths.
	for f in assets_root.rglob("*"):
		if not f.is_file():
			continue
		rel = f.relative_to(assets_root).as_posix()
		asset_key = f"{prefix}/{rel}" if prefix else rel
		assets[asset_key] = f.read_bytes()

	return assets


def get_spec() -> mujoco.MjSpec:
	xml_text = OLI_XML.read_text(encoding="utf-8")
	spec = mujoco.MjSpec.from_string(_sanitize_xml_text(xml_text))
	spec.assets = get_assets(spec.meshdir)
	return spec


##
# Actuator config.
##

# Armature values from OLI XML joints.
ARMATURE_LEG = 0.14125
ARMATURE_WAIST = 0.1845504
ARMATURE_ARM = 0.0886706
ARMATURE_WRIST_HEAD = 0.0153218

# Gains/limits are chosen to be compatible with official deployment style
# (walk/mimic controller groups), while keeping training stable.

OLI_ACTUATOR_LEG_HIP = BuiltinPositionActuatorCfg(
	target_names_expr=(".*_hip_(pitch|roll|yaw)_joint",),
	effort_limit=140.0,
	armature=ARMATURE_LEG,
	stiffness=139.41,
	damping=17.75,
)

OLI_ACTUATOR_LEG_KNEE = BuiltinPositionActuatorCfg(
	target_names_expr=(".*_knee_joint",),
	effort_limit=140.0,
	armature=ARMATURE_LEG,
	stiffness=139.41,
	damping=17.75,
)

OLI_ACTUATOR_LEG_ANKLE = BuiltinPositionActuatorCfg(
	target_names_expr=(".*_ankle_(pitch|roll)_joint",),
	effort_limit=80.0,
	armature=ARMATURE_LEG,
	stiffness=93.65,
	damping=11.92,
)

OLI_ACTUATOR_WAIST_YAW = BuiltinPositionActuatorCfg(
	target_names_expr=("waist_yaw_joint",),
	effort_limit=42.0,
	armature=ARMATURE_WAIST,
	stiffness=93.65,
	damping=11.92,
)

OLI_ACTUATOR_WAIST_ROLL_PITCH = BuiltinPositionActuatorCfg(
	target_names_expr=("waist_roll_joint", "waist_pitch_joint"),
	effort_limit=80.0,
	armature=ARMATURE_WAIST,
	stiffness=93.65,
	damping=11.92,
)

OLI_ACTUATOR_HEAD = BuiltinPositionActuatorCfg(
	target_names_expr=("head_pitch_joint", "head_yaw_joint"),
	effort_limit=19.0,
	armature=ARMATURE_WRIST_HEAD,
	stiffness=15.12,
	damping=1.93,
)

OLI_ACTUATOR_ARMS = BuiltinPositionActuatorCfg(
	target_names_expr=(".*_shoulder_(pitch|roll|yaw)_joint", ".*_elbow_joint"),
	effort_limit=42.0,
	armature=ARMATURE_ARM,
	stiffness=87.51,
	damping=11.14,
)

OLI_ACTUATOR_WRISTS = BuiltinPositionActuatorCfg(
	target_names_expr=(".*_wrist_(yaw|pitch|roll)_joint",),
	effort_limit=19.0,
	armature=ARMATURE_WRIST_HEAD,
	stiffness=15.12,
	damping=1.93,
)


##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
	pos=(0.0, 0.0, 1.0),
	joint_pos={
		# legs
		"left_hip_pitch_joint": -0.15,
		"left_hip_roll_joint": 0.0,
		"left_hip_yaw_joint": -0.05,
		"left_knee_joint": 0.30,
		"left_ankle_pitch_joint": -0.16,
		"left_ankle_roll_joint": 0.0,
		"right_hip_pitch_joint": -0.15,
		"right_hip_roll_joint": 0.0,
		"right_hip_yaw_joint": 0.05,
		"right_knee_joint": 0.30,
		"right_ankle_pitch_joint": -0.16,
		"right_ankle_roll_joint": 0.0,
		# waist
		"waist_yaw_joint": 0.0,
		"waist_roll_joint": 0.0,
		"waist_pitch_joint": 0.0,
		# head
		"head_pitch_joint": 0.0,
		"head_yaw_joint": 0.0,
		# arms
		"left_shoulder_pitch_joint": 0.10,
		"left_shoulder_roll_joint": 0.10,
		"left_shoulder_yaw_joint": -0.20,
		"left_elbow_joint": -0.20,
		"left_wrist_yaw_joint": 0.0,
		"left_wrist_pitch_joint": 0.0,
		"left_wrist_roll_joint": 0.0,
		"right_shoulder_pitch_joint": 0.10,
		"right_shoulder_roll_joint": -0.10,
		"right_shoulder_yaw_joint": 0.20,
		"right_elbow_joint": -0.20,
		"right_wrist_yaw_joint": 0.0,
		"right_wrist_pitch_joint": 0.0,
		"right_wrist_roll_joint": 0.0,
	},
	joint_vel={".*": 0.0},
)


KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
	pos=(0.0, 0.0, 0.92),
	joint_pos={
		"left_hip_pitch_joint": -0.30,
		"left_hip_roll_joint": 0.0,
		"left_hip_yaw_joint": -0.05,
		"left_knee_joint": 0.66,
		"left_ankle_pitch_joint": -0.36,
		"left_ankle_roll_joint": 0.0,
		"right_hip_pitch_joint": -0.30,
		"right_hip_roll_joint": 0.0,
		"right_hip_yaw_joint": 0.05,
		"right_knee_joint": 0.66,
		"right_ankle_pitch_joint": -0.36,
		"right_ankle_roll_joint": 0.0,
		"waist_yaw_joint": 0.0,
		"waist_roll_joint": 0.0,
		"waist_pitch_joint": 0.0,
		"head_pitch_joint": 0.0,
		"head_yaw_joint": 0.0,
		"left_shoulder_pitch_joint": 0.10,
		"left_shoulder_roll_joint": 0.10,
		"left_shoulder_yaw_joint": -0.20,
		"left_elbow_joint": -0.20,
		"left_wrist_yaw_joint": 0.0,
		"left_wrist_pitch_joint": 0.0,
		"left_wrist_roll_joint": 0.0,
		"right_shoulder_pitch_joint": 0.10,
		"right_shoulder_roll_joint": -0.10,
		"right_shoulder_yaw_joint": 0.20,
		"right_elbow_joint": -0.20,
		"right_wrist_yaw_joint": 0.0,
		"right_wrist_pitch_joint": 0.0,
		"right_wrist_roll_joint": 0.0,
	},
	joint_vel={".*": 0.0},
)


##
# Collision config.
##

FULL_COLLISION = CollisionCfg(
	geom_names_expr=(r".*",),
	condim={r"^(left|right)_foot([1-7]_collision)?$": 3, r".*": 1},
	priority={r"^(left|right)_foot([1-7]_collision)?$": 1},
	friction={r"^(left|right)_foot([1-7]_collision)?$": (0.8,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
	geom_names_expr=(r".*",),
	contype=0,
	conaffinity=1,
	condim={r"^(left|right)_foot([1-7]_collision)?$": 3, r".*": 1},
	priority={r"^(left|right)_foot([1-7]_collision)?$": 1},
	friction={r"^(left|right)_foot([1-7]_collision)?$": (0.8,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
	geom_names_expr=(r"^(left|right)_foot([1-7]_collision)?$",),
	contype=0,
	conaffinity=1,
	condim=3,
	priority=1,
	friction=(0.8,),
)


##
# Final config.
##

OLI_ARTICULATION = EntityArticulationInfoCfg(
	actuators=(
		OLI_ACTUATOR_LEG_HIP,
		OLI_ACTUATOR_LEG_KNEE,
		OLI_ACTUATOR_LEG_ANKLE,
		OLI_ACTUATOR_WAIST_YAW,
		OLI_ACTUATOR_WAIST_ROLL_PITCH,
		OLI_ACTUATOR_HEAD,
		OLI_ACTUATOR_ARMS,
		OLI_ACTUATOR_WRISTS,
	),
	soft_joint_pos_limit_factor=0.9,
)

def get_oli_robot_cfg() -> EntityCfg:
  """Get a fresh G1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=OLI_ARTICULATION,
  )


OLI_ACTION_SCALE: dict[str, float] = {}
for a in OLI_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    OLI_ACTION_SCALE[n] = 0.25 * e / s


# OLI deployment joint names (31 DoF)
OLI_JOINTS = [
	"left_hip_pitch_joint",
	"left_hip_roll_joint",
	"left_hip_yaw_joint",
	"left_knee_joint",
	"left_ankle_pitch_joint",
	"left_ankle_roll_joint",
	"right_hip_pitch_joint",
	"right_hip_roll_joint",
	"right_hip_yaw_joint",
	"right_knee_joint",
	"right_ankle_pitch_joint",
	"right_ankle_roll_joint",
	"waist_yaw_joint",
	"waist_roll_joint",
	"waist_pitch_joint",
	"head_pitch_joint",
	"head_yaw_joint",
	"left_shoulder_pitch_joint",
	"left_shoulder_roll_joint",
	"left_shoulder_yaw_joint",
	"left_elbow_joint",
	"left_wrist_yaw_joint",
	"left_wrist_pitch_joint",
	"left_wrist_roll_joint",
	"right_shoulder_pitch_joint",
	"right_shoulder_roll_joint",
	"right_shoulder_yaw_joint",
	"right_elbow_joint",
	"right_wrist_yaw_joint",
	"right_wrist_pitch_joint",
	"right_wrist_roll_joint",
]


if __name__ == "__main__":
	import mujoco.viewer as viewer

	from mjlab.entity.entity import Entity

	robot = Entity(get_oli_robot_cfg())

	viewer.launch(robot.spec.compile())

