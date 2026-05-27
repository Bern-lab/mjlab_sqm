"""Fixed terrain sets and runtime evaluation overrides for velocity tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains.config import flat, pyramid_stairs, pyramid_stairs_inv, random_rough
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg

TerrainKind = Literal["flat", "rough", "upstairs", "downstairs"]


@dataclass(frozen=True)
class EvalTerrainSpec:
  """One fixed terrain used by offline evaluation."""

  name: str
  label: str
  kind: TerrainKind
  height_m: float | None = None
  step_width: float = 0.30
  platform_width: float = 3.0
  terrain_size: tuple[float, float] = (8.0, 8.0)
  border_width: float = 20.0
  stair_riser_levels: int = 10
  stair_border_width: float = 1.0

  def generator_size(self) -> tuple[float, float]:
    """Return the terrain-generator tile size for this fixed terrain."""
    if self.kind not in ("upstairs", "downstairs"):
      return self.terrain_size

    # BoxPyramidStairsTerrainCfg derives num_steps from size. It creates
    # num_steps + 1 riser boundaries, so use one fewer stair ring to expose the
    # requested number of riser levels for by-level collision statistics.
    stair_rings = max(0, self.stair_riser_levels - 1)
    side = (
      2.0 * self.stair_border_width
      + self.platform_width
      + 2.0 * self.step_width * stair_rings
      + 1e-6
    )
    return (side, side)

  def make_subterrain(self) -> SubTerrainCfg:
    """Build a single fixed sub-terrain config."""
    if self.kind == "flat":
      return flat(proportion=1.0)
    if self.kind == "rough":
      return random_rough(
        proportion=1.0,
        noise_range=(0.04, 0.04),
        noise_step=0.01,
        border_width=0.5,
      )

    if self.height_m is None:
      raise ValueError(f"Terrain '{self.name}' needs height_m.")

    common = dict(
      proportion=1.0,
      step_height_range=(self.height_m, self.height_m),
      step_width=self.step_width,
      platform_width=self.platform_width,
      border_width=self.stair_border_width,
    )
    if self.kind == "upstairs":
      # The inverted pyramid starts low at the center, so walking out from the
      # spawn origin encounters ascending stairs.
      return pyramid_stairs_inv(**common)
    if self.kind == "downstairs":
      # The standard pyramid starts high at the center, so walking out from the
      # spawn origin encounters descending stairs.
      return pyramid_stairs(**common)

    raise ValueError(f"Unsupported terrain kind: {self.kind!r}")


EVAL_TERRAIN_SETS: dict[str, tuple[EvalTerrainSpec, ...]] = {#地形类型
  "eval_v1": (
    EvalTerrainSpec(name="flat", label="flat", kind="flat"),
    EvalTerrainSpec(
      name="upstairs_10cm", label="stair_up_10", kind="upstairs", height_m=0.10
    ),
    EvalTerrainSpec(
      name="upstairs_15cm", label="stair_up_15", kind="upstairs", height_m=0.15
    ),
    EvalTerrainSpec(
      name="upstairs_20cm", label="stair_up_20", kind="upstairs", height_m=0.20
    ),
  ),
  "stairs_both_v1": (
    EvalTerrainSpec(name="flat", label="flat", kind="flat"),
    EvalTerrainSpec(
      name="upstairs_10cm", label="stair_up_10", kind="upstairs", height_m=0.10
    ),
    EvalTerrainSpec(
      name="upstairs_15cm", label="stair_up_15", kind="upstairs", height_m=0.15
    ),
    EvalTerrainSpec(
      name="upstairs_20cm", label="stair_up_20", kind="upstairs", height_m=0.20
    ),
    EvalTerrainSpec(
      name="downstairs_10cm",
      label="stair_down_10",
      kind="downstairs",
      height_m=0.10,
    ),
    EvalTerrainSpec(
      name="downstairs_15cm",
      label="stair_down_15",
      kind="downstairs",
      height_m=0.15,
    ),
    EvalTerrainSpec(
      name="downstairs_20cm",
      label="stair_down_20",
      kind="downstairs",
      height_m=0.20,
    ),
  ),
  "cluster_v1": (
    EvalTerrainSpec(name="flat", label="flat", kind="flat"),
    EvalTerrainSpec(name="rough", label="rough", kind="rough"),
    EvalTerrainSpec(
      name="upstairs_10cm", label="stair_up_10", kind="upstairs", height_m=0.10
    ),
    EvalTerrainSpec(
      name="upstairs_15cm", label="stair_up_15", kind="upstairs", height_m=0.15
    ),
    EvalTerrainSpec(
      name="upstairs_20cm", label="stair_up_20", kind="upstairs", height_m=0.20
    ),
    EvalTerrainSpec(
      name="downstairs_10cm",
      label="stair_down_10",
      kind="downstairs",
      height_m=0.10,
    ),
  ),
}


def get_terrain_set(name: str) -> tuple[EvalTerrainSpec, ...]:
  """Return a named fixed evaluation terrain set."""
  try:
    return EVAL_TERRAIN_SETS[name]
  except KeyError as exc:
    names = ", ".join(sorted(EVAL_TERRAIN_SETS))
    raise ValueError(f"Unknown terrain set {name!r}. Available: {names}") from exc


def make_fixed_terrain_generator(
  terrain: EvalTerrainSpec,
  *,
  num_envs: int,
  seed: int,
) -> TerrainGeneratorCfg:
  """Create a deterministic terrain generator with one fixed terrain type.

  Random allocation mode with one row and many columns gives each environment a
  distinct tile along the y axis while keeping the terrain type identical.
  """
  return TerrainGeneratorCfg(
    seed=seed,
    curriculum=False,
    size=terrain.generator_size(),
    border_width=terrain.border_width,
    num_rows=1,
    num_cols=max(1, num_envs),
    difficulty_range=(0.0, 0.0),
    sub_terrains={terrain.name: terrain.make_subterrain()},
    add_lights=True,
  )


def _disable_observation_delays(cfg: ManagerBasedRlEnvCfg) -> None:
  for group_cfg in cfg.observations.values():
    for term_cfg in group_cfg.terms.values():
      term_cfg.delay_min_lag = 0
      term_cfg.delay_max_lag = 0
      term_cfg.delay_hold_prob = 0.0
      term_cfg.delay_update_period = 0


def _disable_actuator_delays(cfg: ManagerBasedRlEnvCfg) -> None:
  robot_cfg = cfg.scene.entities.get("robot")
  if robot_cfg is None or robot_cfg.articulation is None:
    return
  for actuator_cfg in robot_cfg.articulation.actuators:
    for attr, value in (
      ("delay_min_lag", 0),
      ("delay_max_lag", 0),
      ("delay_hold_prob", 0.0),
      ("delay_update_period", 0),
    ):
      if hasattr(actuator_cfg, attr):
        setattr(actuator_cfg, attr, value)


def _fix_reset_events(cfg: ManagerBasedRlEnvCfg) -> None:
  kept_events = {}
  for name in ("reset_scene_to_default", "reset_base", "reset_robot_joints"):
    if name in cfg.events:
      kept_events[name] = cfg.events[name]
  cfg.events = kept_events

  reset_base = cfg.events.get("reset_base")
  if reset_base is not None:
    reset_base.params["pose_range"] = {
      "x": (0.0, 0.0),
      "y": (0.0, 0.0),
      "z": (0.03, 0.03),
      "yaw": (0.0, 0.0),
    }
    reset_base.params["velocity_range"] = {}

  reset_joints = cfg.events.get("reset_robot_joints")
  if reset_joints is not None:
    reset_joints.params["position_range"] = (0.0, 0.0)
    reset_joints.params["velocity_range"] = (0.0, 0.0)


def _fix_velocity_command(
  cfg: ManagerBasedRlEnvCfg,
  *,
  command: tuple[float, float, float],
  max_episode_length_s: float,
) -> None:
  old_twist_cmd = cfg.commands.get("twist")
  if not isinstance(old_twist_cmd, UniformVelocityCommandCfg):
    raise TypeError(
      "Velocity eval currently expects a UniformVelocityCommandCfg named 'twist'."
    )

  vx, vy, wz = command
  twist_cmd = UniformVelocityCommandCfg(
    entity_name=old_twist_cmd.entity_name,
    heading_command=False,
    rel_standing_envs=0.0,
    rel_heading_envs=0.0,
    rel_world_envs=0.0,
    rel_forward_envs=0.0,
    init_velocity_prob=0.0,
    resampling_time_range=(max_episode_length_s + 1.0, max_episode_length_s + 1.0),
    debug_vis=old_twist_cmd.debug_vis,
    heading_control_stiffness=old_twist_cmd.heading_control_stiffness,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(vx, vx),
      lin_vel_y=(vy, vy),
      ang_vel_z=(wz, wz),
      heading=None,
    ),
  )
  twist_cmd.viz.z_offset = old_twist_cmd.viz.z_offset
  twist_cmd.viz.scale = old_twist_cmd.viz.scale
  cfg.commands["twist"] = twist_cmd


def _enable_eval_riser_contact_sensor(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add true foot-vs-terrain contact data for eval-only collision metrics."""
  try:
    from mjlab.tasks.velocity.config.g1.blind_rough_toe_contact_cfg import (
      add_g1_toe_terrain_contact_sensor,
    )
  except ImportError:
    return

  cfg.sim.contact_sensor_maxmatch = max(
    int(getattr(cfg.sim, "contact_sensor_maxmatch", 0) or 0), 256
  )
  cfg.sim.nconmax = max(int(getattr(cfg.sim, "nconmax", 0) or 0), 256)
  add_g1_toe_terrain_contact_sensor(cfg)


def apply_eval_overrides(
  cfg: ManagerBasedRlEnvCfg,
  terrain: EvalTerrainSpec,
  *,
  num_envs: int,
  seed: int,
  max_episode_length_s: float = 20.0,
  command: tuple[float, float, float] = (0.4, 0.0, 0.0),
  clean_observations: bool = True,
  disable_observation_delay: bool = True,
  disable_actuator_delay: bool = True,
  enable_riser_contact_sensor: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Apply deterministic offline-eval overrides to a copied env config."""
  cfg.seed = seed
  cfg.scene.num_envs = num_envs
  cfg.episode_length_s = max_episode_length_s

  if cfg.scene.terrain is None:
    raise ValueError("Velocity eval requires an environment with terrain enabled.")
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = make_fixed_terrain_generator(
    terrain, num_envs=num_envs, seed=seed
  )
  cfg.scene.terrain.max_init_terrain_level = 0

  cfg.curriculum = {}
  _fix_reset_events(cfg)
  _fix_velocity_command(
    cfg, command=command, max_episode_length_s=max_episode_length_s
  )
  if enable_riser_contact_sensor:
    _enable_eval_riser_contact_sensor(cfg)

  if clean_observations:
    for group_cfg in cfg.observations.values():
      group_cfg.enable_corruption = False
  if disable_observation_delay:
    _disable_observation_delays(cfg)
  if disable_actuator_delay:
    _disable_actuator_delays(cfg)

  return cfg
