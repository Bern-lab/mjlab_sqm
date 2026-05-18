from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Literal

import mujoco
import numpy as np

from mjlab.terrains.utils import make_border
from mjlab.utils.color import RGBA

_DARK_GRAY = (0.2, 0.2, 0.2, 1.0)


@dataclass
class FlatPatchSamplingCfg:
  """Configuration for sampling flat patches on a heightfield surface."""

  num_patches: int = 10
  """Number of flat patches to sample per sub-terrain."""
  patch_radius: float = 0.3
  """Radius of the circular footprint used to test flatness, in meters."""
  max_height_diff: float = 0.05
  """Maximum allowed height variation within the patch footprint, in meters."""
  x_range: tuple[float, float] = (-1e6, 1e6)
  """Allowed range of x coordinates for sampled patches, in meters."""
  y_range: tuple[float, float] = (-1e6, 1e6)
  """Allowed range of y coordinates for sampled patches, in meters."""
  z_range: tuple[float, float] = (-1e6, 1e6)
  """Allowed range of z coordinates (world height) for sampled patches, in meters."""
  grid_resolution: float | None = None
  """Resolution of the grid used for flat-patch detection, in meters. When
  ``None`` (default), the terrain's own ``horizontal_scale`` is used. Set to a
  smaller value (e.g. 0.025) for finer boundary precision at the cost of a
  larger intermediate grid."""


@dataclass
class TerrainGeometry:
  geom: mujoco.MjsGeom | None = None
  """MuJoCo geometry spec element, or None."""
  hfield: mujoco.MjsHField | None = None
  """MuJoCo heightfield spec element, or None."""
  color: tuple[float, float, float, float] | None = None
  """RGBA color override for this geometry, or None to use default."""


@dataclass
class TerrainOutput:
  origin: np.ndarray
  """Spawn origin position (x, y, z) in the sub-terrain's local frame."""
  geometries: list[TerrainGeometry]
  """List of geometry elements comprising this terrain."""
  flat_patches: dict[str, np.ndarray] | None = None#关联平坦点，保存找到的点位置，key是点的名字，value是一个(N, 3)的数组，包含N个平坦点的世界坐标。如果某个地形没有生成平坦点，则对应的值为None。
  """Named sets of flat patch positions, each an (N, 3) array. None if not configured."""
  step_boundaries: np.ndarray | None = None
  """Height-discontinuity boundaries as ``[p0_high, p1_high, normal_to_low, z_low, z_high]`` rows."""


@dataclass
class StepDangerVisualizationCfg:
  """Visualization-only geometry for step lip/riser danger zones."""

  enabled: bool = False
  """If True, add non-colliding translucent geoms for step danger zones."""
  lip_radius: float = 0.05
  """Radius of the high-side lip tube, matching the lip reward radius."""
  slab_depth: float = 0.04
  """Depth of the toe-only riser slab extending from the riser to the low side."""
  slab_u_margin: float = 0.02
  """Extra half-width along the step edge for the riser slab visual."""
  slab_v_margin: float = 0.02
  """Extra half-height along the riser for the slab visual."""
  geom_group: int = 4
  """MuJoCo visualization group for danger-zone geoms."""
  lip_rgba: RGBA = (1.0, 0.10, 0.05, 0.35)
  """RGBA color for lip edge tubes."""
  slab_rgba: RGBA = (1.0, 0.75, 0.05, 0.22)
  """RGBA color for riser slabs."""


@dataclass
class SubTerrainCfg(abc.ABC):
  proportion: float = 1.0
  """Robot spawning weight for this terrain type.

  In curriculum mode, controls how many robots are spawned on this terrain's
  column relative to other terrain types. Each terrain type always gets
  exactly one column; proportion only affects spawning distribution.

  In random mode, controls the sampling probability for each patch.
  """
  size: tuple[float, float] = (10.0, 10.0)
  """Width and length of the terrain patch, in meters."""
  flat_patch_sampling: dict[str, FlatPatchSamplingCfg] | None = None
  """Named flat-patch sampling configurations, or None to disable."""

  @abc.abstractmethod
  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    """Generate terrain geometry.

    Returns:
      TerrainOutput containing spawn origin and list of geometries.
    """
    raise NotImplementedError


@dataclass(kw_only=True)
class TerrainGeneratorCfg:
  seed: int | None = None
  """Random seed for terrain generation. None uses a random seed."""
  curriculum: bool = False
  """Controls terrain allocation mode:

  - curriculum=True: Each terrain type gets exactly ONE column. The generator uses
    ``len(sub_terrains)`` columns regardless of ``num_cols``. Difficulty increases
    along rows. The ``proportion`` field controls how many robots are spawned per
    column, not column count.

  - curriculum=False: Every patch is randomly sampled from all terrain types.
    Proportions control sampling probability. Use this for random variety.
  """
  size: tuple[float, float]
  """Width and length of each sub-terrain patch, in meters."""
  border_width: float = 0.0
  """Width of the flat border around the entire terrain grid, in meters."""
  border_height: float = 1.0
  """Height of the border wall around the terrain grid, in meters."""
  num_rows: int = 1
  """Number of sub-terrain rows in the grid. Represents difficulty levels in
  curriculum mode. Note: Environments are randomly assigned to rows, so multiple
  envs can share the same patch."""
  num_cols: int = 1
  """Number of sub-terrain columns in the grid.

  In curriculum mode the generator ignores this value and uses one column per terrain
  type (``len(sub_terrains)``). In random mode it is used as-is."""
  color_scheme: Literal["height", "random", "none"] = "height"
  """Coloring strategy for terrain geometry. "height" colors by elevation,
  "random" assigns random colors, "none" uses uniform gray."""
  sub_terrains: dict[str, SubTerrainCfg] = field(default_factory=dict)
  """Named sub-terrain configurations to populate the grid."""
  difficulty_range: tuple[float, float] = (0.0, 1.0)
  """Min and max difficulty values used when generating sub-terrains."""
  add_lights: bool = False
  """If True, adds a directional light above the terrain grid."""
  step_danger_visualization: StepDangerVisualizationCfg = field(
    default_factory=StepDangerVisualizationCfg
  )
  """Optional non-colliding visual geoms for step lip/riser danger zones."""


class TerrainGenerator:
  """Generates procedural terrain grids with configurable difficulty.

  Creates a grid of terrain patches where each patch can be a different
  terrain type. Supports two modes:

  - **Random mode** (curriculum=False): Every patch independently samples a
    terrain type weighted by proportions. Results in random variety across
    all patches.

  - **Curriculum mode** (curriculum=True): Each terrain type gets exactly one column
    (the generator uses ``len(sub_terrains)`` columns regardless of ``num_cols``).
    Difficulty increases along rows. The ``proportion`` field controls robot spawning
    distribution, not column count.

  Terrain types are weighted by proportion and their geometry is generated
  based on a difficulty value in the configured range. The grid is centered
  at the world origin. A border can be added around the entire grid along with
  optional overhead lighting.
  """

  def __init__(self, cfg: TerrainGeneratorCfg, device: str = "cpu") -> None:
    if len(cfg.sub_terrains) == 0:
      raise ValueError("At least one sub_terrain must be specified.")

    self.cfg = cfg
    self.device = device

    # In curriculum mode, one column per terrain type.
    if self.cfg.curriculum:
      self._num_cols = len(self.cfg.sub_terrains)
    else:
      self._num_cols = self.cfg.num_cols

    for sub_cfg in self.cfg.sub_terrains.values():
      sub_cfg.size = self.cfg.size

    if self.cfg.seed is not None:
      seed = self.cfg.seed
    else:
      seed = np.random.randint(0, 10000)
    self.np_rng = np.random.default_rng(seed)

    self.terrain_origins = np.zeros((self.cfg.num_rows, self._num_cols, 3))
    self._step_boundaries_tiles: list[list[np.ndarray]] = [
      [
        np.zeros((0, 11), dtype=np.float32)
        for _ in range(self._num_cols)
      ]
      for _ in range(self.cfg.num_rows)
    ]
    self.step_boundaries_by_tile = np.zeros(
      (self.cfg.num_rows, self._num_cols, 0, 11), dtype=np.float32
    )
    self.step_boundary_counts = np.zeros(
      (self.cfg.num_rows, self._num_cols), dtype=np.int32
    )

    # Pre-allocate flat patch storage by scanning all sub-terrain configs.
    self.flat_patches: dict[str, np.ndarray] = {}
    self.flat_patch_radii: dict[str, float] = {}
    patch_names: dict[str, int] = {}
    for sub_cfg in self.cfg.sub_terrains.values():
      if sub_cfg.flat_patch_sampling is not None:
        for name, patch_cfg in sub_cfg.flat_patch_sampling.items():
          if name in patch_names:
            patch_names[name] = max(patch_names[name], patch_cfg.num_patches)
          else:
            patch_names[name] = patch_cfg.num_patches
          self.flat_patch_radii[name] = max(
            self.flat_patch_radii.get(name, 0.0), patch_cfg.patch_radius
          )
    for name, max_num_patches in patch_names.items():
      self.flat_patches[name] = np.zeros(
        (self.cfg.num_rows, self._num_cols, max_num_patches, 3)
      )

  def compile(self, spec: mujoco.MjSpec) -> None:
    body = spec.worldbody.add_body(name="terrain")

    if self.cfg.curriculum:
      tic = time.perf_counter()
      self._generate_curriculum_terrains(spec)
      toc = time.perf_counter()
      print(f"Curriculum terrain generation took {toc - tic:.4f} seconds.")

    else:
      tic = time.perf_counter()
      self._generate_random_terrains(spec)
      toc = time.perf_counter()
      print(f"Terrain generation took {toc - tic:.4f} seconds.")

    self._finalize_step_boundaries()
    self._add_terrain_border(spec)
    self._add_grid_lights(spec)

    counter = 0
    for geom in body.geoms:
      geom.name = f"terrain_{counter}"
      # Terrain is static (no joints), so body mass is physically meaningless.
      # Without this, the thousands of dense geoms give the terrain body millions of kg
      # of mass, which inflates stat.meanmass and makes MuJoCo's force arrow
      # visualization invisible (arrows scale as force / meanmass).
      geom.mass = 0
      counter += 1

  def _generate_random_terrains(self, spec: mujoco.MjSpec) -> None:
    # Normalize the proportions of the sub-terrains.
    proportions = np.array(
      [sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()]
    )
    proportions /= np.sum(proportions)

    sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

    # Randomly sample and place sub-terrains in the grid.
    for index in range(self.cfg.num_rows * self._num_cols):
      sub_row, sub_col = np.unravel_index(index, (self.cfg.num_rows, self._num_cols))
      sub_row = int(sub_row)
      sub_col = int(sub_col)

      # Randomly select a sub-terrain type and difficulty.
      sub_index = self.np_rng.choice(len(proportions), p=proportions)
      difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)

      # Calculate the world position for this sub-terrain.
      world_position = self._get_sub_terrain_position(sub_row, sub_col)

      # Create the terrain mesh and get the spawn origin in world coordinates.
      spawn_origin = self._create_terrain_geom(
        spec,
        world_position,
        difficulty,
        sub_terrains_cfgs[sub_index],
        sub_row,
        sub_col,
      )

      # Store the spawn origin for this terrain.
      self.terrain_origins[sub_row, sub_col] = spawn_origin

  def _generate_curriculum_terrains(self, spec: mujoco.MjSpec) -> None:
    # One column per terrain type — proportion is only for spawning.
    sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

    for sub_col in range(self._num_cols):
      for sub_row in range(self.cfg.num_rows):
        lower, upper = self.cfg.difficulty_range
        difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
        difficulty = lower + (upper - lower) * difficulty
        world_position = self._get_sub_terrain_position(sub_row, sub_col)
        spawn_origin = self._create_terrain_geom(
          spec,
          world_position,
          difficulty,
          sub_terrains_cfgs[sub_col],
          sub_row,
          sub_col,
        )
        self.terrain_origins[sub_row, sub_col] = spawn_origin

  def _get_sub_terrain_position(self, row: int, col: int) -> np.ndarray:
    """Get the world position for a sub-terrain at the given grid indices.

    This returns the position of the sub-terrain's corner (not center).
    The entire grid is centered at the world origin.
    """
    # Calculate position relative to grid corner.
    rel_x = row * self.cfg.size[0]
    rel_y = col * self.cfg.size[1]

    # Offset to center the entire grid at world origin.
    grid_offset_x = -self.cfg.num_rows * self.cfg.size[0] * 0.5
    grid_offset_y = -self._num_cols * self.cfg.size[1] * 0.5

    return np.array([grid_offset_x + rel_x, grid_offset_y + rel_y, 0.0])

  def _create_terrain_geom(
    self,
    spec: mujoco.MjSpec,
    world_position: np.ndarray,
    difficulty: float,
    cfg: SubTerrainCfg,
    sub_row: int,
    sub_col: int,
  ) -> np.ndarray:
    """Create a terrain geometry at the specified world position.

    Args:
      spec: MuJoCo spec to add geometry to.
      world_position: World position of the terrain's corner.
      difficulty: Difficulty parameter for terrain generation.
      cfg: Sub-terrain configuration.
      sub_row: Row index in the terrain grid.
      sub_col: Column index in the terrain grid.

    Returns:
      The spawn origin in world coordinates.
    """
    output = cfg.function(difficulty, spec, self.np_rng)
    for terrain_geom in output.geometries:
      if terrain_geom.geom is not None:
        terrain_geom.geom.pos = np.array(terrain_geom.geom.pos) + world_position
        if terrain_geom.geom.material is not None:
          if self.cfg.color_scheme == "height" and terrain_geom.color:
            terrain_geom.geom.rgba[:] = terrain_geom.color
          elif self.cfg.color_scheme == "random":
            terrain_geom.geom.rgba[:3] = self.np_rng.uniform(0.3, 0.8, 3)
            terrain_geom.geom.rgba[3] = 1.0
          elif self.cfg.color_scheme == "none":
            terrain_geom.geom.rgba[:] = (0.5, 0.5, 0.5, 1.0)

    # Collect flat patches into pre-allocated arrays.
    spawn_origin = output.origin + world_position
    for name, arr in self.flat_patches.items():
      if output.flat_patches is not None and name in output.flat_patches:
        patches = output.flat_patches[name]
        arr[sub_row, sub_col, : len(patches)] = patches + world_position
        arr[sub_row, sub_col, len(patches) :] = spawn_origin
      else:
        # Sub-terrain didn't produce patches: fill with spawn origin so that
        # every slot contains a valid position for reset_root_state_from_flat_patches.
        arr[sub_row, sub_col] = spawn_origin

    if output.step_boundaries is not None and len(output.step_boundaries) > 0:
      boundaries = np.asarray(output.step_boundaries, dtype=np.float32).copy()
      if boundaries.ndim != 2 or boundaries.shape[1] != 11:
        raise ValueError(
          "TerrainOutput.step_boundaries must have shape [N, 11], got "
          f"{boundaries.shape}."
        )
      boundaries[:, 0:3] += world_position
      boundaries[:, 3:6] += world_position
      boundaries[:, 9:11] += world_position[2]
      self._step_boundaries_tiles[sub_row][sub_col] = boundaries
      self._add_step_danger_visual_geoms(spec, boundaries)

    return spawn_origin

  def _add_step_danger_visual_geoms(
    self, spec: mujoco.MjSpec, boundaries: np.ndarray
  ) -> None:
    vis_cfg = self.cfg.step_danger_visualization
    if not vis_cfg.enabled:
      return

    body = spec.body("terrain")
    for boundary in boundaries:
      p0 = boundary[0:3]
      p1 = boundary[3:6]
      normal_to_low = boundary[6:9]
      z_low = float(boundary[9])
      z_high = float(boundary[10])

      lip = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=(vis_cfg.lip_radius, 0.0, 0.0),
      )
      lip.fromto[:] = np.concatenate([p0, p1])
      lip.rgba[:] = vis_cfg.lip_rgba
      lip.group = vis_cfg.geom_group
      lip.contype = 0
      lip.conaffinity = 0

      slab_geom = self._make_riser_slab_geom(
        body=body,
        p0=p0,
        p1=p1,
        normal_to_low=normal_to_low,
        z_low=z_low,
        z_high=z_high,
        vis_cfg=vis_cfg,
      )
      if slab_geom is not None:
        slab_geom.rgba[:] = vis_cfg.slab_rgba
        slab_geom.group = vis_cfg.geom_group
        slab_geom.contype = 0
        slab_geom.conaffinity = 0

  def _make_riser_slab_geom(
    self,
    body: mujoco.MjsBody,
    p0: np.ndarray,
    p1: np.ndarray,
    normal_to_low: np.ndarray,
    z_low: float,
    z_high: float,
    vis_cfg: StepDangerVisualizationCfg,
  ) -> mujoco.MjsGeom | None:
    edge = p1 - p0
    edge_len = float(np.linalg.norm(edge))
    height = z_high - z_low
    if edge_len <= 1e-6 or height <= 1e-6 or vis_cfg.slab_depth <= 0.0:
      return None

    x_axis = edge / edge_len
    y_axis = normal_to_low.astype(np.float64)
    y_axis[2] = 0.0
    y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
    y_norm = np.linalg.norm(y_axis)
    if y_norm <= 1e-6:
      return None
    y_axis = y_axis / y_norm
    z_axis = np.cross(x_axis, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm <= 1e-6:
      return None
    z_axis = z_axis / z_norm

    mat = np.column_stack([x_axis, y_axis, z_axis])
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, mat.reshape(-1))

    center = 0.5 * (p0 + p1)
    center = center.astype(np.float64)
    center[2] = 0.5 * (z_low + z_high)
    center += y_axis * (0.5 * vis_cfg.slab_depth)

    size = (
      0.5 * edge_len + vis_cfg.slab_u_margin,
      0.5 * vis_cfg.slab_depth,
      0.5 * height + vis_cfg.slab_v_margin,
    )
    return body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=size,
      pos=center,
      quat=quat,
    )

  def _finalize_step_boundaries(self) -> None:
    max_count = 0
    for row_boundaries in self._step_boundaries_tiles:
      for boundaries in row_boundaries:
        max_count = max(max_count, len(boundaries))

    self.step_boundaries_by_tile = np.zeros(
      (self.cfg.num_rows, self._num_cols, max_count, 11), dtype=np.float32
    )
    self.step_boundary_counts = np.zeros(
      (self.cfg.num_rows, self._num_cols), dtype=np.int32
    )
    if max_count == 0:
      return

    for row in range(self.cfg.num_rows):
      for col in range(self._num_cols):
        boundaries = self._step_boundaries_tiles[row][col]
        count = len(boundaries)
        self.step_boundary_counts[row, col] = count
        if count > 0:
          self.step_boundaries_by_tile[row, col, :count] = boundaries

  def _add_terrain_border(self, spec: mujoco.MjSpec) -> None:
    if self.cfg.border_width <= 0.0:
      return
    body = spec.body("terrain")
    border_size = (
      self.cfg.num_rows * self.cfg.size[0] + 2 * self.cfg.border_width,
      self._num_cols * self.cfg.size[1] + 2 * self.cfg.border_width,
    )
    inner_size = (
      self.cfg.num_rows * self.cfg.size[0],
      self._num_cols * self.cfg.size[1],
    )
    # Border should be centered at origin since the terrain grid is centered.
    border_center = (0, 0, -self.cfg.border_height / 2)
    boxes = make_border(
      body,
      border_size,
      inner_size,
      height=abs(self.cfg.border_height),
      position=border_center,
    )
    for box in boxes:
      if self.cfg.color_scheme == "random":
        box.rgba = RGBA.random(self.np_rng, alpha=1.0)
      else:
        box.rgba = _DARK_GRAY

  def _add_grid_lights(self, spec: mujoco.MjSpec) -> None:
    if not self.cfg.add_lights:
      return

    total_width = self.cfg.size[0] * self.cfg.num_rows
    total_height = self.cfg.size[1] * self._num_cols
    light_height = max(total_width, total_height) * 0.6

    spec.body("terrain").add_light(
      pos=(0, 0, light_height),
      type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
      dir=(0, 0, -1),
    )
