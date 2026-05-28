"""Environment viewer built on MuJoCo's passive viewer.

Overview
--------
The simulation runs on the GPU via mujoco_warp, but MuJoCo's passive viewer
requires CPU ``MjModel`` and ``MjData`` structures. Each frame, this module
copies the GPU state for one environment into CPU buffers, calls
``mj_forward`` to recompute kinematics and contacts, and hands the result to
the render thread via ``v.sync()``. Contacts visible in the viewer are
computed by C MuJoCo on the CPU, not by mujoco_warp on the GPU.

The per frame sync has four steps:

1. ``_sync_env_state_to_mjdata``: copy ``qpos``, ``qvel``, ``mocap``, and
   ``xfrc_applied`` from GPU tensors into CPU ``MjData``.
2. ``mj_forward``: recompute kinematics, collisions, and derived quantities.
3. ``_sync_model_fields``: if domain randomization expanded visual fields
   (geom colors, body poses, camera parameters, etc.), copy the per world
   values from GPU ``sim.model`` into CPU ``MjModel``.
4. ``v.sync()``: hand ``MjModel``/``MjData`` to the passive viewer's render
   thread.

For multi environment scenes, steps 1 and 2 are repeated for neighboring
environments into a secondary ``MjData`` (``self.vd``), and their geoms are
injected via ``mjv_addGeoms``.

External force channels
-----------------------
MuJoCo sums two external force inputs during forward dynamics:

* ``xfrc_applied``: Cartesian forces per body (GPU to CPU, for rendering).
* ``qfrc_applied``: generalized forces per DoF (CPU to GPU, for mouse input).

Programmatic forces (e.g. ``apply_body_impulse``) write to ``xfrc_applied``
on the GPU and flow one way to the CPU ``MjData`` for visualization. Mouse
perturbation forces flow the opposite direction: the viewer computes them on
the CPU via ``mjv_applyPerturbForce``, converts to joint space with
``mj_applyFT``, and writes to ``qfrc_applied`` on the GPU. Each field flows
in one direction only, so the two sources never overwrite each other.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Callable, Optional, Protocol

import mujoco
import mujoco.viewer
import numpy as np
import torch

from mjlab.sensor import CameraSensor
from mjlab.viewer.base import (
  BaseViewer,
  EnvProtocol,
  PolicyProtocol,
  VerbosityLevel,
  ViewerAction,
)
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass(frozen=True)
class PlotCfg:
  """Reward plot configuration."""

  history: int = 300  # Points kept per series.
  p_lo: float = 2.0  # Percentile low.
  p_hi: float = 98.0  # Percentile high.
  pad: float = 0.25  # Pad % of span on both sides.
  min_span: float = 1e-6  # Minimum vertical span.
  init_yrange: tuple[float, float] = (-0.01, 0.01)  # Initial y-range.
  grid_size: tuple[int, int] = (3, 4)  # Grid size (rows, columns).
  max_viewports: int = 12  # Cap number of plots shown.
  min_rows_per_col: int = 4  # Minimum rows for height calc (prevents stretching).
  max_rows_per_col: int = 6  # Stack up to this many per column.
  plot_strip_fraction: float = 1 / 3  # Right-side width reserved for plots.
  background_alpha: float = 0.5  # Background alpha for plots.


def _terrain_geom_mask(mj_model: mujoco.MjModel) -> np.ndarray:
  mask = np.zeros(mj_model.ngeom, dtype=bool)
  for geom_id in range(mj_model.ngeom):
    name = mj_model.geom(geom_id).name or ""
    if name == "terrain" or name.startswith("terrain_"):
      mask[geom_id] = True
  if not mask.any():
    mask = np.asarray(mj_model.geom_group) == 0
  return mask


class _SimDataProtocol(Protocol):
  qpos: "_TensorArrayProtocol"
  qvel: "_TensorArrayProtocol"
  mocap_pos: "_TensorArrayProtocol"
  mocap_quat: "_TensorArrayProtocol"
  cam_xpos: "_TensorArrayProtocol"
  cam_xmat: "_TensorArrayProtocol"
  ctrl: "_TensorArrayProtocol"
  xfrc_applied: "_TensorArrayProtocol"
  qfrc_applied: "_TensorArrayProtocol"


class _CpuArrayProtocol(Protocol):
  def cpu(self) -> "_CpuArrayProtocol": ...
  def numpy(self) -> np.ndarray: ...


class _TensorArrayProtocol(Protocol):
  def __getitem__(self, idx: int) -> _CpuArrayProtocol: ...


class _SimModelProtocol(Protocol):
  def __getattr__(self, name: str) -> object: ...


class _SimProtocol(Protocol):
  data: _SimDataProtocol
  model: _SimModelProtocol
  expanded_fields: set[str]


class NativeMujocoViewer(BaseViewer):
  _DEPTH_CAMERA_PROJECTION_COLOR = (0.02, 0.42, 0.10, 0.72)
  _DEPTH_CAMERA_FRUSTUM_COLOR = (0.25, 0.25, 0.25, 0.45)
  _DEPTH_CAMERA_MAX_RANGE = 5.0
  _DEPTH_CAMERA_PROJECTION_STRIDE = 2
  _DEPTH_CAMERA_POINT_RADIUS = 0.005
  _DEPTH_CAMERA_SURFACE_Z_OFFSET = 0.002
  _DEPTH_CAMERA_FRUSTUM_LENGTH = 0.75
  _SHOW_DEPTH_CAMERA_GROUND_PROJECTION = False

  def __init__(
    self,
    env: EnvProtocol,
    policy: PolicyProtocol,
    frame_rate: float = 60.0,
    key_callback: Optional[Callable[[int], None]] = None,
    plot_cfg: PlotCfg | None = None,
    enable_perturbations: bool = True,
    verbosity: VerbosityLevel = VerbosityLevel.SILENT,
  ):
    super().__init__(env, policy, frame_rate, verbosity)
    self.user_key_callback = key_callback
    self.enable_perturbations = enable_perturbations

    self.mjm: Optional[mujoco.MjModel] = None
    self.mjd: Optional[mujoco.MjData] = None
    self.viewer: Optional[mujoco.viewer.Handle] = None
    self.vd: Optional[mujoco.MjData] = None
    self.vopt: Optional[mujoco.MjvOption] = None
    self.pert: Optional[mujoco.MjvPerturb] = None
    self.catmask: int = mujoco.mjtCatBit.mjCAT_DYNAMIC.value

    self._term_names: list[str] = []
    self._figures: dict[str, mujoco.MjvFigure] = {}  # Per-term figure.
    self._histories: dict[str, deque[float]] = {}  # Per-term ring buffer.
    self._yrange: dict[str, tuple[float, float]] = {}  # Per-term y-range.
    self._scale: dict[str, float] = {}  # Per-term display scale factor.
    self._show_plots: bool = False
    self._show_debug_vis: bool = True
    self._show_all_envs: bool = False
    self._plot_cfg = plot_cfg or PlotCfg()
    self._figures_dirty: bool = False
    self._terrain_geom_mask: np.ndarray | None = None

    self.env_idx = self.cfg.env_idx
    self._mj_lock = Lock()

  def setup(self) -> None:
    """Setup MuJoCo viewer resources."""
    sim = self.env.unwrapped.sim
    self.mjm = sim.mj_model
    self.mjd = sim.mj_data
    assert self.mjm is not None
    if self.cfg.fovy is not None:
      self.mjm.vis.global_.fovy = self.cfg.fovy

    if self.env.unwrapped.num_envs > 1:
      self.vd = mujoco.MjData(self.mjm)

    self.pert = mujoco.MjvPerturb() if self.enable_perturbations else None
    self.vopt = mujoco.MjvOption()

    self._term_names = [
      name
      for name, _ in self.env.unwrapped.reward_manager.get_active_iterable_terms(
        self.env_idx
      )
    ]
    self._init_reward_plots(self._term_names)

    assert self.mjm is not None
    assert self.mjd is not None
    self.viewer = mujoco.viewer.launch_passive(
      self.mjm,
      self.mjd,
      key_callback=self._safe_key_callback,
      show_left_ui=False,
      show_right_ui=False,
    )
    if self.viewer is None:
      raise RuntimeError("Failed to launch MuJoCo viewer")

    if not self.cfg.enable_shadows:
      self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

    self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SCLINERTIA] = 1

    self._setup_camera()

    if self.enable_perturbations:
      self.log("[INFO] Interactive perturbations enabled", VerbosityLevel.INFO)

  def is_running(self) -> bool:
    return bool(self.viewer and self.viewer.is_running())

  def sync_env_to_viewer(self) -> None:
    """Copy env state to viewer; update reward figures; render other envs."""
    v = self.viewer
    assert v is not None
    assert self.mjm is not None and self.mjd is not None and self.vopt is not None

    # Window may have been closed between is_running() check and here.
    if not v.is_running():
      return

    with self._mj_lock:
      sim = self.env.unwrapped.sim
      sim_data = sim.data
      self._sync_env_state_to_mjdata(self.mjd, sim_data, self.env_idx)
      self._sync_model_fields(sim, self.env_idx)
      mujoco.mj_forward(self.mjm, self.mjd)

      # The viewer window can close at any time. Re-check before set_texts
      # / set_figures since those use spin-wait sync with the render loop.
      if not v.is_running():
        return
      self._set_status_overlay(v)
      if not v.is_running():
        return
      self._update_reward_figures(v)

      self._update_debug_visualizers(v)
      self._render_other_env_geoms(v, sim, sim_data)

      # Pin tracking camera to body frame origin so DR-induced COM shifts don't move
      # the camera.
      if sim.expanded_fields & self._INERTIAL_FIELDS:
        self._stabilize_tracking_camera()

      has_visual_dr = bool(sim.expanded_fields & self._VISUAL_FIELDS)
      v.sync(state_only=not has_visual_dr)

  def _set_status_overlay(self, viewer: mujoco.viewer.Handle) -> None:
    status = self.get_status()
    capped = " [CAPPED]" if status.capped else ""
    text_1 = "Env\nStep\nStatus\nSpeed\nTarget RT\nActual RT"
    text_2 = (
      f"{self.env_idx + 1}/{self.env.num_envs}\n"
      f"{status.step_count}\n"
      f"{'PAUSED' if status.paused else 'RUNNING'}{capped}\n"
      f"{status.speed_label}\n"
      f"{status.target_realtime:.2f}x\n"
      f"{status.actual_realtime:.2f}x ({status.smoothed_fps:.0f} FPS)"
    )
    terrain_status = self.get_terrain_status(self.env_idx)
    if terrain_status is not None:
      terrain_label, _is_stairs = terrain_status
      text_1 += "\nTerrain"
      text_2 += f"\n{terrain_label}"
    target_status = self.get_target_heading_status(self.env_idx)
    if target_status is not None:
      target_label = target_status.mode
      if target_status.distance_m is not None:
        target_label += f" {target_status.distance_m:.2f}m"
      if target_status.reached:
        target_label += " reached"
      text_1 += "\nCommand"
      text_2 += f"\n{target_label}"
    overlay = (
      mujoco.mjtFontScale.mjFONTSCALE_150.value,
      mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
      text_1,
      text_2,
    )
    viewer.set_texts(overlay)

  def _update_reward_figures(self, viewer: mujoco.viewer.Handle) -> None:
    if not self._show_plots or not self._term_names:
      # Only send an empty set_figures when transitioning from shown to hidden, to
      # avoid a spin-wait round trip every frame.
      if self._figures_dirty:
        viewer.set_figures([])
        self._figures_dirty = False
      return

    terms = list(
      self.env.unwrapped.reward_manager.get_active_iterable_terms(self.env_idx)
    )
    if not self._is_paused:
      for name, arr in terms:
        if name in self._histories:
          self._append_point(name, float(arr[0]))
          self._write_history_to_figure(name)

    viewports = compute_viewports(
      len(self._term_names), viewer.viewport, self._plot_cfg
    )
    viewport_figs = [
      (viewports[i], self._figures[self._term_names[i]])
      for i in range(
        min(len(viewports), len(self._term_names), self._plot_cfg.max_viewports)
      )
    ]
    viewer.set_figures(viewport_figs)
    self._figures_dirty = True

  def _update_debug_visualizers(self, viewer: mujoco.viewer.Handle) -> None:
    viewer.user_scn.ngeom = 0
    assert self.mjm is not None
    visualizer = MujocoNativeDebugVisualizer(
      viewer.user_scn, self.mjm, self.env_idx, self._show_all_envs
    )
    if self._show_debug_vis and hasattr(self.env.unwrapped, "update_visualizers"):
      self.env.unwrapped.update_visualizers(visualizer)
      self._update_depth_camera_visualizers(visualizer)
    self._draw_terrain_flag_marker(visualizer)

  def _draw_terrain_flag_marker(
    self, visualizer: MujocoNativeDebugVisualizer
  ) -> None:
    marker = self.get_terrain_flag_marker(self.env_idx)
    if marker is None:
      return

    color = (1.0, 0.58, 0.10, 1.0) if marker.is_stairs else (0.15, 0.85, 0.35, 1.0)
    center = marker.position
    visualizer.add_sphere(center, 0.12, (*color[:3], 0.28))
    if marker.text == "1":
      self._draw_digit_segments(
        visualizer,
        center,
        color,
        (
          ((0.00, -0.25), (0.00, 0.25)),
          ((-0.07, 0.18), (0.00, 0.25)),
        ),
      )
      return

    self._draw_digit_segments(
      visualizer,
      center,
      color,
      (
        ((-0.12, -0.25), (0.12, -0.25)),
        ((0.12, -0.25), (0.12, 0.25)),
        ((0.12, 0.25), (-0.12, 0.25)),
        ((-0.12, 0.25), (-0.12, -0.25)),
      ),
    )

  @staticmethod
  def _draw_digit_segments(
    visualizer: MujocoNativeDebugVisualizer,
    center: np.ndarray,
    color: tuple[float, float, float, float],
    segments: tuple[tuple[tuple[float, float], tuple[float, float]], ...],
  ) -> None:
    for (x0, z0), (x1, z1) in segments:
      start = center + np.array([x0, 0.0, z0], dtype=np.float32)
      end = center + np.array([x1, 0.0, z1], dtype=np.float32)
      visualizer.add_cylinder(start, end, 0.018, color)

  def _update_depth_camera_visualizers(
    self, visualizer: MujocoNativeDebugVisualizer
  ) -> None:
    if not self.cfg.show_depth_camera_visualizers:
      return

    sim_data = self.env.unwrapped.sim.data
    for sensor in self.env.unwrapped.scene.sensors.values():
      if not isinstance(sensor, CameraSensor) or "depth" not in sensor.cfg.data_types:
        continue
      camera_data = sensor.data
      depth = camera_data.depth
      if depth is None:
        continue
      segmentation = camera_data.segmentation

      cam_id = sensor.camera_idx
      cam_pos = sim_data.cam_xpos[self.env_idx, cam_id].cpu().numpy()
      cam_mat = sim_data.cam_xmat[self.env_idx, cam_id].cpu().numpy().reshape(3, 3)
      self._draw_depth_camera_frustum(visualizer, sensor, cam_pos, cam_mat)
      if self._SHOW_DEPTH_CAMERA_GROUND_PROJECTION:
        self._draw_depth_camera_ground_projection(
          visualizer=visualizer,
          sensor=sensor,
          depth=depth[self.env_idx, :, :, 0].cpu().numpy(),
          segmentation=(
            segmentation[self.env_idx, :, :, 0].cpu().numpy()
            if segmentation is not None
            else None
          ),
          cam_pos=cam_pos,
          cam_mat=cam_mat,
        )

  def _draw_depth_camera_frustum(
    self,
    visualizer: MujocoNativeDebugVisualizer,
    sensor: CameraSensor,
    cam_pos: np.ndarray,
    cam_mat: np.ndarray,
  ) -> None:
    assert self.mjm is not None
    cam_id = sensor.camera_idx
    fovy = math.radians(float(self.mjm.cam_fovy[cam_id]))
    aspect = sensor.cfg.width / sensor.cfg.height
    tan_v = math.tan(fovy * 0.5)
    tan_h = tan_v * aspect
    local_dirs = np.array(
      [
        [-tan_h, -tan_v, -1.0],
        [tan_h, -tan_v, -1.0],
        [tan_h, tan_v, -1.0],
        [-tan_h, tan_v, -1.0],
      ],
      dtype=np.float64,
    )
    local_dirs /= np.linalg.norm(local_dirs, axis=1, keepdims=True)
    corners = cam_pos + (cam_mat @ local_dirs.T).T * self._DEPTH_CAMERA_FRUSTUM_LENGTH

    for corner in corners:
      visualizer.add_cylinder(
        cam_pos,
        corner,
        radius=0.006,
        color=self._DEPTH_CAMERA_FRUSTUM_COLOR,
      )
    for i in range(4):
      visualizer.add_cylinder(
        corners[i],
        corners[(i + 1) % 4],
        radius=0.005,
        color=self._DEPTH_CAMERA_FRUSTUM_COLOR,
      )

  def _draw_depth_camera_ground_projection(
    self,
    visualizer: MujocoNativeDebugVisualizer,
    sensor: CameraSensor,
    depth: np.ndarray,
    segmentation: np.ndarray | None,
    cam_pos: np.ndarray,
    cam_mat: np.ndarray,
  ) -> None:
    height, width = depth.shape
    stride = self._DEPTH_CAMERA_PROJECTION_STRIDE
    ys = np.arange(0, height, stride)
    xs = np.arange(0, width, stride)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")

    local_dirs = self._depth_camera_pixel_dirs(
      sensor=sensor,
      pixel_x=grid_x.reshape(-1),
      pixel_y=grid_y.reshape(-1),
    )
    depth_values = depth[grid_y.reshape(-1), grid_x.reshape(-1)].astype(np.float64)
    ray_scale = depth_values / np.maximum(-local_dirs[:, 2], 1.0e-6)
    valid = (
      np.isfinite(depth_values)
      & (depth_values > 1.0e-4)
      & (ray_scale <= self._DEPTH_CAMERA_MAX_RANGE)
    )
    if segmentation is not None:
      geom_ids = segmentation[grid_y.reshape(-1), grid_x.reshape(-1)].astype(np.int64)
      valid &= self._terrain_hit_mask(geom_ids)
    if not valid.any():
      return

    local_dirs = local_dirs[valid]
    ray_scale = ray_scale[valid]
    local_points = local_dirs * ray_scale[:, None]
    points = (cam_mat @ local_points.T).T + cam_pos
    points[:, 2] += self._DEPTH_CAMERA_SURFACE_Z_OFFSET

    for point in points:
      visualizer.add_sphere(
        point,
        radius=self._DEPTH_CAMERA_POINT_RADIUS,
        color=self._DEPTH_CAMERA_PROJECTION_COLOR,
      )

  def _depth_camera_pixel_dirs(
    self,
    sensor: CameraSensor,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
  ) -> np.ndarray:
    assert self.mjm is not None
    cam_id = sensor.camera_idx
    fovy = math.radians(float(self.mjm.cam_fovy[cam_id]))
    aspect = sensor.cfg.width / sensor.cfg.height
    tan_v = math.tan(fovy * 0.5)
    tan_h = tan_v * aspect
    u = 2.0 * pixel_x / max(sensor.cfg.width - 1, 1) - 1.0
    v = 2.0 * pixel_y / max(sensor.cfg.height - 1, 1) - 1.0
    dirs = np.stack(
      [
        u * tan_h,
        v * tan_v,
        -np.ones_like(u, dtype=np.float64),
      ],
      axis=1,
    )
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs

  def _terrain_hit_mask(self, geom_ids: np.ndarray) -> np.ndarray:
    assert self.mjm is not None
    if (
      self._terrain_geom_mask is None
      or self._terrain_geom_mask.shape[0] != self.mjm.ngeom
    ):
      self._terrain_geom_mask = _terrain_geom_mask(self.mjm)

    terrain_hit = np.zeros(geom_ids.shape, dtype=bool)
    in_bounds = (geom_ids >= 0) & (geom_ids < self._terrain_geom_mask.shape[0])
    terrain_hit[in_bounds] = self._terrain_geom_mask[geom_ids[in_bounds]]
    return terrain_hit

  def _sync_env_state_to_mjdata(
    self, target_data: mujoco.MjData, sim_data: _SimDataProtocol, env_idx: int
  ) -> None:
    """Copy one environment state from batched sim data into a MjData buffer."""
    assert self.mjm is not None
    if self.mjm.nq > 0:
      target_data.qpos[:] = sim_data.qpos[env_idx].cpu().numpy()
      target_data.qvel[:] = sim_data.qvel[env_idx].cpu().numpy()
    if self.mjm.nu > 0:
      target_data.ctrl[:] = sim_data.ctrl[env_idx].cpu().numpy()
    if self.mjm.nmocap > 0:
      target_data.mocap_pos[:] = sim_data.mocap_pos[env_idx].cpu().numpy()
      target_data.mocap_quat[:] = sim_data.mocap_quat[env_idx].cpu().numpy()
    target_data.xfrc_applied[:] = sim_data.xfrc_applied[env_idx].cpu().numpy()

  def _render_other_env_geoms(
    self,
    viewer: mujoco.viewer.Handle,
    sim: _SimProtocol,
    sim_data: _SimDataProtocol,
  ) -> None:
    """Render non-selected environments into the native viewer scene."""
    if self.vd is None:
      return
    assert self.mjm is not None
    assert self.vopt is not None
    assert self.pert is not None

    for i in range(self.env.unwrapped.num_envs):
      if i == self.env_idx:
        continue
      self._sync_env_state_to_mjdata(self.vd, sim_data, i)
      self._sync_model_fields(sim, i)
      mujoco.mj_forward(self.mjm, self.vd)
      mujoco.mjv_addGeoms(
        self.mjm, self.vd, self.vopt, self.pert, self.catmask, viewer.user_scn
      )

    # Restore main env's model fields.
    self._sync_model_fields(sim, self.env_idx)

  # Inertial fields that shift subtree_com (and thus the tracking camera).
  _INERTIAL_FIELDS = frozenset({"body_ipos", "body_mass"})

  # Fields that affect rendering. Physics-only fields (geom_aabb,
  # geom_rbound, dof_*, jnt_*, actuator_*, tendon_*, etc.) are skipped.
  _VISUAL_FIELDS = frozenset(
    {
      "qpos0",  # Needed for correct mj_forward kinematics (qpos - qpos0).
      "geom_rgba",
      "geom_size",
      "geom_pos",
      "geom_quat",
      "mat_rgba",
      "site_pos",
      "site_quat",
      "body_pos",
      "body_quat",
      "body_ipos",
      "body_inertia",
      "body_iquat",
      "body_mass",
      "cam_pos",
      "cam_quat",
      "cam_fovy",
      "cam_intrinsic",
      "light_pos",
      "light_dir",
    }
  )

  def _sync_model_fields(self, sim: _SimProtocol, env_idx: int) -> None:
    """Sync visually-relevant DR'd model fields from GPU to MjModel."""
    for field_name in sim.expanded_fields & self._VISUAL_FIELDS:
      src = getattr(sim.model, field_name)[env_idx].cpu().numpy()
      dst = getattr(self.mjm, field_name)
      dst[:] = src.reshape(dst.shape)

  def _stabilize_tracking_camera(self) -> None:
    """Pin the tracked body's subtree_com to its frame origin (xpos).

    MuJoCo's tracking camera centers on ``subtree_com[trackbodyid]``, which
    shifts when inertial fields (body_ipos, body_mass) are domain randomized.
    Overwriting that single entry with ``xpos`` (the body frame origin,
    unaffected by body_ipos) keeps the camera stable.
    """
    assert self.mjd is not None
    if not (
      self.viewer and self.viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING.value
    ):
      return
    bid = self.viewer.cam.trackbodyid
    if bid >= 0:
      self.mjd.subtree_com[bid] = self.mjd.xpos[bid]

  def sync_viewer_to_env(self) -> None:
    """Sync mouse perturbation to sim via ``qfrc_applied``.

    Mouse perturbation forces are converted from Cartesian body space
    (``xfrc_applied``) to generalized joint space (``qfrc_applied``)
    so that they coexist with programmatic forces on ``xfrc_applied``.
    See the module docstring for details on the channel separation.
    """
    v = self.viewer
    if v is None or self.mjm is None or self.mjd is None:
      return

    sim_data = self.env.unwrapped.sim.data
    pert = v.perturb

    if pert.active != 0 and pert.select > 0:
      # Compute mouse perturbation force in Cartesian space.
      mujoco.mjv_applyPerturbForce(self.mjm, self.mjd, pert)

      body_id = pert.select
      force = self.mjd.xfrc_applied[body_id, :3].copy()
      torque = self.mjd.xfrc_applied[body_id, 3:].copy()
      point = self.mjd.xipos[body_id].copy()

      # Convert to generalized forces.
      qfrc = np.zeros(self.mjm.nv)
      mujoco.mj_applyFT(self.mjm, self.mjd, force, torque, point, body_id, qfrc)

      sim_data.qfrc_applied[self.env_idx] = torch.from_numpy(qfrc).to(
        device=sim_data.qfrc_applied.device
      )

      # Clear so _sync_env_state_to_mjdata writes clean programmatic
      # forces next frame.
      self.mjd.xfrc_applied[body_id] = 0.0
    else:
      sim_data.qfrc_applied[self.env_idx] = 0.0

  def close(self) -> None:
    """Close viewer and cleanup."""
    v = self.viewer
    self.viewer = None
    if v:
      v.close()
    self.log("[INFO] MuJoCo viewer closed", VerbosityLevel.INFO)

  def reset_environment(self) -> None:
    """Extend BaseViewer.reset_environment to clear reward histories."""
    super().reset_environment()
    self._clear_histories()

  def _safe_key_callback(self, key: int) -> None:
    """Runs on MuJoCo viewer thread; must not touch env/sim directly."""
    from mjlab.viewer.native.keys import (
      KEY_A,
      KEY_COMMA,
      KEY_ENTER,
      KEY_EQUAL,
      KEY_MINUS,
      KEY_P,
      KEY_PERIOD,
      KEY_R,
      KEY_RIGHT,
      KEY_SPACE,
    )

    if key == KEY_ENTER:
      self.request_reset()
    elif key == KEY_SPACE:
      self.request_toggle_pause()
    elif key == KEY_MINUS:
      self.request_speed_down()
    elif key == KEY_EQUAL:
      self.request_speed_up()
    elif key == KEY_COMMA:
      self.request_action("PREV_ENV")
    elif key == KEY_PERIOD:
      self.request_action("NEXT_ENV")
    elif key == KEY_P:
      self.request_action("TOGGLE_PLOTS")
    elif key == KEY_R:
      self.request_action("TOGGLE_DEBUG_VIS")
    elif key == KEY_A:
      self.request_action("TOGGLE_SHOW_ALL_ENVS")
    elif key == KEY_RIGHT:
      self.request_single_step()

    if self.user_key_callback:
      try:
        self.user_key_callback(key)
      except Exception as e:
        self.log(f"[WARN] user key_callback raised: {e}", VerbosityLevel.INFO)

  def _forward_paused(self) -> None:
    """Run forward kinematics while paused to keep perturbation visuals accurate."""
    if self.mjm is not None and self.mjd is not None:
      with self._mj_lock:
        mujoco.mj_forward(self.mjm, self.mjd)

  def _handle_custom_action(self, action: ViewerAction, payload: object | None) -> bool:
    del payload
    if action == ViewerAction.PREV_ENV and self.env.unwrapped.num_envs > 1:
      self.env_idx = (self.env_idx - 1) % self.env.unwrapped.num_envs
      self._clear_histories()
      self.log(f"[INFO] Switched to environment {self.env_idx}", VerbosityLevel.INFO)
      return True
    if action == ViewerAction.NEXT_ENV and self.env.unwrapped.num_envs > 1:
      self.env_idx = (self.env_idx + 1) % self.env.unwrapped.num_envs
      self._clear_histories()
      self.log(f"[INFO] Switched to environment {self.env_idx}", VerbosityLevel.INFO)
      return True
    if action == ViewerAction.TOGGLE_PLOTS:
      self._show_plots = not self._show_plots
      self.log(
        f"[INFO] Reward plots {'shown' if self._show_plots else 'hidden'}",
        VerbosityLevel.INFO,
      )
      return True
    if action == ViewerAction.TOGGLE_DEBUG_VIS:
      self._show_debug_vis = not self._show_debug_vis
      self.log(
        f"[INFO] Debug visualization {'shown' if self._show_debug_vis else 'hidden'}",
        VerbosityLevel.INFO,
      )
      return True
    if action == ViewerAction.TOGGLE_SHOW_ALL_ENVS:
      self._show_all_envs = not self._show_all_envs
      self.log(
        f"[INFO] Show all envs {'enabled' if self._show_all_envs else 'disabled'}",
        VerbosityLevel.INFO,
      )
      return True
    return False

  def _setup_camera(self) -> None:
    """Configure native viewer camera from viewer config."""
    assert self.viewer is not None
    self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD.value

    if not self.cfg or not hasattr(self.cfg, "origin_type"):
      self._set_camera_auto_track()
      return

    if self.cfg.origin_type == self.cfg.OriginType.AUTO:
      self._set_camera_auto_track()
    elif self.cfg.origin_type == self.cfg.OriginType.WORLD:
      self._set_camera_world()
    elif self.cfg.origin_type == self.cfg.OriginType.ASSET_ROOT:
      self._set_camera_asset_root()
    else:  # ASSET_BODY
      self._set_camera_asset_body()

    self.viewer.cam.lookat = getattr(self.cfg, "lookat", self.viewer.cam.lookat)
    self.viewer.cam.elevation = getattr(
      self.cfg, "elevation", self.viewer.cam.elevation
    )
    self.viewer.cam.azimuth = getattr(self.cfg, "azimuth", self.viewer.cam.azimuth)
    self.viewer.cam.distance = getattr(self.cfg, "distance", self.viewer.cam.distance)

  def _set_camera_auto_track(self) -> None:
    """Track first non-fixed body; fall back to free camera if none exists."""
    assert self.viewer is not None
    assert self.mjm is not None
    for body_id in range(self.mjm.nbody):
      is_weld = self.mjm.body_weldid[body_id] == 0
      root_id = self.mjm.body_rootid[body_id]
      root_is_mocap = self.mjm.body_mocapid[root_id] >= 0
      if not (is_weld and not root_is_mocap):
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
        self.viewer.cam.trackbodyid = body_id
        self.viewer.cam.fixedcamid = -1
        return
    self._set_camera_world()

  def _set_camera_world(self) -> None:
    """Configure free camera in world frame."""
    assert self.viewer is not None
    self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE.value
    self.viewer.cam.fixedcamid = -1
    self.viewer.cam.trackbodyid = -1

  def _set_camera_asset_root(self) -> None:
    """Track the root body of a configured asset."""
    assert self.viewer is not None
    if not self.cfg.entity_name:
      raise ValueError("Asset name must be specified for ASSET_ROOT origin type")
    robot: Entity = self.env.unwrapped.scene[self.cfg.entity_name]
    body_id = robot.indexing.root_body_id
    self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
    self.viewer.cam.trackbodyid = body_id
    self.viewer.cam.fixedcamid = -1

  def _set_camera_asset_body(self) -> None:
    """Track a specific body of a configured asset."""
    assert self.viewer is not None
    if not self.cfg.entity_name or not self.cfg.body_name:
      raise ValueError("entity_name/body_name required for ASSET_BODY origin type")
    robot: Entity = self.env.unwrapped.scene[self.cfg.entity_name]
    if self.cfg.body_name not in robot.body_names:
      raise ValueError(
        f"Body '{self.cfg.body_name}' not found in asset '{self.cfg.entity_name}'"
      )
    body_id_list, _ = robot.find_bodies(self.cfg.body_name)
    body_id = robot.indexing.bodies[body_id_list[0]].id

    self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
    self.viewer.cam.trackbodyid = body_id
    self.viewer.cam.fixedcamid = -1

  # Reward plotting helpers.

  def _init_reward_plots(self, term_names: list[str]) -> None:
    """Create per-term figures and histories."""
    self._figures.clear()
    self._histories.clear()
    self._yrange.clear()
    self._scale.clear()
    for name in term_names:
      self._figures[name] = make_empty_figure(
        name,
        self._plot_cfg.grid_size,
        self._plot_cfg.init_yrange,
        self._plot_cfg.history,
        self._plot_cfg.background_alpha,
      )
      self._histories[name] = deque(maxlen=self._plot_cfg.history)
      self._yrange[name] = self._plot_cfg.init_yrange
      self._scale[name] = 1.0

  def _clear_histories(self) -> None:
    """Clear histories and reset figures."""
    for name in self._term_names:
      self._histories[name].clear()
      self._yrange[name] = self._plot_cfg.init_yrange
      self._scale[name] = 1.0
      fig = self._figures[name]
      fig.title = name
      fig.linepnt[0] = 0
      fig.range[1][0] = float(self._plot_cfg.init_yrange[0])
      fig.range[1][1] = float(self._plot_cfg.init_yrange[1])

  def _append_point(self, name: str, value: float) -> None:
    """Append a new point to the ring buffer."""
    if not np.isfinite(value):
      return
    self._histories[name].append(float(value))

  def _write_history_to_figure(self, name: str) -> None:
    """Copy history into figure and autoscale y-axis."""
    fig = self._figures[name]
    hist = self._histories[name]
    n = min(len(hist), self._plot_cfg.history)

    # Autoscale y-axis (in raw units).
    if n >= 5:
      data = np.fromiter(hist, dtype=float, count=n)
      lo = float(np.percentile(data, self._plot_cfg.p_lo))
      hi = float(np.percentile(data, self._plot_cfg.p_hi))
      span = max(hi - lo, self._plot_cfg.min_span)
      lo -= self._plot_cfg.pad * span
      hi += self._plot_cfg.pad * span
    elif n >= 1:
      v = float(hist[-1])
      span = max(abs(v), 1e-3)
      lo, hi = v - span, v + span
    else:
      lo, hi = self._plot_cfg.init_yrange

    # Compute scale factor so axis labels avoid scientific notation.
    scale = _display_scale(lo, hi)
    self._scale[name] = scale

    # Write scaled data into figure.
    fig.linepnt[0] = n
    for i in range(n):
      fig.linedata[0][2 * i] = float(-i)
      fig.linedata[0][2 * i + 1] = float(hist[-1 - i]) * scale

    fig.range[1][0] = float(lo * scale)
    fig.range[1][1] = float(hi * scale)

    # Update title with scale suffix.
    if scale == 1.0:
      fig.title = name
    else:
      exp = round(math.log10(1.0 / scale))
      fig.title = f"{name} (1e{exp})"


def _display_scale(lo: float, hi: float) -> float:
  """Return a power-of-10 multiplier that brings *lo*/*hi* into a readable range.

  Values in [0.01, 100] render as clean decimals in MuJoCo's ``%g``
  tick labels, so we only rescale outside that band.
  """
  max_abs = max(abs(lo), abs(hi))
  if max_abs < 1e-15:
    return 1.0
  exp = math.floor(math.log10(max_abs))
  if -2 <= exp <= 2:
    return 1.0
  return 10.0 ** (-exp)


def compute_viewports(
  num_plots: int,
  rect: mujoco.MjrRect,
  cfg: PlotCfg,
) -> list[mujoco.MjrRect]:
  """Lay plots in a strip on the right."""
  if num_plots <= 0:
    return []
  cols = 1 if num_plots <= cfg.max_rows_per_col else 2
  rows = min(cfg.max_rows_per_col, (num_plots + cols - 1) // cols)

  vp_w = int(rect.width * cfg.plot_strip_fraction) // 2
  strip_w = vp_w * cols
  vp_h = rect.height // max(rows, cfg.min_rows_per_col)

  left0 = rect.left + rect.width - strip_w
  vps: list[mujoco.MjrRect] = []
  for idx in range(min(num_plots, cfg.max_viewports)):
    c = idx // rows
    r = idx % rows
    left = left0 + (cols - 1 - c) * vp_w
    bottom = rect.bottom + rect.height - (r + 1) * vp_h
    vps.append(mujoco.MjrRect(left=left, bottom=bottom, width=vp_w, height=vp_h))
  return vps


def make_empty_figure(
  title: str,
  grid_size: tuple[int, int],
  yrange: tuple[float, float],
  history: int,
  alpha: float,
) -> mujoco.MjvFigure:
  fig = mujoco.MjvFigure()
  mujoco.mjv_defaultFigure(fig)
  fig.flg_extend = 1
  fig.gridsize[0] = grid_size[0]
  fig.gridsize[1] = grid_size[1]
  fig.range[1][0] = float(yrange[0])
  fig.range[1][1] = float(yrange[1])
  fig.figurergba[3] = alpha
  fig.title = title
  # Pre-fill x coordinates; y's will be written on update.
  for i in range(history):
    fig.linedata[0][2 * i] = -float(i)
    fig.linedata[0][2 * i + 1] = 0.0
  fig.linepnt[0] = 0
  return fig
