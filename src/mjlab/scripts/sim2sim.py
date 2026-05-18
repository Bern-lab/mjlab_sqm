"""Sim2Sim validation script for exported TorchScript policies.

This script loads an exported policy (.pt) and a motion (.npz), then runs
policy inference in a MuJoCo simulation loop for deployment-side validation.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Keep Warp initialization quiet (must be set before imports that may load Warp)
os.environ.setdefault("WARP_PRINT_LAUNCHES", "0")
os.environ.setdefault("WARP_QUIET", "1")

import mujoco
import numpy as np
import torch

from mjlab.asset_zoo.robots.booster_k1.k1_constants import K1_ACTION_SCALE, get_k1_robot_cfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.asset_zoo.robots.engineai_pm01.pm01_constants import PM01_ACTION_SCALE, get_pm01_robot_cfg
from mjlab.asset_zoo.robots.limx_oli.oli_constants import OLI_ACTION_SCALE, get_oli_robot_cfg
from mjlab.asset_zoo.robots.fourier_gr3.gr3_constants import GR3_ACTION_SCALE, get_gr3_robot_cfg
from mjlab.asset_zoo.robots.noetix_bumi.bumi_constants import BUMI_ACTION_SCALE, get_bumi_robot_cfg
from mjlab.asset_zoo.robots.noetix_e1.e1_constants import E1_ACTION_SCALE, get_e1_robot_cfg

from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat, matrix_from_quat, subtract_frame_transforms


@dataclass(frozen=True)
class RobotProfile:
  robot_cfg: EntityCfg
  action_scale: dict[str, float]
  base_link_name: str


TASK_ROBOT_PROFILES: dict[str, RobotProfile] = {
  "Mjlab-Tracking-Flat-Booster-K1": RobotProfile(
    robot_cfg=get_k1_robot_cfg(),
    action_scale=K1_ACTION_SCALE,
    base_link_name="base_link",
  ),
  "Mjlab-Tracking-Flat-Booster-K1-No-State-Estimation": RobotProfile(
    robot_cfg=get_k1_robot_cfg(),
    action_scale=K1_ACTION_SCALE,
    base_link_name="base_link",
  ),
  "Mjlab-Tracking-Flat-Unitree-G1": RobotProfile(
    robot_cfg=get_g1_robot_cfg(),
    action_scale=G1_ACTION_SCALE,
    base_link_name="torso_link",
  ),
  "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation": RobotProfile(
    robot_cfg=get_g1_robot_cfg(),
    action_scale=G1_ACTION_SCALE,
    base_link_name="torso_link",
  ),
  "Mjlab-Tracking-Flat-Engineai-PM01": RobotProfile(
    robot_cfg=get_pm01_robot_cfg(),
    action_scale=PM01_ACTION_SCALE,
    base_link_name="LINK_BASE",
  ),
  "Mjlab-Tracking-Flat-Engineai-PM01-No-State-Estimation": RobotProfile(
    robot_cfg=get_pm01_robot_cfg(),
    action_scale=PM01_ACTION_SCALE,
    base_link_name="LINK_BASE",
  ),
  "Mjlab-Tracking-Flat-Limx-OLI": RobotProfile(
    robot_cfg=get_oli_robot_cfg(),
    action_scale=OLI_ACTION_SCALE,
    base_link_name="base_link",
  ),
  "Mjlab-Tracking-Flat-Limx-OLI-No-State-Estimation": RobotProfile(
    robot_cfg=get_oli_robot_cfg(),
    action_scale=OLI_ACTION_SCALE,
    base_link_name="base_link",
  ),
  "Mjlab-Tracking-Flat-Fourier-GR3": RobotProfile(
    robot_cfg=get_gr3_robot_cfg(),
    action_scale=GR3_ACTION_SCALE,
    base_link_name="base_link",
  ),
  "Mjlab-Tracking-Flat-Fourier-GR3-No-State-Estimation": RobotProfile(
    robot_cfg=get_gr3_robot_cfg(),
    action_scale=GR3_ACTION_SCALE,
    base_link_name="base_link",
  ),
  "Mjlab-Tracking-Flat-Noetix-BUMI": RobotProfile(
    robot_cfg=get_bumi_robot_cfg(),
    action_scale=BUMI_ACTION_SCALE,
    base_link_name="base_link",
  ),
  "Mjlab-Tracking-Flat-Noetix-BUMI-No-State-Estimation": RobotProfile(
    robot_cfg=get_bumi_robot_cfg(),
    action_scale=BUMI_ACTION_SCALE,
    base_link_name="base_link",
  ),
  "Mjlab-Tracking-Flat-Noetix-E1": RobotProfile(
    robot_cfg=get_e1_robot_cfg(),
    action_scale=E1_ACTION_SCALE,
    base_link_name="base_link",
  ),
  "Mjlab-Tracking-Flat-Noetix-E1-No-State-Estimation": RobotProfile(
    robot_cfg=get_e1_robot_cfg(),
    action_scale=E1_ACTION_SCALE,
    base_link_name="base_link",
  ),


}


def _pick_latest_pt(candidates: list[Path]) -> Path:
  if not candidates:
    raise FileNotFoundError("No .pt file found.")

  def _key(path: Path) -> tuple[int, float]:
    m = re.search(r"model_(\d+)\.pt$", path.name)
    step = int(m.group(1)) if m else -1
    return (step, path.stat().st_mtime)

  return sorted(candidates, key=_key)[-1]


def resolve_policy_path(policy_arg: str) -> Path:
  """Resolve policy argument to an exported policy file."""
  p = Path(policy_arg)

  if p.is_absolute() and p.exists():
    if p.is_file() and p.suffix == ".pt":
      return p
    if p.is_dir():
      exported = p / "exported"
      if exported.exists():
        return _pick_latest_pt(list(exported.glob("*.pt")))
      return _pick_latest_pt(list(p.glob("*.pt")))

  if p.exists():
    if p.is_file() and p.suffix == ".pt":
      return p
    if p.is_dir():
      exported = p / "exported"
      if exported.exists():
        return _pick_latest_pt(list(exported.glob("*.pt")))
      return _pick_latest_pt(list(p.glob("*.pt")))

  log_root = Path("logs")
  if log_root.exists():
    direct = log_root / policy_arg
    if direct.exists() and direct.is_dir():
      exported = direct / "exported"
      if exported.exists():
        return _pick_latest_pt(list(exported.glob("*.pt")))
    matching_dirs = [d for d in log_root.glob(f"**/{policy_arg}") if d.is_dir()]
    if matching_dirs:
      exported = matching_dirs[0] / "exported"
      if exported.exists():
        return _pick_latest_pt(list(exported.glob("*.pt")))

  raise FileNotFoundError(f"Policy not found: {policy_arg}")


def load_motion_file_from_log(policy_file: Path) -> str | None:
  """Try loading motion_file from <log_dir>/params/env.yaml."""
  log_dir = policy_file.parent
  if log_dir.name == "exported":
    log_dir = log_dir.parent

  env_yaml = log_dir / "params" / "env.yaml"
  if not env_yaml.exists():
    return None

  try:
    content = env_yaml.read_text(encoding="utf-8")
    match = re.search(r"motion_file:\s*(.+\.npz)", content)
    if match:
      motion_file = match.group(1).strip()
      if Path(motion_file).exists():
        return motion_file
  except Exception as exc:
    print(f"[WARN] Failed to parse motion_file from {env_yaml}: {exc}")

  return None


def load_policy(policy_file: Path, device: str) -> torch.nn.Module:
  print(f"[INFO] Loading policy: {policy_file}")
  policy = torch.jit.load(str(policy_file), map_location=device)
  policy.eval()
  return policy


def load_motion_data(motion_file: Path) -> dict[str, torch.Tensor]:
  print(f"[INFO] Loading motion: {motion_file}")
  data = np.load(motion_file)
  motion = {
    "joint_pos": torch.from_numpy(data["joint_pos"]).float(),
    "joint_vel": torch.from_numpy(data["joint_vel"]).float(),
    "body_pos_w": torch.from_numpy(data["body_pos_w"]).float(),
    "body_quat_w": torch.from_numpy(data["body_quat_w"]).float(),
    "body_lin_vel_w": torch.from_numpy(data["body_lin_vel_w"]).float(),
    "body_ang_vel_w": torch.from_numpy(data["body_ang_vel_w"]).float(),
  }
  print(f"[INFO] Motion frames: {motion['joint_pos'].shape[0]}")
  return motion


class Sim2SimTester:
  """Sim2Sim tester for tracking tasks with exported TorchScript policies."""

  def __init__(
    self,
    profile: RobotProfile,
    motion_data: dict[str, torch.Tensor],
    policy: torch.nn.Module,
    device: str,
    dt: float,
    render: bool,
    fps_limit: float | None,
    verbose: bool,
  ) -> None:
    self.profile = profile
    self.motion_data = motion_data
    self.policy = policy
    self.device = device
    self.dt = dt
    self.render = render
    self.fps_limit = fps_limit
    self.verbose = verbose

    scene_cfg = SceneCfg(
      num_envs=1,
      entities={"robot": profile.robot_cfg},
      terrain=TerrainEntityCfg(terrain_type="plane", num_envs=1),
    )
    self.scene = Scene(scene_cfg, device=device)
    self.robot = self.scene["robot"]

    sim_cfg = SimulationCfg(
      nconmax=200_000,
      njmax=8192,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    )
    mj_model = self.scene.compile()
    self.sim = Simulation(num_envs=1, cfg=sim_cfg, model=mj_model, device=device)
    self.scene.initialize(mj_model, self.sim.model, self.sim.data)

    self.joint_names = self.robot.joint_names
    self.num_joints = len(self.joint_names)
    self.ctrl_ids = self.robot.indexing.ctrl_ids

    try:
      self.robot_anchor_body_idx = self.robot.body_names.index(profile.base_link_name)
    except ValueError:
      self.robot_anchor_body_idx = 0
    self.motion_anchor_body_idx = 0

    self.default_joint_pos = self.robot.data.default_joint_pos[:, : self.num_joints].clone()
    self.action_scale = torch.ones(self.num_joints, dtype=torch.float32, device=device)
    for i, joint_name in enumerate(self.joint_names):
      if joint_name in profile.action_scale:
        self.action_scale[i] = float(profile.action_scale[joint_name])
      else:
        for pattern, scale in profile.action_scale.items():
          if re.match(pattern, joint_name):
            self.action_scale[i] = float(scale)
            break

    self.viewer = None
    self.paused = False
    self.last_render_time = 0.0
    self.timestep = 0
    self.max_timesteps = int(motion_data["joint_pos"].shape[0])
    self.last_action = torch.zeros(1, self.num_joints, device=device)

  def _sync_warp_data_to_mj_data(self) -> None:
    self.sim.mj_data.qpos[:] = self.sim.data.qpos[0].cpu().numpy().astype(np.float64)
    self.sim.mj_data.qvel[:] = self.sim.data.qvel[0].cpu().numpy().astype(np.float64)
    mujoco.mj_forward(self.sim.mj_model, self.sim.mj_data)

  def reset(self) -> None:
    self.timestep = 0
    self.last_action.zero_()

    init_joint_pos = self.motion_data["joint_pos"][0].unsqueeze(0).to(self.device)
    init_joint_vel = self.motion_data["joint_vel"][0].unsqueeze(0).to(self.device)

    joint_pos = self.robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)
    joint_pos[:, : self.num_joints] = init_joint_pos
    joint_vel[:, : self.num_joints] = init_joint_vel
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

    body_pos = self.motion_data["body_pos_w"][0, 0].to(self.device)
    body_quat = self.motion_data["body_quat_w"][0, 0].to(self.device)
    body_lin_vel = self.motion_data["body_lin_vel_w"][0, 0].to(self.device)
    body_ang_vel = self.motion_data["body_ang_vel_w"][0, 0].to(self.device)

    root_state = torch.zeros(1, 13, device=self.device)
    root_state[:, :3] = body_pos
    root_state[:, 3:7] = body_quat
    root_state[:, 7:10] = body_lin_vel
    root_state[:, 10:13] = body_ang_vel
    self.robot.write_root_state_to_sim(root_state)

    self.sim.forward()
    if self.verbose:
      _, _, yaw = euler_xyz_from_quat(body_quat.unsqueeze(0))
      print(f"[INFO] Reset complete. Initial yaw: {yaw.item() * 180.0 / np.pi:.2f} deg")

  def get_observation(self) -> torch.Tensor:
    t = min(self.timestep, self.max_timesteps - 1)

    joint_pos = self.robot.data.joint_pos[:, : self.num_joints]
    joint_vel = self.robot.data.joint_vel[:, : self.num_joints]
    joint_pos_rel = joint_pos - self.default_joint_pos

    target_joint_pos = self.motion_data["joint_pos"][t].unsqueeze(0).to(self.device)
    target_joint_vel = self.motion_data["joint_vel"][t].unsqueeze(0).to(self.device)

    base_ang_vel = self.robot.data.root_link_ang_vel_b

    body_poses = self.robot.data.body_link_pose_w
    robot_anchor_pos = body_poses[:, self.robot_anchor_body_idx, :3]
    robot_anchor_quat = body_poses[:, self.robot_anchor_body_idx, 3:7]

    motion_anchor_pos = self.motion_data["body_pos_w"][t, self.motion_anchor_body_idx].unsqueeze(0).to(self.device)
    motion_anchor_quat = self.motion_data["body_quat_w"][t, self.motion_anchor_body_idx].unsqueeze(0).to(self.device)

    _, relative_quat = subtract_frame_transforms(
      robot_anchor_pos,
      robot_anchor_quat,
      motion_anchor_pos,
      motion_anchor_quat,
    )
    relative_rot_mat = matrix_from_quat(relative_quat)
    motion_anchor_ori_b = relative_rot_mat[:, :, :2].reshape(1, 6)

    obs = torch.cat(
      [
        target_joint_pos,
        target_joint_vel,
        motion_anchor_ori_b,
        base_ang_vel,
        joint_pos_rel,
        joint_vel,
        self.last_action,
      ],
      dim=-1,
    )
    return obs

  def step(self, action: torch.Tensor) -> None:
    scaled_action = action * self.action_scale
    target_pos = self.default_joint_pos + scaled_action

    # Use the framework's actuator pipeline for consistency with training/play.
    self.robot.set_joint_position_target(target_pos)

    substeps = max(1, int(round(self.dt / self.sim.cfg.mujoco.timestep)))
    for _ in range(substeps):
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(self.sim.cfg.mujoco.timestep)

    self.timestep += 1

  def run(self, num_steps: int | None) -> dict[str, list[float]]:
    steps_to_run = num_steps if num_steps is not None else self.max_timesteps
    print(f"[INFO] Running sim2sim for {steps_to_run} steps")

    if self.render:
      from mujoco import viewer as mj_viewer

      def key_callback(keycode: int):
        if keycode == 32:
          self.paused = not self.paused
          if self.verbose:
            print(f"[INFO] {'Paused' if self.paused else 'Resumed'}")

      self.viewer = mj_viewer.launch_passive(
        self.sim.mj_model,
        self.sim.mj_data,
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False,
      )
      self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL.value] = 0
      self.viewer.cam.distance = 3.0
      self.viewer.cam.azimuth = -90
      self.viewer.cam.elevation = -15

    self.reset()

    errors: dict[str, list[float]] = {"joint_pos": [], "joint_vel": []}

    step = 0
    while step < steps_to_run:
      if self.render and self.viewer is not None and not self.viewer.is_running():
        print("[INFO] Viewer closed, stopping simulation")
        break

      if self.paused:
        if self.render and self.viewer is not None and self.viewer.is_running():
          self.viewer.sync()
        continue

      obs = self.get_observation()
      with torch.no_grad():
        action = self.policy(obs)
      self.step(action)
      self.last_action = action

      t = min(step, self.max_timesteps - 1)
      cur_pos = self.robot.data.joint_pos[:, : self.num_joints]
      cur_vel = self.robot.data.joint_vel[:, : self.num_joints]
      tgt_pos = self.motion_data["joint_pos"][t].unsqueeze(0).to(self.device)
      tgt_vel = self.motion_data["joint_vel"][t].unsqueeze(0).to(self.device)
      pos_err = torch.abs(cur_pos - tgt_pos).mean().item()
      vel_err = torch.abs(cur_vel - tgt_vel).mean().item()
      errors["joint_pos"].append(pos_err)
      errors["joint_vel"].append(vel_err)

      if self.render and self.viewer is not None and self.viewer.is_running() and step % 2 == 0:
        self._sync_warp_data_to_mj_data()
        self.viewer.sync()

      if self.verbose and step % 100 == 0:
        print(
          f"[INFO] step={step}, pos_err={pos_err:.4f} rad ({pos_err * 180.0 / np.pi:.2f} deg), "
          f"vel_err={vel_err:.4f} rad/s"
        )
        if step == 0:
          body_poses = self.robot.data.body_link_pose_w
          robot_quat = body_poses[0, self.robot_anchor_body_idx, 3:7].cpu()
          motion_quat = self.motion_data["body_quat_w"][t, self.motion_anchor_body_idx].cpu()
          _, _, robot_yaw = euler_xyz_from_quat(robot_quat.unsqueeze(0))
          _, _, motion_yaw = euler_xyz_from_quat(motion_quat.unsqueeze(0))
          print(f"[INFO] robot_yaw={robot_yaw.item() * 180.0 / np.pi:.2f} deg")
          print(f"[INFO] motion_yaw={motion_yaw.item() * 180.0 / np.pi:.2f} deg")
          print(f"[INFO] obs_mean={obs.mean().item():.4f}, obs_std={obs.std().item():.4f}")
          print(f"[INFO] action_mean={action.mean().item():.4f}, action_std={action.std().item():.4f}")

      step += 1

      if self.render and self.fps_limit is not None:
        import time

        now = time.time()
        if self.last_render_time > 0:
          frame_duration = 1.0 / self.fps_limit
          elapsed = now - self.last_render_time
          if frame_duration > elapsed:
            time.sleep(frame_duration - elapsed)
        self.last_render_time = time.time()

    if self.render and self.viewer is not None:
      self.viewer.close()

    print("[INFO] Sim2sim complete")
    print(
      "[INFO] Mean errors: "
      f"pos={float(np.mean(errors['joint_pos'])):.4f} rad, "
      f"vel={float(np.mean(errors['joint_vel'])):.4f} rad/s"
    )
    print(
      "[INFO] Max  errors: "
      f"pos={float(np.max(errors['joint_pos'])):.4f} rad, "
      f"vel={float(np.max(errors['joint_vel'])):.4f} rad/s"
    )
    return errors


def _normalize_device(device_arg: str) -> str:
  if device_arg == "cpu":
    return "cpu"
  try:
    return f"cuda:{int(device_arg)}"
  except ValueError:
    return device_arg


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Sim2Sim validation for exported TorchScript policies",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=(
      "Examples:\n"
      "  uv run python src/mjlab/scripts/sim2sim.py -t Mjlab-Tracking-Flat-Booster-K1-No-State-Estimation "
      "-p logs/rsl_rl/k1_tracking/2026-04-13_19-58-18\n"
      "  uv run python src/mjlab/scripts/sim2sim.py -t Mjlab-Tracking-Flat-Booster-K1-No-State-Estimation "
      "-p logs/rsl_rl/k1_tracking/2026-04-13_19-58-18/exported/2026-04-13_19-58-18.pt "
      "--motion /path/to/motion.npz --device cuda:0"
    ),
  )
  parser.add_argument("-t", "--task", required=True, choices=sorted(TASK_ROBOT_PROFILES.keys()))
  parser.add_argument("-p", "--policy", required=True, help="Run dir, exported dir, or policy .pt")
  parser.add_argument("--motion", default=None, help="Motion .npz path; auto-loaded from env.yaml if omitted")
  parser.add_argument("--device", default="cuda:0", help="cpu / cuda:0 / 0")
  parser.add_argument("--num-steps", type=int, default=None, help="Number of control steps")
  parser.add_argument("--dt", type=float, default=0.02, help="Control timestep in seconds")
  parser.add_argument("--no-render", action="store_true", help="Disable interactive MuJoCo viewer")
  parser.add_argument("--fps", type=float, default=None, help="Viewer FPS cap")
  parser.add_argument("--verbose", action="store_true", help="Verbose logs")
  return parser


def main() -> None:
  parser = _build_parser()
  args = parser.parse_args()

  device = _normalize_device(args.device)
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARN] CUDA unavailable, fallback to cpu")
    device = "cpu"

  policy_file = resolve_policy_path(args.policy)
  print(f"[INFO] Resolved policy: {policy_file}")

  motion_path = args.motion
  if motion_path is None:
    motion_path = load_motion_file_from_log(policy_file)
    if motion_path is not None:
      print(f"[INFO] Motion from log: {motion_path}")
  if motion_path is None:
    raise ValueError("Motion file required: set --motion or place policy under a log dir with params/env.yaml")

  motion_file = Path(motion_path)
  if not motion_file.exists():
    raise FileNotFoundError(f"Motion file not found: {motion_file}")

  profile = TASK_ROBOT_PROFILES[args.task]
  policy = load_policy(policy_file, device=device)
  motion_data = load_motion_data(motion_file)

  tester = Sim2SimTester(
    profile=profile,
    motion_data=motion_data,
    policy=policy,
    device=device,
    dt=args.dt,
    render=not args.no_render,
    fps_limit=args.fps,
    verbose=args.verbose,
  )
  tester.run(num_steps=args.num_steps)


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")
    sys.exit(0)
