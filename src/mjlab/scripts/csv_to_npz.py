from typing import Any, Literal
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from tqdm import tqdm
import os,json

import mjlab

from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.k1.env_cfgs import booster_k1_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.bumi.env_cfgs import noetix_bumi_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.pm01.env_cfgs import engineai_pm01_flat_tracking_env_cfg
# from mjlab.tasks.tracking.config.n1.env_cfgs import N1FlatEnvCfg
from mjlab.tasks.tracking.config.gr3.env_cfgs import fourier_gr3_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.e1.env_cfgs import noetix_e1_flat_tracking_env_cfg
# from mjlab.tasks.tracking.config.z1.env_cfgs import Z1FlatEnvCfg
from mjlab.tasks.tracking.config.oli.env_cfgs import limx_oli_flat_tracking_env_cfg

RobotType = Literal["unitree_g1", "booster_k1", "noetix_bumi", "engineai_pm01", "fourier_n1", "fourier_gr3","noetix_e1", "magicbot_z1","limx_oli"]
robot_json_dict = {}
current_path = os.path.abspath(__file__) #/home/ubt2204/work/mjlab_pm01_v1/
current_dir_path = os.path.dirname(current_path)
root_path = os.path.abspath(os.path.join(current_dir_path, os.pardir))
#TODO 需要确认新路径是否正确
with open(root_path+os.path.join('/tasks/tracking/config/body_name.json'),'r') as f:
    robot_json_dict = json.load(f)

from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
  quat_slerp,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig
from pathlib import Path
import os
SRC_ROOT = Path(__file__).resolve().parents[3]


def quat_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
  """Convert quaternion to rotation matrix.

Args:
  quat: [N, 4] tensor in [w, x, y, z] format
Returns:
  rotmat: [N, 3, 3] rotation matrices
"""
  # Normalize quaternion
  quat = F.normalize(quat, dim=-1)

  w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

  # Compute rotation matrix elements
  xx, yy, zz = x * x, y * y, z * z
  xy, xz, yz = x * y, x * z, y * z
  wx, wy, wz = w * x, w * y, w * z

  # Build rotation matrix
  rotmat = torch.zeros(*quat.shape[:-1], 3, 3, device=quat.device)
  rotmat[..., 0, 0] = 1 - 2 * (yy + zz)
  rotmat[..., 0, 1] = 2 * (xy - wz)
  rotmat[..., 0, 2] = 2 * (xz + wy)
  rotmat[..., 1, 0] = 2 * (xy + wz)
  rotmat[..., 1, 1] = 1 - 2 * (xx + zz)
  rotmat[..., 1, 2] = 2 * (yz - wx)
  rotmat[..., 2, 0] = 2 * (xz - wy)
  rotmat[..., 2, 1] = 2 * (yz + wx)
  rotmat[..., 2, 2] = 1 - 2 * (xx + yy)

  return rotmat


def matrix_to_euler_zyx(rotmat: torch.Tensor) -> torch.Tensor:
  """Convert rotation matrix to Euler angles in ZYX order.

Args:
  rotmat: [N, 3, 3] rotation matrices
Returns:
  euler: [N, 3] Euler angles [z, y, x]
"""
  sy = torch.sqrt(rotmat[..., 0, 0] ** 2 + rotmat[..., 1, 0] ** 2)

  singular = sy < 1e-6

  x = torch.where(singular,
                  torch.atan2(-rotmat[..., 1, 2], rotmat[..., 1, 1]),
                  torch.atan2(rotmat[..., 2, 1], rotmat[..., 2, 2]))
  y = torch.where(singular,
                  torch.atan2(-rotmat[..., 2, 0], sy),
                  torch.atan2(-rotmat[..., 2, 0], sy))
  z = torch.where(singular,
                  torch.zeros_like(rotmat[..., 0, 0]),
                  torch.atan2(rotmat[..., 1, 0], rotmat[..., 0, 0]))

  return torch.stack([z, y, x], dim=-1)


def euler_to_rotation_matrix_zyx(euler: torch.Tensor) -> torch.Tensor:
  """Convert Euler angles in ZYX order to rotation matrix.

Args:
  euler: [N, 3] Euler angles [z, y, x]
Returns:
  rotmat: [N, 3, 3] rotation matrices
"""
  z, y, x = euler[..., 0], euler[..., 1], euler[..., 2]

  # Compute sin and cos
  cx, sx = torch.cos(x), torch.sin(x)
  cy, sy = torch.cos(y), torch.sin(y)
  cz, sz = torch.cos(z), torch.sin(z)

  # Build rotation matrix (ZYX order: Rz * Ry * Rx)
  rotmat = torch.zeros(*euler.shape[:-1], 3, 3, device=euler.device)

  rotmat[..., 0, 0] = cz * cy
  rotmat[..., 0, 1] = cz * sy * sx - sz * cx
  rotmat[..., 0, 2] = cz * sy * cx + sz * sx
  rotmat[..., 1, 0] = sz * cy
  rotmat[..., 1, 1] = sz * sy * sx + cz * cx
  rotmat[..., 1, 2] = sz * sy * cx - cz * sx
  rotmat[..., 2, 0] = -sy
  rotmat[..., 2, 1] = cy * sx
  rotmat[..., 2, 2] = cy * cx

  return rotmat


def process_rotation_matrices_with_yaw_compensation(root_quat: torch.Tensor) -> torch.Tensor:
  """Process body quaternions to rotation matrices with yaw angle compensation.

将第一帧的yaw设置为0，后续帧的yaw值相对于第一帧进行调整

Args:
  root_quat: [N, 4] root body quaternions in [w, x, y, z] format
Returns:
  rot_mat: [N, 3, 3] processed rotation matrices with yaw compensation
"""
  # Step 1: Convert quaternion to rotation matrix
  traj_root_rotmat = quat_to_rotation_matrix(root_quat)  # [N, 3, 3]

  # Step 2: Convert all rotation matrices to Euler angles (ZYX order)
  all_euler = matrix_to_euler_zyx(traj_root_rotmat)  # [N, 3] -> [z, y, x]

  # Step 3: Get the initial yaw angle from the first frame
  init_yaw = all_euler[0, 0].clone()  # First frame's yaw (z component)

  # Step 4: Subtract the initial yaw from all frames' yaw angles
  # This makes the first frame's yaw = 0, and preserves relative yaw changes
  compensated_euler = all_euler.clone()
  compensated_euler[:, 0] = all_euler[:, 0] - init_yaw  # Subtract initial yaw from all yaw angles

  # Step 5: Convert compensated Euler angles back to rotation matrices
  rot_mat = euler_to_rotation_matrix_zyx(compensated_euler)  # [N, 3, 3]

  return rot_mat

class MotionLoader:
  def __init__(
    self,
    motion_file: str,
    input_fps: int,
    output_fps: int,
    device: torch.device | str,
    line_range: tuple[int, int] | None = None,
  ):
    self.motion_file = motion_file
    self.input_fps = input_fps
    self.output_fps = output_fps
    self.input_dt = 1.0 / self.input_fps
    self.output_dt = 1.0 / self.output_fps
    self.current_idx = 0
    self.device = device
    self.line_range = line_range
    self._load_motion()
    self._interpolate_motion()
    self._compute_velocities()

  def _load_motion(self):
    """Loads the motion from the csv file."""
    if self.line_range is None:
      motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
    else:
      motion = torch.from_numpy(
        np.loadtxt(
          self.motion_file,
          delimiter=",",
          skiprows=self.line_range[0] - 1,
          max_rows=self.line_range[1] - self.line_range[0] + 1,
        )
      )
    motion = motion.to(torch.float32).to(self.device)
    # motion[:, 2] -= 0.05
    self.motion_base_poss_input = motion[:, :3]
    self.motion_base_rots_input = motion[:, 3:7]
    self.motion_base_rots_input = self.motion_base_rots_input[
      :, [3, 0, 1, 2]
    ]  # convert to wxyz
    self.motion_dof_poss_input = motion[:, 7:]

    self.input_frames = motion.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt

  def _interpolate_motion(self):
    """Interpolates the motion to the output fps."""
    times = torch.arange(
      0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
    )
    self.output_frames = times.shape[0]
    index_0, index_1, blend = self._compute_frame_blend(times)
    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[index_0],
      self.motion_base_poss_input[index_1],
      blend.unsqueeze(1),
    )
    self.motion_base_rots = self._slerp(
      self.motion_base_rots_input[index_0],
      self.motion_base_rots_input[index_1],
      blend,
    )
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0],
      self.motion_dof_poss_input[index_1],
      blend.unsqueeze(1),
    )
    print(
      f"Motion interpolated, input frames: {self.input_frames}, "
      f"input fps: {self.input_fps}, "
      f"output frames: {self.output_frames}, "
      f"output fps: {self.output_fps}"
    )

  def _lerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Linear interpolation between two tensors."""
    return a * (1 - blend) + b * blend

  def _slerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Spherical linear interpolation between two quaternions."""
    slerped_quats = torch.zeros_like(a)
    for i in range(a.shape[0]):
      slerped_quats[i] = quat_slerp(a[i], b[i], float(blend[i]))
    return slerped_quats

  def _compute_frame_blend(
    self, times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the frame blend for the motion."""
    phase = times / self.duration
    index_0 = (phase * (self.input_frames - 1)).floor().long()
    index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
    blend = phase * (self.input_frames - 1) - index_0
    return index_0, index_1, blend

  def _compute_velocities(self):
    """Computes the velocities of the motion."""
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_base_ang_vels = self._so3_derivative(
      self.motion_base_rots, self.output_dt
    )

  def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
    """Computes the derivative of a sequence of SO3 rotations.

    Args:
      rotations: shape (B, 4).
      dt: time step.
    Returns:
      shape (B, 3).
    """
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
    omega = torch.cat(
      [omega[:1], omega, omega[-1:]], dim=0
    )  # repeat first and last sample
    return omega

  def get_next_state(
    self,
  ) -> tuple[
    tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
    ],
    bool,
  ]:
    """Gets the next state of the motion."""
    state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    self.current_idx += 1
    reset_flag = False
    if self.current_idx >= self.output_frames:
      self.current_idx = 0
      reset_flag = True
    return state, reset_flag


def run_sim(
  sim: Simulation,
  scene: Scene,
  joint_names,
  input_file,
  input_fps,
  output_fps,
  output_name,
  render,
  line_range,
  renderer: OffscreenRenderer | None = None,
):
  motion = MotionLoader(
    motion_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
    line_range=line_range,
  )

  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

  log: dict[str, Any] = {
    "fps": [output_fps],
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }
  file_saved = False

  frames = []
  scene.reset()

  print(f"\nStarting simulation with {motion.output_frames} frames...")
  if render:
    print("Rendering enabled - generating video frames...")

  # Create progress bar
  pbar = tqdm(
    total=motion.output_frames,
    desc="Processing frames",
    unit="frame",
    ncols=100,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
  )

  frame_count = 0
  while not file_saved:
    (
      (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
      ),
      reset_flag,
    ) = motion.get_next_state()

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = motion_base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = motion_base_rot
    root_states[:, 7:10] = motion_base_lin_vel
    root_states[:, 10:] = motion_base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = motion_dof_pos
    joint_vel[:, robot_joint_indexes] = motion_dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)
    if render and renderer is not None:
      renderer.update(sim.data)
      frames.append(renderer.render())

    if not file_saved:
      log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
      log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
      log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
      log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
      log["body_lin_vel_w"].append(
        robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
      )
      log["body_ang_vel_w"].append(
        robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
      )

      torch.testing.assert_close(
        robot.data.body_link_lin_vel_w[0, 0], motion_base_lin_vel[0]
      )
      torch.testing.assert_close(
        robot.data.body_link_ang_vel_w[0, 0], motion_base_ang_vel[0]
      )

      frame_count += 1
      pbar.update(1)

      if frame_count % 100 == 0:  # Update every 100 frames to avoid spam
        elapsed_time = frame_count / output_fps
        pbar.set_description(f"Processing frames (t={elapsed_time:.1f}s)")

      if reset_flag and not file_saved:
        file_saved = True
        pbar.close()

        print("\nStacking arrays and saving data...")
        for k in (
          "joint_pos",
          "joint_vel",
          "body_pos_w",
          "body_quat_w",
          "body_lin_vel_w",
          "body_ang_vel_w",
        ):
          log[k] = np.stack(log[k], axis=0)
        from pathlib import Path
        input_dir = Path(input_file).parent
        output_npz = input_dir / f"{output_name}.npz"
        # 检查路径是否存在
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving motion data to {output_npz}...")
        np.savez(output_npz, **log)
        # Generate PT trajectory file for C++ deployment
        print("\n[INFO]: Generating PT trajectory file for C++ deployment...")

        # Convert numpy arrays to torch tensors (CPU)
        joint_pos_tensor = torch.from_numpy(log["joint_pos"]).float()  # [N, num_joints]
        joint_vel_tensor = torch.from_numpy(log["joint_vel"]).float()  # [N, num_joints]
        body_quat_w_array = log["body_quat_w"]  # numpy array

        print(f"[INFO]: body_quat_w shape: {body_quat_w_array.shape}")

        # Extract root body quaternion
        # body_quat_w shape is [N, num_bodies, 4] (from body_link_quat_w)
        if len(body_quat_w_array.shape) == 3:  # [N, num_bodies, 4]
          root_quat = torch.from_numpy(body_quat_w_array[:, 0, :]).float()  # [N, 4] - first body
        else:  # [N, num_bodies * 4]
          root_quat = torch.from_numpy(body_quat_w_array[:, :4]).float()  # [N, 4] - first 4 elements

        print(f"[INFO]: Root quaternion shape: {root_quat.shape}")

        # Apply yaw compensation to rotation matrices
        rot_mat_tensor = process_rotation_matrices_with_yaw_compensation(root_quat)  # [N, 3, 3]
        print(f"[INFO]: Rotation matrix shape after yaw compensation: {rot_mat_tensor.shape}")

        # Print yaw compensation details
        if rot_mat_tensor.shape[0] > 0:
          # Get yaw angles before and after compensation for verification
          original_rotmat = quat_to_rotation_matrix(root_quat)
          original_euler = matrix_to_euler_zyx(original_rotmat)
          compensated_euler = matrix_to_euler_zyx(rot_mat_tensor)

          original_first_yaw = torch.rad2deg(original_euler[0, 0])
          original_last_yaw = torch.rad2deg(original_euler[-1, 0])
          compensated_first_yaw = torch.rad2deg(compensated_euler[0, 0])
          compensated_last_yaw = torch.rad2deg(compensated_euler[-1, 0])

          print(f"[INFO]: Yaw compensation applied:")
          print(
            f"        Original - First frame: {original_first_yaw.item():.2f}°, Last frame: {original_last_yaw.item():.2f}°")
          print(
            f"        Compensated - First frame: {compensated_first_yaw.item():.2f}°, Last frame: {compensated_last_yaw.item():.2f}°")
          print(f"        Yaw offset removed: {original_first_yaw.item():.2f}°")

        # Apply IsaacLab joint reordering for C++ deployment compatibility
        # This mapping converts from MJLab/CSV order to IsaacLab order
        # IsaacLab groups joints by type: yaw axes, pitch axes, elbows, etc.

        isaaclab_joint_mapping = []
        if 'k1' in robot.spec.modelname:
          isaaclab_joint_mapping = robot_json_dict['booster_k1']["isaaclab_joint_mapping"]
        elif 'bumi' in robot.spec.modelname:
          isaaclab_joint_mapping = robot_json_dict['noetix_bumi']["isaaclab_joint_mapping"]
        elif 'pm01' in robot.spec.modelname:
          isaaclab_joint_mapping = robot_json_dict['engineai_pm01']["isaaclab_joint_mapping"]
        elif 'n1' in robot.spec.modelname:
          isaaclab_joint_mapping = robot_json_dict['fourier_n1']["isaaclab_joint_mapping"]
        elif 'gr3' in robot.spec.modelname:
          isaaclab_joint_mapping = robot_json_dict['fourier_gr3']["isaaclab_joint_mapping"]
        elif 'e1' in robot.spec.modelname:
          isaaclab_joint_mapping = robot_json_dict['noetix_e1']["isaaclab_joint_mapping"]
        elif 'z1' in robot.spec.modelname:
          isaaclab_joint_mapping = robot_json_dict['magicbot_z1']["isaaclab_joint_mapping"]
        elif 'oli' in robot.spec.modelname:
          isaaclab_joint_mapping = robot_json_dict['limx_oli']["isaaclab_joint_mapping"]
        # Reorder joints for IsaacLab compatibility
        joint_pos_isaaclab = joint_pos_tensor[:, isaaclab_joint_mapping]
        joint_vel_isaaclab = joint_vel_tensor[:, isaaclab_joint_mapping]

        print(f"[INFO]: Applied IsaacLab joint reordering for C++ compatibility")

        # Create PT data dictionary (CPU version, IsaacLab joint order)
        pt_data_cpu = {
          "dof_pos": joint_pos_isaaclab,
          "dof_vel": joint_vel_isaaclab,
          "rot_mat": rot_mat_tensor
        }

        # Save CPU version as PT file (IsaacLab joint order for C++ deployment)
        output_pt = input_dir / f"{output_name}_traj.pt"
        torch.save(pt_data_cpu, output_pt)
        print(f"[INFO]: PT trajectory file saved to {output_pt} (IsaacLab joint order)")
        print(
          f"[INFO]: PT data shapes - dof_pos: {joint_pos_isaaclab.shape}, dof_vel: {joint_vel_isaaclab.shape}, rot_mat: {rot_mat_tensor.shape}")

        if render:
          from moviepy import ImageSequenceClip
          output_video = input_dir / f"{output_name}.mp4"
          print(f"Creating video: {output_video}...")
          clip = ImageSequenceClip(frames, fps=output_fps)
          clip.write_videofile(str(output_video))
          print(f"[INFO]: Video saved to {output_video}")




def main(
  input_file: str,
  output_name: str,
  robot_type: RobotType = "noetix_bumi",
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  line_range: tuple[int, int] | None = None,
):
  """Replay motion from CSV file and output to npz file.

  Args:
    input_file: Path to the input CSV file.
    output_name: Path to the output npz file.
    input_fps: Frame rate of the CSV file.
    output_fps: Desired output frame rate.
    device: Device to use.
    render: Whether to render the simulation and save a video.
    line_range: Range of lines to process from the CSV file.
  """
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA is not available. Falling back to CPU. This may be slow.")
    device = "cpu"

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  env_cfg = None
  # Get robot-specific configuration
  if robot_type == "unitree_g1":
    env_cfg = unitree_g1_flat_tracking_env_cfg()
  elif robot_type == "booster_k1":
    env_cfg = booster_k1_flat_tracking_env_cfg()
  elif robot_type == "engineai_pm01":
    env_cfg = engineai_pm01_flat_tracking_env_cfg()
  elif robot_type == "limx_oli":
    env_cfg = limx_oli_flat_tracking_env_cfg()
  elif robot_type == "noetix_bumi":
    env_cfg = noetix_bumi_flat_tracking_env_cfg()
  elif robot_type == "noetix_e1":
    env_cfg = noetix_e1_flat_tracking_env_cfg()
  elif robot_type == "fourier_gr3":
    env_cfg = fourier_gr3_flat_tracking_env_cfg()
  #TODO 记得新增机型时需要更新此处

  else:
    raise ValueError(f"Unknown robot type: {robot_type}")

  scene = Scene(env_cfg.scene, device=device)
  model = scene.compile()

  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)

  scene.initialize(sim.mj_model, sim.model, sim.data)

  renderer = None
  if render:
    viewer_cfg = ViewerConfig(
      height=480,
      width=640,
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      entity_name="robot",
      distance=2.0,
      elevation=-5.0,
      azimuth=-90,
    )
    renderer = OffscreenRenderer(
      model=sim.mj_model,
      cfg=viewer_cfg,
      scene=scene,
    )
    renderer.initialize()

  joint_names = []
  # Robot-specific joint names
  if robot_type == "unitree_g1":
    joint_names = robot_json_dict['unitree_g1']['joint_names']
  elif robot_type == "booster_k1":
    joint_names = robot_json_dict['booster_k1']['joint_names']
  elif robot_type == "noetix_bumi":
    joint_names = robot_json_dict['noetix_bumi']['joint_names']
  elif robot_type == "engineai_pm01":
    joint_names = robot_json_dict['engineai_pm01']['joint_names']
  elif robot_type == "fourier_n1":
    joint_names = robot_json_dict['fourier_n1']['joint_names']
  elif robot_type == "fourier_gr3":
    joint_names = robot_json_dict['fourier_gr3']['joint_names']
  elif robot_type == "noetix_e1":
    joint_names = robot_json_dict['noetix_e1']['joint_names']
  elif robot_type == "magicbot_z1":
    joint_names = robot_json_dict['magicbot_z1']['joint_names']
  elif robot_type == "limx_oli":
    joint_names = robot_json_dict['limx_oli']['joint_names']

  run_sim(
    sim=sim,
    scene=scene,
    joint_names=joint_names,
    input_fps=input_fps,
    input_file=input_file,
    output_fps=output_fps,
    output_name=output_name,
    render=render,
    line_range=line_range,
    renderer=renderer,
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
