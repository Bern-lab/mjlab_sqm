from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class _TeacherTargetHeadingCommandTerm(Protocol):
  target_pos_w: torch.Tensor
  is_target_env: torch.Tensor
  target_reached: torch.Tensor


def teacher_target_progress(
  env: "ManagerBasedRlEnv",
  command_name: str,
  min_distance: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward signed world-frame velocity projected toward the active target."""
  asset: Entity = env.scene[asset_cfg.name]
  command_term = env.command_manager.get_term(command_name)
  if (
    command_term is None
    or not hasattr(command_term, "target_pos_w")
    or not hasattr(command_term, "is_target_env")
  ):
    return torch.zeros(env.num_envs, device=env.device)

  target_command = cast(_TeacherTargetHeadingCommandTerm, command_term)
  target_pos_w = target_command.target_pos_w
  is_target_env = target_command.is_target_env
  delta_xy = target_pos_w[:, :2] - asset.data.root_link_pos_w[:, :2]
  distance = torch.linalg.norm(delta_xy, dim=1)
  direction = delta_xy / torch.clamp(distance, min=min_distance).unsqueeze(1)
  progress = torch.sum(asset.data.root_link_lin_vel_w[:, :2] * direction, dim=1)
  progress = torch.clamp(progress, min=-1.0, max=1.0)

  active = is_target_env & (distance > min_distance)
  return torch.where(active, progress, torch.zeros_like(progress))


def teacher_target_reached_bonus(
  env: "ManagerBasedRlEnv",
  command_name: str,
) -> torch.Tensor:
  """Return a one-step bonus for target-heading envs that reached a target."""
  command_term = env.command_manager.get_term(command_name)
  if command_term is None or not hasattr(command_term, "target_reached"):
    return torch.zeros(env.num_envs, device=env.device)
  target_command = cast(_TeacherTargetHeadingCommandTerm, command_term)
  return target_command.target_reached.float()
