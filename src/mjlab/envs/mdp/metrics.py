from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def mean_action_acc(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Mean absolute action acceleration.

  Lower values indicate smoother actions.

  Returns:
    Per-environment scalar. Shape: ``(B,)``.
  """
  # Discrete second derivative: a_t - 2 * a_{t-1} + a_{t-2}.  (B, N)
  action_acc = (
    env.action_manager.action
    - 2 * env.action_manager.prev_action
    + env.action_manager.prev_prev_action
  )
  return torch.mean(torch.abs(action_acc), dim=-1)  # (B,)


def joint_pos_deg(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  mode: str = "mean",
) -> torch.Tensor:
  """Joint position diagnostic in degrees.

  ``mode`` selects how multiple matched joints are reduced into one scalar per
  environment.
  """
  asset = env.scene[asset_cfg.name]
  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  joint_pos_deg = torch.rad2deg(joint_pos)

  if mode == "mean":
    return torch.mean(joint_pos_deg, dim=1)
  if mode == "max":
    return torch.max(joint_pos_deg, dim=1).values
  if mode == "min":
    return torch.min(joint_pos_deg, dim=1).values
  if mode == "absmax":
    return torch.max(torch.abs(joint_pos_deg), dim=1).values

  raise ValueError(f"Unsupported joint_pos_deg mode: {mode}")
