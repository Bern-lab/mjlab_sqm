"""Policy loading helpers for offline evaluation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from mjlab.utils.lstm import reset_policy_state
from mjlab.utils.os import get_task_log_root, get_wandb_checkpoint_path


def make_inference_train_cfg(agent_cfg: Any) -> dict[str, Any]:
  """Convert a runner config into an inference-only train config.

  Teacher-KL checkpoints should not be required for offline actor inference, so
  PPOTeacherKL configs are locally converted to PPO while preserving the actor
  and critic model definitions needed to build the network.
  """
  cfg = asdict(agent_cfg) if is_dataclass(agent_cfg) else deepcopy(agent_cfg)
  cfg["upload_model"] = False

  algorithm_cfg = cfg.get("algorithm", {})
  if algorithm_cfg.get("class_name") == "PPOTeacherKL":
    algorithm_cfg = dict(algorithm_cfg)
    algorithm_cfg["class_name"] = "PPO"
    algorithm_cfg.pop("teacher_kl_cfg", None)
    cfg["algorithm"] = algorithm_cfg
    cfg.pop("teacher", None)
    cfg["obs_groups"] = {
      key: value
      for key, value in cfg.get("obs_groups", {}).items()
      if key in ("actor", "critic")
    }

  return cfg


def resolve_checkpoint_path(
  *,
  task_id: str,
  agent_cfg: Any,
  checkpoint_file: str | None,
  wandb_run_path: str | None,
  wandb_checkpoint_name: str | None,
) -> Path:
  """Resolve either a local checkpoint or a W&B checkpoint."""
  if checkpoint_file is not None:
    path = Path(checkpoint_file).expanduser()
    if not path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {path}")
    return path

  if wandb_run_path is None:
    raise ValueError("Provide either --checkpoint-file or --wandb-run-path.")

  log_root_path = get_task_log_root(agent_cfg.experiment_name, task_id).resolve()
  path, _ = get_wandb_checkpoint_path(
    log_root_path, Path(wandb_run_path), wandb_checkpoint_name
  )
  return path


def load_inference_policy(
  *,
  env: RslRlVecEnvWrapper,
  task_id: str,
  agent_cfg: Any,
  checkpoint_path: Path,
  device: str,
):
  """Build a runner, load actor weights, and return the inference policy."""
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  train_cfg = make_inference_train_cfg(agent_cfg)
  runner = runner_cls(env, train_cfg, device=device)
  runner.load(
    str(checkpoint_path),
    load_cfg={
      "actor": True,
      "critic": False,
      "optimizer": False,
      "iteration": True,
      "rnd": False,
    },
    strict=True,
    map_location=device,
  )
  policy = runner.get_inference_policy(device=device)
  policy.eval()
  reset_policy_state(policy)
  return policy, runner

