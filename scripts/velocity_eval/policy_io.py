"""Policy loading helpers for offline evaluation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any

from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_runner_cls
from mjlab.utils.lstm import reset_policy_state
from mjlab.utils.os import get_task_log_root, get_wandb_checkpoint_path


def _slugify(value: str) -> str:
  value = value.strip().lower()
  value = re.sub(r"^mjlab[-_]*", "", value)
  value = re.sub(r"[-_]*unitree[-_]*g1$", "", value)
  value = re.sub(r"[^a-z0-9]+", "_", value)
  value = re.sub(r"_+", "_", value).strip("_")
  return value or "policy"


def get_policy_output_name(
  *,
  task_id: str,
  agent_cfg: Any,
  checkpoint_path: Path | None = None,
) -> str:
  """Return a concise folder name for outputs from one trained policy family."""
  experiment_name = getattr(agent_cfg, "experiment_name", None)
  if experiment_name:
    return _slugify(str(experiment_name))

  if checkpoint_path is not None:
    parts = checkpoint_path.parts
    if "rsl_rl" in parts:
      idx = parts.index("rsl_rl")
      if len(parts) > idx + 1:
        return _slugify(parts[idx + 1])

  return _slugify(task_id)


def make_timestamped_policy_output_dir(
  *,
  output_root: str | Path,
  task_id: str,
  agent_cfg: Any,
  checkpoint_path: Path | None = None,
) -> Path:
  """Create ``output_root / policy_name / timestamp`` with collision suffixes."""
  policy_name = get_policy_output_name(
    task_id=task_id,
    agent_cfg=agent_cfg,
    checkpoint_path=checkpoint_path,
  )
  root = Path(output_root) / policy_name
  timestamp = datetime.now().strftime("%m%d_%H%M%S")
  output_dir = root / timestamp
  suffix = 1
  while output_dir.exists():
    output_dir = root / f"{timestamp}_{suffix:02d}"
    suffix += 1
  output_dir.mkdir(parents=True, exist_ok=False)
  return output_dir


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
