"""Utilities for stateful recurrent policies such as LSTM actors."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def is_recurrent_policy(policy: Any) -> bool:
  """Return whether a policy advertises recurrent state."""
  return bool(getattr(policy, "is_recurrent", False))


def extract_dones(step_result: Any) -> torch.Tensor | None:
  """Extract done flags from wrapped or raw vector-env step returns."""
  if not isinstance(step_result, tuple | list):
    return None

  dones: torch.Tensor | None = None
  if len(step_result) == 4 and torch.is_tensor(step_result[2]):
    dones = step_result[2]
  elif (
    len(step_result) >= 5
    and torch.is_tensor(step_result[2])
    and torch.is_tensor(step_result[3])
  ):
    dones = torch.logical_or(step_result[2].bool(), step_result[3].bool())

  if dones is None:
    return None
  if dones.ndim > 1 and dones.shape[-1] == 1:
    dones = dones.squeeze(-1)
  return dones


def reset_policy_state(policy: Any, dones: torch.Tensor | None = None) -> bool:
  """Reset a policy's recurrent state.

  Passing ``dones`` resets only finished environments. Passing ``None`` resets all
  environments, which is appropriate after a manual environment reset or checkpoint
  hot-swap.
  """
  reset_fn = getattr(policy, "reset", None)
  if reset_fn is None:
    return False

  if dones is None:
    reset_fn()
    return True

  if not is_recurrent_policy(policy):
    return False

  reset_fn(dones)
  return True


def reset_policy_state_from_step(policy: Any, step_result: Any) -> bool:
  """Reset recurrent policy state using done flags returned by ``env.step``."""
  dones = extract_dones(step_result)
  if dones is None:
    return False
  return reset_policy_state(policy, dones)


def _get_rnn_module(policy: Any) -> nn.Module | None:
  rnn = getattr(policy, "rnn", None)
  if rnn is None:
    return None
  return getattr(rnn, "rnn", rnn)


def get_recurrent_policy_metadata(policy: Any) -> dict[str, list | str | float]:
  """Return ONNX metadata that tells deployment code how to carry policy state."""
  if not is_recurrent_policy(policy):
    return {"policy_is_recurrent": "false"}

  rnn_module = _get_rnn_module(policy)
  rnn_type = getattr(policy, "rnn_type", None)
  if rnn_type is None and rnn_module is not None:
    if isinstance(rnn_module, nn.LSTM):
      rnn_type = "lstm"
    elif isinstance(rnn_module, nn.GRU):
      rnn_type = "gru"
    else:
      rnn_type = type(rnn_module).__name__.lower()
  rnn_type = str(rnn_type or "unknown").lower()

  hidden_size = getattr(policy, "hidden_size", None)
  num_layers = getattr(policy, "num_layers", None)
  if rnn_module is not None:
    hidden_size = hidden_size or getattr(rnn_module, "hidden_size", None)
    num_layers = num_layers or getattr(rnn_module, "num_layers", None)

  metadata: dict[str, list | str | float] = {
    "policy_is_recurrent": "true",
    "policy_recurrent_type": rnn_type,
    "policy_recurrent_hidden_size": str(hidden_size or ""),
    "policy_recurrent_num_layers": str(num_layers or ""),
    "policy_recurrent_state_batch_axis": "1",
    "policy_recurrent_state_step_rule": "feed h_out/c_out back as next h_in/c_in; zero state on reset",
  }

  if rnn_type == "lstm":
    metadata.update(
      {
        "policy_onnx_input_names": ["obs", "h_in", "c_in"],
        "policy_onnx_output_names": ["actions", "h_out", "c_out"],
        "policy_recurrent_state_names": ["h", "c"],
      }
    )
  elif rnn_type == "gru":
    metadata.update(
      {
        "policy_onnx_input_names": ["obs", "h_in"],
        "policy_onnx_output_names": ["actions", "h_out"],
        "policy_recurrent_state_names": ["h"],
      }
    )

  return metadata


__all__ = [
  "extract_dones",
  "get_recurrent_policy_metadata",
  "is_recurrent_policy",
  "reset_policy_state",
  "reset_policy_state_from_step",
]
