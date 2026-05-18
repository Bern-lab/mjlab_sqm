from __future__ import annotations

"""Convert an mjlab PM01 policy to an EngineAI-compatible MNN model.

This script solves the PM01 joint-order mismatch between:

- mjlab exported policies / ONNX metadata order:
  J00, J01, J02, ..., J23
- EngineAI pm01_edu rl_dance_example runtime order:
  J00, J06, J12, J01, ...

The EngineAI runtime does *not* reorder every observation block the same way:
- ``command`` stays in the trajectory's original mjlab order
- ``motion_anchor_ori_b`` and ``base_ang_vel`` are order-independent
- ``joint_pos``, ``joint_vel``, and previous ``actions`` follow the EngineAI YAML

To adapt an mjlab policy for the EngineAI runtime, this script inserts a thin
reordering wrapper:

- input obs:
  EngineAI runtime layout -> mjlab policy layout
- output actions:
  mjlab policy layout -> EngineAI runtime layout

Supported inputs:
- ``policy.onnx`` exported from mjlab
- exported TorchScript ``.pt`` policies (not raw training checkpoints)

Examples
--------
ONNX -> MNN:

  uv run python src/mjlab/scripts/convert_pm01_policy_to_mnn.py \\
    -i logs/rsl_rl/pm01_tracking/run/policy.onnx \\
    --output-onnx /tmp/pm01_engineai.onnx \\
    --output-mnn /home/ubt2204/work_ljh/pm01-engineai_robotics_native_sdk-main/assets/config/pm01_edu/rl_dance_example/dance_22/policy/policy.mnn

TorchScript -> ONNX -> MNN:

  uv run python src/mjlab/scripts/convert_pm01_policy_to_mnn.py \\
    -i logs/rsl_rl/pm01_tracking/run/exported/run.pt \\
    --output-onnx /tmp/pm01_engineai.onnx \\
    --output-mnn /home/ubt2204/work_ljh/pm01-engineai_robotics_native_sdk-main/assets/config/pm01_edu/rl_dance_example/dance_22/policy/policy.mnn
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import onnx
import torch
from onnx import TensorProto, checker, helper

MJLAB_PM01_JOINT_ORDER = [
  "J00_HIP_PITCH_L",
  "J01_HIP_ROLL_L",
  "J02_HIP_YAW_L",
  "J03_KNEE_PITCH_L",
  "J04_ANKLE_PITCH_L",
  "J05_ANKLE_ROLL_L",
  "J06_HIP_PITCH_R",
  "J07_HIP_ROLL_R",
  "J08_HIP_YAW_R",
  "J09_KNEE_PITCH_R",
  "J10_ANKLE_PITCH_R",
  "J11_ANKLE_ROLL_R",
  "J12_WAIST_YAW",
  "J13_SHOULDER_PITCH_L",
  "J14_SHOULDER_ROLL_L",
  "J15_SHOULDER_YAW_L",
  "J16_ELBOW_PITCH_L",
  "J17_ELBOW_YAW_L",
  "J18_SHOULDER_PITCH_R",
  "J19_SHOULDER_ROLL_R",
  "J20_SHOULDER_YAW_R",
  "J21_ELBOW_PITCH_R",
  "J22_ELBOW_YAW_R",
  "J23_HEAD_YAW",
]

ENGINEAI_PM01_DANCE_JOINT_ORDER = [
  "J00_HIP_PITCH_L",
  "J06_HIP_PITCH_R",
  "J12_WAIST_YAW",
  "J01_HIP_ROLL_L",
  "J07_HIP_ROLL_R",
  "J13_SHOULDER_PITCH_L",
  "J18_SHOULDER_PITCH_R",
  "J23_HEAD_YAW",
  "J02_HIP_YAW_L",
  "J08_HIP_YAW_R",
  "J14_SHOULDER_ROLL_L",
  "J19_SHOULDER_ROLL_R",
  "J03_KNEE_PITCH_L",
  "J09_KNEE_PITCH_R",
  "J15_SHOULDER_YAW_L",
  "J20_SHOULDER_YAW_R",
  "J04_ANKLE_PITCH_L",
  "J10_ANKLE_PITCH_R",
  "J16_ELBOW_PITCH_L",
  "J21_ELBOW_PITCH_R",
  "J05_ANKLE_ROLL_L",
  "J11_ANKLE_ROLL_R",
  "J17_ELBOW_YAW_L",
  "J22_ELBOW_YAW_R",
]

# Tailored to pm01_edu/rl_dance_example/default.yaml observation_names:
# command(48), motion_anchor_ori_b(6), base_ang_vel(3), joint_pos(24), joint_vel(24), actions(24)
PM01_ENGINEAI_DANCE_OBS_DIM = 48 + 6 + 3 + 24 + 24 + 24
PM01_ENGINEAI_DANCE_INPUT_BLOCK_OFFSETS = (57, 81, 105)


def _read_onnx_metadata(model: onnx.ModelProto) -> dict[str, str]:
  return {entry.key: entry.value for entry in model.metadata_props}


def _csv_list(value: str) -> list[str]:
  return [item.strip() for item in value.split(",") if item.strip()]


def _load_source_joint_order_from_onnx(model: onnx.ModelProto) -> list[str] | None:
  metadata = _read_onnx_metadata(model)
  joint_names = metadata.get("joint_names")
  if not joint_names:
    return None
  parsed = _csv_list(joint_names)
  return parsed if len(parsed) == 24 else None


def _ensure_same_joint_set(source_order: list[str], target_order: list[str]) -> None:
  if len(source_order) != 24 or len(target_order) != 24:
    raise ValueError("PM01 conversion expects exactly 24 joints in both orders.")
  if set(source_order) != set(target_order):
    raise ValueError("Source and target joint orders do not contain the same joints.")


def _compute_input_permutation(
  source_order: list[str], target_order: list[str]
) -> list[int]:
  """Map EngineAI runtime obs layout to the mjlab policy layout.

  ``command`` stays in mjlab order already, so only joint_pos/joint_vel/actions are
  reordered block-wise from target -> source.
  """
  _ensure_same_joint_set(source_order, target_order)

  joint_block_perm = [target_order.index(name) for name in source_order]
  permutation = list(range(PM01_ENGINEAI_DANCE_INPUT_BLOCK_OFFSETS[0]))

  for offset in PM01_ENGINEAI_DANCE_INPUT_BLOCK_OFFSETS:
    permutation.extend(offset + idx for idx in joint_block_perm)

  if len(permutation) != PM01_ENGINEAI_DANCE_OBS_DIM:
    raise RuntimeError(
      f"Unexpected PM01 observation permutation length: {len(permutation)}"
    )
  return permutation


def _compute_output_permutation(
  source_order: list[str], target_order: list[str]
) -> list[int]:
  """Map mjlab policy action order to the EngineAI runtime order."""
  _ensure_same_joint_set(source_order, target_order)
  return [source_order.index(name) for name in target_order]


def _replace_graph_inputs(
  graph: onnx.GraphProto, old_name: str, new_name: str
) -> None:
  for node in graph.node:
    for i, input_name in enumerate(node.input):
      if input_name == old_name:
        node.input[i] = new_name


def _replace_graph_outputs(
  graph: onnx.GraphProto, old_name: str, new_name: str
) -> None:
  for node in graph.node:
    for i, output_name in enumerate(node.output):
      if output_name == old_name:
        node.output[i] = new_name


def _make_int64_initializer(name: str, values: list[int]) -> onnx.TensorProto:
  return helper.make_tensor(name, TensorProto.INT64, [len(values)], values)


def _find_value_info(
  value_infos: list[onnx.ValueInfoProto], name: str
) -> onnx.ValueInfoProto | None:
  for value_info in value_infos:
    if value_info.name == name:
      return value_info
  return None


def _backward_reachable_nodes(
  graph: onnx.GraphProto, required_outputs: list[str]
) -> tuple[set[str], set[int]]:
  graph_input_names = {value.name for value in graph.input}
  initializer_names = {init.name for init in graph.initializer}
  output_to_node: dict[str, onnx.NodeProto] = {}
  for node in graph.node:
    for output_name in node.output:
      if output_name:
        output_to_node[output_name] = node

  needed_tensors = set(required_outputs)
  needed_nodes: set[int] = set()
  changed = True
  while changed:
    changed = False
    for tensor_name in list(needed_tensors):
      if not tensor_name:
        continue
      if tensor_name in graph_input_names or tensor_name in initializer_names:
        continue
      node = output_to_node.get(tensor_name)
      if node is None:
        continue
      node_id = id(node)
      if node_id not in needed_nodes:
        needed_nodes.add(node_id)
        changed = True
      for input_name in node.input:
        if input_name and input_name not in needed_tensors:
          needed_tensors.add(input_name)
          changed = True
  return needed_tensors, needed_nodes


def _prune_onnx_to_obs_actions(model: onnx.ModelProto) -> onnx.ModelProto:
  """Keep only the obs -> actions subgraph if the model still contains tracking extras."""
  graph = model.graph
  input_names = [value.name for value in graph.input]
  output_names = [value.name for value in graph.output]
  if input_names == ["obs"] and output_names == ["actions"]:
    return model
  if "obs" not in input_names or "actions" not in output_names:
    raise ValueError(
      f"Unsupported ONNX inputs/outputs for PM01 conversion. inputs={input_names}, outputs={output_names}"
    )

  needed_tensors, needed_node_ids = _backward_reachable_nodes(graph, ["actions"])

  if "time_step" in needed_tensors:
    raise RuntimeError(
      "The actions output still depends on time_step; please export a pure obs->actions ONNX first."
    )

  kept_nodes = [node for node in graph.node if id(node) in needed_node_ids]
  kept_initializers = [init for init in graph.initializer if init.name in needed_tensors]
  obs_input = _find_value_info(list(graph.input), "obs")
  actions_output = _find_value_info(list(graph.output), "actions")
  if obs_input is None or actions_output is None:
    raise RuntimeError("Failed to locate obs/actions value infos while pruning ONNX.")

  kept_value_info = [value for value in graph.value_info if value.name in needed_tensors]

  new_graph = helper.make_graph(
    nodes=kept_nodes,
    name=(graph.name + "_obs_actions") if graph.name else "pm01_obs_actions",
    inputs=[obs_input],
    outputs=[actions_output],
    initializer=kept_initializers,
    value_info=kept_value_info,
  )
  new_model = helper.make_model(
    new_graph,
    producer_name=model.producer_name,
    producer_version=model.producer_version,
    domain=model.domain,
    model_version=model.model_version,
    doc_string=model.doc_string,
  )
  new_model.ir_version = model.ir_version

  del new_model.opset_import[:]
  for opset in model.opset_import:
    new_model.opset_import.append(opset)

  del new_model.metadata_props[:]
  for prop in model.metadata_props:
    entry = new_model.metadata_props.add()
    entry.key = prop.key
    entry.value = prop.value

  checker.check_model(new_model)
  return new_model


def _reorder_metadata_list(
  metadata: dict[str, str], key: str, target_order: list[str], source_order: list[str]
) -> str | None:
  raw = metadata.get(key)
  if raw is None:
    return None
  items = _csv_list(raw)
  if len(items) != 24:
    return None
  index_map = [source_order.index(name) for name in target_order]
  reordered = [items[idx] for idx in index_map]
  return ",".join(reordered)


def rewrite_onnx_joint_order(
  input_onnx: Path,
  output_onnx: Path,
  source_order: list[str] | None = None,
  target_order: list[str] | None = None,
) -> dict[str, Any]:
  target_order = target_order or ENGINEAI_PM01_DANCE_JOINT_ORDER

  model = onnx.load(str(input_onnx))
  model = _prune_onnx_to_obs_actions(model)
  metadata = _read_onnx_metadata(model)

  if source_order is None:
    source_order = _load_source_joint_order_from_onnx(model) or MJLAB_PM01_JOINT_ORDER

  _ensure_same_joint_set(source_order, target_order)
  input_perm = _compute_input_permutation(source_order, target_order)
  output_perm = _compute_output_permutation(source_order, target_order)

  graph = model.graph
  input_names = [value.name for value in graph.input]
  output_names = [value.name for value in graph.output]
  if input_names != ["obs"] or output_names != ["actions"]:
    raise ValueError(
      f"Expected obs->actions ONNX after pruning, got inputs={input_names}, outputs={output_names}"
    )

  original_input = graph.input[0]
  original_output = graph.output[0]
  input_internal_name = "obs_mjlab_internal"
  output_internal_name = "actions_mjlab_internal"

  _replace_graph_inputs(graph, original_input.name, input_internal_name)
  _replace_graph_outputs(graph, original_output.name, output_internal_name)

  input_indices_name = "pm01_engineai_input_indices"
  output_indices_name = "pm01_engineai_output_indices"
  graph.initializer.extend(
    [
      _make_int64_initializer(input_indices_name, input_perm),
      _make_int64_initializer(output_indices_name, output_perm),
    ]
  )

  input_gather = helper.make_node(
    "Gather",
    inputs=["obs", input_indices_name],
    outputs=[input_internal_name],
    axis=1,
    name="pm01_engineai_input_reorder",
  )
  output_gather = helper.make_node(
    "Gather",
    inputs=[output_internal_name, output_indices_name],
    outputs=["actions"],
    axis=1,
    name="pm01_engineai_output_reorder",
  )

  new_input = helper.make_tensor_value_info(
    "obs", TensorProto.FLOAT, [1, PM01_ENGINEAI_DANCE_OBS_DIM]
  )
  new_output = helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 24])

  del graph.input[:]
  graph.input.extend([new_input])
  del graph.output[:]
  graph.output.extend([new_output])
  graph.node.insert(0, input_gather)
  graph.node.append(output_gather)

  for prop in list(model.metadata_props):
    if prop.key == "joint_names":
      prop.value = ",".join(target_order)
    elif prop.key in {
      "default_joint_pos",
      "joint_stiffness",
      "joint_damping",
      "action_scale",
    }:
      reordered = _reorder_metadata_list(metadata, prop.key, target_order, source_order)
      if reordered is not None:
        prop.value = reordered

  reorder_note = model.metadata_props.add()
  reorder_note.key = "pm01_engineai_conversion"
  reorder_note.value = (
    "Reordered joint_pos/joint_vel/actions inputs and actions output from "
    "mjlab PM01 order to EngineAI pm01_edu rl_dance_example order."
  )

  checker.check_model(model)
  output_onnx.parent.mkdir(parents=True, exist_ok=True)
  onnx.save(model, str(output_onnx))

  return {
    "source_joint_order": source_order,
    "target_joint_order": target_order,
    "input_permutation": input_perm,
    "output_permutation": output_perm,
    "input_dim": PM01_ENGINEAI_DANCE_OBS_DIM,
    "output_dim": 24,
    "input_onnx": str(input_onnx),
    "output_onnx": str(output_onnx),
  }


def _infer_torchscript_input_dim(module: torch.jit.RecursiveScriptModule) -> int:
  for _, parameter in module.named_parameters():
    if parameter.ndim == 2:
      return int(parameter.shape[1])
  raise RuntimeError("Unable to infer TorchScript input dimension from model parameters.")


def convert_torchscript_to_onnx_with_joint_reorder(
  input_pt: Path,
  output_onnx: Path,
  source_order: list[str] | None = None,
  target_order: list[str] | None = None,
) -> dict[str, Any]:
  source_order = source_order or MJLAB_PM01_JOINT_ORDER
  target_order = target_order or ENGINEAI_PM01_DANCE_JOINT_ORDER
  _ensure_same_joint_set(source_order, target_order)

  try:
    policy = torch.jit.load(str(input_pt), map_location="cpu")
  except Exception as exc:
    raise RuntimeError(
      f"Failed to load TorchScript model from {input_pt}. "
      "Only exported TorchScript .pt policies are supported, not raw checkpoints."
    ) from exc

  input_dim = _infer_torchscript_input_dim(policy)
  if input_dim != PM01_ENGINEAI_DANCE_OBS_DIM:
    raise ValueError(
      f"Expected PM01 no-state-estimation obs dim {PM01_ENGINEAI_DANCE_OBS_DIM}, "
      f"but the TorchScript model appears to expect {input_dim}."
    )

  dummy_obs = torch.zeros(1, input_dim)
  output_onnx.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(prefix="pm01_mjlab_raw_onnx_") as temp_dir:
    temp_raw_onnx = Path(temp_dir) / f"{input_pt.stem}.mjlab_raw.onnx"
    torch.onnx.export(
      policy,
      dummy_obs,
      str(temp_raw_onnx),
      export_params=True,
      opset_version=11,
      input_names=["obs"],
      output_names=["actions"],
      dynamic_axes={},
      dynamo=False,
    )

    report = rewrite_onnx_joint_order(
      input_onnx=temp_raw_onnx,
      output_onnx=output_onnx,
      source_order=source_order,
      target_order=target_order,
    )

  report["input_torchscript"] = str(input_pt)
  report["intermediate_raw_onnx"] = "<temporary>"
  return report


def _find_mnnconvert_binary(preferred: str | None = None) -> str:
  candidates = [preferred] if preferred else []
  candidates.extend(["mnnconvert", "MNNConvert"])
  for candidate in candidates:
    if not candidate:
      continue
    resolved = shutil.which(candidate)
    if resolved:
      return resolved
  raise FileNotFoundError(
    "Could not find mnnconvert/MNNConvert in PATH. "
    "Please install MNN tools or pass --mnnconvert-bin."
  )


def convert_onnx_to_mnn(
  input_onnx: Path,
  output_mnn: Path,
  mnnconvert_bin: str | None = None,
) -> list[str]:
  converter = _find_mnnconvert_binary(mnnconvert_bin)
  output_mnn.parent.mkdir(parents=True, exist_ok=True)
  command = [
    converter,
    "-f",
    "ONNX",
    "--modelFile",
    str(input_onnx),
    "--MNNModel",
    str(output_mnn),
    "--bizCode",
    "biz",
  ]
  result = subprocess.run(command, capture_output=True, text=True)
  if result.returncode != 0:
    raise RuntimeError(
      "mnnconvert failed.\n"
      f"Command: {' '.join(command)}\n"
      f"stdout:\n{result.stdout}\n"
      f"stderr:\n{result.stderr}"
    )
  return command


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Convert an mjlab PM01 policy to an EngineAI-compatible MNN model.",
  )
  parser.add_argument(
    "-i",
    "--input",
    required=True,
    help="Input policy file (.onnx or exported TorchScript .pt).",
  )
  parser.add_argument(
    "--output-onnx",
    default=None,
    help="Intermediate ONNX output path. Defaults to <input_stem>.engineai_pm01.onnx",
  )
  parser.add_argument(
    "--output-mnn",
    default=None,
    help="Optional MNN output path. If omitted, only the reordered ONNX is written.",
  )
  parser.add_argument(
    "--mnnconvert-bin",
    default=None,
    help="Optional path/name for mnnconvert or MNNConvert.",
  )
  parser.add_argument(
    "--report",
    default=None,
    help="Optional JSON report path. Defaults to <output_onnx>.report.json",
  )
  parser.add_argument(
    "--source-order",
    choices=("auto", "mjlab"),
    default="auto",
    help="How to determine the source joint order. 'auto' uses ONNX metadata when available.",
  )
  parser.add_argument(
    "--target-order",
    choices=("engineai_pm01_dance", "mjlab"),
    default="engineai_pm01_dance",
    help="Target joint order for the converted model.",
  )
  return parser.parse_args()


def _resolve_output_onnx_path(input_path: Path, output_onnx: str | None) -> Path:
  if output_onnx:
    return Path(output_onnx)
  return input_path.with_suffix(".engineai_pm01.onnx")


def _resolve_report_path(output_onnx: Path, report: str | None) -> Path:
  if report:
    return Path(report)
  return output_onnx.with_suffix(output_onnx.suffix + ".report.json")


def _resolve_target_order(name: str) -> list[str]:
  if name == "engineai_pm01_dance":
    return ENGINEAI_PM01_DANCE_JOINT_ORDER
  return MJLAB_PM01_JOINT_ORDER


def _resolve_source_order(name: str) -> list[str] | None:
  if name == "mjlab":
    return MJLAB_PM01_JOINT_ORDER
  return None


def main() -> None:
  args = parse_args()

  input_path = Path(args.input)
  if not input_path.exists():
    raise FileNotFoundError(f"Input file not found: {input_path}")

  output_onnx = _resolve_output_onnx_path(input_path, args.output_onnx)
  report_path = _resolve_report_path(output_onnx, args.report)
  source_order = _resolve_source_order(args.source_order)
  target_order = _resolve_target_order(args.target_order)

  if input_path.suffix.lower() == ".onnx":
    report = rewrite_onnx_joint_order(
      input_onnx=input_path,
      output_onnx=output_onnx,
      source_order=source_order,
      target_order=target_order,
    )
  elif input_path.suffix.lower() == ".pt":
    report = convert_torchscript_to_onnx_with_joint_reorder(
      input_pt=input_path,
      output_onnx=output_onnx,
      source_order=source_order or MJLAB_PM01_JOINT_ORDER,
      target_order=target_order,
    )
  else:
    raise ValueError(
      f"Unsupported input suffix {input_path.suffix!r}. Expected .onnx or exported TorchScript .pt"
    )

  report["target_runtime"] = "pm01-engineai_robotics_native_sdk-main/assets/config/pm01_edu/rl_dance_example"

  if args.output_mnn:
    output_mnn = Path(args.output_mnn)
    command = convert_onnx_to_mnn(
      input_onnx=output_onnx,
      output_mnn=output_mnn,
      mnnconvert_bin=args.mnnconvert_bin,
    )
    report["output_mnn"] = str(output_mnn)
    report["mnnconvert_command"] = command

  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

  print(f"[INFO] Reordered ONNX written to: {output_onnx}")
  print(f"[INFO] Report written to: {report_path}")
  if args.output_mnn:
    print(f"[INFO] EngineAI-compatible MNN written to: {args.output_mnn}")


if __name__ == "__main__":
  main()
