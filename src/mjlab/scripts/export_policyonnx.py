#!/usr/bin/env python3
from __future__ import annotations
"""
删除 ONNX 模型中的 time_step 输入，并且只保留 obs -> actions 这条主干。

用法:
    python remove_timestep_keep_obs.py -i input.onnx -o output.onnx

可选:
    --keep-input obs
    --keep-output actions
    --remove-input time_step

依赖:
    pip install onnx
"""
import argparse
import sys
from typing import Dict, List, Set
import pathlib

def _lazy_import_onnx():
    import onnx
    from onnx import helper, checker
    return onnx, helper, checker

def names_from_value_infos(value_infos) -> List[str]:
    return [v.name for v in value_infos]

def find_value_info_by_name(value_infos, name: str):
    for v in value_infos:
        if v.name == name:
            return v
    return None

def build_output_to_node_map(nodes) -> Dict[str, object]:
    mapping = {}
    for node in nodes:
        for out_name in node.output:
            if out_name:
                mapping[out_name] = node
    return mapping

def build_initializer_name_set(graph) -> Set[str]:
    return {init.name for init in graph.initializer}

def build_sparse_initializer_name_set(graph) -> Set[str]:
    sparse = getattr(graph, "sparse_initializer", [])
    names = set()
    for s in sparse:
        if s.values.name:
            names.add(s.values.name)
    return names

def backward_reachable_nodes(required_outputs, output_to_node, graph_input_names, initializer_names, sparse_initializer_names):
    needed_tensors = set(required_outputs)
    needed_nodes = set()
    changed = True
    while changed:
        changed = False
        for tensor_name in list(needed_tensors):
            if not tensor_name:
                continue
            if tensor_name in graph_input_names or tensor_name in initializer_names or tensor_name in sparse_initializer_names:
                continue
            node = output_to_node.get(tensor_name)
            if node is None:
                continue
            node_id = id(node)
            if node_id not in needed_nodes:
                needed_nodes.add(node_id)
                changed = True
            for inp in node.input:
                if inp and inp not in needed_tensors:
                    needed_tensors.add(inp)
                    changed = True
    return needed_tensors, needed_nodes

def prune_model(model, keep_input: str, keep_output: str, remove_input: str):
    onnx, helper, checker = _lazy_import_onnx()
    graph = model.graph

    graph_input_names = set(names_from_value_infos(graph.input))
    graph_output_names = set(names_from_value_infos(graph.output))
    initializer_names = build_initializer_name_set(graph)
    sparse_initializer_names = build_sparse_initializer_name_set(graph)
    output_to_node = build_output_to_node_map(graph.node)

    if keep_input not in graph_input_names:
        raise ValueError(f"找不到要保留的输入 {keep_input!r}。当前输入: {sorted(graph_input_names)}")
    if keep_output not in graph_output_names:
        raise ValueError(f"找不到要保留的输出 {keep_output!r}。当前输出: {sorted(graph_output_names)}")
    if remove_input not in graph_input_names:
        print(f"[警告] 没有找到要删除的输入 {remove_input!r}，继续执行。", file=sys.stderr)

    needed_tensors, needed_node_ids = backward_reachable_nodes(
        required_outputs=[keep_output],
        output_to_node=output_to_node,
        graph_input_names=graph_input_names,
        initializer_names=initializer_names,
        sparse_initializer_names=sparse_initializer_names,
    )

    if remove_input in needed_tensors:
        raise RuntimeError(
            f"输出 {keep_output!r} 仍然依赖输入 {remove_input!r}，无法直接删除。"
            "这通常说明你选错了 keep_output，或者该模型的主干真的依赖 time_step。"
        )

    kept_nodes = [node for node in graph.node if id(node) in needed_node_ids]
    kept_initializers = [init for init in graph.initializer if init.name in needed_tensors]
    sparse_inits = getattr(graph, "sparse_initializer", [])
    kept_sparse_initializers = [s for s in sparse_inits if s.values.name in needed_tensors]

    input_vi = find_value_info_by_name(graph.input, keep_input)
    output_vi = find_value_info_by_name(graph.output, keep_output)
    if input_vi is None or output_vi is None:
        raise RuntimeError("内部错误：无法找到保留输入或输出。")

    kept_value_info = [vi for vi in graph.value_info if vi.name in needed_tensors]

    new_graph = helper.make_graph(
        nodes=kept_nodes,
        name=(graph.name + "_obs_only") if graph.name else "obs_only_graph",
        inputs=[input_vi],
        outputs=[output_vi],
        initializer=kept_initializers,
        value_info=kept_value_info,
        sparse_initializer=kept_sparse_initializers,
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
        p = new_model.metadata_props.add()
        p.key = prop.key
        p.value = prop.value

    try:
        new_model = onnx.shape_inference.infer_shapes(new_model)
    except Exception as e:
        print(f"[警告] shape inference 失败，但模型仍会保存: {e}", file=sys.stderr)

    checker.check_model(new_model)
    return new_model

def parse_args():
    parser = argparse.ArgumentParser(description="删除 time_step 及其依赖输出，只保留 obs -> actions 子图")
    parser.add_argument("-i", "--input", required=True, help="输入 ONNX 文件路径")
    parser.add_argument("-o", "--output", required=True, help="输出 ONNX 文件路径")
    parser.add_argument("--keep-input", default="obs", help="要保留的输入名，默认 obs")
    parser.add_argument("--keep-output", default="actions", help="要保留的输出名，默认 actions")
    parser.add_argument("--remove-input", default="time_step", help="要删除的输入名，默认 time_step")
    return parser.parse_args()

def main():
    args = parse_args()
    onnx, _, _ = _lazy_import_onnx()
    model = onnx.load(args.input)
    new_model = prune_model(
        model=model,
        keep_input=args.keep_input,
        keep_output=args.keep_output,
        remove_input=args.remove_input,
    )
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(new_model, str(out_path))
    print("处理完成")
    print(f"输入模型:  {args.input}")
    print(f"输出模型:  {args.output}")
    print(f"保留输入:  {args.keep_input}")
    print(f"保留输出:  {args.keep_output}")
    print(f"删除输入:  {args.remove_input}")

if __name__ == "__main__":
    main()
