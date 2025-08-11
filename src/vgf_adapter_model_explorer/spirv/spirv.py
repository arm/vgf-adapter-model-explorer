# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

from typing import Dict

from mlir import ir
from model_explorer import graph_builder as gb

from .. import constants as cn
from ..generic import append_attr_to_metadata_list, create_node


def get_spirv_functions(module):
    """Extracts all spirv.func operations from the module."""
    functions = []

    for op in module.body.operations:
        for region in op.regions:
            for block in region.blocks:
                for inner_op in block.operations:
                    if inner_op.name == "spirv.ARM.Graph":
                        functions.append(inner_op)

    return functions


def add_graph_inputs(function) -> gb.GraphNode:
    """Add graph inputs node as a singular graph node."""

    node = gb.GraphNode(
        id=cn.GRAPH_INPUT_ANNOTATION,
        label=cn.GRAPH_INPUT_ANNOTATION,
    )
    assert node is not None, "[Bug]: Input node couldn't be created."

    for arg_idx, arg in enumerate(function.operands):
        append_attr_to_metadata_list(
            metadata_list=node.outputsMetadata,
            uid=str(arg_idx),
            attr=gb.KeyValue(key=cn.GRAPH_TENSOR_IDX, value=str(arg_idx)),
        )
        append_attr_to_metadata_list(
            metadata_list=node.outputsMetadata,
            uid=str(arg_idx),
            attr=gb.KeyValue(key=cn.GRAPH_TENSOR_TYPE, value=str(arg.type)),
        )

    return node


def build_function_graph(
    function,
) -> gb.Graph:
    """Create a sub-graph for a given MLIR `spirv.ARM.Graph`."""

    function_name = str(function.attributes["sym_name"]).replace('"', "")
    function_graph = gb.Graph(id=function_name, nodes=[])

    input_node = add_graph_inputs(function)
    function_graph.nodes.append(input_node)

    traversed_ops: Dict[ir.Operation, str] = {}
    for operation in function.regions[0].blocks[0].operations:
        node = create_node(
            operation=operation,
            namespace=function_name,
            traversed_ops=traversed_ops,
        )

        if "Tosa" not in operation.name:
            continue

        function_graph.nodes.append(node)

    return function_graph
