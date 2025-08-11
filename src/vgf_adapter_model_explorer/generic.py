# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

from typing import (
    Dict,
    List,
)

from mlir import ir
from model_explorer import graph_builder as gb

from . import constants as cn


def is_terminator(operation: ir.Operation) -> bool:
    """Check if an operator is a terminator operation."""
    return operation.name in cn.TERMINATOR_OPS


def append_attr_to_metadata_list(
    metadata_list: List[gb.MetadataItem],
    uid: str,
    attr: gb.KeyValue,
) -> None:
    """Append attrs to a given metadata list."""

    for metadata in metadata_list:
        if metadata.id == uid:
            metadata.attrs.append(attr)
            return

    metadata_list.append(gb.MetadataItem(id=uid, attrs=[attr]))


def add_operation_attrs(node: gb.GraphNode, op: ir.Operation) -> None:
    """Annotate graph node with operation attributes."""

    node_attrs = []

    for named_attr in op.attributes:
        name = named_attr.name
        attr = str(named_attr.attr)
        node_attrs.append(gb.KeyValue(key=name, value=attr))

    node_attrs.append(gb.KeyValue(key="loc", value=str(op.location)))

    node.attrs.extend(node_attrs)


def add_incoming_edges(
    node: gb.GraphNode,
    operation: ir.Operation,
    traversed_ops: Dict[ir.Operation, str],
) -> None:
    """Add incoming edges/connections to a graph node."""
    for operand_idx, operand in enumerate(operation.operands):
        if isinstance(operand.owner, ir.Operation):
            src_node_id = traversed_ops[operand.owner]
            src_node_output_id = ir.OpResult(operand).result_number
        elif ir.BlockArgument.isinstance(operand):
            block_arg = ir.BlockArgument(operand)
            src_node_id = cn.GRAPH_INPUT_ANNOTATION
            src_node_output_id = block_arg.arg_number
        else:
            raise AssertionError("[Bug]: Unhandled operand type.")

        node.incomingEdges.append(
            gb.IncomingEdge(
                sourceNodeId=src_node_id,
                sourceNodeOutputId=str(src_node_output_id),
                targetNodeInputId=str(operand_idx),
            )
        )


def add_inputs_metadata(
    node: gb.GraphNode,
    operation: ir.Operation,
) -> None:
    """Add metadata to all the inputs of a graph node."""

    for operand_idx, operand in enumerate(operation.operands):
        append_attr_to_metadata_list(
            metadata_list=node.inputsMetadata,
            uid=str(operand_idx),
            attr=gb.KeyValue(
                key=cn.GRAPH_TENSOR_TYPE, value=str(operand.type)
            ),
        )


def add_outputs_metadata(
    node: gb.GraphNode,
    operation: ir.Operation,
) -> None:
    """Add metadata to all the outputs of a graph node."""

    for result_idx, result in enumerate(operation.results):
        append_attr_to_metadata_list(
            metadata_list=node.outputsMetadata,
            uid=str(result_idx),
            attr=gb.KeyValue(key=cn.GRAPH_TENSOR_IDX, value=str(result_idx)),
        )
        append_attr_to_metadata_list(
            metadata_list=node.outputsMetadata,
            uid=str(result_idx),
            attr=gb.KeyValue(key=cn.GRAPH_TENSOR_TYPE, value=str(result.type)),
        )


def create_node(
    operation: ir.Operation,
    namespace: str,
    traversed_ops: Dict[ir.Operation, str],
) -> gb.GraphNode:
    """Create a graph node for a given MLIR operation."""

    node_id = str(len(traversed_ops))
    node_label = operation.name.split(".")[-1]
    node_namespace = namespace

    if is_terminator(operation):
        node_id = cn.GRAPH_OUTPUT_ANNOTATION
        node_label = cn.GRAPH_OUTPUT_ANNOTATION
        node_namespace = ""

    node = gb.GraphNode(
        id=node_id,
        label=node_label,
        namespace=node_namespace,
    )
    assert node is not None, "[Bug]: Operation node couldn't be created"

    traversed_ops[operation] = node.id

    add_operation_attrs(node, operation)
    add_incoming_edges(node, operation, traversed_ops)
    add_inputs_metadata(node, operation)
    add_outputs_metadata(node, operation)

    return node
