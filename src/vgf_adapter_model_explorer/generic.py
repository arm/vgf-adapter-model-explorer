# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

from typing import Any, Dict, List, Optional

import numpy as np
from mlir import ir
from model_explorer import graph_builder as gb

from . import constants as cn

ATTRIBUTE_PREFIX = "attribute"
ENUM_SUFFIX = "_t"


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

    for name, attr in op.attributes.items():
        node_attrs.append(gb.KeyValue(key=name, value=str(attr)))

    node_attrs.append(gb.KeyValue(key="loc", value=str(op.location)))

    node.attrs.extend(node_attrs)


def extract_enum_attribute(
    attribute: ir.Attribute, enum_values: List[str]
) -> Optional[str]:
    """
    Maps an integer enum attribute value to its name using the TOSA spec enum list.
    TOSA enums are 1-indexed.
    """
    if not ir.IntegerAttr.isinstance(attribute):
        return None
    index = int(ir.IntegerAttr(attribute)) - 1
    return enum_values[index] if 0 <= index < len(enum_values) else None


# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
def extract_tensor_values(
    attribute: ir.Attribute, shaped_type: ir.ShapedType
) -> List[Any]:
    """Materialize Tensor to Python lists (handles splat/non-splat)."""
    is_dense = ir.DenseElementsAttr.isinstance(attribute)
    if is_dense and not attribute.is_splat:
        values = list(attribute)
    else:
        v = (
            attribute.get_splat_value().value
            if is_dense
            else getattr(attribute, "value", attribute)
        )
        values = np.full(shaped_type.shape, v).tolist()
    return values


def get_tosa_attribute_value(
    operand: ir.Value,
    operand_value: ir.Attribute,
    enum_values: Optional[List[str]] = None,
) -> str:
    """Get attribute value for a given operand."""
    if enum_values:
        mapped = extract_enum_attribute(operand_value, enum_values)
        return mapped if mapped is not None else str(operand_value)

    if ir.ShapedType.isinstance(operand.type) and str(operand.type).startswith(
        "!spirv.arm.tensor"
    ):
        return str(
            extract_tensor_values(operand_value, ir.ShapedType(operand.type))
        )

    return str(operand_value)


def get_operands_info(op_name: str) -> Optional[List[Dict[str, Any]]]:
    """Get operand info for a given operation."""
    normalised_operation_name = op_name.split(".")[-1].lower()
    operands_info = cn.TOSA_OPERAND_INFO["operations"].get(
        normalised_operation_name
    )
    return operands_info if operands_info else None


def add_tosa_operation_attrs(node: gb.GraphNode, op: ir.Operation) -> None:
    """Annotate graph node with TOSA operation attributes."""
    operands_info = get_operands_info(op.name)
    if not operands_info:
        return

    tosa_attrs: List[gb.KeyValue] = []
    for operand, operand_info in zip(op.operands, operands_info, strict=False):
        if not operand_info["category"].startswith(ATTRIBUTE_PREFIX):
            continue
        operation_attributes = operand.owner.attributes

        if "value" not in operation_attributes:
            continue

        attribute_name = operand_info["name"]
        enum_key = f"{attribute_name}{ENUM_SUFFIX}"
        enum_values = cn.TOSA_OPERAND_INFO["enums"].get(enum_key)

        tosa_attrs.append(
            gb.KeyValue(
                key=attribute_name,
                value=get_tosa_attribute_value(
                    operand, operation_attributes["value"], enum_values
                ),
            )
        )

    node.attrs.extend(tosa_attrs)


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
    add_tosa_operation_attrs(node, operation)
    add_incoming_edges(node, operation, traversed_ops)
    add_inputs_metadata(node, operation)
    add_outputs_metadata(node, operation)

    return node
