# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

from mlir import ir
from model_explorer import graph_builder as gb

from ..constants import GRAPH_INPUT_ANNOTATION
from ..exec.mlir_translate import exec_mlir_translate
from ..exec.vgf_dump import exec_vgf_dump
from ..parser.types import Vgf
from .spirv import build_function_graph, get_spirv_functions


def build_spirv_nodes(vgf_data: Vgf, module_index: int) -> list[gb.GraphNode]:
    """Build model-explorer nodes from a SPIR-V module."""
    dumped_spirv = exec_vgf_dump(
        vgf_data.file_path, dump_spirv_index=module_index
    )
    spirv_content = exec_mlir_translate(dumped_spirv)
    module = _load_mlir_module(spirv_content)

    nodes: list[gb.GraphNode] = []
    functions = get_spirv_functions(module)
    for function in functions:
        function_graph = build_function_graph(function)

        for node in function_graph.nodes:
            if node.id != GRAPH_INPUT_ANNOTATION:
                nodes.append(node)

    return nodes


def _load_mlir_module(file_content: str) -> ir.Module:
    """Loads the MLIR module from the given file content."""
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(file_content, ctx)
    return module
