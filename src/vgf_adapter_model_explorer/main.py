# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

from typing import Dict

from model_explorer import (
    Adapter,
    AdapterMetadata,
    ModelExplorerGraphs,
)

from .builder.builder import VgfGraphBuilder
from .exec.vgf_dump import exec_vgf_dump
from .parser.parser import Parser
from .spirv.spirv_node_builder import build_spirv_nodes


class VGFAdapter(Adapter):  # pylint: disable=too-few-public-methods
    """Adapter for VGF format."""

    metadata = AdapterMetadata(
        id="vgf_adapter_model_explorer",
        name="VGF Adapter",
        description="VGF adapter for Model Explorer",
        fileExts=["vgf"],
    )

    # pylint: disable-next=unused-argument
    def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
        """Convert a given model to a model-explorer compatible format."""

        vgf = Parser(model_path, exec_vgf_dump).vgf
        return {
            "graphs": VgfGraphBuilder(
                vgf, build_spirv_nodes
            ).graph_collection.graphs
        }
