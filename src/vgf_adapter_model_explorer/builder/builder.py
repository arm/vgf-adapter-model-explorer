# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

import os
from typing import Callable

from model_explorer import graph_builder as gb

from ..constants import GRAPH_INPUT_ANNOTATION, GRAPH_OUTPUT_ANNOTATION
from ..generic import append_attr_to_metadata_list
from ..parser.types import IOBase, Module, Resource, Vgf
from .utils import extend_resource, find_item, format_index


class VgfGraphBuilder:
    """Builds a graph from a VGF model."""

    def __init__(
        self,
        vgf_data: Vgf,
        build_spirv_nodes: Callable[[Vgf, int], list[gb.GraphNode]],
    ):
        """Builds a Model Explorer GraphCollection from VGF data."""

        self.vgf_data = vgf_data
        self._build_spirv_nodes = build_spirv_nodes
        self.graph_collection = self._build_graph_collection()

    def _build_graph_collection(self) -> gb.GraphCollection:
        """Build VGF graph collection with main graph."""
        return gb.GraphCollection(
            label=os.path.splitext(os.path.basename(self.vgf_data.file_path))[
                0
            ],
            graphs=[gb.Graph(id="Main", nodes=self._build_nodes())],
        )

    def _build_nodes(self) -> list[gb.GraphNode]:
        """Build all nodes for the graph."""
        nodes = [self._build_graph_input_node()]

        for segment in self.vgf_data.model_sequence.segments:
            module = find_item(segment.module_index, self.vgf_data.modules)
            if not module:
                continue

            input_nodes = self._build_segment_input_nodes(
                segment.inputs, module
            )
            spirv_nodes = self._build_segment_spirv_nodes(input_nodes, module)
            nodes.extend(spirv_nodes)

            output_nodes = self._build_segment_output_nodes(
                segment.outputs, module, spirv_nodes
            )
            nodes.extend(
                input_nodes + output_nodes + [self._build_graph_output_node()]
            )

        return nodes

    def _build_segment_spirv_nodes(
        self, input_nodes: list[gb.GraphNode], module: Module
    ) -> list[gb.GraphNode]:
        """Builds SPIR-V nodes for a segment."""
        if not module.has_spirv:
            return []
        spirv_nodes = self._build_spirv_nodes(self.vgf_data, module.index)
        if not spirv_nodes:
            return []
        self._clean_spirv_edges(spirv_nodes)
        self._connect_spirv_node(spirv_nodes[0], input_nodes)
        return spirv_nodes

    def _clean_spirv_edges(self, spirv_nodes: list[gb.GraphNode]) -> None:
        """Clean incoming edges of SPIR-V nodes."""
        for node in spirv_nodes:
            new_incoming_edges = []
            for e in node.incomingEdges:
                if e.sourceNodeId != GRAPH_INPUT_ANNOTATION:
                    new_incoming_edges.append(e)
            node.incomingEdges[:] = new_incoming_edges

    def _connect_spirv_node(
        self, spirv_node: gb.GraphNode, input_nodes: list[gb.GraphNode]
    ) -> None:
        """Connect SPIR-V node to the graph."""
        for node in input_nodes:
            spirv_node.incomingEdges.append(
                gb.IncomingEdge(sourceNodeId=node.id, targetNodeInputId="0")
            )

    def _build_segment_input_nodes(
        self, inputs: list[IOBase], module: Module
    ) -> list[gb.GraphNode]:
        """Build input nodes for the segment."""
        nodes: list[gb.GraphNode] = []

        for input in inputs:
            resource = find_item(input.mrt_index, self.vgf_data.resources)
            if not resource:
                continue
            nodes.append(self._build_node(resource, module, input.mrt_index))

        return nodes

    def _build_segment_output_nodes(
        self, outputs, module, spirv_nodes: list[gb.GraphNode]
    ) -> list[gb.GraphNode]:
        """Build output nodes for the segment."""
        output_nodes = []
        source_id = spirv_nodes[-1].id if spirv_nodes else None

        for output in outputs:
            resource = find_item(output.mrt_index, self.vgf_data.resources)
            if not resource:
                continue
            output_nodes.append(
                self._build_node(
                    resource,
                    module,
                    output.mrt_index,
                    output=True,
                    source_id=source_id,
                )
            )

        return output_nodes

    def _build_graph_input_node(self) -> gb.GraphNode:
        """Build the input node for the graph."""
        return gb.GraphNode(id=GRAPH_INPUT_ANNOTATION, label="Graph Inputs")

    def _build_graph_output_node(self) -> gb.GraphNode:
        """Build the output node for the graph."""
        incoming_edges: list[gb.IncomingEdge] = []
        for output in self.vgf_data.model_sequence.outputs:
            incoming_edges.append(
                gb.IncomingEdge(
                    sourceNodeId=format_index("mrt", output.mrt_index)
                )
            )
        return gb.GraphNode(
            id=GRAPH_OUTPUT_ANNOTATION,
            label="Graph Outputs",
            incomingEdges=incoming_edges,
        )

    def _build_node(
        self,
        resource: Resource,
        module: Module,
        mrt_index: int,
        output: bool = False,
        source_id: str | None = None,
    ) -> gb.GraphNode:
        """Build a single graph node."""
        if source_id:
            incoming_edges = [
                gb.IncomingEdge(sourceNodeId=source_id, targetNodeInputId="0")
            ]
        else:
            incoming_edges = self._build_incoming_edges(resource)

        node = gb.GraphNode(
            id=format_index("mrt", mrt_index),
            label=resource.vk_descriptor_type,
            incomingEdges=incoming_edges,
        )

        if not output:
            extend_resource(node, resource)
            append_attr_to_metadata_list(
                node.outputsMetadata,
                uid="0",
                attr=gb.KeyValue(key="type", value=module.type),
            )

        return node

    def _build_incoming_edges(
        self, resource: Resource
    ) -> list[gb.IncomingEdge]:
        """Build incoming edges for a resource."""
        incoming_edges: list[gb.IncomingEdge] = []

        if resource.category == "INPUT":
            incoming_edges = [
                gb.IncomingEdge(sourceNodeId=GRAPH_INPUT_ANNOTATION)
            ]

        return incoming_edges
