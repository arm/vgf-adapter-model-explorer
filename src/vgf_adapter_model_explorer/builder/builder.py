# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

import os
import re
import struct
from collections import Counter, defaultdict
from itertools import pairwise
from math import prod

from model_explorer import graph_builder as gb
from spirv_adapter_model_explorer.builder import (
    EntryPointKind,
    build_spirv_graph_from_bytes,
)

from ..parser.types import (
    Constant,
    IOBase,
    Model_Sequence_IO,
    Module,
    ModuleType,
    Resource,
    ResourceCategory,
    Segment,
    Vgf,
    VkFormat,
)

_GRAPH_CONSTANT_RE = re.compile(r"^<graph constant (\d+)>(?:\s|$)")
_ARG_ATTR_RE = re.compile(r"^arg(\d+)(?:\s|$)")

_FORMAT_DECODE = {
    VkFormat.VK_FORMAT_R8_SINT: "b",
    VkFormat.VK_FORMAT_R8_UINT: "B",
    VkFormat.VK_FORMAT_R8_BOOL_ARM: "?",
    VkFormat.VK_FORMAT_R16_SINT: "<h",
    VkFormat.VK_FORMAT_R16_UINT: "<H",
    VkFormat.VK_FORMAT_R16_SFLOAT: "<e",
    VkFormat.VK_FORMAT_R32_SINT: "<i",
    VkFormat.VK_FORMAT_R32_UINT: "<I",
    VkFormat.VK_FORMAT_R32_SFLOAT: "<f",
    VkFormat.VK_FORMAT_R64_SINT: "<q",
    VkFormat.VK_FORMAT_R64_UINT: "<Q",
    VkFormat.VK_FORMAT_R64_SFLOAT: "<d",
}
_CONSTANT_VALUE_PREVIEW_LENGTH = 1024


def _format_constant_value(constant: Constant, resource: Resource) -> str:
    """Format a VGF constant's raw bytes for display."""
    if constant.sparsity_dimension != -1:
        return (
            f"<{len(constant.data)} sparse bytes, "
            f"sparsity_dimension={constant.sparsity_dimension}>"
        )
    decoded = _decode_constant_data(constant.data, resource)
    if decoded is None:
        return _unsupported_constant_value(resource, len(constant.data))
    return _format_sequence_value(decoded, resource.shape)


def _decode_constant_data(
    data: bytes, resource: Resource
) -> tuple[object, ...] | None:
    fmt = _FORMAT_DECODE.get(resource.vk_format)
    if fmt is None or len(data) % struct.calcsize(fmt):
        return None
    return tuple(item[0] for item in struct.iter_unpack(fmt, data))


def _unsupported_constant_value(resource: Resource, byte_count: int) -> str:
    if not resource.shape:
        return f"<{byte_count} bytes of {resource.vk_format.name}>"
    return f"<{prod(resource.shape)} elements of {resource.vk_format.name}>"


def _format_sequence_value(value: tuple[object, ...], shape: list[int]) -> str:
    length = len(value)
    if length <= _CONSTANT_VALUE_PREVIEW_LENGTH:
        shaped_value = _reshape_sequence(value, shape)
        return _format_nested_sequence(shaped_value)
    if all(item == 0 for item in value):
        return f"[<all zeroes>] len={length}"
    return (
        "["
        + ", ".join(
            _format_value_literal(item)
            for item in value[:_CONSTANT_VALUE_PREVIEW_LENGTH]
        )
        + f", ...] len={length}"
    )


def _reshape_sequence(value: tuple[object, ...], shape: list[int]) -> object:
    if not shape or prod(shape) != len(value):
        return value

    items = iter(value)

    def reshape(dimensions: list[int]) -> object:
        if not dimensions:
            return next(items)
        return tuple(reshape(dimensions[1:]) for _ in range(dimensions[0]))

    return reshape(shape)


def _format_nested_sequence(value: object, indent: int = 0) -> str:
    if not isinstance(value, tuple):
        return _format_value_literal(value)
    if not value:
        return "[]"
    if not any(isinstance(item, tuple) for item in value):
        return (
            "["
            + ", ".join(_format_value_literal(item) for item in value)
            + "]"
        )

    child_indent = indent + 2
    prefix = " " * child_indent
    suffix = " " * indent
    return (
        "[\n"
        + ",\n".join(
            prefix + _format_nested_sequence(item, child_indent)
            for item in value
        )
        + "\n"
        + suffix
        + "]"
    )


def _format_value_literal(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class VgfGraphBuilder:
    """Builds a Model Explorer graph for a VGF container."""

    def __init__(self, vgf_data: Vgf, show_constants: bool = False):
        """Builds a Model Explorer GraphCollection from VGF data."""

        self.vgf_data = vgf_data
        self.show_constants = show_constants
        self.vgf_resources = {r.index: r for r in vgf_data.resources}
        self.vgf_constants = {c.index: c for c in vgf_data.constants}
        self.vgf_constants_by_mrt = {
            c.mrt_index: c for c in vgf_data.constants
        }
        self.model_inputs = self._model_ios_by_mrt(
            vgf_data.model_sequence.inputs
        )
        self.model_outputs = self._model_ios_by_mrt(
            vgf_data.model_sequence.outputs
        )
        self.segment_namespaces = self._segment_namespaces(
            vgf_data.model_sequence.segments
        )
        self.graph_collection = self._build_graph_collection()

    def _build_graph_collection(self) -> gb.GraphCollection:
        """Build VGF graph collection with one top-level graph."""
        main_graph = gb.Graph(
            id="Main",
            nodes=[],
            groupNodeAttributes={},
        )
        nodes_by_id: dict[str, gb.GraphNode] = {}
        modules = {m.index: m for m in self.vgf_data.modules}

        for segment in self.vgf_data.model_sequence.segments:
            self._add_segment(segment, modules, main_graph, nodes_by_id)

        self._connect_alias_groups(nodes_by_id)
        main_graph.nodes = list(nodes_by_id.values())
        return gb.GraphCollection(
            label=os.path.splitext(os.path.basename(self.vgf_data.file_path))[
                0
            ],
            graphs=[main_graph],
        )

    def _add_segment(
        self,
        segment: Segment,
        modules: dict[int, Module],
        main_graph: gb.Graph,
        nodes_by_id: dict[str, gb.GraphNode],
    ) -> None:
        module = modules.get(segment.module_index)
        if module is None:
            return

        for segment_input in segment.inputs:
            self._ensure_resource_node(segment_input.mrt_index, nodes_by_id)

        if self.show_constants:
            for constant_index in segment.constants:
                constant = self.vgf_constants.get(constant_index)
                if constant is not None:
                    self._ensure_resource_node(constant.mrt_index, nodes_by_id)

        if not module.has_spirv or module.spirv_code is None:
            self._add_placeholder_segment(
                segment,
                module,
                nodes_by_id,
                "SPIR-V code is not available for this segment.",
            )
            return

        self._add_spirv_segment(segment, module, main_graph, nodes_by_id)

    def _add_spirv_segment(
        self,
        segment: Segment,
        module: Module,
        main_graph: gb.Graph,
        nodes_by_id: dict[str, gb.GraphNode],
    ) -> None:
        if module.spirv_code is None:
            return

        try:
            spirv_graph = build_spirv_graph_from_bytes(
                module.spirv_code,
                entry_point=module.entry_point,
                entry_point_kind=self._spirv_entry_point_kind(module.type),
                namespace=self.segment_namespaces[segment.index],
                node_id_prefix=f"{segment.index}__",
            )
        except Exception as err:
            self._add_placeholder_segment(
                segment,
                module,
                nodes_by_id,
                f"SPIR-V graph could not be built: {err}",
            )
            return

        self._merge_group_node_attributes(
            main_graph, spirv_graph.groupNodeAttributes
        )
        nodes_by_id.update({node.id: node for node in spirv_graph.nodes})
        terminal_node_ids = self._connect_segment_inputs(
            segment, spirv_graph, nodes_by_id
        )
        self._process_segment_constants(segment, spirv_graph, nodes_by_id)
        terminal_node_ids.update(
            self._connect_segment_outputs(segment, spirv_graph, nodes_by_id)
        )
        self._remove_terminal_nodes(terminal_node_ids, nodes_by_id)

    @staticmethod
    def _merge_group_node_attributes(
        graph: gb.Graph, incoming: dict[str, dict[str, str]] | None
    ) -> None:
        if not incoming:
            return
        graph.groupNodeAttributes = graph.groupNodeAttributes or {}
        for group, attrs in incoming.items():
            graph.groupNodeAttributes.setdefault(group, {}).update(attrs)

    def _connect_segment_inputs(
        self,
        segment: Segment,
        spirv_graph: gb.Graph,
        nodes_by_id: dict[str, gb.GraphNode],
    ) -> set[str]:
        terminal_node_ids: set[str] = set()
        metadata_id = "0" if segment.type == ModuleType.GRAPH else "resource"
        for segment_input, graph_inputs in self._segment_io_nodes(
            segment, spirv_graph, segment.inputs, "logical input idx"
        ):
            resource_node = self._ensure_resource_node(
                segment_input.mrt_index, nodes_by_id
            )
            if resource_node is None:
                continue
            for graph_input in graph_inputs:
                terminal_node_ids.add(graph_input.id)
                self._redirect_terminal_input_consumers(
                    graph_input,
                    resource_node,
                    nodes_by_id,
                    self._resource_tensor_shape_metadata(
                        metadata_id, segment_input.mrt_index
                    ),
                )
        return terminal_node_ids

    def _process_segment_constants(
        self,
        segment: Segment,
        spirv_graph: gb.Graph,
        nodes_by_id: dict[str, gb.GraphNode],
    ) -> None:
        segment_constant_indexes = set(segment.constants)
        for node in spirv_graph.nodes:
            for attr in tuple(node.attrs):
                if not isinstance(attr.key, str) or not isinstance(
                    attr.value, str
                ):
                    continue
                input_id, input_tag = self._graph_constant_input(attr.key)
                constant_index = self._graph_constant_index(attr.value)
                if input_id is None or constant_index is None:
                    continue
                if constant_index not in segment_constant_indexes:
                    continue
                constant = self.vgf_constants.get(constant_index)
                if constant is None:
                    continue
                if not self.show_constants:
                    attr.value = self._graph_constant_attr_value(
                        attr.value, constant
                    )
                    continue
                resource_node = self._ensure_resource_node(
                    constant.mrt_index, nodes_by_id
                )
                if resource_node is None:
                    continue
                self._append_unique_edge(
                    node,
                    gb.IncomingEdge(
                        sourceNodeId=resource_node.id,
                        sourceNodeOutputId="0",
                        targetNodeInputId=input_id,
                    ),
                )
                self._upsert_metadata(
                    node.inputsMetadata,
                    self._constant_input_metadata(
                        input_id, input_tag, constant
                    ),
                )
                node.attrs.remove(attr)

    def _graph_constant_attr_value(
        self, attr_value: str, constant: Constant
    ) -> str:
        resource = self.vgf_resources.get(constant.mrt_index)
        if resource is None:
            return attr_value
        return f"{attr_value}: {_format_constant_value(constant, resource)}"

    def _connect_segment_outputs(
        self,
        segment: Segment,
        spirv_graph: gb.Graph,
        nodes_by_id: dict[str, gb.GraphNode],
    ) -> set[str]:
        terminal_node_ids: set[str] = set()
        for output, graph_outputs in self._segment_io_nodes(
            segment, spirv_graph, segment.outputs, "logical output idx"
        ):
            output_node = self._ensure_resource_node(
                output.mrt_index, nodes_by_id
            )
            if output_node is None:
                continue
            for graph_output in graph_outputs:
                terminal_node_ids.add(graph_output.id)
                self._connect_segment_output_node(
                    output, output_node, graph_output
                )
        return terminal_node_ids

    def _connect_segment_output_node(
        self,
        output: IOBase,
        output_node: gb.GraphNode,
        graph_output: gb.GraphNode,
    ) -> None:
        for incoming_edge in graph_output.incomingEdges:
            self._append_unique_edge(
                output_node,
                gb.IncomingEdge(
                    sourceNodeId=incoming_edge.sourceNodeId,
                    sourceNodeOutputId=incoming_edge.sourceNodeOutputId,
                    targetNodeInputId="0",
                ),
            )
        self._upsert_metadata(
            output_node.inputsMetadata,
            self._resource_tensor_shape_metadata("0", output.mrt_index),
        )

    def _redirect_terminal_input_consumers(
        self,
        terminal_node: gb.GraphNode,
        resource_node: gb.GraphNode,
        nodes_by_id: dict[str, gb.GraphNode],
        metadata: gb.MetadataItem,
    ) -> None:
        for node in nodes_by_id.values():
            if node.id == terminal_node.id:
                continue
            for edge in node.incomingEdges:
                if edge.sourceNodeId != terminal_node.id:
                    continue
                edge.sourceNodeId = resource_node.id
                edge.sourceNodeOutputId = "0"
                self._upsert_metadata(
                    node.inputsMetadata,
                    gb.MetadataItem(
                        id=edge.targetNodeInputId, attrs=metadata.attrs
                    ),
                )

    @staticmethod
    def _remove_terminal_nodes(
        terminal_node_ids: set[str], nodes_by_id: dict[str, gb.GraphNode]
    ) -> None:
        for terminal_node_id in terminal_node_ids:
            nodes_by_id.pop(terminal_node_id, None)
        for node in nodes_by_id.values():
            node.incomingEdges = [
                edge
                for edge in node.incomingEdges
                if edge.sourceNodeId not in terminal_node_ids
            ]

    def _segment_io_nodes(
        self,
        segment: Segment,
        graph: gb.Graph,
        ios: list[IOBase],
        logical_index_attr: str,
    ) -> list[tuple[IOBase, tuple[gb.GraphNode, ...]]]:
        if segment.type == ModuleType.GRAPH:
            nodes = self._nodes_by_attr(graph, logical_index_attr)
            return [
                (io, (nodes[str(io.index)],) if str(io.index) in nodes else ())
                for io in ios
            ]
        if segment.type == ModuleType.COMPUTE:
            nodes = (
                self._nodes_by_attr(graph, "descriptor")
                if graph.id == "compute"
                else {}
            )
            nodes_by_mrt: dict[int, list[gb.GraphNode]] = defaultdict(list)
            # Match the shader ABI bindings, not the segment's logical slots.
            for descriptor_set in segment.descriptor_set_infos:
                for binding in descriptor_set.bindings:
                    descriptor = (
                        f"set {descriptor_set.set_index}, "
                        f"binding {binding.binding}"
                    )
                    if descriptor in nodes:
                        nodes_by_mrt[binding.mrt_index].append(
                            nodes[descriptor]
                        )
            return [(io, tuple(nodes_by_mrt[io.mrt_index])) for io in ios]
        raise ValueError(f"Unsupported VGF segment type: {segment.type!r}")

    @staticmethod
    def _nodes_by_attr(graph: gb.Graph, key: str) -> dict[str, gb.GraphNode]:
        nodes: dict[str, gb.GraphNode] = {}
        for node in graph.nodes:
            for attr in node.attrs:
                if attr.key == key and isinstance(attr.value, str):
                    nodes[attr.value] = node
        return nodes

    @staticmethod
    def _graph_constant_input(key: str) -> tuple[str | None, str | None]:
        match = _ARG_ATTR_RE.match(key)
        if match is None:
            return None, None
        input_id = match.group(1)
        input_tag = key[match.end() :].strip() or None
        return input_id, input_tag

    @staticmethod
    def _graph_constant_index(value: str) -> int | None:
        match = _GRAPH_CONSTANT_RE.match(value)
        return None if match is None else int(match.group(1))

    def _ensure_resource_node(
        self, mrt_index: int, nodes_by_id: dict[str, gb.GraphNode]
    ) -> gb.GraphNode | None:
        """Get or create a VGF resource node."""
        resource = self.vgf_resources.get(mrt_index)
        if resource is None:
            return None
        if (
            resource.category == ResourceCategory.CONSTANT
            and not self.show_constants
        ):
            return None

        node_id = self._resource_node_id(mrt_index)
        node = nodes_by_id.get(node_id)
        if node is None:
            node = self._build_resource_node(resource)
            nodes_by_id[node_id] = node
        return node

    def _connect_alias_groups(
        self, nodes_by_id: dict[str, gb.GraphNode]
    ) -> None:
        alias_groups: dict[int, list[Resource]] = defaultdict(list)
        for resource in self.vgf_resources.values():
            if resource.alias_group_id is not None:
                alias_groups[resource.alias_group_id].append(resource)

        for resources in alias_groups.values():
            present_resources = [
                resource
                for resource in sorted(resources, key=lambda item: item.index)
                if self._resource_node_id(resource.index) in nodes_by_id
            ]
            for source, target in pairwise(present_resources):
                source_node = nodes_by_id[self._resource_node_id(source.index)]
                target_node = nodes_by_id[self._resource_node_id(target.index)]
                self._append_alias_edge(target_node, source_node)
                self._append_alias_edge(source_node, target_node)

    def _append_alias_edge(
        self, node: gb.GraphNode, source_node: gb.GraphNode
    ) -> None:
        self._append_unique_edge(
            node,
            gb.IncomingEdge(
                sourceNodeId=source_node.id,
                sourceNodeOutputId="alias",
                targetNodeInputId="alias",
            ),
        )

    def _add_placeholder_segment(
        self,
        segment: Segment,
        module: Module,
        nodes_by_id: dict[str, gb.GraphNode],
        reason: str,
    ) -> None:
        node = self._build_placeholder_segment_node(segment, module, reason)
        nodes_by_id[node.id] = node

        for segment_input in segment.inputs:
            self._ensure_resource_node(segment_input.mrt_index, nodes_by_id)

        for output in segment.outputs:
            output_node = self._ensure_resource_node(
                output.mrt_index, nodes_by_id
            )
            if output_node is None:
                continue
            self._append_unique_edge(
                output_node,
                gb.IncomingEdge(
                    sourceNodeId=node.id,
                    sourceNodeOutputId=str(output.index),
                    targetNodeInputId="0",
                ),
            )
            self._upsert_metadata(
                output_node.inputsMetadata,
                self._resource_tensor_shape_metadata("0", output.mrt_index),
            )

    def _build_placeholder_segment_node(
        self, segment: Segment, module: Module, reason: str
    ) -> gb.GraphNode:
        label = segment.name or module.name or f"segment_{segment.index}"
        return gb.GraphNode(
            id=f"{segment.index}__segment",
            label=f"{label} (unavailable)",
            namespace=self.segment_namespaces[segment.index],
            attrs=[
                gb.KeyValue(key="module", value=module.name),
                gb.KeyValue(key="module type", value=module.type.name),
                gb.KeyValue(key="code type", value=module.code_type.value),
                gb.KeyValue(key="reason", value=reason),
            ],
            incomingEdges=[
                gb.IncomingEdge(
                    sourceNodeId=self._resource_node_id(
                        segment_input.mrt_index
                    ),
                    sourceNodeOutputId="0",
                    targetNodeInputId=str(segment_input.index),
                )
                for segment_input in segment.inputs
            ],
            inputsMetadata=[
                self._resource_tensor_shape_metadata(
                    str(segment_input.index), segment_input.mrt_index
                )
                for segment_input in segment.inputs
            ],
            outputsMetadata=[
                self._resource_tensor_shape_metadata(
                    str(output.index), output.mrt_index
                )
                for output in segment.outputs
            ],
            config=gb.GraphNodeConfig(pinToGroupTop=True),
        )

    def _build_resource_node(self, resource: Resource) -> gb.GraphNode:
        """Build a shared VGF resource node."""
        node = gb.GraphNode(
            id=self._resource_node_id(resource.index),
            label=self._resource_node_label(resource),
            namespace=self._alias_group_namespace(resource),
            attrs=[
                gb.KeyValue(key="category", value=resource.category.name),
                gb.KeyValue(key="format", value=resource.vk_format.name),
                gb.KeyValue(key="shape", value=str(resource.shape)),
            ],
            outputsMetadata=[
                gb.MetadataItem(
                    id="0",
                    attrs=self._resource_tensor_shape_attrs(resource.index),
                )
            ],
        )
        self._extend_constant_value(node, resource.index)
        self._extend_model_io(node, resource.index)
        return node

    def _extend_constant_value(
        self, node: gb.GraphNode, mrt_index: int
    ) -> None:
        """Add constant value metadata to constant resource nodes."""
        constant = self.vgf_constants_by_mrt.get(mrt_index)
        if constant is None:
            return
        resource = self.vgf_resources.get(mrt_index)
        if resource is None:
            return
        node.attrs.append(
            gb.KeyValue(
                key="value",
                value=_format_constant_value(constant, resource),
            )
        )

    def _resource_node_label(self, resource: Resource) -> str:
        """Build a display label for a VGF resource node."""
        if resource.category == ResourceCategory.INPUT:
            return self._model_resource_label(
                self.model_inputs.get(resource.index, []), "model input"
            )
        if resource.category == ResourceCategory.OUTPUT:
            return self._model_resource_label(
                self.model_outputs.get(resource.index, []), "model output"
            )
        if resource.category == ResourceCategory.CONSTANT:
            constant = self.vgf_constants_by_mrt.get(resource.index)
            if constant is not None:
                return f"<constant {constant.index}>"
            return "<constant>"
        if resource.category == ResourceCategory.INTERMEDIATE:
            return f"<intermediate {resource.index}>"
        return "<unknown>"

    @staticmethod
    def _alias_group_namespace(resource: Resource) -> str:
        if resource.alias_group_id is None:
            return ""
        return f"alias_group_{resource.alias_group_id}"

    @staticmethod
    def _model_resource_label(
        model_ios: list[Model_Sequence_IO], fallback: str
    ) -> str:
        for model_io in model_ios:
            if model_io.name.strip():
                return model_io.name
        if model_ios:
            return f"<{fallback} {model_ios[0].index}>"
        return f"<{fallback}>"

    def _extend_model_io(self, node: gb.GraphNode, mrt_index: int) -> None:
        """Add model-level input/output annotations to resource nodes."""
        model_io_attrs = [
            gb.KeyValue(
                key="model input", value=self._format_model_io(model_input)
            )
            for model_input in self.model_inputs.get(mrt_index, [])
        ]
        model_io_attrs.extend(
            gb.KeyValue(
                key="model output", value=self._format_model_io(model_output)
            )
            for model_output in self.model_outputs.get(mrt_index, [])
        )
        insert_at = self._resource_attr_insert_index(node, "category")
        node.attrs[insert_at:insert_at] = model_io_attrs

    @staticmethod
    def _resource_attr_insert_index(node: gb.GraphNode, key: str) -> int:
        for index, attr in enumerate(node.attrs):
            if attr.key == key:
                return index + 1
        return len(node.attrs)

    def _constant_input_metadata(
        self, input_id: str, input_tag: str | None, constant: Constant
    ) -> gb.MetadataItem:
        """Build metadata for a constant edge into a segment."""
        attrs = self._resource_tensor_shape_attrs(constant.mrt_index)
        if input_tag is not None:
            attrs.insert(0, gb.KeyValue(key="__tensor_tag", value=input_tag))
        return gb.MetadataItem(id=input_id, attrs=attrs)

    def _resource_tensor_shape_metadata(
        self, metadata_id: str, mrt_index: int
    ) -> gb.MetadataItem:
        return gb.MetadataItem(
            id=metadata_id,
            attrs=self._resource_tensor_shape_attrs(mrt_index),
        )

    def _resource_tensor_shape_attrs(
        self, mrt_index: int
    ) -> list[gb.KeyValue]:
        resource = self.vgf_resources.get(mrt_index)
        if resource is None:
            return []
        return [gb.KeyValue(key="tensor_shape", value=str(resource.shape))]

    @staticmethod
    def _upsert_metadata(
        metadata_items: list[gb.MetadataItem], metadata: gb.MetadataItem
    ) -> None:
        for index, item in enumerate(metadata_items):
            if item.id == metadata.id:
                metadata_items[index] = metadata
                return
        metadata_items.append(metadata)

    @staticmethod
    def _append_unique_edge(node: gb.GraphNode, edge: gb.IncomingEdge) -> None:
        if edge not in node.incomingEdges:
            node.incomingEdges.append(edge)

    @staticmethod
    def _model_ios_by_mrt(
        model_ios: list[Model_Sequence_IO],
    ) -> dict[int, list[Model_Sequence_IO]]:
        ios_by_mrt: dict[int, list[Model_Sequence_IO]] = defaultdict(list)
        for model_io in model_ios:
            ios_by_mrt[model_io.mrt_index].append(model_io)
        return ios_by_mrt

    @staticmethod
    def _format_model_io(model_io: Model_Sequence_IO) -> str:
        name = model_io.name or f"io_{model_io.index}"
        return f"{name} (index {model_io.index}, binding {model_io.binding})"

    @staticmethod
    def _resource_node_id(mrt_index: int) -> str:
        return f"mrt_{mrt_index}"

    @staticmethod
    def _segment_namespaces(segments: list[Segment]) -> dict[int, str]:
        # VGF segment names are display strings and are not guaranteed to be
        # unique. Prefer readable name-first namespaces, but append the segment
        # index when a name is duplicated. Empty names use a stable generated
        # name. Escape Model Explorer's namespace separator to keep arbitrary
        # segment names, such as "encoder/block", as one visual group component.
        base_names = {
            segment.index: segment.name.replace("/", r"\/")
            for segment in segments
        }
        base_counts = Counter(base_names.values())
        namespaces = {}
        for index, name in base_names.items():
            if not name:
                name = f"segment_{index}"
            elif base_counts[name] > 1:
                name = f"{name}_{index}"
            namespaces[index] = name
        return namespaces

    @staticmethod
    def _spirv_entry_point_kind(module_type: ModuleType) -> EntryPointKind:
        if module_type == ModuleType.GRAPH:
            return "graph"
        if module_type == ModuleType.COMPUTE:
            return "regular"
        raise ValueError(f"Unsupported VGF module type: {module_type!r}")
