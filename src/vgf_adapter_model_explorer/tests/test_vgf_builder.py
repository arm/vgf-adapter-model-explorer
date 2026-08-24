# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

import io
import struct
from unittest.mock import patch

import pytest
from model_explorer import graph_builder as gb
from spirv_adapter_model_explorer.assembly import assemble_spirv_text
from spirv_adapter_model_explorer.validation import validate_spirv

from ..builder.builder import VgfGraphBuilder, _format_constant_value
from ..parser.parser import Parser
from ..parser.types import (
    Constant,
    DescriptorSetInfo,
    Header,
    IOBase,
    Model_Sequence_IO,
    ModelSequence,
    Module,
    ModuleCodeType,
    ModuleType,
    Resource,
    ResourceCategory,
    Segment,
    Vgf,
    VkDescriptorType,
    VkFormat,
)
from .test_vgf_parser import _write_multi_segment_vgf, vgfpy


def test_builder_creates_referenced_resource_nodes_and_placeholder_segments():
    graph = VgfGraphBuilder(_dataflow_vgf()).graph_collection.graphs[0]
    nodes = _nodes_by_id(graph)

    assert list(nodes) == [
        "mrt_0",
        "0__segment",
        "mrt_1",
        "1__segment",
        "mrt_3",
    ]
    assert all(not node.subgraphIds for node in graph.nodes)
    assert _edge_sources(nodes["0__segment"]) == [("mrt_0", "0", "0")]
    assert nodes["0__segment"].label == "graph_segment_0 (unavailable)"
    assert _edge_sources(nodes["mrt_1"]) == [("0__segment", "0", "0")]
    assert _edge_sources(nodes["1__segment"]) == [("mrt_1", "0", "0")]
    assert _edge_sources(nodes["mrt_3"]) == [("1__segment", "0", "0")]


def test_builder_deduplicates_intermediate_resource_nodes():
    graph = VgfGraphBuilder(_dataflow_vgf()).graph_collection.graphs[0]

    assert [node.id for node in graph.nodes].count("mrt_1") == 1


def test_builder_does_not_group_unaliased_resource_or_segment_nodes_by_namespace():
    graph = VgfGraphBuilder(_dataflow_vgf()).graph_collection.graphs[0]

    assert all(
        node.namespace == ""
        for node in graph.nodes
        if node.id.startswith("mrt_")
    )
    assert _nodes_by_id(graph)["0__segment"].namespace == "graph_segment_0"


def test_builder_connects_alias_group_resources_bidirectionally_in_namespace():
    vgf = _dataflow_vgf()
    vgf.resources[1].alias_group_id = 7
    vgf.resources[3].alias_group_id = 7

    graph = VgfGraphBuilder(vgf).graph_collection.graphs[0]
    nodes = _nodes_by_id(graph)

    assert nodes["mrt_1"].namespace == "alias_group_7"
    assert nodes["mrt_3"].namespace == "alias_group_7"
    assert _edge_sources(nodes["mrt_1"]) == [
        ("0__segment", "0", "0"),
        ("mrt_3", "alias", "alias"),
    ]
    assert _edge_sources(nodes["mrt_3"]) == [
        ("1__segment", "0", "0"),
        ("mrt_1", "alias", "alias"),
    ]


def test_builder_adds_model_level_input_output_attrs_to_resource_nodes():
    graph = VgfGraphBuilder(_dataflow_vgf()).graph_collection.graphs[0]
    nodes = _nodes_by_id(graph)

    assert _attrs(nodes["mrt_0"])["model input"] == (
        "model_input (index 0, binding 0)"
    )
    assert _attrs(nodes["mrt_3"])["model output"] == (
        "model_output (index 0, binding 3)"
    )


def test_builder_orders_model_input_resource_attrs():
    graph = VgfGraphBuilder(_dataflow_vgf()).graph_collection.graphs[0]

    assert _attr_keys(_nodes_by_id(graph)["mrt_0"]) == [
        "category",
        "model input",
        "format",
        "shape",
    ]


def test_builder_orders_model_output_resource_attrs():
    graph = VgfGraphBuilder(_dataflow_vgf()).graph_collection.graphs[0]

    assert _attr_keys(_nodes_by_id(graph)["mrt_3"]) == [
        "category",
        "model output",
        "format",
        "shape",
    ]


def test_builder_labels_model_input_resource_with_model_sequence_name():
    graph = VgfGraphBuilder(_dataflow_vgf()).graph_collection.graphs[0]

    assert _nodes_by_id(graph)["mrt_0"].label == "model_input"


def test_builder_labels_model_output_resource_with_model_sequence_name():
    graph = VgfGraphBuilder(_dataflow_vgf()).graph_collection.graphs[0]

    assert _nodes_by_id(graph)["mrt_3"].label == "model_output"


def test_builder_labels_intermediate_resource_with_mrt_index():
    graph = VgfGraphBuilder(_dataflow_vgf()).graph_collection.graphs[0]

    assert _nodes_by_id(graph)["mrt_1"].label == "<intermediate 1>"


def test_builder_labels_input_resource_with_blank_model_sequence_name_as_model_input():
    graph = VgfGraphBuilder(
        _dataflow_vgf(input_name="")
    ).graph_collection.graphs[0]

    assert _nodes_by_id(graph)["mrt_0"].label == "<model input 0>"


def test_builder_labels_output_resource_with_blank_model_sequence_name_as_model_output():
    graph = VgfGraphBuilder(
        _dataflow_vgf(output_name="")
    ).graph_collection.graphs[0]

    assert _nodes_by_id(graph)["mrt_3"].label == "<model output 0>"


def test_builder_labels_input_resource_missing_from_model_sequence_as_model_input():
    graph = VgfGraphBuilder(
        _dataflow_vgf(include_model_input=False)
    ).graph_collection.graphs[0]

    assert _nodes_by_id(graph)["mrt_0"].label == "<model input>"


def test_builder_labels_output_resource_missing_from_model_sequence_as_model_output():
    graph = VgfGraphBuilder(
        _dataflow_vgf(include_model_output=False)
    ).graph_collection.graphs[0]

    assert _nodes_by_id(graph)["mrt_3"].label == "<model output>"


def test_builder_hides_constant_resource_nodes_by_default():
    graph = VgfGraphBuilder(_dataflow_vgf()).graph_collection.graphs[0]

    assert "mrt_2" not in _nodes_by_id(graph)


def test_builder_labels_constant_resource_with_constant_table_index():
    graph = VgfGraphBuilder(
        _dataflow_vgf(), show_constants=True
    ).graph_collection.graphs[0]

    assert _nodes_by_id(graph)["mrt_2"].label == "<constant 0>"


def test_builder_adds_constant_value_attr():
    graph = VgfGraphBuilder(
        _dataflow_vgf(), show_constants=True
    ).graph_collection.graphs[0]

    assert _attrs(_nodes_by_id(graph)["mrt_2"])["value"] == ("[97, 98, 99]")


def test_builder_orders_constant_resource_attrs():
    graph = VgfGraphBuilder(
        _dataflow_vgf(), show_constants=True
    ).graph_collection.graphs[0]

    assert _attr_keys(_nodes_by_id(graph)["mrt_2"]) == [
        "category",
        "format",
        "shape",
        "value",
    ]


@pytest.mark.parametrize(
    ("vk_format", "data", "expected"),
    [
        (VkFormat.VK_FORMAT_R8_SINT, bytes([255, 0, 1]), "[-1, 0, 1]"),
        (VkFormat.VK_FORMAT_R8_UINT, bytes([255, 0, 1]), "[255, 0, 1]"),
        (VkFormat.VK_FORMAT_R8_BOOL_ARM, bytes([1, 0]), "[true, false]"),
        (VkFormat.VK_FORMAT_R16_SINT, b"\xff\xff\x02\x00", "[-1, 2]"),
        (VkFormat.VK_FORMAT_R16_UINT, b"\xff\xff\x02\x00", "[65535, 2]"),
        (VkFormat.VK_FORMAT_R16_SFLOAT, b"\x00<\x00\xc0", "[1.0, -2.0]"),
        (
            VkFormat.VK_FORMAT_R32_SINT,
            b"\xff\xff\xff\xff\x02\x00\x00\x00",
            "[-1, 2]",
        ),
        (
            VkFormat.VK_FORMAT_R32_UINT,
            b"\xff\xff\xff\xff\x02\x00\x00\x00",
            "[4294967295, 2]",
        ),
        (
            VkFormat.VK_FORMAT_R32_SFLOAT,
            b"\x00\x00\x80?\x00\x00\x00\xc0",
            "[1.0, -2.0]",
        ),
        (
            VkFormat.VK_FORMAT_R64_SINT,
            b"\xff\xff\xff\xff\xff\xff\xff\xff",
            "[-1]",
        ),
        (
            VkFormat.VK_FORMAT_R64_UINT,
            b"\xff\xff\xff\xff\xff\xff\xff\xff",
            "[18446744073709551615]",
        ),
        (
            VkFormat.VK_FORMAT_R64_SFLOAT,
            b"\x00\x00\x00\x00\x00\x00\xf0?",
            "[1.0]",
        ),
    ],
)
def test_format_constant_value_decodes_single_channel_formats(
    vk_format, data, expected
):
    constant = Constant(index=0, mrt_index=0, sparsity_dimension=-1, data=data)
    resource = Resource(
        category=ResourceCategory.CONSTANT,
        index=0,
        shape=[1],
        stride=[],
        vk_descriptor_type=None,
        vk_format=vk_format,
    )

    assert _format_constant_value(constant, resource) == expected


def test_format_constant_value_uses_multidimensional_shape():
    constant = Constant(
        index=0,
        mrt_index=0,
        sparsity_dimension=-1,
        data=bytes(range(6)),
    )
    resource = Resource(
        category=ResourceCategory.CONSTANT,
        index=0,
        shape=[2, 3],
        stride=[],
        vk_descriptor_type=None,
        vk_format=VkFormat.VK_FORMAT_R8_UINT,
    )

    assert (
        _format_constant_value(constant, resource)
        == "[\n  [0, 1, 2],\n  [3, 4, 5]\n]"
    )


def test_format_constant_value_uses_nested_multidimensional_shape():
    constant = Constant(
        index=0,
        mrt_index=0,
        sparsity_dimension=-1,
        data=bytes(range(8)),
    )
    resource = Resource(
        category=ResourceCategory.CONSTANT,
        index=0,
        shape=[2, 2, 2],
        stride=[],
        vk_descriptor_type=None,
        vk_format=VkFormat.VK_FORMAT_R8_UINT,
    )

    assert _format_constant_value(constant, resource) == (
        "[\n"
        "  [\n"
        "    [0, 1],\n"
        "    [2, 3]\n"
        "  ],\n"
        "  [\n"
        "    [4, 5],\n"
        "    [6, 7]\n"
        "  ]\n"
        "]"
    )


def test_format_constant_value_summarizes_large_all_zero_constants():
    constant = Constant(
        index=0, mrt_index=0, sparsity_dimension=-1, data=bytes(1025)
    )
    resource = Resource(
        category=ResourceCategory.CONSTANT,
        index=0,
        shape=[1025],
        stride=[],
        vk_descriptor_type=None,
        vk_format=VkFormat.VK_FORMAT_R8_UINT,
    )

    assert (
        _format_constant_value(constant, resource) == "[<all zeroes>] len=1025"
    )


def test_format_constant_value_truncates_large_constants():
    constant = Constant(
        index=0,
        mrt_index=0,
        sparsity_dimension=-1,
        data=bytes(index % 256 for index in range(1025)),
    )
    resource = Resource(
        category=ResourceCategory.CONSTANT,
        index=0,
        shape=[1025],
        stride=[],
        vk_descriptor_type=None,
        vk_format=VkFormat.VK_FORMAT_R8_UINT,
    )

    expected_preview = ", ".join(str(index % 256) for index in range(1024))
    assert _format_constant_value(constant, resource) == (
        f"[{expected_preview}, ...] len=1025"
    )


def test_format_constant_value_uses_sparse_fallback():
    constant = Constant(
        index=0, mrt_index=0, sparsity_dimension=2, data=b"abc"
    )
    resource = Resource(
        category=ResourceCategory.CONSTANT,
        index=0,
        shape=[3],
        stride=[],
        vk_descriptor_type=None,
        vk_format=VkFormat.VK_FORMAT_R8_SINT,
    )

    assert _format_constant_value(constant, resource) == (
        "<3 sparse bytes, sparsity_dimension=2>"
    )


def test_format_constant_value_uses_element_count_fallback_for_unsupported_format():
    constant = Constant(
        index=0, mrt_index=0, sparsity_dimension=-1, data=b"abc"
    )
    resource = Resource(
        category=ResourceCategory.CONSTANT,
        index=0,
        shape=[2, 3],
        stride=[],
        vk_descriptor_type=None,
        vk_format=VkFormat.VK_FORMAT_R8G8_UINT,
    )

    assert _format_constant_value(constant, resource) == (
        "<6 elements of VK_FORMAT_R8G8_UINT>"
    )


def test_format_constant_value_uses_byte_count_fallback_without_shape():
    constant = Constant(
        index=0, mrt_index=0, sparsity_dimension=-1, data=b"abc"
    )
    resource = Resource(
        category=ResourceCategory.CONSTANT,
        index=0,
        shape=[],
        stride=[],
        vk_descriptor_type=None,
        vk_format=VkFormat.VK_FORMAT_R8G8_UINT,
    )

    assert _format_constant_value(constant, resource) == (
        "<3 bytes of VK_FORMAT_R8G8_UINT>"
    )


def test_builder_keeps_resource_attrs_on_outputs():
    graph = VgfGraphBuilder(_dataflow_vgf()).graph_collection.graphs[0]
    nodes = _nodes_by_id(graph)

    assert _attrs(nodes["mrt_3"])["category"] == "OUTPUT"
    assert _attrs(nodes["mrt_3"])["shape"] == "[1, 4]"
    assert _attrs(nodes["mrt_3"])["format"] == "VK_FORMAT_R8_UINT"


def test_spirv_entry_point_kind_rejects_unknown_module_type():
    with pytest.raises(ValueError, match="Unsupported VGF module type"):
        VgfGraphBuilder._spirv_entry_point_kind(ModuleType(99))


def test_builder_inlines_spirv_nodes_in_segment_namespace():
    spirv_graph = gb.Graph(
        id="0_0_spirv",
        nodes=[gb.GraphNode(id="0__spirv_inst_0", label="OpCapability")],
        groupNodeAttributes={"graph_segment_0": {"Validation": "Passed"}},
    )

    with patch(
        "vgf_adapter_model_explorer.builder.builder.build_spirv_graph_from_bytes",
        return_value=spirv_graph,
    ) as mock_build_spirv_graph:
        graph_collection = VgfGraphBuilder(
            _dataflow_vgf(has_spirv=True)
        ).graph_collection

    main_graph = graph_collection.graphs[0]
    nodes = _nodes_by_id(main_graph)

    assert [graph.id for graph in graph_collection.graphs] == ["Main"]
    assert nodes["0__spirv_inst_0"].label == "OpCapability"
    assert main_graph.groupNodeAttributes["graph_segment_0"] == {
        "Validation": "Passed"
    }
    mock_build_spirv_graph.assert_called_once_with(
        b"SPIR-V bytes are mocked in this test",
        entry_point="graph_partition_0",
        entry_point_kind="graph",
        namespace="graph_segment_0",
        node_id_prefix="0__",
    )


def test_builder_escapes_segment_name_namespace_separators():
    vgf = _dataflow_vgf(has_spirv=True)
    vgf.model_sequence.segments[0].name = "encoder/block"

    with patch(
        "vgf_adapter_model_explorer.builder.builder.build_spirv_graph_from_bytes",
        return_value=gb.Graph(id="0_0_spirv", nodes=[]),
    ) as mock_build_spirv_graph:
        VgfGraphBuilder(vgf)

    assert (
        mock_build_spirv_graph.call_args.kwargs["namespace"]
        == r"encoder\/block"
    )


def test_builder_appends_segment_index_suffix_only_for_duplicate_names():
    vgf = _dataflow_vgf()
    first, second = vgf.model_sequence.segments

    assert VgfGraphBuilder(vgf).segment_namespaces == {
        0: "graph_segment_0",
        1: "compute_segment_1",
    }

    first.name = "shared"
    second.name = "shared"
    assert VgfGraphBuilder(vgf).segment_namespaces == {
        0: "shared_0",
        1: "shared_1",
    }

    first.name = ""
    second.name = "named"
    assert VgfGraphBuilder(vgf).segment_namespaces == {
        0: "segment_0",
        1: "named",
    }


def test_builder_adds_tosa_arg_and_vgf_shape_metadata_to_constant_edges():
    spirv_graph = gb.Graph(
        id="0_0_spirv",
        nodes=[
            gb.GraphNode(
                id="0__conv",
                label="CONV2D",
                attrs=[
                    gb.KeyValue(
                        key="arg6 weight",
                        value="<graph constant 0> uint8[3]",
                    )
                ],
            )
        ],
    )

    with patch(
        "vgf_adapter_model_explorer.builder.builder.build_spirv_graph_from_bytes",
        return_value=spirv_graph,
    ):
        graph = VgfGraphBuilder(
            _dataflow_vgf(has_spirv=True), show_constants=True
        ).graph_collection.graphs[0]

    conv = _nodes_by_id(graph)["0__conv"]

    assert _edge_sources(conv) == [("mrt_2", "0", "6")]
    assert "arg6 weight" not in _attrs(conv)
    assert _attrs(_metadata_by_id(conv)["6"]) == {
        "__tensor_tag": "weight",
        "tensor_shape": "[3]",
    }
    assert _attrs(
        _output_metadata_by_id(_nodes_by_id(graph)["mrt_2"])["0"]
    ) == {"tensor_shape": "[3]"}


def test_builder_adds_vgf_shape_metadata_to_input_output_and_intermediate_edges():
    spirv_graph = gb.Graph(
        id="0_0_spirv",
        nodes=[
            gb.GraphNode(
                id="0__graph_input",
                label="Input 0",
                attrs=[gb.KeyValue(key="logical input idx", value="0")],
            ),
            gb.GraphNode(
                id="0__consumer",
                label="Consumer",
                incomingEdges=[gb.IncomingEdge("0__graph_input", "0", "0")],
            ),
            gb.GraphNode(
                id="0__graph_output",
                label="Output 0",
                attrs=[gb.KeyValue(key="logical output idx", value="0")],
                incomingEdges=[gb.IncomingEdge("0__consumer", "0", "0")],
            ),
        ],
    )

    with patch(
        "vgf_adapter_model_explorer.builder.builder.build_spirv_graph_from_bytes",
        return_value=spirv_graph,
    ):
        graph = VgfGraphBuilder(
            _dataflow_vgf(has_spirv=True)
        ).graph_collection.graphs[0]

    nodes = _nodes_by_id(graph)

    assert "0__graph_input" not in nodes
    assert "0__graph_output" not in nodes
    assert _edge_sources(nodes["0__consumer"]) == [("mrt_0", "0", "0")]
    assert _edge_sources(nodes["mrt_1"]) == [("0__consumer", "0", "0")]

    input_metadata = _attrs(_metadata_by_id(nodes["0__consumer"])["0"])
    assert input_metadata == {"tensor_shape": "[1, 4]"}
    assert _attrs(_output_metadata_by_id(nodes["mrt_0"])["0"]) == {
        "tensor_shape": "[1, 4]"
    }

    output_metadata = _attrs(_metadata_by_id(nodes["mrt_1"])["0"])
    assert output_metadata == {"tensor_shape": "[1, 4]"}


def test_builder_bypasses_terminal_nodes_for_multiple_consumers_and_producers():
    spirv_graph = gb.Graph(
        id="0_0_spirv",
        nodes=[
            gb.GraphNode(
                id="0__graph_input",
                label="Input 0",
                attrs=[gb.KeyValue(key="logical input idx", value="0")],
            ),
            gb.GraphNode(
                id="0__consumer_a",
                label="Consumer A",
                incomingEdges=[gb.IncomingEdge("0__graph_input", "0", "0")],
            ),
            gb.GraphNode(
                id="0__consumer_b",
                label="Consumer B",
                incomingEdges=[gb.IncomingEdge("0__graph_input", "0", "1")],
            ),
            gb.GraphNode(id="0__producer_a", label="Producer A"),
            gb.GraphNode(id="0__producer_b", label="Producer B"),
            gb.GraphNode(
                id="0__graph_output",
                label="Output 0",
                attrs=[gb.KeyValue(key="logical output idx", value="0")],
                incomingEdges=[
                    gb.IncomingEdge("0__producer_a", "0", "0"),
                    gb.IncomingEdge("0__producer_b", "1", "1"),
                ],
            ),
        ],
    )

    with patch(
        "vgf_adapter_model_explorer.builder.builder.build_spirv_graph_from_bytes",
        return_value=spirv_graph,
    ):
        graph = VgfGraphBuilder(
            _dataflow_vgf(has_spirv=True)
        ).graph_collection.graphs[0]

    nodes = _nodes_by_id(graph)

    assert "0__graph_input" not in nodes
    assert "0__graph_output" not in nodes
    assert _edge_sources(nodes["0__consumer_a"]) == [("mrt_0", "0", "0")]
    assert _edge_sources(nodes["0__consumer_b"]) == [("mrt_0", "0", "1")]
    assert _edge_sources(nodes["mrt_1"]) == [
        ("0__producer_a", "0", "0"),
        ("0__producer_b", "1", "0"),
    ]
    assert _attrs(_metadata_by_id(nodes["0__consumer_a"])["0"]) == {
        "tensor_shape": "[1, 4]"
    }
    assert _attrs(_metadata_by_id(nodes["0__consumer_b"])["1"]) == {
        "tensor_shape": "[1, 4]"
    }


def test_builder_keeps_model_viewable_when_spirv_graph_build_fails():
    with patch(
        "vgf_adapter_model_explorer.builder.builder.build_spirv_graph_from_bytes",
        side_effect=ValueError("bad SPIR-V"),
    ):
        graph = VgfGraphBuilder(
            _dataflow_vgf(has_spirv=True)
        ).graph_collection.graphs[0]

    nodes = _nodes_by_id(graph)

    assert "0__segment" in nodes
    assert _attrs(nodes["0__segment"])["reason"] == (
        "SPIR-V graph could not be built: bad SPIR-V"
    )
    assert _edge_sources(nodes["0__segment"]) == [("mrt_0", "0", "0")]
    assert _edge_sources(nodes["mrt_1"]) == [("0__segment", "0", "0")]


def test_builder_connects_compute_segment_io_by_descriptor_mrt_not_logical_binding():
    vgf = _dataflow_vgf(has_spirv=False)
    vgf.resources.append(
        Resource(
            category=ResourceCategory.INTERMEDIATE,
            index=4,
            shape=[2, 4],
            stride=[],
            vk_descriptor_type=VkDescriptorType.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            vk_format=VkFormat.VK_FORMAT_R8_UINT,
        )
    )
    vgf.modules[1].has_spirv = True
    vgf.modules[1].code_size = 4
    vgf.modules[1].code_available = True
    vgf.modules[1].code_type = ModuleCodeType.SPIRV
    vgf.modules[1].spirv_code = b"compute SPIR-V bytes are mocked in this test"
    # These are logical VGF binding slots, not shader descriptor bindings.
    # Keep them different from the descriptor_set_infos bindings below so the
    # test fails if compute matching accidentally starts using io.binding.
    vgf.model_sequence.segments[1].inputs[0].binding = 22
    vgf.model_sequence.segments[1].outputs[0].binding = 33
    vgf.model_sequence.segments[1].inputs.append(
        IOBase(binding=44, index=1, mrt_index=4)
    )
    vgf.model_sequence.segments[1].outputs.append(
        IOBase(binding=44, index=1, mrt_index=4)
    )
    vgf.model_sequence.segments[1].descriptor_set_infos[0].bindings.append(
        IOBase(binding=4, index=2, mrt_index=4)
    )

    spirv_graph = gb.Graph(
        id="compute",
        nodes=[
            gb.GraphNode(
                id="1__resource_input",
                label="Input input_buffer",
                attrs=[
                    gb.KeyValue(key="descriptor", value="set 6, binding 2")
                ],
            ),
            gb.GraphNode(
                id="1__resource_output",
                label="Output output_buffer",
                attrs=[
                    gb.KeyValue(key="descriptor", value="set 6, binding 3")
                ],
                incomingEdges=[gb.IncomingEdge("1__compute", "0", "0")],
            ),
            gb.GraphNode(
                id="1__resource_state",
                label="Input/Output state_buffer",
                attrs=[
                    gb.KeyValue(key="descriptor", value="set 6, binding 4")
                ],
                incomingEdges=[gb.IncomingEdge("1__compute", "1", "0")],
            ),
            gb.GraphNode(
                id="1__compute",
                label="<compute>",
                incomingEdges=[
                    gb.IncomingEdge("1__resource_input", "0", "0"),
                    gb.IncomingEdge("1__resource_state", "0", "1"),
                ],
            ),
        ],
    )

    with patch(
        "vgf_adapter_model_explorer.builder.builder.build_spirv_graph_from_bytes",
        return_value=spirv_graph,
    ) as mock_build_spirv_graph:
        graph = VgfGraphBuilder(vgf).graph_collection.graphs[0]

    nodes = _nodes_by_id(graph)
    mock_build_spirv_graph.assert_called_once_with(
        b"compute SPIR-V bytes are mocked in this test",
        entry_point="compute_partition_1",
        entry_point_kind="regular",
        namespace="compute_segment_1",
        node_id_prefix="1__",
    )
    assert "1__resource_input" not in nodes
    assert "1__resource_output" not in nodes
    assert "1__resource_state" not in nodes
    assert _edge_sources(nodes["1__compute"]) == [
        ("mrt_1", "0", "0"),
        ("mrt_4", "0", "1"),
    ]
    assert _edge_sources(nodes["mrt_3"]) == [("1__compute", "0", "0")]
    assert _edge_sources(nodes["mrt_4"]) == [("1__compute", "1", "0")]
    assert _attrs(_metadata_by_id(nodes["1__compute"])["1"]) == {
        "tensor_shape": "[2, 4]"
    }


def test_builder_inlines_multi_segment_vgf_without_constants_by_default(
    tmp_path,
):
    vgf_path = tmp_path / "multi_segment.vgf"
    _write_multi_segment_vgf(vgf_path)

    graph_collection = VgfGraphBuilder(
        Parser(str(vgf_path)).vgf
    ).graph_collection
    main_graph = graph_collection.graphs[0]
    nodes = _nodes_by_id(main_graph)

    assert [graph.id for graph in graph_collection.graphs] == ["Main"]
    assert set(main_graph.groupNodeAttributes) == {
        "graph_preprocess",
        "compute_preprocess",
        "graph_fuse",
        "side_branch",
        "graph_final",
    }
    assert (
        main_graph.groupNodeAttributes["graph_preprocess"]["Validation"]
        == "Passed"
    )
    assert not any(node_id.startswith("segment_") for node_id in nodes)
    assert all(
        node.namespace == "graph_preprocess"
        for node in main_graph.nodes
        if node.id.startswith("0__")
    )

    assert not any("__input_" in node_id for node_id in nodes)
    assert not any("__output_" in node_id for node_id in nodes)
    assert not any("__resource_" in node_id for node_id in nodes)
    assert _edge_sources(nodes["0__op_17"]) == [("mrt_0", "0", "0")]
    assert (
        _attrs(nodes["0__op_17"])["arg1 input2"]
        == "<graph constant 0> uint8[1]"
    )
    assert "mrt_10" not in nodes
    assert nodes["mrt_3"].namespace == "alias_group_42"
    assert nodes["mrt_9"].namespace == "alias_group_42"
    assert _edge_sources(nodes["mrt_3"]) == [
        ("0__op_18", "0", "0"),
        ("mrt_9", "alias", "alias"),
    ]
    assert _edge_sources(nodes["mrt_9"]) == [("mrt_3", "alias", "alias")]
    assert _edge_sources(nodes["1__compute"]) == [("mrt_1", "0", "0")]
    assert _edge_sources(nodes["mrt_4"]) == [("1__compute", "0", "0")]
    assert _edge_sources(nodes["3__compute"]) == [
        ("mrt_2", "0", "0"),
        ("mrt_9", "0", "1"),
    ]
    assert _edge_sources(nodes["mrt_6"]) == [("3__compute", "0", "0")]
    assert _edge_sources(nodes["mrt_5"]) == [("2__op_25", "0", "0")]
    assert _edge_sources(nodes["mrt_8"]) == [("2__op_25", "0", "0")]
    assert _edge_sources(nodes["mrt_7"]) == [("4__op_22", "0", "0")]

    compute_input_metadata = _attrs(_metadata_by_id(nodes["3__compute"])["1"])
    fuse_input_metadata = _attrs(_metadata_by_id(nodes["2__op_22"])["0"])
    final_output_metadata = _attrs(_metadata_by_id(nodes["mrt_7"])["0"])
    aux_output_metadata = _attrs(_metadata_by_id(nodes["mrt_8"])["0"])

    assert compute_input_metadata == {"tensor_shape": "[1, 16, 16, 8]"}
    assert fuse_input_metadata == {"tensor_shape": "[1, 16, 16, 8]"}
    assert final_output_metadata == {"tensor_shape": "[1, 16, 16, 16]"}
    assert aux_output_metadata == {"tensor_shape": "[1, 16, 16, 16]"}


def test_builder_inlines_multi_segment_vgf_with_constants_shown(tmp_path):
    vgf_path = tmp_path / "multi_segment.vgf"
    _write_multi_segment_vgf(vgf_path)

    graph_collection = VgfGraphBuilder(
        Parser(str(vgf_path)).vgf, show_constants=True
    ).graph_collection
    nodes = _nodes_by_id(graph_collection.graphs[0])

    assert "0__input_0" not in nodes
    assert _edge_sources(nodes["0__op_17"]) == [
        ("mrt_0", "0", "0"),
        ("mrt_10", "0", "1"),
    ]
    assert "arg1 input2" not in _attrs(nodes["0__op_17"])
    assert _attrs(nodes["mrt_10"])["value"] == "[10, 11, 12, 13]"
    assert _edge_sources(nodes["2__op_24"]) == [
        ("2__op_23", "0", "0"),
        ("mrt_11", "0", "1"),
    ]
    assert _edge_sources(nodes["4__op_22"]) == [
        ("4__op_21", "0", "0"),
        ("mrt_14", "0", "1"),
    ]

    assert "arg1 input2" not in _attrs(nodes["2__op_24"])
    assert "arg1 input2" not in _attrs(nodes["4__op_22"])
    assert _attrs(nodes["mrt_11"])["value"] == "[20, 21, 22, 23]"
    assert _attrs(nodes["mrt_14"])["value"] == "[50, 51]"
    assert _attrs(_metadata_by_id(nodes["2__op_24"])["1"]) == {
        "__tensor_tag": "input2",
        "tensor_shape": "[4]",
    }
    assert _attrs(_metadata_by_id(nodes["4__op_22"])["1"]) == {
        "__tensor_tag": "input2",
        "tensor_shape": "[2]",
    }


def _dataflow_vgf(
    has_spirv: bool = False,
    input_name: str = "model_input",
    output_name: str = "model_output",
    include_model_input: bool = True,
    include_model_output: bool = True,
) -> Vgf:
    return Vgf(
        file_path="hello_vgf.vgf",
        header=Header(
            major=0,
            minor=4,
            patch=3,
            encoder_vulkan_headers_version=123,
            is_latest_version=True,
            is_valid=True,
            check_version=True,
        ),
        resources=[
            Resource(
                category=ResourceCategory.INPUT,
                index=0,
                shape=[1, 4],
                stride=[],
                vk_descriptor_type=VkDescriptorType.VK_DESCRIPTOR_TYPE_TENSOR_ARM,
                vk_format=VkFormat.VK_FORMAT_R8_SINT,
            ),
            Resource(
                category=ResourceCategory.INTERMEDIATE,
                index=1,
                shape=[1, 4],
                stride=[],
                vk_descriptor_type=VkDescriptorType.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                vk_format=VkFormat.VK_FORMAT_R8_UINT,
            ),
            Resource(
                category=ResourceCategory.CONSTANT,
                index=2,
                shape=[3],
                stride=[],
                vk_descriptor_type=None,
                vk_format=VkFormat.VK_FORMAT_R8_SINT,
            ),
            Resource(
                category=ResourceCategory.OUTPUT,
                index=3,
                shape=[1, 4],
                stride=[],
                vk_descriptor_type=VkDescriptorType.VK_DESCRIPTOR_TYPE_TENSOR_ARM,
                vk_format=VkFormat.VK_FORMAT_R8_UINT,
            ),
        ],
        constants=[
            Constant(index=0, mrt_index=2, sparsity_dimension=-1, data=b"abc"),
        ],
        modules=[
            Module(
                code_size=541,
                entry_point="graph_partition_0",
                has_spirv=has_spirv,
                index=0,
                name="graph_partition_0",
                type=ModuleType.GRAPH,
                code_type=ModuleCodeType.SPIRV,
                code_available=True,
                spirv_code=b"SPIR-V bytes are mocked in this test"
                if has_spirv
                else None,
            ),
            Module(
                code_size=0,
                entry_point="compute_partition_1",
                has_spirv=False,
                index=1,
                name="compute_partition_1",
                type=ModuleType.COMPUTE,
                code_type=ModuleCodeType.NONE,
                code_available=False,
            ),
        ],
        model_sequence=ModelSequence(
            inputs=[
                Model_Sequence_IO(
                    binding=0,
                    index=0,
                    mrt_index=0,
                    name=input_name,
                )
            ]
            if include_model_input
            else [],
            outputs=[
                Model_Sequence_IO(
                    binding=3,
                    index=0,
                    mrt_index=3,
                    name=output_name,
                )
            ]
            if include_model_output
            else [],
            segments=[
                Segment(
                    constants=[0],
                    descriptor_set_infos=[
                        DescriptorSetInfo(
                            index=0,
                            set_index=5,
                            bindings=[
                                IOBase(binding=0, index=0, mrt_index=0),
                                IOBase(binding=1, index=1, mrt_index=1),
                            ],
                        )
                    ],
                    index=0,
                    dispatch_shape=[0, 0, 0],
                    inputs=[IOBase(binding=0, index=0, mrt_index=0)],
                    outputs=[IOBase(binding=1, index=0, mrt_index=1)],
                    module_index=0,
                    name="graph_segment_0",
                    type=ModuleType.GRAPH,
                    push_constant_ranges=[],
                ),
                Segment(
                    constants=[],
                    descriptor_set_infos=[
                        DescriptorSetInfo(
                            index=0,
                            set_index=6,
                            bindings=[
                                IOBase(binding=2, index=0, mrt_index=1),
                                IOBase(binding=3, index=1, mrt_index=3),
                            ],
                        )
                    ],
                    index=1,
                    dispatch_shape=[1, 1, 1],
                    inputs=[IOBase(binding=2, index=0, mrt_index=1)],
                    outputs=[IOBase(binding=3, index=0, mrt_index=3)],
                    module_index=1,
                    name="compute_segment_1",
                    type=ModuleType.COMPUTE,
                    push_constant_ranges=[],
                ),
            ],
        ),
    )


def _nodes_by_id(graph: gb.Graph) -> dict[str, gb.GraphNode]:
    return {node.id: node for node in graph.nodes}


def _attrs(item) -> dict[str, str]:
    return {attr.key: attr.value for attr in item.attrs}


def _edge_sources(node: gb.GraphNode) -> list[tuple[str, str, str]]:
    return [
        (edge.sourceNodeId, edge.sourceNodeOutputId, edge.targetNodeInputId)
        for edge in node.incomingEdges
    ]


def _metadata_by_id(node: gb.GraphNode) -> dict[str, gb.MetadataItem]:
    return {metadata.id: metadata for metadata in node.inputsMetadata}


def _output_metadata_by_id(node: gb.GraphNode) -> dict[str, gb.MetadataItem]:
    return {metadata.id: metadata for metadata in node.outputsMetadata}


def _attr_keys(item) -> list[str]:
    return [attr.key for attr in item.attrs]


def test_builder_uses_vgf_module_entry_point_for_multi_entry_spirv(
    tmp_path,
):
    vgf_path = tmp_path / "multi_entry_module.vgf"
    _write_multi_entry_point_vgf(vgf_path)

    graph = VgfGraphBuilder(Parser(str(vgf_path)).vgf).graph_collection.graphs[
        0
    ]
    nodes = _nodes_by_id(graph)

    assert (
        graph.groupNodeAttributes["first_segment"]["entry_points"] == "first"
    )
    assert (
        graph.groupNodeAttributes["second_segment"]["entry_points"] == "second"
    )
    assert not any("__input_" in node_id for node_id in nodes)
    assert not any("__output_" in node_id for node_id in nodes)
    assert set(nodes) == {"mrt_0", "mrt_1", "mrt_2", "mrt_3"}
    assert _edge_sources(nodes["mrt_1"]) == [("mrt_0", "0", "0")]
    assert _edge_sources(nodes["mrt_3"]) == [("mrt_2", "0", "0")]


def _write_multi_entry_point_vgf(path) -> None:
    encoder = vgfpy.CreateEncoder(123)
    spirv_words = _multi_entry_point_spirv_words()

    first_module = encoder.AddModule(
        vgfpy.ModuleType.Graph,
        "shared_graph_mod_first",
        "first",
        spirv_words,
    )
    second_module = encoder.AddModule(
        vgfpy.ModuleType.Graph,
        "shared_graph_mod_second",
        "second",
        spirv_words,
    )

    first_input = encoder.AddInputResource(
        VkDescriptorType.VK_DESCRIPTOR_TYPE_TENSOR_ARM,
        VkFormat.VK_FORMAT_R8_SINT,
        [1],
        [],
    )
    first_output = encoder.AddIntermediateResource(
        VkDescriptorType.VK_DESCRIPTOR_TYPE_TENSOR_ARM,
        VkFormat.VK_FORMAT_R8_SINT,
        [1],
        [],
    )
    second_input = encoder.AddIntermediateResource(
        VkDescriptorType.VK_DESCRIPTOR_TYPE_TENSOR_ARM,
        VkFormat.VK_FORMAT_R8_SINT,
        [1],
        [],
    )
    second_output = encoder.AddOutputResource(
        VkDescriptorType.VK_DESCRIPTOR_TYPE_TENSOR_ARM,
        VkFormat.VK_FORMAT_R8_SINT,
        [1],
        [],
    )

    first_input_slot = encoder.AddBindingSlot(0, first_input)
    first_output_slot = encoder.AddBindingSlot(1, first_output)
    second_input_slot = encoder.AddBindingSlot(2, second_input)
    second_output_slot = encoder.AddBindingSlot(3, second_output)

    first_descriptor = encoder.AddDescriptorSetInfo(
        [first_input_slot, first_output_slot],
        0,
    )
    second_descriptor = encoder.AddDescriptorSetInfo(
        [second_input_slot, second_output_slot],
        0,
    )

    encoder.AddSegmentInfo(
        first_module,
        "first_segment",
        [first_descriptor],
        [first_input_slot],
        [first_output_slot],
        [],
        [1, 1, 1],
        [],
    )
    encoder.AddSegmentInfo(
        second_module,
        "second_segment",
        [second_descriptor],
        [second_input_slot],
        [second_output_slot],
        [],
        [1, 1, 1],
        [],
    )
    encoder.AddModelSequenceInputsOutputs(
        [first_input_slot],
        ["first_input"],
        [second_output_slot],
        ["second_output"],
    )

    encoder.Finish()
    stream = io.BytesIO()
    assert encoder.WriteTo(stream)
    path.write_bytes(stream.getvalue())


def _multi_entry_point_spirv_words() -> list[int]:
    asm = """
; SPIR-V
; Version: 1.0
; Bound: 100
; Schema: 0
OpCapability Shader
OpCapability TensorsARM
OpCapability GraphARM
OpCapability Int8
OpExtension "SPV_ARM_tensors"
OpExtension "SPV_ARM_graph"
OpMemoryModel Logical GLSL450

OpDecorate %first_input_ptr DescriptorSet 0
OpDecorate %first_input_ptr Binding 0
OpDecorate %first_output_ptr DescriptorSet 0
OpDecorate %first_output_ptr Binding 1
OpDecorate %second_input_ptr DescriptorSet 0
OpDecorate %second_input_ptr Binding 2
OpDecorate %second_output_ptr DescriptorSet 0
OpDecorate %second_output_ptr Binding 3

%uint = OpTypeInt 32 0
%uint8 = OpTypeInt 8 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%shape_type = OpTypeArray %uint %uint_1
%shape = OpConstantComposite %shape_type %uint_1
%tensor = OpTypeTensorARM %uint8 %uint_1 %shape
%ptr_tensor = OpTypePointer UniformConstant %tensor
%graph_type = OpTypeGraphARM 1 %tensor %tensor

%first_input_ptr = OpVariable %ptr_tensor UniformConstant
%first_output_ptr = OpVariable %ptr_tensor UniformConstant
%second_input_ptr = OpVariable %ptr_tensor UniformConstant
%second_output_ptr = OpVariable %ptr_tensor UniformConstant

OpGraphEntryPointARM %first_graph "first" %first_input_ptr %first_output_ptr
OpGraphEntryPointARM %second_graph "second" %second_input_ptr %second_output_ptr
%first_graph = OpGraphARM %graph_type
%first_input = OpGraphInputARM %tensor %uint_0
OpGraphSetOutputARM %first_input %uint_0
OpGraphEndARM
%second_graph = OpGraphARM %graph_type
%second_input = OpGraphInputARM %tensor %uint_0
OpGraphSetOutputARM %second_input %uint_0
OpGraphEndARM
"""
    result = assemble_spirv_text(asm)
    assert result.ok, result.diagnostics
    validation = validate_spirv(result.data)
    assert validation.ok, validation.diagnostics
    return list(
        struct.unpack("<" + "I" * (len(result.data) // 4), result.data)
    )
