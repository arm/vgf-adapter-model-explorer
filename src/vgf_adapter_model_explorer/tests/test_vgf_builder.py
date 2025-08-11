# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

from dataclasses import asdict
from unittest.mock import Mock

from model_explorer import graph_builder as gb

from ..builder.builder import VgfGraphBuilder
from ..parser.types import (
    Constant,
    IOBase,
    Model_Sequence_IO,
    ModelSequence,
    Module,
    Resource,
    Segment,
    Vgf,
)

test_vgf_data = Vgf(
    file_path="hello_vgf.vgf",
    resources=[
        Resource(
            category="CONSTANT",
            index=0,
            shape=[1, 16],
            stride=[],
            vk_descriptor_type="none",
            vk_format="VK_FORMAT_R8_SINT",
        ),
        Resource(
            category="INPUT",
            index=3,
            shape=[1, 1],
            stride=[],
            vk_descriptor_type="VK_DESCRIPTOR_TYPE_STORAGE_TENSOR_ARM",
            vk_format="VK_FORMAT_R8_SINT",
        ),
        Resource(
            category="OUTPUT",
            index=4,
            shape=[1, 1],
            stride=[],
            vk_descriptor_type="VK_DESCRIPTOR_TYPE_STORAGE_TENSOR_ARM",
            vk_format="VK_FORMAT_R8_UINT",
        ),
    ],
    constants=[
        Constant(index=0, mrt_index=0, sparsity_dimension=-1),
    ],
    modules=[
        Module(
            code_size=541,
            entry_point="graph_partition_0",
            has_spirv=True,
            index=0,
            name="graph_partition_0",
            type="GRAPH",
        )
    ],
    model_sequence=ModelSequence(
        inputs=[
            Model_Sequence_IO(binding=0, index=0, mrt_index=3, name="input_0")
        ],
        outputs=[
            Model_Sequence_IO(binding=1, index=0, mrt_index=4, name="output_0")
        ],
        segments=[
            Segment(
                constants=[0, 1, 2],
                descriptor_set_infos=[
                    IOBase(binding=0, index=0, mrt_index=3),
                    IOBase(binding=1, index=1, mrt_index=4),
                ],
                index=0,
                dispatch_shape=[0, 0, 0],
                inputs=[IOBase(binding=0, index=0, mrt_index=3)],
                outputs=[IOBase(binding=1, index=0, mrt_index=4)],
                module_index=0,
                name="graph_segment_0",
                type="GRAPH",
                push_constant_ranges=[],
            )
        ],
    ),
)


def test_builder():
    mock_get_spirv_nodes = Mock(
        return_value=[
            gb.GraphNode(
                id="0",
                label="spirv",
                namespace="",
                subgraphIds=[],
                attrs=[],
                incomingEdges=[],
                outputsMetadata=[
                    gb.MetadataItem(
                        id="0",
                        attrs=[gb.KeyValue(key="shape", value="[1, 1]")],
                    )
                ],
                inputsMetadata=[],
                style=None,
                config=None,
            )
        ]
    )

    graph_collection = VgfGraphBuilder(
        test_vgf_data, mock_get_spirv_nodes
    ).graph_collection

    got = asdict(graph_collection)

    expected = gb.GraphCollection(
        label="hello_vgf",
        graphs=[
            gb.Graph(
                id="Main",
                nodes=[
                    gb.GraphNode(
                        id="GraphInputs",
                        label="Graph Inputs",
                        namespace="",
                        subgraphIds=[],
                        attrs=[],
                        incomingEdges=[],
                        outputsMetadata=[],
                        inputsMetadata=[],
                        style=None,
                        config=None,
                    ),
                    gb.GraphNode(
                        id="0",
                        label="spirv",
                        namespace="",
                        subgraphIds=[],
                        attrs=[],
                        incomingEdges=[
                            gb.IncomingEdge(
                                sourceNodeId="mrt_3",
                                sourceNodeOutputId="0",
                                targetNodeInputId="0",
                            )
                        ],
                        outputsMetadata=[
                            gb.MetadataItem(
                                id="0",
                                attrs=[
                                    gb.KeyValue(key="shape", value="[1, 1]")
                                ],
                            )
                        ],
                        inputsMetadata=[],
                        style=None,
                        config=None,
                    ),
                    gb.GraphNode(
                        id="mrt_3",
                        label="VK_DESCRIPTOR_TYPE_STORAGE_TENSOR_ARM",
                        namespace="",
                        subgraphIds=[],
                        attrs=[
                            gb.KeyValue(key="Shape", value="[1, 1]"),
                            gb.KeyValue(key="Category", value="INPUT"),
                            gb.KeyValue(
                                key="Format", value="VK_FORMAT_R8_SINT"
                            ),
                        ],
                        incomingEdges=[
                            gb.IncomingEdge(
                                sourceNodeId="GraphInputs",
                                sourceNodeOutputId="0",
                                targetNodeInputId="0",
                            )
                        ],
                        outputsMetadata=[
                            gb.MetadataItem(
                                id="0",
                                attrs=[gb.KeyValue(key="type", value="GRAPH")],
                            )
                        ],
                        inputsMetadata=[],
                        style=None,
                        config=None,
                    ),
                    gb.GraphNode(
                        id="mrt_4",
                        label="VK_DESCRIPTOR_TYPE_STORAGE_TENSOR_ARM",
                        namespace="",
                        subgraphIds=[],
                        attrs=[],
                        incomingEdges=[
                            gb.IncomingEdge(
                                sourceNodeId="0",
                                sourceNodeOutputId="0",
                                targetNodeInputId="0",
                            )
                        ],
                        outputsMetadata=[],
                        inputsMetadata=[],
                        style=None,
                        config=None,
                    ),
                    gb.GraphNode(
                        id="GraphOutputs",
                        label="Graph Outputs",
                        namespace="",
                        subgraphIds=[],
                        attrs=[],
                        incomingEdges=[
                            gb.IncomingEdge(
                                sourceNodeId="mrt_4",
                                sourceNodeOutputId="0",
                                targetNodeInputId="0",
                            )
                        ],
                        outputsMetadata=[],
                        inputsMetadata=[],
                        style=None,
                        config=None,
                    ),
                ],
                groupNodeAttributes=None,
            )
        ],
    )

    assert got == asdict(expected)
