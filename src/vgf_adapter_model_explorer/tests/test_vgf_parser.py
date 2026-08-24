# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

import io
import struct

import pytest
from spirv_adapter_model_explorer.assembly import assemble_spirv_text
from spirv_adapter_model_explorer.validation import validate_spirv

vgfpy = pytest.importorskip("vgfpy")

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
    PushConstantRange,
    Resource,
    ResourceCategory,
    SamplerConfig,
    Segment,
    Vgf,
    VkBorderColor,
    VkDescriptorType,
    VkFilter,
    VkFormat,
    VkSamplerAddressMode,
)


def test_parser_decodes_synthetic_feature_coverage_vgf(tmp_path):
    vgf_path = tmp_path / "synthetic_feature_coverage.vgf"
    graph_spirv_words = [0x07230203, 0x00010000, 0, 5, 0, 0x00020011]
    compute_spirv_words = [
        0x07230203,
        0x00010000,
        0,
        5,
        0,
        0x00020011,
        0x0003000E,
    ]
    _write_synthetic_feature_coverage_vgf(
        vgf_path, graph_spirv_words, compute_spirv_words
    )

    assert Parser(str(vgf_path)).vgf == Vgf(
        file_path=str(vgf_path),
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
                shape=[1, 2, 3],
                stride=[6, 3, 1],
                vk_descriptor_type=VkDescriptorType.VK_DESCRIPTOR_TYPE_TENSOR_ARM,
                vk_format=VkFormat.VK_FORMAT_R8_SINT,
                alias_group_id=77,
                sampler_config=SamplerConfig(
                    min_filter=VkFilter.VK_FILTER_NEAREST,
                    mag_filter=VkFilter.VK_FILTER_LINEAR,
                    address_mode_u=(
                        VkSamplerAddressMode.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
                    ),
                    address_mode_v=(
                        VkSamplerAddressMode.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER
                    ),
                    border_color=VkBorderColor.VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
                ),
            ),
            Resource(
                category=ResourceCategory.INTERMEDIATE,
                index=1,
                shape=[4],
                stride=[1],
                vk_descriptor_type=VkDescriptorType.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                vk_format=VkFormat.VK_FORMAT_R8_UINT,
                alias_group_id=88,
            ),
            Resource(
                category=ResourceCategory.OUTPUT,
                index=2,
                shape=[5, 6],
                stride=[],
                vk_descriptor_type=VkDescriptorType.VK_DESCRIPTOR_TYPE_TENSOR_ARM,
                vk_format=VkFormat.VK_FORMAT_R32_SINT,
            ),
            Resource(
                category=ResourceCategory.CONSTANT,
                index=3,
                shape=[2],
                stride=[1],
                vk_descriptor_type=None,
                vk_format=VkFormat.VK_FORMAT_R8_SINT,
            ),
            Resource(
                category=ResourceCategory.CONSTANT,
                index=4,
                shape=[1],
                stride=[],
                vk_descriptor_type=None,
                vk_format=VkFormat.VK_FORMAT_R32_SINT,
            ),
        ],
        constants=[
            Constant(
                index=0,
                mrt_index=3,
                sparsity_dimension=-1,
                data=b"\x01\x02",
            ),
            Constant(
                index=1,
                mrt_index=4,
                sparsity_dimension=0,
                data=b"\x03\x00\x00\x00",
            ),
        ],
        model_sequence=ModelSequence(
            inputs=[
                Model_Sequence_IO(
                    binding=0,
                    index=0,
                    mrt_index=0,
                    name="input_name",
                )
            ],
            outputs=[
                Model_Sequence_IO(
                    binding=3,
                    index=0,
                    mrt_index=2,
                    name="output_name",
                )
            ],
            segments=[
                Segment(
                    constants=[0, 1],
                    descriptor_set_infos=[
                        DescriptorSetInfo(
                            index=0,
                            set_index=5,
                            bindings=[
                                IOBase(binding=0, index=0, mrt_index=0),
                                IOBase(binding=1, index=1, mrt_index=1),
                            ],
                        ),
                        DescriptorSetInfo(
                            index=1,
                            set_index=9,
                            bindings=[
                                IOBase(binding=4, index=0, mrt_index=3),
                            ],
                        ),
                    ],
                    index=0,
                    dispatch_shape=[1, 2, 3],
                    inputs=[IOBase(binding=0, index=0, mrt_index=0)],
                    outputs=[IOBase(binding=1, index=0, mrt_index=1)],
                    module_index=0,
                    name="graph_segment",
                    type=ModuleType.GRAPH,
                    push_constant_ranges=[
                        PushConstantRange(
                            index=0,
                            stage_flags=7,
                            offset=16,
                            size=32,
                        )
                    ],
                ),
                Segment(
                    constants=[],
                    descriptor_set_infos=[
                        DescriptorSetInfo(
                            index=0,
                            set_index=6,
                            bindings=[
                                IOBase(binding=2, index=0, mrt_index=1),
                                IOBase(binding=3, index=1, mrt_index=2),
                            ],
                        ),
                    ],
                    index=1,
                    dispatch_shape=[4, 5, 6],
                    inputs=[IOBase(binding=2, index=0, mrt_index=1)],
                    outputs=[IOBase(binding=3, index=0, mrt_index=2)],
                    module_index=1,
                    name="compute_segment",
                    type=ModuleType.COMPUTE,
                    push_constant_ranges=[],
                ),
            ],
        ),
        modules=[
            Module(
                code_size=len(graph_spirv_words) * 4,
                entry_point="graph_main",
                has_spirv=True,
                index=0,
                name="graph_mod",
                type=ModuleType.GRAPH,
                code_type=ModuleCodeType.SPIRV,
                code_available=True,
                spirv_code=_words_to_bytes(graph_spirv_words),
            ),
            Module(
                code_size=len(compute_spirv_words) * 4,
                entry_point="compute_main",
                has_spirv=True,
                index=1,
                name="compute_mod",
                type=ModuleType.COMPUTE,
                code_type=ModuleCodeType.SPIRV,
                code_available=True,
                spirv_code=_words_to_bytes(compute_spirv_words),
            ),
            Module(
                code_size=0,
                entry_point="placeholder_main",
                has_spirv=False,
                index=2,
                name="placeholder_mod",
                type=ModuleType.COMPUTE,
                code_type=ModuleCodeType.SPIRV,
                code_available=False,
            ),
            Module(
                code_size=len("void main(){}"),
                entry_point="glsl_main",
                has_spirv=False,
                index=3,
                name="glsl_mod",
                type=ModuleType.COMPUTE,
                code_type=ModuleCodeType.GLSL,
                code_available=True,
                shader_code="void main(){}",
            ),
        ],
    )


def _write_synthetic_feature_coverage_vgf(
    path,
    graph_spirv_words: list[int],
    compute_spirv_words: list[int],
) -> None:
    encoder = vgfpy.CreateEncoder(123)

    graph_module = encoder.AddModule(
        vgfpy.ModuleType.Graph,
        "graph_mod",
        "graph_main",
        graph_spirv_words,
    )
    compute_module = encoder.AddModule(
        vgfpy.ModuleType.Compute,
        "compute_mod",
        "compute_main",
        compute_spirv_words,
    )
    encoder.AddModule(
        vgfpy.ModuleType.Compute,
        "placeholder_mod",
        "placeholder_main",
    )
    encoder.AddModule(
        vgfpy.ModuleType.Compute,
        "glsl_mod",
        "glsl_main",
        vgfpy.ShaderType.Glsl,
        "void main(){}",
    )

    input_resource = encoder.AddInputResource(
        VkDescriptorType.VK_DESCRIPTOR_TYPE_TENSOR_ARM,
        VkFormat.VK_FORMAT_R8_SINT,
        [1, 2, 3],
        [6, 3, 1],
        77,
    )
    intermediate_resource = encoder.AddIntermediateResource(
        VkDescriptorType.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VkFormat.VK_FORMAT_R8_UINT,
        [4],
        [1],
    )
    output_resource = encoder.AddOutputResource(
        VkDescriptorType.VK_DESCRIPTOR_TYPE_TENSOR_ARM,
        VkFormat.VK_FORMAT_R32_SINT,
        [5, 6],
        [],
    )
    constant_resource = encoder.AddConstantResource(
        VkFormat.VK_FORMAT_R8_SINT,
        [2],
        [1],
    )
    sparse_constant_resource = encoder.AddConstantResource(
        VkFormat.VK_FORMAT_R32_SINT,
        [1],
        [],
    )

    encoder.AddSamplerConfig(
        input_resource,
        VkFilter.VK_FILTER_NEAREST,
        VkFilter.VK_FILTER_LINEAR,
        VkSamplerAddressMode.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        VkSamplerAddressMode.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        VkBorderColor.VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
    )
    encoder.SetAliasGroup(intermediate_resource, 88)

    constant = encoder.AddConstant(constant_resource, bytearray([1, 2]), -1)
    sparse_constant = encoder.AddConstant(
        sparse_constant_resource,
        bytearray([3, 0, 0, 0]),
        0,
    )

    input_slot = encoder.AddBindingSlot(0, input_resource)
    graph_output_slot = encoder.AddBindingSlot(1, intermediate_resource)
    compute_input_slot = encoder.AddBindingSlot(2, intermediate_resource)
    output_slot = encoder.AddBindingSlot(3, output_resource)
    constant_slot = encoder.AddBindingSlot(4, constant_resource)

    graph_descriptor = encoder.AddDescriptorSetInfo(
        [input_slot, graph_output_slot],
        5,
    )
    graph_constant_descriptor = encoder.AddDescriptorSetInfo(
        [constant_slot],
        9,
    )
    compute_descriptor = encoder.AddDescriptorSetInfo(
        [compute_input_slot, output_slot],
        6,
    )
    push_constant = encoder.AddPushConstRange(7, 16, 32)

    encoder.AddSegmentInfo(
        graph_module,
        "graph_segment",
        [graph_descriptor, graph_constant_descriptor],
        [input_slot],
        [graph_output_slot],
        [constant, sparse_constant],
        [1, 2, 3],
        [push_constant],
    )
    encoder.AddSegmentInfo(
        compute_module,
        "compute_segment",
        [compute_descriptor],
        [compute_input_slot],
        [output_slot],
        [],
        [4, 5, 6],
        [],
    )
    encoder.AddModelSequenceInputsOutputs(
        [input_slot],
        ["input_name"],
        [output_slot],
        ["output_name"],
    )

    encoder.Finish()
    stream = io.BytesIO()
    assert encoder.WriteTo(stream)
    path.write_bytes(stream.getvalue())


def _words_to_bytes(words: list[int]) -> bytes:
    return struct.pack("<" + "I" * len(words), *words)


def test_parser_decodes_multi_segment_vgf(tmp_path):
    vgf_path = tmp_path / "multi_segment.vgf"
    _write_multi_segment_vgf(vgf_path)

    vgf = Parser(str(vgf_path)).vgf

    assert [segment.name for segment in vgf.model_sequence.segments] == [
        "graph_preprocess",
        "compute_preprocess",
        "graph_fuse",
        "side_branch",
        "graph_final",
    ]
    assert [segment.constants for segment in vgf.model_sequence.segments] == [
        [0, 2],
        [],
        [0, 1, 3],
        [],
        [1, 4],
    ]


def _graph_spirv_words(
    *,
    graph_type: str,
    input_count: int,
    output_count: int,
    constants: list[int],
) -> list[int]:
    input_types = " ".join("%tensor" for _index in range(input_count))
    output_types = " ".join("%tensor" for _index in range(output_count))
    variables = "\n".join(
        f"%input{index}_ptr = OpVariable %ptr_tensor UniformConstant"
        for index in range(input_count)
    )
    variables += "\n" + "\n".join(
        f"%output{index}_ptr = OpVariable %ptr_tensor UniformConstant"
        for index in range(output_count)
    )
    interfaces = " ".join(
        [
            *(f"%input{index}_ptr" for index in range(input_count)),
            *(f"%output{index}_ptr" for index in range(output_count)),
        ]
    )
    graph_inputs = "\n".join(
        f"%input{index} = OpGraphInputARM %tensor %uint_{index}"
        for index in range(input_count)
    )
    graph_constants = "\n".join(
        f"%constant{constant} = OpGraphConstantARM %tensor {constant}"
        for constant in constants
    )
    operands = [
        *(f"%input{index}" for index in range(input_count)),
        *(f"%constant{constant}" for constant in constants),
    ]
    operations = []
    current = operands[0]
    for index, operand in enumerate(operands[1:]):
        result = f"%op{index}"
        operations.append(
            f"{result} = OpExtInst %tensor %tosa ADD {current} {operand}"
        )
        current = result
    graph_outputs = "\n".join(
        f"OpGraphSetOutputARM {current} %uint_{index}"
        for index in range(output_count)
    )
    uint_constants = "\n".join(
        f"%uint_{index} = OpConstant %uint {index}"
        for index in range(max(input_count, output_count, 1) + 1)
    )
    asm = f'''
; SPIR-V
; Version: 1.0
; Bound: 200
; Schema: 0
OpCapability Shader
OpCapability TensorsARM
OpCapability GraphARM
OpCapability Int8
OpExtension "SPV_ARM_tensors"
OpExtension "SPV_ARM_graph"
%tosa = OpExtInstImport "TOSA.001000.1"
OpMemoryModel Logical GLSL450
%uint = OpTypeInt 32 0
%uint8 = OpTypeInt 8 0
{uint_constants}
%uint_array_l1 = OpTypeArray %uint %uint_1
%shape = OpConstantComposite %uint_array_l1 %uint_1
%tensor = OpTypeTensorARM %uint8 %uint_1 %shape
%ptr_tensor = OpTypePointer UniformConstant %tensor
%graph_type = OpTypeGraphARM {input_count} {input_types} {output_types}
{variables}
{graph_constants}
OpGraphEntryPointARM %graph "{graph_type}" {interfaces}
%graph = OpGraphARM %graph_type
{graph_inputs}
{chr(10).join(operations)}
{graph_outputs}
OpGraphEndARM
'''
    result = assemble_spirv_text(asm)
    assert result.ok, result.diagnostics
    validation = validate_spirv(result.data)
    assert validation.ok, validation.diagnostics
    return list(
        struct.unpack("<" + "I" * (len(result.data) // 4), result.data)
    )


def _compute_spirv_words(
    entry_point: str,
    *,
    input_bindings: list[int],
    output_bindings: list[int],
) -> list[int]:
    names = "\n".join(
        [
            *(
                f'OpName %input{index} "input{index}"'
                for index in range(len(input_bindings))
            ),
            *(
                f'OpName %output{index} "output{index}"'
                for index in range(len(output_bindings))
            ),
        ]
    )
    input_decorations = "\n".join(
        f"""OpDecorate %input{index} DescriptorSet 0
OpDecorate %input{index} Binding {binding}
OpDecorate %input{index} NonWritable"""
        for index, binding in enumerate(input_bindings)
    )
    output_decorations = "\n".join(
        f"""OpDecorate %output{index} DescriptorSet 0
OpDecorate %output{index} Binding {binding}
OpDecorate %output{index} NonReadable"""
        for index, binding in enumerate(output_bindings)
    )
    variables = "\n".join(
        [
            *(
                f"%input{index} = OpVariable %ptr_buffer StorageBuffer"
                for index in range(len(input_bindings))
            ),
            *(
                f"%output{index} = OpVariable %ptr_buffer StorageBuffer"
                for index in range(len(output_bindings))
            ),
        ]
    )
    asm = f'''
; SPIR-V
; Version: 1.0
; Bound: 100
; Schema: 0
OpCapability Shader
OpExtension "SPV_KHR_storage_buffer_storage_class"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "{entry_point}"
OpExecutionMode %main LocalSize 1 1 1
{names}
{input_decorations}
{output_decorations}
OpDecorate %runtime ArrayStride 4
OpDecorate %buffer Block
OpMemberDecorate %buffer 0 Offset 0
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%runtime = OpTypeRuntimeArray %uint
%buffer = OpTypeStruct %runtime
%ptr_buffer = OpTypePointer StorageBuffer %buffer
%func_ty = OpTypeFunction %void
{variables}
%main = OpFunction %void None %func_ty
%label = OpLabel
OpReturn
OpFunctionEnd
'''
    result = assemble_spirv_text(asm)
    assert result.ok, result.diagnostics
    validation = validate_spirv(result.data)
    assert validation.ok, validation.diagnostics
    return list(
        struct.unpack("<" + "I" * (len(result.data) // 4), result.data)
    )


def _write_multi_segment_vgf(path) -> None:
    encoder = vgfpy.CreateEncoder(123)

    graph_preprocess_module = encoder.AddModule(
        vgfpy.ModuleType.Graph,
        "graph_preprocess_mod",
        "graph_preprocess_main",
        _graph_spirv_words(
            graph_type="graph_preprocess_main",
            input_count=1,
            output_count=1,
            constants=[0, 2],
        ),
    )
    compute_preprocess_module = encoder.AddModule(
        vgfpy.ModuleType.Compute,
        "compute_preprocess_mod",
        "compute_preprocess_main",
        _compute_spirv_words(
            "compute_preprocess_main",
            input_bindings=[1],
            output_bindings=[4],
        ),
    )
    graph_fuse_module = encoder.AddModule(
        vgfpy.ModuleType.Graph,
        "graph_fuse_mod",
        "graph_fuse_main",
        _graph_spirv_words(
            graph_type="graph_fuse_main",
            input_count=2,
            output_count=2,
            constants=[0, 1, 3],
        ),
    )
    side_branch_module = encoder.AddModule(
        vgfpy.ModuleType.Compute,
        "side_branch_mod",
        "side_branch_main",
        _compute_spirv_words(
            "side_branch_main",
            input_bindings=[2, 3],
            output_bindings=[6],
        ),
    )
    graph_final_module = encoder.AddModule(
        vgfpy.ModuleType.Graph,
        "graph_final_mod",
        "graph_final_main",
        _graph_spirv_words(
            graph_type="graph_final_main",
            input_count=2,
            output_count=1,
            constants=[1, 4],
        ),
    )

    input_0 = encoder.AddInputResource(
        VkDescriptorType.VK_DESCRIPTOR_TYPE_TENSOR_ARM,
        VkFormat.VK_FORMAT_R8_SINT,
        [1, 16, 16, 8],
        [],
    )
    input_1 = encoder.AddInputResource(
        VkDescriptorType.VK_DESCRIPTOR_TYPE_TENSOR_ARM,
        VkFormat.VK_FORMAT_R8_SINT,
        [1, 16, 16, 8],
        [],
    )
    input_2 = encoder.AddInputResource(
        VkDescriptorType.VK_DESCRIPTOR_TYPE_TENSOR_ARM,
        VkFormat.VK_FORMAT_R8_SINT,
        [1, 8, 8, 4],
        [],
    )
    graph_preprocess_out = encoder.AddIntermediateResource(
        VkDescriptorType.VK_DESCRIPTOR_TYPE_TENSOR_ARM,
        VkFormat.VK_FORMAT_R8_SINT,
        [1, 16, 16, 8],
        [],
    )
    compute_preprocess_out = encoder.AddIntermediateResource(
        VkDescriptorType.VK_DESCRIPTOR_TYPE_TENSOR_ARM,
        VkFormat.VK_FORMAT_R8_SINT,
        [1, 16, 16, 8],
        [],
    )
    fuse_out = encoder.AddIntermediateResource(
        VkDescriptorType.VK_DESCRIPTOR_TYPE_TENSOR_ARM,
        VkFormat.VK_FORMAT_R8_SINT,
        [1, 16, 16, 16],
        [],
    )
    side_branch_out = encoder.AddIntermediateResource(
        VkDescriptorType.VK_DESCRIPTOR_TYPE_TENSOR_ARM,
        VkFormat.VK_FORMAT_R8_SINT,
        [1, 8, 8, 8],
        [],
    )
    output_main = encoder.AddOutputResource(
        VkDescriptorType.VK_DESCRIPTOR_TYPE_TENSOR_ARM,
        VkFormat.VK_FORMAT_R8_SINT,
        [1, 16, 16, 16],
        [],
    )
    output_aux = encoder.AddOutputResource(
        VkDescriptorType.VK_DESCRIPTOR_TYPE_TENSOR_ARM,
        VkFormat.VK_FORMAT_R8_SINT,
        [1, 16, 16, 16],
        [],
    )
    graph_preprocess_out_alias = encoder.AddIntermediateResource(
        VkDescriptorType.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VkFormat.VK_FORMAT_R8_SINT,
        [1, 16, 16, 8],
        [],
    )
    encoder.SetAliasGroup(graph_preprocess_out, 42)
    encoder.SetAliasGroup(graph_preprocess_out_alias, 42)

    const_shared_preprocess_fuse_resource = encoder.AddConstantResource(
        VkFormat.VK_FORMAT_R8_SINT,
        [4],
        [],
    )
    const_shared_fuse_final_resource = encoder.AddConstantResource(
        VkFormat.VK_FORMAT_R8_SINT,
        [4],
        [],
    )
    const_preprocess_only_resource = encoder.AddConstantResource(
        VkFormat.VK_FORMAT_R8_SINT,
        [2],
        [],
    )
    const_fuse_only_resource = encoder.AddConstantResource(
        VkFormat.VK_FORMAT_R8_SINT,
        [2],
        [],
    )
    const_final_only_resource = encoder.AddConstantResource(
        VkFormat.VK_FORMAT_R8_SINT,
        [2],
        [],
    )

    const_shared_preprocess_fuse = encoder.AddConstant(
        const_shared_preprocess_fuse_resource,
        bytearray([10, 11, 12, 13]),
        -1,
    )
    const_shared_fuse_final = encoder.AddConstant(
        const_shared_fuse_final_resource,
        bytearray([20, 21, 22, 23]),
        -1,
    )
    const_preprocess_only = encoder.AddConstant(
        const_preprocess_only_resource,
        bytearray([30, 31]),
        -1,
    )
    const_fuse_only = encoder.AddConstant(
        const_fuse_only_resource,
        bytearray([40, 41]),
        -1,
    )
    const_final_only = encoder.AddConstant(
        const_final_only_resource,
        bytearray([50, 51]),
        -1,
    )

    input_0_slot = encoder.AddBindingSlot(0, input_0)
    input_1_slot = encoder.AddBindingSlot(1, input_1)
    input_2_slot = encoder.AddBindingSlot(2, input_2)
    graph_preprocess_out_slot = encoder.AddBindingSlot(3, graph_preprocess_out)
    graph_preprocess_out_alias_slot = encoder.AddBindingSlot(
        3, graph_preprocess_out_alias
    )
    compute_preprocess_out_slot = encoder.AddBindingSlot(
        4, compute_preprocess_out
    )
    fuse_out_slot = encoder.AddBindingSlot(5, fuse_out)
    side_branch_out_slot = encoder.AddBindingSlot(6, side_branch_out)
    output_main_slot = encoder.AddBindingSlot(7, output_main)
    output_aux_slot = encoder.AddBindingSlot(8, output_aux)

    graph_preprocess_descriptor = encoder.AddDescriptorSetInfo(
        [input_0_slot, graph_preprocess_out_slot],
        0,
    )
    compute_preprocess_descriptor = encoder.AddDescriptorSetInfo(
        [input_1_slot, compute_preprocess_out_slot],
        0,
    )
    graph_fuse_descriptor = encoder.AddDescriptorSetInfo(
        [
            graph_preprocess_out_slot,
            compute_preprocess_out_slot,
            fuse_out_slot,
            output_aux_slot,
        ],
        0,
    )
    side_branch_descriptor = encoder.AddDescriptorSetInfo(
        [input_2_slot, graph_preprocess_out_alias_slot, side_branch_out_slot],
        0,
    )
    graph_final_descriptor = encoder.AddDescriptorSetInfo(
        [fuse_out_slot, side_branch_out_slot, output_main_slot],
        0,
    )

    encoder.AddSegmentInfo(
        graph_preprocess_module,
        "graph_preprocess",
        [graph_preprocess_descriptor],
        [input_0_slot],
        [graph_preprocess_out_slot],
        [const_shared_preprocess_fuse, const_preprocess_only],
        [1, 1, 1],
        [],
    )
    encoder.AddSegmentInfo(
        compute_preprocess_module,
        "compute_preprocess",
        [compute_preprocess_descriptor],
        [input_1_slot],
        [compute_preprocess_out_slot],
        [],
        [2, 2, 1],
        [],
    )
    encoder.AddSegmentInfo(
        graph_fuse_module,
        "graph_fuse",
        [graph_fuse_descriptor],
        [graph_preprocess_out_slot, compute_preprocess_out_slot],
        [fuse_out_slot, output_aux_slot],
        [
            const_shared_preprocess_fuse,
            const_shared_fuse_final,
            const_fuse_only,
        ],
        [1, 1, 1],
        [],
    )
    encoder.AddSegmentInfo(
        side_branch_module,
        "side_branch",
        [side_branch_descriptor],
        [input_2_slot, graph_preprocess_out_alias_slot],
        [side_branch_out_slot],
        [],
        [4, 2, 1],
        [],
    )
    encoder.AddSegmentInfo(
        graph_final_module,
        "graph_final",
        [graph_final_descriptor],
        [fuse_out_slot, side_branch_out_slot],
        [output_main_slot],
        [const_shared_fuse_final, const_final_only],
        [1, 1, 1],
        [],
    )

    encoder.AddModelSequenceInputsOutputs(
        [input_0_slot, input_1_slot, input_2_slot],
        ["input_0", "input_1", "input_2"],
        [output_main_slot, output_aux_slot],
        ["output_main", "output_aux"],
    )

    encoder.Finish()
    stream = io.BytesIO()
    assert encoder.WriteTo(stream)
    path.write_bytes(stream.getvalue())
