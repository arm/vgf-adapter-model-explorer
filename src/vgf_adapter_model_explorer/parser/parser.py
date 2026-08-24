# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Iterable
from pathlib import Path

import vgfpy  # type: ignore[reportMissingImports]

from .types import (
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


class Parser:
    def __init__(self, model_path: str):
        self.model_name = os.path.basename(model_path)
        self._data = memoryview(Path(model_path).read_bytes())
        self._header_decoder = self._create_header_decoder()
        self.vgf = self._parse_vgf(model_path)

    def _create_header_decoder(self):
        header = vgfpy.CreateHeaderDecoder(
            self._data, vgfpy.HeaderSize(), len(self._data)
        )
        if header is None:
            raise ValueError(
                "Invalid VGF file: header or section verification failed"
            )
        return header

    def _parse_vgf(self, file_path: str) -> Vgf:
        return Vgf(
            header=self._parse_header(),
            resources=self._parse_resources(),
            constants=self._parse_constants(),
            modules=self._parse_modules(),
            model_sequence=self._parse_model_sequence(),
            file_path=file_path,
        )

    def _parse_header(self) -> Header:
        return Header(
            major=self._header_decoder.GetMajor(),
            minor=self._header_decoder.GetMinor(),
            patch=self._header_decoder.GetPatch(),
            encoder_vulkan_headers_version=(
                self._header_decoder.GetEncoderVulkanHeadersVersion()
            ),
            is_latest_version=self._header_decoder.IsLatestVersion(),
            is_valid=self._header_decoder.IsValid(),
            check_version=self._header_decoder.CheckVersion(),
        )

    def _parse_modules(self) -> list[Module]:
        decoder = vgfpy.CreateModuleTableDecoder(
            self._data[self._header_decoder.GetModuleTableOffset() :],
            self._header_decoder.GetModuleTableSize(),
        )
        if decoder is None:
            raise ValueError(
                "Invalid VGF file: module table could not be decoded"
            )

        modules = []
        for index in range(decoder.size()):
            spirv_code = _spirv_code_bytes(decoder, index)
            code_type = _module_code_type(decoder, index)
            shader_code = _shader_code(decoder, index, code_type)
            modules.append(
                Module(
                    code_size=len(spirv_code or b"")
                    if code_type == ModuleCodeType.SPIRV
                    else len(shader_code or ""),
                    entry_point=decoder.getModuleEntryPoint(index),
                    has_spirv=decoder.isSPIRV(index)
                    and decoder.hasSPIRVCode(index),
                    index=index,
                    name=decoder.getModuleName(index),
                    type=_module_type(decoder.getModuleType(index)),
                    code_type=code_type,
                    code_available=_module_code_available(
                        decoder, index, code_type
                    ),
                    spirv_code=spirv_code,
                    shader_code=shader_code,
                )
            )
        return modules

    def _parse_model_sequence(self) -> ModelSequence:
        decoder = vgfpy.CreateModelSequenceTableDecoder(
            self._data[self._header_decoder.GetModelSequenceTableOffset() :],
            self._header_decoder.GetModelSequenceTableSize(),
        )
        if decoder is None:
            raise ValueError(
                "Invalid VGF file: model sequence table could not be decoded"
            )

        return ModelSequence(
            inputs=self._parse_model_sequence_io(
                decoder,
                decoder.getModelSequenceInputBindingSlotsHandle(),
                decoder.getModelSequenceInputNamesHandle(),
            ),
            outputs=self._parse_model_sequence_io(
                decoder,
                decoder.getModelSequenceOutputBindingSlotsHandle(),
                decoder.getModelSequenceOutputNamesHandle(),
            ),
            segments=[
                self._parse_segment(decoder, index)
                for index in range(decoder.modelSequenceTableSize())
            ],
        )

    def _parse_model_sequence_io(
        self,
        decoder,
        bindings_handle,
        names_handle,
    ) -> list[Model_Sequence_IO]:
        names_size = decoder.getNamesSize(names_handle)
        return [
            Model_Sequence_IO(
                binding=decoder.getBindingSlotBinding(bindings_handle, index),
                index=index,
                mrt_index=decoder.getBindingSlotMrtIndex(
                    bindings_handle, index
                ),
                name=decoder.getName(names_handle, index)
                if index < names_size
                else "",
            )
            for index in range(decoder.getBindingsSize(bindings_handle))
        ]

    def _parse_resources(self) -> list[Resource]:
        decoder = vgfpy.CreateModelResourceTableDecoder(
            self._data[self._header_decoder.GetModelResourceTableOffset() :],
            self._header_decoder.GetModelResourceTableSize(),
        )
        if decoder is None:
            raise ValueError(
                "Invalid VGF file: model resource table could not be decoded"
            )

        return [
            Resource(
                category=_resource_category(decoder.getCategory(index)),
                index=index,
                shape=_list_or_empty(decoder.getTensorShape(index)),
                stride=_list_or_empty(decoder.getTensorStride(index)),
                vk_descriptor_type=_descriptor_type(
                    decoder.getDescriptorType(index)
                ),
                vk_format=VkFormat(decoder.getVkFormat(index)),
                alias_group_id=decoder.getAliasGroupId(index),
                sampler_config=_sampler_config(decoder, index),
            )
            for index in range(decoder.size())
        ]

    def _parse_constants(self) -> list[Constant]:
        decoder = vgfpy.CreateConstantDecoder(
            self._data[self._header_decoder.GetConstantsOffset() :],
            self._header_decoder.GetConstantsSize(),
        )
        if decoder is None:
            raise ValueError(
                "Invalid VGF file: constants could not be decoded"
            )

        return [
            Constant(
                index=index,
                mrt_index=decoder.getConstantMrtIndex(index),
                sparsity_dimension=decoder.getConstantSparsityDimension(index)
                if decoder.isSparseConstant(index)
                else -1,
                data=bytes(decoder.getConstant(index)),
            )
            for index in range(decoder.size())
        ]

    def _parse_segment(self, decoder, index: int) -> Segment:
        return Segment(
            constants=_list_or_empty(decoder.getSegmentConstantIndexes(index)),
            descriptor_set_infos=self._parse_descriptor_set_infos(
                decoder, index
            ),
            index=index,
            dispatch_shape=_list_or_empty(
                decoder.getSegmentDispatchShape(index)
            ),
            inputs=self._parse_binding_slots(
                decoder, decoder.getSegmentInputBindingSlotsHandle(index)
            ),
            outputs=self._parse_binding_slots(
                decoder, decoder.getSegmentOutputBindingSlotsHandle(index)
            ),
            module_index=decoder.getSegmentModuleIndex(index),
            name=decoder.getSegmentName(index),
            type=_module_type(decoder.getSegmentType(index)),
            push_constant_ranges=self._parse_push_constant_ranges(
                decoder, index
            ),
        )

    def _parse_descriptor_set_infos(
        self, decoder, segment_index: int
    ) -> list[DescriptorSetInfo]:
        descriptor_set_infos: list[DescriptorSetInfo] = []
        for descriptor_index in range(
            decoder.getSegmentDescriptorSetInfosSize(segment_index)
        ):
            handle = decoder.getDescriptorBindingSlotsHandle(
                segment_index, descriptor_index
            )
            descriptor_set_infos.append(
                DescriptorSetInfo(
                    index=descriptor_index,
                    set_index=decoder.getSegmentDescriptorSetIndex(
                        segment_index, descriptor_index
                    ),
                    bindings=self._parse_binding_slots(decoder, handle),
                )
            )
        return descriptor_set_infos

    def _parse_push_constant_ranges(
        self, decoder, segment_index: int
    ) -> list[PushConstantRange]:
        handle = decoder.getSegmentPushConstRange(segment_index)
        return [
            PushConstantRange(
                index=index,
                stage_flags=decoder.getPushConstRangeStageFlags(handle, index),
                offset=decoder.getPushConstRangeOffset(handle, index),
                size=decoder.getPushConstRangeSize(handle, index),
            )
            for index in range(decoder.getPushConstRangesSize(handle))
        ]

    def _parse_binding_slots(self, decoder, handle) -> list[IOBase]:
        return [
            IOBase(
                binding=decoder.getBindingSlotBinding(handle, index),
                index=index,
                mrt_index=decoder.getBindingSlotMrtIndex(handle, index),
            )
            for index in range(decoder.getBindingsSize(handle))
        ]


def _spirv_code_bytes(decoder, index: int) -> bytes | None:
    if not decoder.isSPIRV(index) or not decoder.hasSPIRVCode(index):
        return None
    code = decoder.getSPIRVModuleCode(index)
    return None if code is None else bytes(code)


def _shader_code(decoder, index: int, code_type: ModuleCodeType) -> str | None:
    if code_type == ModuleCodeType.GLSL and decoder.hasGLSLCode(index):
        return decoder.getGLSLModuleCode(index)
    if code_type == ModuleCodeType.HLSL and decoder.hasHLSLCode(index):
        return decoder.getHLSLModuleCode(index)
    return None


def _module_code_type(decoder, index: int) -> ModuleCodeType:
    if decoder.isSPIRV(index):
        return ModuleCodeType.SPIRV
    if decoder.isGLSL(index):
        return ModuleCodeType.GLSL
    if decoder.isHLSL(index):
        return ModuleCodeType.HLSL
    return ModuleCodeType.NONE


def _module_code_available(
    decoder, index: int, code_type: ModuleCodeType
) -> bool:
    if code_type == ModuleCodeType.SPIRV:
        return decoder.hasSPIRVCode(index)
    if code_type == ModuleCodeType.GLSL:
        return decoder.hasGLSLCode(index)
    if code_type == ModuleCodeType.HLSL:
        return decoder.hasHLSLCode(index)
    return False


def _module_type(value) -> ModuleType:
    return ModuleType(value.value)


def _resource_category(value) -> ResourceCategory:
    return ResourceCategory(value.value)


def _descriptor_type(value: int | None) -> VkDescriptorType | None:
    if value is None:
        return None
    return VkDescriptorType(value)


def _list_or_empty(value: Iterable | None) -> list:
    return [] if value is None else list(value)


def _sampler_config(decoder, index: int) -> SamplerConfig | None:
    handle = decoder.getSamplerConfigHandle(index)
    if handle is None:
        return None
    return SamplerConfig(
        min_filter=VkFilter(decoder.getSamplerConfigMinFilter(handle)),
        mag_filter=VkFilter(decoder.getSamplerConfigMagFilter(handle)),
        address_mode_u=VkSamplerAddressMode(
            decoder.getSamplerConfigAddressModeU(handle)
        ),
        address_mode_v=VkSamplerAddressMode(
            decoder.getSamplerConfigAddressModeV(handle)
        ),
        border_color=VkBorderColor(
            decoder.getSamplerConfigBorderColor(handle)
        ),
    )
