# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

from dataclasses import dataclass


@dataclass
class Constant:
    index: int
    mrt_index: int
    sparsity_dimension: int


@dataclass
class IOBase:
    binding: int
    index: int
    mrt_index: int


@dataclass
class Model_Sequence_IO(IOBase):
    name: str


@dataclass
class Segment:
    constants: list[int]
    descriptor_set_infos: list[IOBase]
    index: int
    dispatch_shape: list[int]
    inputs: list[IOBase]
    outputs: list[IOBase]
    module_index: int
    name: str
    type: str
    push_constant_ranges: list


@dataclass
class ModelSequence:
    inputs: list[Model_Sequence_IO]
    outputs: list[Model_Sequence_IO]
    segments: list[Segment]


@dataclass
class Module:
    code_size: int
    entry_point: str
    has_spirv: bool
    index: int
    name: str
    type: str


@dataclass
class Resource:
    category: str
    index: int
    shape: list[int]
    stride: list[int]
    vk_descriptor_type: str
    vk_format: str


@dataclass
class Vgf:
    file_path: str
    resources: list[Resource]
    constants: list[Constant]
    model_sequence: ModelSequence
    modules: list[Module]
