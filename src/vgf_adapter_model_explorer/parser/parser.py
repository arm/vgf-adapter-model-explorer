# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

import os
from typing import Any, Callable, Dict

from .types import (
    Constant,
    IOBase,
    Model_Sequence_IO,
    ModelSequence,
    Module,
    Resource,
    Segment,
    Vgf,
)


class Parser:
    def __init__(
        self,
        model_path: str,
        spirv_binary_extractor: Callable[[str], Dict[str, Any]],
    ):
        vgf_json = spirv_binary_extractor(model_path)
        self.model_name = os.path.basename(model_path)
        self.vgf = self._parse_vgf(vgf_json, model_path)

    def _parse_vgf(self, data: Dict[str, Any], file_path: str) -> Vgf:
        return Vgf(
            resources=self._parse_resources(data),
            constants=self._parse_constants(data),
            modules=self._parse_modules(data),
            model_sequence=self._parse_model_sequence(data),
            file_path=file_path,
        )

    def _parse_model_sequence(self, data: Dict[str, Any]) -> ModelSequence:
        data = data["model_sequence"]
        return ModelSequence(
            inputs=[Model_Sequence_IO(**i) for i in data["inputs"]],
            outputs=[Model_Sequence_IO(**o) for o in data["outputs"]],
            segments=[self._parse_segment(s) for s in data["segments"]],
        )

    def _parse_modules(self, data: Dict[str, Any]) -> list[Module]:
        return [Module(**m) for m in data["modules"]]

    def _parse_resources(self, data: Dict[str, Any]) -> list[Resource]:
        return [Resource(**r) for r in data["resources"]]

    def _parse_constants(self, data: Dict[str, Any]) -> list[Constant]:
        return [Constant(**c) for c in data["constants"]]

    def _parse_segment(self, data: Dict[str, Any]) -> Segment:
        return Segment(
            constants=data["constants"],
            descriptor_set_infos=data["descriptor_set_infos"],
            index=data["index"],
            dispatch_shape=data["dispatch_shape"],
            inputs=[IOBase(**i) for i in data["inputs"]],
            outputs=[IOBase(**o) for o in data["outputs"]],
            module_index=data["module_index"],
            name=data["name"],
            type=data["type"],
            push_constant_ranges=data["push_constant_ranges"],
        )
