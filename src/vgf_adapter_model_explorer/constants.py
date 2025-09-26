# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

import importlib.resources
import json
from pathlib import Path
from typing import Any, Dict, List, Set

GRAPH_INPUT_ANNOTATION: str = "GraphInputs"
GRAPH_OUTPUT_ANNOTATION: str = "GraphOutputs"
GRAPH_TENSOR_IDX: str = "tensor_index"
GRAPH_TENSOR_TYPE: str = "tensor_shape"
GRAPH_TENSOR_TAG: str = "__tensor_tag"

TERMINATOR_OPS: Set[str] = {"func.return"}

with open(
    Path(
        str(
            importlib.resources.files(
                "vgf_adapter_model_explorer.resources"
            ).joinpath("tosa_1_0_operand_info.json")
        )
    ),
    mode="r",
    encoding="utf-8",
) as operand_file:
    TOSA_OPERAND_INFO: Dict[str, Dict[str, List[Any]]] = json.load(
        operand_file
    )
