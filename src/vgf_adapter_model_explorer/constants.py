# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

from typing import (
    Set,
)

GRAPH_INPUT_ANNOTATION: str = "GraphInputs"
GRAPH_OUTPUT_ANNOTATION: str = "GraphOutputs"
GRAPH_TENSOR_IDX: str = "tensor_index"
GRAPH_TENSOR_TYPE: str = "tensor_shape"
GRAPH_TENSOR_TAG: str = "__tensor_tag"

TERMINATOR_OPS: Set[str] = {"func.return"}
