from typing import (
    Set,
)

GRAPH_INPUT_ANNOTATION: str = "GraphInputs"
GRAPH_OUTPUT_ANNOTATION: str = "GraphOutputs"
GRAPH_TENSOR_IDX: str = "tensor_index"
GRAPH_TENSOR_TYPE: str = "tensor_shape"
GRAPH_TENSOR_TAG: str = "__tensor_tag"

TERMINATOR_OPS: Set[str] = {"func.return"}
