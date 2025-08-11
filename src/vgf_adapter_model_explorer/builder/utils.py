# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

from typing import Iterable, TypeVar

from model_explorer import graph_builder as gb

from ..parser.types import Module, Resource

T = TypeVar("T")


def find_item(index: int, items: Iterable[T]) -> T | None:
    """Finds the item at the given index."""
    return next(
        filter(lambda x: getattr(x, "index", None) == index, items), None
    )


def format_index(prefix: str, index: int) -> str:
    """Creates an id from the index and prefix."""
    return f"{prefix}_{index}"


def extend_resource(node: gb.GraphNode, resource: Resource) -> None:
    """Extends model resource nodes with attributes."""
    node.attrs.extend(
        [
            gb.KeyValue(key="Shape", value=str(resource.shape)),
            gb.KeyValue(key="Category", value=resource.category),
            gb.KeyValue(key="Format", value=resource.vk_format),
        ]
    )


def extend_module(node: gb.GraphNode, module: Module) -> None:
    """Extends module nodes with attributes."""
    node.attrs.extend(
        [
            gb.KeyValue(key="Has Spirv", value=str(module.has_spirv)),
            gb.KeyValue(key="Type", value=str(module.type)),
            gb.KeyValue(key="Entry Point", value=str(module.entry_point)),
        ]
    )
