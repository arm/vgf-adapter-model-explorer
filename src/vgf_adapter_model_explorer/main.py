# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from model_explorer import (
    Adapter,
    AdapterMetadata,
    ModelExplorerGraphs,
)

try:
    from model_explorer import (
        AdapterGetConfigEditorsResult,  # type: ignore[reportAttributeAccessIssue]
    )
    from model_explorer.config_editor import SlideToggleConfigEditor
except ImportError:
    # Older Model Explorer versions do not support adapter-provided settings.
    # Keep this adapter importable there; conversion will simply use the
    # default value from settings.get(..., False) below.
    @dataclass
    class AdapterGetConfigEditorsResult:  # type: ignore[no-redef]
        """Fallback shape for older Model Explorer versions."""

        configEditors: list | None = None
        error: str = ""

    SlideToggleConfigEditor = None  # type: ignore[assignment]

from .builder.builder import VgfGraphBuilder
from .parser.parser import Parser

SHOW_CONSTANTS_SETTING = "show_constants"


class VGFAdapter(Adapter):  # pylint: disable=too-few-public-methods
    """Adapter for VGF format."""

    metadata = AdapterMetadata(
        id="vgf_adapter_model_explorer",
        name="VGF Adapter",
        description="VGF adapter for Model Explorer",
        fileExts=["vgf"],
    )

    def get_config_editors(self) -> AdapterGetConfigEditorsResult:
        """Expose VGF-specific conversion settings when supported."""

        if SlideToggleConfigEditor is None:
            return AdapterGetConfigEditorsResult(configEditors=[])

        return AdapterGetConfigEditorsResult(
            configEditors=[
                SlideToggleConfigEditor(
                    id=SHOW_CONSTANTS_SETTING,
                    label="Show constants",
                    defaultValue=False,
                    help="Show VGF constants as graph nodes.",
                )
            ]
        )

    def convert(self, model_path: str, settings: dict) -> ModelExplorerGraphs:
        """Convert a given model to a model-explorer compatible format."""

        vgf = Parser(model_path).vgf
        show_constants = bool(settings.get(SHOW_CONSTANTS_SETTING, False))
        return {
            "graphCollections": [
                VgfGraphBuilder(
                    vgf, show_constants=show_constants
                ).graph_collection
            ]
        }
