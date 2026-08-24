# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch, sentinel

from ..main import SHOW_CONSTANTS_SETTING, VGFAdapter


def test_convert_hides_constants_by_default():
    with (
        patch("vgf_adapter_model_explorer.main.Parser") as mock_parser,
        patch(
            "vgf_adapter_model_explorer.main.VgfGraphBuilder"
        ) as mock_builder,
    ):
        mock_builder.return_value.graph_collection = sentinel.graph_collection

        graphs = VGFAdapter().convert("model.vgf", {})

    mock_builder.assert_called_once_with(
        mock_parser.return_value.vgf, show_constants=False
    )
    assert graphs == {"graphCollections": [sentinel.graph_collection]}


def test_convert_shows_constants_when_enabled():
    with (
        patch("vgf_adapter_model_explorer.main.Parser") as mock_parser,
        patch(
            "vgf_adapter_model_explorer.main.VgfGraphBuilder"
        ) as mock_builder,
    ):
        mock_builder.return_value.graph_collection = sentinel.graph_collection

        graphs = VGFAdapter().convert(
            "model.vgf", {SHOW_CONSTANTS_SETTING: True}
        )

    mock_builder.assert_called_once_with(
        mock_parser.return_value.vgf, show_constants=True
    )
    assert graphs == {"graphCollections": [sentinel.graph_collection]}
