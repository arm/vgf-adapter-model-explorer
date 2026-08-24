# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

import glob
import json
import os
from dataclasses import asdict

import pytest

from ..builder import builder
from ..parser import parser

FIXTURES_ROOT = os.path.join(os.path.dirname(__file__), "fixtures")

test_case_dirs = [
    d for d in glob.glob(os.path.join(FIXTURES_ROOT, "*")) if os.path.isdir(d)
]


@pytest.mark.parametrize(
    "case_dir", test_case_dirs, ids=lambda d: os.path.basename(d)
)
def test_e2e(case_dir):
    """Test parsing for each VGF file and compare against expected graph output."""

    input_vgf = os.path.join(case_dir, "input.vgf")
    expected_json = os.path.join(case_dir, "expected.json")

    vgf = parser.Parser(input_vgf).vgf
    graph_collection = builder.VgfGraphBuilder(vgf).graph_collection

    got = asdict(graph_collection)
    # Model Explorer 0.1.32 added this default field to GraphCollection. It is
    # not adapter output, so keep the fixture stable across supported versions.
    got.pop("graphSorting", None)

    with open(expected_json) as f:
        expected = json.load(f)

    assert got == expected, (
        f"Test failed for {input_vgf}. Expected and actual output differ."
    )
