# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

import importlib.resources
import sys
from pathlib import Path

from vgf_adapter_model_explorer.exec.exec_cmd import exec_cmd


def get_binary_path(binary_name: str) -> Path:
    """Get path to bundled binary, accounting for platform extensions."""
    if sys.platform.startswith("win"):
        binary_name = binary_name + ".exe"

    return Path(
        str(
            importlib.resources.files(
                "vgf_adapter_model_explorer.bin"
            ).joinpath(binary_name)
        )
    )


def exec_mlir_translate(spirv_path: Path) -> str:
    """
    Deserialize SPIR-V using mlir-translate.
    - `spirv_path` is a file path to the SPIR-V binary.
    """
    mlir_translate = get_binary_path("mlir-translate")

    res = exec_cmd(
        [str(mlir_translate), "--deserialize-spirv", str(spirv_path)],
        input=None,
        text=False,
    )
    return res.stdout.decode("utf-8").rstrip("\r\n")
