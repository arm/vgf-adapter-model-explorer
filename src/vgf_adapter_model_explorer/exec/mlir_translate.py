# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

from pathlib import Path

from vgf_adapter_model_explorer.exec.exec_cmd import exec_cmd
from vgf_adapter_model_explorer.exec.utils import get_binary_path


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
