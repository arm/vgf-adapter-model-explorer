# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

import importlib.resources
from pathlib import Path

from vgf_adapter_model_explorer.exec.exec_cmd import exec_cmd


def get_binary_path(binary_name: str) -> Path:
    """Get path to bundled binary."""
    return Path(
        str(
            importlib.resources.files(
                "vgf_adapter_model_explorer.bin"
            ).joinpath(binary_name)
        )
    )


def exec_mlir_translate(dumped_spirv_module):
    """Deserializes SPIR-V module using mlir-translate"""

    mlir_translate_path = get_binary_path("mlir-translate")
    result = exec_cmd(
        [str(mlir_translate_path), "--deserialize-spirv"],
        input=dumped_spirv_module,
        text=None,
    )

    return result.stdout.decode("utf-8").strip()
