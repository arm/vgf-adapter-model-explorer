# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

import importlib.resources
import json
from pathlib import Path
from typing import Any, Dict, Optional

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


def exec_vgf_dump(
    file_path: str, dump_spirv_index: Optional[int] = None
) -> Dict[str, Any]:
    """Execute the vgf_dump binary and return the JSON output."""

    is_dump_spirv_set = dump_spirv_index is not None

    vgf_dump_path = get_binary_path("vgf_dump")
    command = [str(vgf_dump_path), "-i", file_path]
    if is_dump_spirv_set:
        command.extend(["--dump-spirv", str(dump_spirv_index)])

    result = exec_cmd(command, text=is_dump_spirv_set == False, input=None)
    return (
        result.stdout.strip()
        if is_dump_spirv_set
        else json.loads(result.stdout)
    )
