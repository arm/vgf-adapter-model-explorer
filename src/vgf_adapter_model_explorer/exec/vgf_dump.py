# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

import tempfile
from pathlib import Path
from typing import Optional

from vgf_adapter_model_explorer.exec.exec_cmd import exec_cmd
from vgf_adapter_model_explorer.exec.utils import get_binary_path


def exec_vgf_dump(
    file_path: str, dump_spirv_index: Optional[int] = None
) -> Path:
    """
    Dump SPIR-V to a temp file and return its Path (caller must unlink).
    """
    vgf_dump = get_binary_path("vgf_dump")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".spv") as f:
        out_path = Path(f.name)

    cmd = [str(vgf_dump), "-i", file_path, "-o", str(out_path)]
    if dump_spirv_index is not None:
        cmd += ["--dump-spirv", str(dump_spirv_index)]

    exec_cmd(cmd, input=None, text=False)

    return out_path
