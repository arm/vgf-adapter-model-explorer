# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

import subprocess


def exec_cmd(cmd: list[str], input, text: bool | None):
    """Run a command and return CompletedProcess. Trust the exit code, not stderr chatter."""
    try:
        return subprocess.run(
            cmd,
            input=input,
            text=text,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Binary execution failed.\n"
            f"  cmd: {e.cmd}\n"
            f"  returncode: {e.returncode}\n"
            f"  stdout: {e.stdout}\n"
            f"  stderr: {e.stderr}"
        ) from e
