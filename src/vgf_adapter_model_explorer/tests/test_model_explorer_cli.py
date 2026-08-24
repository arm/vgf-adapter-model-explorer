# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

import os
import queue
import subprocess
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPO_ROOT / "src"


@pytest.mark.timeout(15)
def test_model_explorer_smoke():
    """
    Launch the server with the vgf_adapter_model_explorer extension,
    verify the expected stdout markers, then stop the server gracefully.
    """

    cmd = [
        "model-explorer",
        "--no_open_in_browser",
        "--extensions=vgf_adapter_model_explorer",
        "--host=127.0.0.1",
        "--skip_health_check",
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SOURCE_ROOT), env.get("PYTHONPATH", "")]
    )

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=0,
        universal_newlines=True,
        env=env,
    )

    # Confirm adapter was loaded and server starts (order is important here).
    curr_searched = 0
    searched_lines = [
        "VGF Adapter",
        "http://127.0.0.1",
    ]
    seen_lines = dict.fromkeys(searched_lines, False)

    assert proc.stdout is not None

    output_lines: list[str] = []
    line_queue: queue.Queue[str] = queue.Queue()

    def read_stdout() -> None:
        assert proc.stdout is not None
        for stdout_line in proc.stdout:
            line_queue.put(stdout_line)

    reader = threading.Thread(target=read_stdout, daemon=True)
    reader.start()

    deadline = time.monotonic() + 10
    try:
        while time.monotonic() < deadline and curr_searched < len(
            searched_lines
        ):
            try:
                line = line_queue.get(timeout=0.2)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue

            output_lines.append(line)
            if searched_lines[curr_searched] in line:
                seen_lines[searched_lines[curr_searched]] = True
                curr_searched += 1
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    assert all(seen_lines.values()), (
        f"Not all expected lines were seen: {seen_lines}\n"
        f"Output:\n{''.join(output_lines)}"
    )
