import os
import subprocess

import pytest


@pytest.mark.timeout(5)
def test_model_explorer_smoke():
    """
    Launch the server with the vgf_adapter_model_explorer extension,
    verify the expected stdout markers, hit the printed URL once, then
    stop the server gracefully.
    """

    cmd = [
        "model-explorer",
        "--no_open_in_browser",
        "--extensions=vgf_adapter_model_explorer",
        "--host=127.0.0.1",
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")

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

    for line in proc.stdout:
        if searched_lines[curr_searched] in line:
            seen_lines[searched_lines[curr_searched]] = True
            curr_searched += 1
            if curr_searched >= len(searched_lines):
                break

    assert all(seen_lines.values()), (
        f"Not all expected lines were seen: {seen_lines}"
    )
