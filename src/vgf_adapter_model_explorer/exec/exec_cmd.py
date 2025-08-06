import json
import subprocess
from typing import Optional


def exec_cmd(cmd: list[str], input, text: Optional[bool]):
    """Executes command"""
    try:
        result = subprocess.run(
            cmd, input=input, text=text, capture_output=True, check=True
        )

        if result.stderr:
            raise RuntimeError(f"Error from binary: {result.stderr.strip()}")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Binary execution failed: {e.stderr or e}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse JSON output: {e}") from e

    return result
