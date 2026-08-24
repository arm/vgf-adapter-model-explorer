# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.machinery
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = REPO_ROOT / os.environ.get(
    "VGF_NATIVE_BUILD_DIR", ".native/build"
)
STAGE_ROOT = REPO_ROOT / "src"


def extension_suffixes() -> tuple[str, ...]:
    return tuple(importlib.machinery.EXTENSION_SUFFIXES)


def find_built_vgfpy() -> Path:
    suffixes = extension_suffixes()
    matches = sorted(
        path
        for path in BUILD_ROOT.rglob("vgfpy*")
        if path.is_file()
        and path.name.startswith("vgfpy")
        and path.name.endswith(suffixes)
    )
    if not matches:
        raise FileNotFoundError(
            f"Could not find a built vgfpy extension under {BUILD_ROOT}. "
            "Run `hatch run build-native` first."
        )

    # Prefer build outputs under VGF library's generated build tree over any
    # copied or temporary files that may exist elsewhere below .native/build.
    for path in matches:
        if "vgf_library-build" in path.parts:
            return path

    return matches[0]


def remove_stale_staged_extensions() -> None:
    suffixes = extension_suffixes()
    for path in STAGE_ROOT.glob("vgfpy*"):
        if path.is_file() and path.name.endswith(suffixes):
            path.unlink()


def main() -> None:
    source = find_built_vgfpy()
    target = STAGE_ROOT / source.name

    remove_stale_staged_extensions()
    shutil.copy2(source, target)
    print(f"Staged {source} -> {target}")


if __name__ == "__main__":
    main()
