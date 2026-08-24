#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Add Arm creator metadata and disclaimer text to an SBOM JSON file."""

import argparse
import json
from pathlib import Path

ARM_CREATOR = "Organization: Arm Limited (open-source-office@arm.com)"
SBOM_DISCLAIMER = (
    "THIS SOFTWARE BILL OF MATERIALS (SBOM) IS PROVIDED BY ARM LIMITED "
    "AS IS AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT "
    "LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A "
    "PARTICULAR PURPOSE, AND NONINFRINGEMENT ARE DISCLAIMED. IN NO EVENT "
    "SHALL ARM LIMITED BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, "
    "SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT "
    "LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, "
    "DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY "
    "THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT "
    "(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE "
    "OF THIS SBOM, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add Arm SBOM disclaimer metadata to SPDX or CycloneDX JSON"
    )
    parser.add_argument("sbom", type=Path, help="Path to SBOM JSON file")
    return parser.parse_args()


def add_spdx_disclaimer(data: dict) -> None:
    creation_info = data.setdefault("creationInfo", {})

    creators = creation_info.setdefault("creators", [])
    if ARM_CREATOR not in creators:
        creators.insert(0, ARM_CREATOR)

    existing_comment = creation_info.get("comment")
    comment = SBOM_DISCLAIMER
    if existing_comment and SBOM_DISCLAIMER not in existing_comment:
        comment = f"{existing_comment}\n\n{SBOM_DISCLAIMER}"
    elif existing_comment:
        comment = existing_comment

    creation_info["comment"] = comment


def upsert_cyclonedx_property(properties: list, name: str, value: str) -> None:
    for prop in properties:
        if prop.get("name") == name:
            prop["value"] = value
            return
    properties.append({"name": name, "value": value})


def add_cyclonedx_disclaimer(data: dict) -> None:
    metadata = data.setdefault("metadata", {})
    properties = metadata.setdefault("properties", [])
    upsert_cyclonedx_property(properties, "arm:sbom:creator", ARM_CREATOR)
    upsert_cyclonedx_property(
        properties, "arm:sbom:disclaimer", SBOM_DISCLAIMER
    )


def add_disclaimer(sbom: Path) -> None:
    data = json.loads(sbom.read_text(encoding="utf-8"))

    if "spdxVersion" in data or "creationInfo" in data:
        add_spdx_disclaimer(data)
    elif data.get("bomFormat") == "CycloneDX":
        add_cyclonedx_disclaimer(data)
    else:
        raise ValueError(f"Unsupported SBOM format: {sbom}")

    sbom.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    add_disclaimer(args.sbom)


if __name__ == "__main__":
    main()
