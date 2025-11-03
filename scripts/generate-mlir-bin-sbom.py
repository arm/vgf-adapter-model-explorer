#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

import argparse
import logging
from datetime import datetime, timezone
from typing import List

from spdx_tools.common.spdx_licensing import spdx_licensing
from spdx_tools.spdx.model import (
    Actor,
    ActorType,
    CreationInfo,
    Document,
    Package,
    PackagePurpose,
    Relationship,
    RelationshipType,
)
from spdx_tools.spdx.validation.document_validator import (
    validate_full_spdx_document,
)
from spdx_tools.spdx.validation.validation_message import ValidationMessage
from spdx_tools.spdx.writer.write_anything import write_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate SPDX document for MLIR builds with LLVM model converter patch"
    )
    parser.add_argument("--llvm-ref", required=True, help="LLVM reference")
    parser.add_argument(
        "--model-converter-ref",
        required=True,
        help="Model converter reference",
    )
    parser.add_argument(
        "--wheel-version",
        required=True,
        help="Version of the vgf-adapter-model-explorer wheel",
    )
    parser.add_argument(
        "--output", default="mlir-builds.spdx.json", help="Output file path"
    )

    args = parser.parse_args()

    build_time = datetime.now(timezone.utc)
    build_time_str = build_time.strftime("%Y%m%d-%H%M%S")
    binary_version = f"{args.wheel_version}-llvm-{args.llvm_ref}-model_converter-{args.model_converter_ref}-{build_time_str}"

    creation_info = CreationInfo(
        spdx_version="SPDX-2.3",
        spdx_id="SPDXRef-DOCUMENT",
        name="vgf-adapter-model-explorer-mlir-builds-with-llvm-model-converter-patch",
        data_license="CC0-1.0",
        document_namespace=f"https://arm.com/spdx/vgf-adapter-model-explorer-mlir-llvm-{args.llvm_ref}-model_converter-{args.model_converter_ref}-{build_time_str}",
        creators=[
            Actor(
                ActorType.ORGANIZATION,
                "Arm Limited",
                "open-source-office@arm.com",
            )
        ],
        creator_comment="THIS SOFTWARE BILL OF MATERIALS (\"SBOM\") IS PROVIDED BY ARM LIMITED \"AS IS\" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT ARE DISCLAIMED. IN NO EVENT SHALL ARM LIMITED BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SBOM, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.",
        created=build_time,
    )
    document = Document(creation_info)

    llvm_pkg = Package(
        name="llvm",
        spdx_id="SPDXRef-LLVM",
        download_location="https://github.com/llvm/llvm-project",
        version=args.llvm_ref,
        files_analyzed=False,
        license_declared=spdx_licensing.parse(
            "Apache-2.0 with LLVM-exception"
        ),
        primary_package_purpose=PackagePurpose.LIBRARY,
        description="Base LLVM package used for MLIR builds.",
    )

    model_conv_pkg = Package(
        name="model-converter",
        spdx_id="SPDXRef-ModelConverter",
        download_location="https://github.com/ARM-software/model-converter",
        version=args.model_converter_ref,
        files_analyzed=False,
        license_declared=spdx_licensing.parse("Apache-2.0"),
        primary_package_purpose=PackagePurpose.OTHER,
        description="Package providing an LLVM patch for model conversion tooling.",
        originator=Actor(
            ActorType.ORGANIZATION, "Arm Limited", "open-source-office@arm.com"
        ),
    )

    mlir_translate_bin = Package(
        name="mlir-translate",
        spdx_id="SPDXRef-MLIRTranslateBin",
        download_location="https://pypi.org/project/vgf-adapter-model-explorer/",
        version=binary_version,
        files_analyzed=False,
        license_declared=spdx_licensing.parse(
            "Apache-2.0 with LLVM-exception"
        ),
        primary_package_purpose=PackagePurpose.APPLICATION,
        description="Binary build of mlir-translate tool, distributed as part of vgf-adapter-model-explorer Python wheel.",
        originator=Actor(
            ActorType.ORGANIZATION, "Arm Limited", "open-source-office@arm.com"
        ),
    )

    mlir_py_bindings_bin = Package(
        name="mlir-python-bindings",
        spdx_id="SPDXRef-MLIRPyBindingsBin",
        download_location="https://pypi.org/project/vgf-adapter-model-explorer/",
        version=binary_version,
        files_analyzed=False,
        license_declared=spdx_licensing.parse(
            "Apache-2.0 with LLVM-exception"
        ),
        primary_package_purpose=PackagePurpose.APPLICATION,
        description="Binary build of MLIR Python Bindings, distributed as part of vgf-adapter-model-explorer Python wheel.",
        originator=Actor(
            ActorType.ORGANIZATION, "Arm Limited", "open-source-office@arm.com"
        ),
    )

    document.packages = [
        llvm_pkg,
        model_conv_pkg,
        mlir_translate_bin,
        mlir_py_bindings_bin,
    ]

    relationships = [
        Relationship(
            "SPDXRef-DOCUMENT",
            RelationshipType.DESCRIBES,
            "SPDXRef-MLIRTranslateBin",
        ),
        Relationship(
            "SPDXRef-DOCUMENT",
            RelationshipType.DESCRIBES,
            "SPDXRef-MLIRPyBindingsBin",
        ),
        Relationship(
            "SPDXRef-LLVM",
            RelationshipType.PATCH_APPLIED,
            "SPDXRef-ModelConverter",
        ),
        Relationship(
            "SPDXRef-MLIRTranslateBin",
            RelationshipType.DEPENDS_ON,
            "SPDXRef-LLVM",
        ),
        Relationship(
            "SPDXRef-MLIRTranslateBin",
            RelationshipType.DEPENDS_ON,
            "SPDXRef-ModelConverter",
        ),
        Relationship(
            "SPDXRef-MLIRPyBindingsBin",
            RelationshipType.DEPENDS_ON,
            "SPDXRef-LLVM",
        ),
        Relationship(
            "SPDXRef-MLIRPyBindingsBin",
            RelationshipType.DEPENDS_ON,
            "SPDXRef-ModelConverter",
        ),
    ]
    document.relationships = relationships

    validation_messages: List[ValidationMessage] = validate_full_spdx_document(
        document
    )
    for m in validation_messages:
        logging.warning(m.validation_message)
        logging.warning(m.context)

    assert not validation_messages, (
        "SPDX document validation failed. See above for details."
    )

    write_file(document, args.output)
    print(f"SPDX document generated: {args.output}")


if __name__ == "__main__":
    main()
