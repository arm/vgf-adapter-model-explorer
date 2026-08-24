#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Download SBOM release assets for dependency versions."""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ADAPTER_REPOSITORIES = {
    "spirv-adapter-model-explorer": "arm/spirv-adapter-model-explorer",
}

NATIVE_REPOSITORIES = {
    "ai-ml-sdk-vgf-library": "arm/ai-ml-sdk-vgf-library",
}

SBOM_ASSET_SUFFIXES = (".spdx.json", ".cdx.json")
NATIVE_SBOM_ARCHIVE_PATTERN = "Linux_x86_64.tar.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download SBOM assets for dependency versions in pyproject.toml "
            "and CMakeLists.txt."
        )
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help="Path to the package pyproject.toml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sbom/dependencies"),
        help="Directory to write downloaded dependency SBOMs.",
    )
    parser.add_argument(
        "--cmake",
        type=Path,
        default=Path("CMakeLists.txt"),
        help="Path to the native dependency CMakeLists.txt.",
    )
    parser.add_argument(
        "--github-token",
        default=None,
        help="Optional GitHub token for release asset downloads.",
    )
    return parser.parse_args()


def pinned_adapter_versions(pyproject: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    dependency_pattern = re.compile(
        r'^\s*"(?P<name>[A-Za-z0-9_.-]+)\s*(?:==|>=)\s*'
        r'(?P<version>[^",]+)",?\s*$'
    )

    in_dependencies = False
    for line in pyproject.read_text(encoding="utf-8").splitlines():
        if line.strip() == "dependencies = [":
            in_dependencies = True
            continue
        if in_dependencies and line.strip() == "]":
            break
        if not in_dependencies:
            continue

        match = dependency_pattern.match(line)
        if not match:
            continue
        name = match.group("name")
        if name in ADAPTER_REPOSITORIES:
            versions[name] = match.group("version").strip()

    missing = sorted(set(ADAPTER_REPOSITORIES) - set(versions))
    if missing:
        raise RuntimeError(
            "Missing adapter dependency versions in pyproject.toml: "
            + ", ".join(missing)
        )
    return versions


def pinned_native_versions(cmake: Path) -> dict[str, str]:
    cmake_text = cmake.read_text(encoding="utf-8")
    vgf_match = re.search(
        r'set\(\s*VGF_LIBRARY_TAG\s+"(?P<tag>[^"]+)"',
        cmake_text,
    )
    if not vgf_match:
        raise RuntimeError(f"Missing VGF_LIBRARY_TAG in {cmake}")

    return {"ai-ml-sdk-vgf-library": vgf_match.group("tag").strip()}


def github_json(url: str, token: str | None) -> object:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(url, headers=headers)
    # The URL is built from the fixed GitHub repository map above.
    with urlopen(request, timeout=30) as response:  # nosec B310
        return json.loads(response.read().decode("utf-8"))


def download_file(url: str, token: str | None) -> bytes:
    headers = {"Accept": "application/octet-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(url, headers=headers)
    # GitHub's API supplies the release asset URL.
    with urlopen(request, timeout=60) as response:  # nosec B310
        return response.read()


def release_by_tag(repo: str, tag: str, token: str | None) -> dict:
    release_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    release = github_json(release_url, token)
    if not isinstance(release, dict):
        raise RuntimeError(f"Unexpected release response for {repo} {tag}")
    return release


def release_assets(release: dict, repo: str, tag: str) -> list[dict]:
    assets = release.get("assets", [])
    if not isinstance(assets, list):
        raise RuntimeError(f"Unexpected assets response for {repo} {tag}")

    return [asset for asset in assets if isinstance(asset, dict)]


def sbom_assets(release: dict, repo: str, tag: str) -> list[dict]:
    return [
        asset
        for asset in release_assets(release, repo, tag)
        if str(asset.get("name", "")).endswith(SBOM_ASSET_SUFFIXES)
    ]


def download_adapter_sboms(
    versions: dict[str, str],
    output_dir: Path,
    token: str | None,
) -> dict[str, list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, list[str]] = {}

    for package_name, version in sorted(versions.items()):
        repo = ADAPTER_REPOSITORIES[package_name]
        tag = version if version.startswith("v") else f"v{version}"
        release = release_by_tag(repo, tag, token)
        assets = sbom_assets(release, repo, tag)
        if not assets:
            raise RuntimeError(f"No JSON SBOM assets found for {repo} {tag}")

        downloaded: list[str] = []
        package_output_dir = output_dir / package_name
        package_output_dir.mkdir(parents=True, exist_ok=True)
        for asset in assets:
            asset_name = str(asset["name"])
            browser_download_url = str(asset["browser_download_url"])
            output = package_output_dir / asset_name
            output.write_bytes(download_file(browser_download_url, token))
            downloaded.append(str(output))
            print(f"Downloaded {repo} {tag} SBOM: {output}")

        manifest[f"{package_name}=={version}"] = downloaded

    return manifest


def download_native_sboms(
    versions: dict[str, str],
    output_dir: Path,
    token: str | None,
) -> dict[str, list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, list[str]] = {}

    for package_name, tag in sorted(versions.items()):
        repo = NATIVE_REPOSITORIES[package_name]
        release = release_by_tag(repo, tag, token)
        downloaded: list[str] = []
        package_output_dir = output_dir / package_name
        package_output_dir.mkdir(parents=True, exist_ok=True)

        for asset in sbom_assets(release, repo, tag):
            asset_name = str(asset["name"])
            browser_download_url = str(asset["browser_download_url"])
            output = package_output_dir / asset_name
            output.write_bytes(download_file(browser_download_url, token))
            downloaded.append(str(output))
            print(f"Downloaded {repo} {tag} SBOM: {output}")

        downloaded.extend(
            extract_native_sbom_from_linux_archive(
                release, repo, tag, package_output_dir, token
            )
        )

        if not downloaded:
            raise RuntimeError(f"No JSON SBOMs found for {repo} {tag}")

        manifest[f"{package_name}=={tag}"] = downloaded

    return manifest


def extract_native_sbom_from_linux_archive(
    release: dict,
    repo: str,
    tag: str,
    output_dir: Path,
    token: str | None,
) -> list[str]:
    matching_assets = [
        asset
        for asset in release_assets(release, repo, tag)
        if str(asset.get("name", "")).endswith(NATIVE_SBOM_ARCHIVE_PATTERN)
    ]
    if not matching_assets:
        raise RuntimeError(
            f"No {NATIVE_SBOM_ARCHIVE_PATTERN} asset found for {repo} {tag}"
        )

    asset = matching_assets[0]
    asset_name = str(asset["name"])
    browser_download_url = str(asset["browser_download_url"])
    archive_bytes = download_file(browser_download_url, token)
    downloaded: list[str] = []

    with tarfile.open(
        fileobj=io.BytesIO(archive_bytes), mode="r:gz"
    ) as archive:
        for member in archive.getmembers():
            member_name = member.name
            if not member.isfile() or not member_name.endswith(
                SBOM_ASSET_SUFFIXES
            ):
                continue
            member_file = archive.extractfile(member)
            if member_file is None:
                continue

            output = output_dir / Path(member_name).name
            output.write_bytes(member_file.read())
            downloaded.append(str(output))
            print(f"Extracted {asset_name} SBOM: {output}")

    return downloaded


def main() -> None:
    args = parse_args()
    try:
        manifest = {}
        manifest.update(
            download_adapter_sboms(
                pinned_adapter_versions(args.pyproject),
                args.output_dir,
                args.github_token,
            )
        )
        manifest.update(
            download_native_sboms(
                pinned_native_versions(args.cmake),
                args.output_dir,
                args.github_token,
            )
        )
    except (HTTPError, URLError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    manifest_path = args.output_dir / "adapter-sbom-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote adapter SBOM manifest: {manifest_path}")


if __name__ == "__main__":
    main()
