#!/usr/bin/env python3
"""
Script to replace symbolic links to Python files with the actual files.
This will recursively process a directory and its subdirectories.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def is_python_file(filepath):
    """Check if a file is a Python file based on extension."""
    return filepath.suffix.lower() in [".py", ".pyw", ".pyi", ".pyx"]


def replace_symlink_with_file(symlink_path):
    """Replace a symlink with the actual file it points to."""
    try:
        # Get the target of the symlink
        target_path = Path(os.readlink(symlink_path))

        # If the target is relative, resolve it relative to the symlink's directory
        if not target_path.is_absolute():
            target_path = symlink_path.parent / target_path

        # Resolve any .. or . in the path
        target_path = target_path.resolve()

        # Check if target exists
        if not target_path.exists():
            print(
                f"Warning: Target does not exist for {symlink_path} -> {target_path}"
            )
            return False

        # Check if target is a Python file
        if not is_python_file(target_path):
            print(f"Skipping: {symlink_path} (target is not a Python file)")
            return False

        # Remove the symlink
        symlink_path.unlink()

        # Copy the target file to the symlink location
        shutil.copy2(target_path, symlink_path)

        print(f"Replaced: {symlink_path} -> {target_path}")
        return True

    except Exception as e:
        print(f"Error processing {symlink_path}: {e}")
        return False


def process_directory(directory, dry_run=False):
    """Recursively process directory to replace Python symlinks."""
    directory = Path(directory).resolve()

    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        return

    if not directory.is_dir():
        print(f"Error: {directory} is not a directory")
        return

    replaced_count = 0
    error_count = 0

    # Walk through directory tree
    for root, _dirs, files in os.walk(directory):
        root_path = Path(root)

        for filename in files:
            filepath = root_path / filename

            # Check if it's a symlink
            if filepath.is_symlink():
                # Get the target to check if it's a Python file
                try:
                    target = Path(os.readlink(filepath))
                    if not target.is_absolute():
                        target = filepath.parent / target
                    target = target.resolve()

                    if target.exists() and is_python_file(target):
                        if dry_run:
                            print(f"Would replace: {filepath} -> {target}")
                            replaced_count += 1
                        else:
                            if replace_symlink_with_file(filepath):
                                replaced_count += 1
                            else:
                                error_count += 1

                except Exception as e:
                    print(f"Error reading symlink {filepath}: {e}")
                    error_count += 1

    print("\nSummary:")
    print(
        f"  Symlinks {'would be ' if dry_run else ''}replaced: {replaced_count}"
    )
    if error_count > 0:
        print(f"  Errors encountered: {error_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Replace symbolic links to Python files with the actual files"
    )
    parser.add_argument(
        "directory", help="Directory to process (including subdirectories)"
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    # Confirm with user unless --yes is specified
    if not args.dry_run and not args.yes:
        response = input(
            f"This will replace all Python symlinks in '{args.directory}' with actual files.\n"
            "This operation cannot be undone. Continue? [y/N]: "
        )
        if response.lower() != "y":
            print("Operation cancelled.")
            sys.exit(0)

    process_directory(args.directory, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
