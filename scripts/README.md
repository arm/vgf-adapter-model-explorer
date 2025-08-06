# Scripts

## replace_symlinks_in_llvm_python_package.py

A utility script to replace symbolic links to Python files with actual copies of the files they point to. This is useful when working with LLVM Python packages that contain symlinks, which may cause issues in certain deployment scenarios.

### Usage

```bash
# Dry run to see what would be changed
python replace_symlinks_in_llvm_python_package.py /path/to/directory --dry-run

# Replace symlinks with confirmation prompt
python replace_symlinks_in_llvm_python_package.py /path/to/directory

# Replace symlinks without confirmation
python replace_symlinks_in_llvm_python_package.py /path/to/directory --yes
```

### Options

- `directory`: Target directory to process (includes subdirectories)
- `-n, --dry-run`: Show what would be done without making changes
- `-y, --yes`: Skip confirmation prompt

### What it does

1. Recursively walks through the specified directory
2. Identifies symbolic links that point to Python files (`.py`, `.pyw`, `.pyi`, `.pyx`)
3. Replaces each symlink with a copy of the actual file
4. Preserves file metadata using `shutil.copy2()`
5. Provides summary of operations performed

### Safety features

- Validates that target files exist before replacement
- Only processes Python files
- Includes dry-run mode for testing
- Requires confirmation before making changes (unless `--yes` is used)
- Comprehensive error handling and reporting

**Warning**: This operation cannot be undone. Always use `--dry-run` first to verify the expected changes.