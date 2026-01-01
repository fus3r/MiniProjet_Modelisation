#!/usr/bin/env python3
"""
Create the Simulateur_Sujet3GrD.zip deliverable.

Includes:
    - README.md
    - requirements.txt
    - src/sidthe/
    - scripts/ (excluding __pycache__)
    - outputs/figures/ (only .png and .pdf)

Excludes:
    - __pycache__
    - .DS_Store
    - .git
    - .venv
    - *.pyc

Usage (from repository root):
    python scripts/make_zip.py
"""
import os
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ZIP_NAME = "Simulateur_Sujet3GrD.zip"

# Files/folders to exclude
EXCLUDE_PATTERNS = {
    "__pycache__",
    ".DS_Store",
    ".git",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    "*.pyc",
    "*.pyo",
    ".gitignore",
}

# Empty placeholder files to exclude (not used in the project)
EMPTY_FILES_TO_SKIP = {
    "checks.py",
    "plotting.py",
    "scenarios.py",
    "simulate.py",
}


def should_exclude(path: Path) -> bool:
    """Check if path should be excluded from zip."""
    for part in path.parts:
        if part in EXCLUDE_PATTERNS:
            return True
        for pattern in EXCLUDE_PATTERNS:
            if "*" in pattern and path.match(pattern):
                return True
    return False


def main() -> int:
    zip_path = REPO_ROOT / ZIP_NAME
    
    print("=" * 60)
    print(f"Creating {ZIP_NAME}")
    print("=" * 60)
    
    # Items to include
    include_items = [
        "README.md",
        "requirements.txt",
        "src/",
        "scripts/",
        "outputs/figures/",
    ]
    
    files_added = 0
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in include_items:
            item_path = REPO_ROOT / item
            
            if not item_path.exists():
                print(f"  WARNING: {item} does not exist, skipping")
                continue
            
            if item_path.is_file():
                # Add single file
                arcname = item
                zf.write(item_path, arcname)
                print(f"  + {arcname}")
                files_added += 1
            else:
                # Add directory recursively
                for root, dirs, files in os.walk(item_path):
                    root_path = Path(root)
                    
                    # Filter out excluded directories
                    dirs[:] = [d for d in dirs if d not in EXCLUDE_PATTERNS]
                    
                    for file in files:
                        file_path = root_path / file
                        
                        if should_exclude(file_path):
                            continue
                        
                        # Skip empty placeholder files
                        if file in EMPTY_FILES_TO_SKIP and file_path.stat().st_size == 0:
                            print(f"  - {file} (empty, skipped)")
                            continue
                        
                        # For outputs/figures, only include .png and .pdf
                        if "outputs/figures" in str(root_path):
                            if file_path.suffix not in (".png", ".pdf"):
                                continue
                        
                        arcname = str(file_path.relative_to(REPO_ROOT))
                        zf.write(file_path, arcname)
                        print(f"  + {arcname}")
                        files_added += 1
    
    # Report
    zip_size = zip_path.stat().st_size / 1024 / 1024
    print("-" * 60)
    print(f"Created: {zip_path.name}")
    print(f"Files: {files_added}")
    print(f"Size: {zip_size:.2f} MB")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
