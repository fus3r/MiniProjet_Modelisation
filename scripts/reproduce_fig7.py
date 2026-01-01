#!/usr/bin/env python3
"""
Reproduce Figure 7 from the paper.

Convenience wrapper for fig7_vanilla_infeasibility.py.

Usage (from repository root):
    python scripts/reproduce_fig7.py
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    print("=" * 60)
    print("Reproducing Figure 7")
    print("=" * 60)
    
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "fig7_vanilla_infeasibility.py")],
        cwd=str(REPO_ROOT),
    )
    
    if result.returncode != 0:
        print("ERROR: fig7_vanilla_infeasibility.py failed")
        return 1
    
    print("\n" + "=" * 60)
    print("Done! Figure saved in outputs/figures/")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
