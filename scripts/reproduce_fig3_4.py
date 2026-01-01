#!/usr/bin/env python3
"""
Reproduce Figures 3 and 4 from the paper.

Convenience script that runs both fig3_robust.py and fig4_recourse.py.

Usage (from repository root):
    python scripts/reproduce_fig3_4.py
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    scripts = ["fig3_robust.py", "fig4_recourse.py"]
    
    print("=" * 60)
    print("Reproducing Figures 3 and 4")
    print("=" * 60)
    
    for script in scripts:
        print(f"\n>>> Running {script}...")
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / script)],
            cwd=str(REPO_ROOT),
        )
        if result.returncode != 0:
            print(f"ERROR: {script} failed")
            return 1
    
    print("\n" + "=" * 60)
    print("Done! Figures saved in outputs/figures/")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
