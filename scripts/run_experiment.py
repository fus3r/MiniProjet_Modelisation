#!/usr/bin/env python3
"""
Canonical experiment script: runs all main simulations and generates figures.

This is the ONE command to reproduce the key results.

Usage (from repository root):
    python scripts/run_experiment.py [--seed SEED]

Outputs:
    - outputs/figures/quickstart_T.png
    - outputs/figures/fig3_robust.png/pdf
    - outputs/figures/fig4_recourse.png/pdf
    - outputs/figures/fig7_vanilla_infeasibility.png/pdf
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def run_script(script_name: str) -> bool:
    """Run a Python script and return True if successful."""
    script_path = SCRIPTS_DIR / script_name
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print("=" * 60)
    
    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(REPO_ROOT),
    )
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"✓ {script_name} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"✗ {script_name} FAILED (exit code {result.returncode})")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run canonical SIDTHE experiment"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run only quickstart (skip MPC figures)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SIDTHE Canonical Experiment")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    scripts = ["quickstart.py"]
    if not args.quick:
        scripts.extend([
            "fig3_robust.py",
            "fig4_recourse.py",
            "fig7_vanilla_infeasibility.py",
        ])

    success_count = 0
    for script in scripts:
        if run_script(script):
            success_count += 1

    print("\n" + "=" * 60)
    print(f"SUMMARY: {success_count}/{len(scripts)} scripts succeeded")
    print("=" * 60)

    # List generated figures
    figures_dir = REPO_ROOT / "outputs" / "figures"
    if figures_dir.exists():
        print("\nGenerated figures:")
        for f in sorted(figures_dir.glob("*")):
            if f.suffix in (".png", ".pdf"):
                print(f"  - {f.name}")

    return 0 if success_count == len(scripts) else 1


if __name__ == "__main__":
    sys.exit(main())
