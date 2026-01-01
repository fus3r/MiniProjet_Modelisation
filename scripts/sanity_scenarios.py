#!/usr/bin/env python3
"""
Sanity check for scenario generation.

Verifies that generate_scenarios produces 729 scenarios (3^6)
with correct min/max bounds per parameter.

Usage (from repository root):
    python3 scripts/sanity_scenarios.py
"""
import sys
from pathlib import Path

# Add src/ to path for imports when running from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from sidthe.params import theta_nom, generate_scenarios


def main() -> int:
    """Run scenario generation sanity checks."""
    rel = 0.05
    thetas_array, probs = generate_scenarios(theta_nom, rel=rel)

    print("=" * 60)
    print("Scenario Generation Sanity Check")
    print("=" * 60)

    # Check shape
    expected_shape = (729, 6)
    print(f"thetas_array.shape = {thetas_array.shape}")
    print(f"Expected shape: {expected_shape}")
    assert thetas_array.shape == expected_shape, (
        f"Shape mismatch: got {thetas_array.shape}, expected {expected_shape}"
    )
    print("✓ Shape check passed")
    print("-" * 60)

    # Check probs
    print(f"probs.shape = {probs.shape}")
    print(f"sum(probs) = {probs.sum():.12f}")
    assert np.abs(probs.sum() - 1.0) < 1e-12, "Probabilities must sum to 1"
    print("✓ Probability check passed")
    print("-" * 60)

    # Check min/max per parameter
    theta_nom_arr = theta_nom.to_array()
    param_names = ["alpha", "gamma", "lam", "delta", "sigma", "tau"]
    tol = 1e-12

    print("Parameter bounds check:")
    all_ok = True
    for i, name in enumerate(param_names):
        nominal = theta_nom_arr[i]
        expected_min = 0.95 * nominal
        expected_max = 1.05 * nominal
        actual_min = thetas_array[:, i].min()
        actual_max = thetas_array[:, i].max()

        min_ok = np.abs(actual_min - expected_min) < tol
        max_ok = np.abs(actual_max - expected_max) < tol

        status = "✓" if (min_ok and max_ok) else "✗"
        print(f"  {name:6s}: min={actual_min:.6e}, max={actual_max:.6e} "
              f"(expected [{expected_min:.6e}, {expected_max:.6e}]) {status}")

        if not (min_ok and max_ok):
            all_ok = False

    print("-" * 60)
    if all_ok:
        print("✓ All checks passed!")
        print("=" * 60)
        return 0
    else:
        print("✗ Some checks failed!")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
