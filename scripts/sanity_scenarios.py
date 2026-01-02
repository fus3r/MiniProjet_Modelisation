#!/usr/bin/env python3
"""
Vérification de la génération de scénarios.

Vérifie que generate_scenarios produit 729 scénarios (3^6)
avec les bonnes bornes min/max par paramètre.

Usage (depuis la racine du dépôt) :
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
    """Lance les vérifications de génération de scénarios."""
    rel = 0.05
    thetas_array, probs = generate_scenarios(theta_nom, rel=rel)

    print("=" * 60)
    print("Scenario Generation Sanity Check")
    print("=" * 60)

    # Vérification de la forme
    expected_shape = (729, 6)
    print(f"thetas_array.shape = {thetas_array.shape}")
    print(f"Forme attendue : {expected_shape}")
    assert thetas_array.shape == expected_shape, (
        f"Forme incorrecte : {thetas_array.shape}, attendu {expected_shape}"
    )
    print("✓ Vérification forme OK")
    print("-" * 60)

    # Vérification des probabilités
    print(f"probs.shape = {probs.shape}")
    print(f"sum(probs) = {probs.sum():.12f}")
    assert np.abs(probs.sum() - 1.0) < 1e-12, "Les probabilités doivent sommer à 1"
    print("✓ Vérification probabilités OK")
    print("-" * 60)

    # Vérification min/max par paramètre
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
