#!/usr/bin/env python3
"""
Smoke test: quick validation that core functionality works.

Runs a minimal simulation in < 30s and verifies outputs are generated.
Exit code 0 = PASS, 1 = FAIL.

Usage (from repository root):
    python scripts/smoke_test.py
"""
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from sidthe.params import x0, theta_nom, T_MAX, T_NPI, DT, generate_scenarios, SIDTHEParams
from sidthe.integrators import simulate_days
from sidthe.mpc import build_controller, MPCConfig


def test_simulation() -> bool:
    """Test basic SIDTHE simulation."""
    print("Test 1: Basic simulation (50 days, u=0)")
    ts, xs = simulate_days(x0, theta_nom, np.zeros(50), dt=DT)
    
    # Check shapes
    assert ts.shape == (51,), f"ts shape mismatch: {ts.shape}"
    assert xs.shape == (51, 6), f"xs shape mismatch: {xs.shape}"
    
    # Check mass conservation
    mass_drift = np.abs(xs.sum(axis=1) - x0.sum()).max()
    assert mass_drift < 1e-10, f"Mass drift too large: {mass_drift}"
    
    print(f"  ✓ Shapes correct, mass drift = {mass_drift:.2e}")
    return True


def test_scenario_generation() -> bool:
    """Test scenario generation."""
    print("Test 2: Scenario generation (729 scenarios)")
    thetas, probs = generate_scenarios(theta_nom, rel=0.05)
    
    assert thetas.shape == (729, 6), f"thetas shape: {thetas.shape}"
    assert probs.shape == (729,), f"probs shape: {probs.shape}"
    assert np.abs(probs.sum() - 1.0) < 1e-12, "Probs don't sum to 1"
    
    print(f"  ✓ 729 scenarios, probs sum = {probs.sum():.12f}")
    return True


def test_mpc_quick() -> bool:
    """Test MPC controller (minimal config)."""
    print("Test 3: MPC controller (3 scenarios, 28-day horizon)")
    
    # Minimal setup
    thetas, probs = generate_scenarios(theta_nom, rel=0.05)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(thetas), 3, replace=False)
    thetas_mini = thetas[idx]
    probs_mini = probs[idx]
    probs_mini /= probs_mini.sum()
    
    config = MPCConfig(
        horizon_days=28,  # 2 blocks only
        npi_days=14,
        enforce_icu_daily=True,
    )
    
    solve = build_controller("robust", thetas_mini, probs_mini, config)
    result = solve(x0)
    
    assert result["status"], f"MPC failed: {result}"
    assert 0 <= result["u0_applied"] <= config.u_max, "u out of bounds"
    
    print(f"  ✓ MPC solved, u0 = {result['u0_applied']:.4f}")
    return True


def test_output_dir() -> bool:
    """Test output directory creation."""
    print("Test 4: Output directory")
    output_dir = REPO_ROOT / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    test_file = output_dir / "_smoke_test_marker.txt"
    test_file.write_text("smoke test OK\n")
    
    assert test_file.exists(), "Could not create test file"
    test_file.unlink()  # Clean up
    
    print(f"  ✓ Output directory writable: {output_dir}")
    return True


def main() -> int:
    print("=" * 60)
    print("SIDTHE Smoke Test")
    print("=" * 60)
    
    start = time.time()
    tests = [
        test_simulation,
        test_scenario_generation,
        test_mpc_quick,
        test_output_dir,
    ]
    
    passed = 0
    for test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
    
    elapsed = time.time() - start
    
    print("-" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed in {elapsed:.1f}s")
    print("=" * 60)
    
    if passed == len(tests):
        print("✓ SMOKE TEST PASSED")
        return 0
    else:
        print("✗ SMOKE TEST FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
