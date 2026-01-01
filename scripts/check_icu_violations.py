#!/usr/bin/env python3
"""
Check ICU constraint violations for MPC simulations.

Validates that robust and recourse SNMPC respect T <= T_MAX daily.
Runs quick simulations and reports max(T - T_MAX) across all trajectories.

Exit code 0: All violations <= tolerance (1e-10)
Exit code 1: Violations exceed tolerance

Usage (from repository root):
    python3 scripts/check_icu_violations.py
"""
import sys
from pathlib import Path

# Add src/ to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from sidthe.params import x0, theta_nom, T_MAX, T_NPI, DT, generate_scenarios, SIDTHEParams
from sidthe.integrators import simulate_days
from sidthe.mpc import build_controller, MPCConfig


TOLERANCE = 1e-10


def run_mpc_check(mode: str, n_mpc_scenarios: int, n_sim_scenarios: int, total_days: int) -> float:
    """
    Run MPC simulation and return max ICU violation.
    
    Returns
    -------
    max_violation : float
        Maximum value of (T - T_MAX) across all trajectories and days.
        Negative or zero means no violation.
    """
    # Generate scenarios
    thetas_all, probs_all = generate_scenarios(theta_nom, rel=0.05)
    
    # Select MPC scenarios
    rng_mpc = np.random.default_rng(123)
    mpc_indices = rng_mpc.choice(len(thetas_all), n_mpc_scenarios, replace=False)
    thetas_mpc = thetas_all[mpc_indices]
    probs_mpc = probs_all[mpc_indices]
    probs_mpc = probs_mpc / probs_mpc.sum()
    
    # Config with daily ICU enforcement
    if mode == "robust":
        config = MPCConfig(
            horizon_days=84,
            npi_days=T_NPI,
            enforce_icu_daily=True,
            max_scenarios_robust=None,
        )
    else:  # recourse
        config = MPCConfig(
            horizon_days=84,
            npi_days=T_NPI,
            enforce_icu_daily=True,
            max_scenarios_recourse=None,
        )
    
    # Build controller
    solve_mpc = build_controller(mode, thetas_mpc, probs_mpc, config)
    
    # Select simulation scenarios
    rng_sim = np.random.default_rng(42)
    sim_indices = rng_sim.choice(len(thetas_all), n_sim_scenarios, replace=False)
    
    max_violation = -np.inf
    
    for sim_idx in sim_indices:
        true_theta = SIDTHEParams.from_array(thetas_all[sim_idx])
        
        x_current = x0.copy()
        u_applied = 0.0
        
        for day in range(total_days):
            if day % T_NPI == 0:
                result = solve_mpc(x_current)
                if result["status"]:
                    u_applied = result["u0_applied"]
                else:
                    u_applied = config.u_max  # Fallback
            
            # Simulate one day
            u_seq = np.array([u_applied])
            _, xs_step = simulate_days(x_current, true_theta, u_seq, dt=DT)
            x_current = xs_step[-1]
            
            # Check ICU constraint
            T_val = x_current[3]
            violation = T_val - T_MAX
            max_violation = max(max_violation, violation)
    
    return max_violation


def main() -> int:
    """Check ICU violations for robust and recourse MPC."""
    print("=" * 60)
    print("ICU Constraint Violation Check")
    print("=" * 60)
    print(f"T_MAX = {T_MAX:.6f}")
    print(f"Tolerance = {TOLERANCE:.2e}")
    print("-" * 60)
    
    results = {}
    
    # Check robust MPC
    print("\nChecking robust SNMPC...")
    max_viol_robust = run_mpc_check(
        mode="robust",
        n_mpc_scenarios=20,
        n_sim_scenarios=10,  # Reduced for quick check
        total_days=200,
    )
    results["robust"] = max_viol_robust
    print(f"  max(T - T_MAX) = {max_viol_robust:.6e}")
    
    # Check recourse MPC
    print("\nChecking recourse SNMPC...")
    max_viol_recourse = run_mpc_check(
        mode="recourse",
        n_mpc_scenarios=15,
        n_sim_scenarios=10,
        total_days=200,
    )
    results["recourse"] = max_viol_recourse
    print(f"  max(T - T_MAX) = {max_viol_recourse:.6e}")
    
    # Summary
    print("-" * 60)
    max_overall = max(results.values())
    
    if max_overall <= TOLERANCE:
        print(f"\n✓ PASS: All ICU violations <= {TOLERANCE:.2e}")
        print(f"  max_overall = {max_overall:.6e}")
        return 0
    else:
        print(f"\n✗ FAIL: ICU violations exceed tolerance")
        print(f"  max_overall = {max_overall:.6e}")
        print(f"  tolerance   = {TOLERANCE:.2e}")
        for mode, viol in results.items():
            if viol > TOLERANCE:
                print(f"  - {mode}: {viol:.6e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
