#!/usr/bin/env python3
"""
Figure 4: Recourse SNMPC simulation.

Demonstrates scenario-based NMPC with recourse (Eq 13) where only the
first control block is shared (non-anticipativity), subsequent blocks
can adapt per scenario.

Reference: Paper Figure 4, page 6.

Usage (from repository root):
    python3 scripts/fig4_recourse.py
"""
import sys
from pathlib import Path

# Add src/ to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt

from sidthe.params import x0, theta_nom, T_MAX, T_NPI, DT, generate_scenarios, SIDTHEParams
from sidthe.integrators import simulate_days
from sidthe.mpc import build_controller, MPCConfig


def main() -> None:
    """Run recourse SNMPC simulation."""
    print("=" * 60)
    print("Figure 4: Recourse SNMPC Simulation")
    print("=" * 60)

    # Configuration
    n_sim_scenarios = 25
    total_days = 350
    
    # Strategy A: simulation ⊂ MPC scenarios (guarantees ICU respect)
    n_mpc_scenarios = 30  # Must be >= n_sim_scenarios
    scenario_seed = 123   # For reproducibility
    
    # Config: no internal reduction (max_scenarios_recourse=None)
    config = MPCConfig(
        horizon_days=84,
        npi_days=T_NPI,
        enforce_icu_daily=True,
        max_scenarios_recourse=None,  # Use exactly what we pass
    )

    # Generate all 729 scenarios
    thetas_all, probs_all = generate_scenarios(theta_nom, rel=0.05)

    # EXPLICIT: Select subset of scenarios for recourse MPC
    rng = np.random.default_rng(scenario_seed)
    mpc_indices = rng.choice(len(thetas_all), n_mpc_scenarios, replace=False)
    thetas_mpc = thetas_all[mpc_indices]
    probs_mpc = probs_all[mpc_indices]
    probs_mpc = probs_mpc / probs_mpc.sum()

    print(f"Building recourse controller...")
    print(f"  - n_mpc_scenarios: {n_mpc_scenarios}")
    print(f"  - scenario_seed: {scenario_seed}")
    print(f"  - enforce_icu_daily: {config.enforce_icu_daily}")
    solve_mpc = build_controller("recourse", thetas_mpc, probs_mpc, config)
    
    # Verify actual scenarios used
    test_result = solve_mpc(x0)
    print(f"  - n_scenarios_used (actual): {test_result['n_scenarios_used']}")

    # Strategy A: sim_indices is a SUBSET of mpc_indices
    sim_rng = np.random.default_rng(42)
    sim_indices = sim_rng.choice(mpc_indices, n_sim_scenarios, replace=False)
    print(f"\nSimulation ⊂ MPC scenarios: {n_sim_scenarios} of {n_mpc_scenarios}")

    print(f"Simulating {n_sim_scenarios} scenarios...")
    print(f"Horizon: {config.horizon_days} days, NPI block: {T_NPI} days")
    print("-" * 60)

    trajectories = []
    max_T_all = -np.inf  # Track max T across all trajectories

    for idx, true_idx in enumerate(sim_indices):
        true_theta = SIDTHEParams.from_array(thetas_all[true_idx])

        x_current = x0.copy()
        xs_traj = [x_current.copy()]
        us_traj = []
        ts_traj = [0.0]
        u_applied = 0.0

        day = 0
        while day < total_days:
            if day % T_NPI == 0:
                result = solve_mpc(x_current)
                if result["status"]:
                    u_applied = result["u0_applied"]
                else:
                    u_applied = config.u_max
                    print(f"  Scenario {idx+1}: infeasible at day {day}, using u_max")

            u_seq = np.array([u_applied])
            _, xs_step = simulate_days(x_current, true_theta, u_seq, dt=DT)
            x_current = xs_step[-1]

            xs_traj.append(x_current.copy())
            us_traj.append(u_applied)
            ts_traj.append(day + 1)
            day += 1

        traj_xs = np.array(xs_traj)
        trajectories.append({
            "ts": np.array(ts_traj),
            "xs": traj_xs,
            "us": np.array(us_traj),
        })
        max_T_all = max(max_T_all, traj_xs[:, 3].max())

        if (idx + 1) % 5 == 0:
            print(f"  Completed {idx+1}/{n_sim_scenarios} scenarios")

    # Report max ICU violation
    max_violation = max_T_all - T_MAX
    print("-" * 60)
    print(f"max(T - T_MAX) = {max_violation:.6e}")
    if max_violation > 1e-6:  # Warn only for visible violations (not numerical noise)
        print(f"  WARNING: ICU threshold exceeded!")

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: % ICU
    ax1 = axes[0]
    for traj in trajectories:
        ts = traj["ts"]
        T_vals = traj["xs"][:, 3] * 100
        ax1.plot(ts, T_vals, "forestgreen", alpha=0.4, linewidth=0.8)

    ax1.axhline(y=100 * T_MAX, color="r", linestyle="--", linewidth=1.5,
                label=f"ICU threshold ({100*T_MAX:.1f}%)")
    ax1.set_ylabel("% ICU", fontsize=12)
    ax1.set_title("Recourse SNMPC (Eq 13): ICU Occupancy", fontsize=14)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Bottom: Control
    ax2 = axes[1]
    for traj in trajectories:
        ts_u = traj["ts"][:-1]
        ax2.plot(ts_u, traj["us"], "forestgreen", alpha=0.4, linewidth=0.8)

    ax2.set_xlabel("Time [days]", fontsize=12)
    ax2.set_ylabel("α reduction (u)", fontsize=12)
    ax2.set_title("Recourse SNMPC: Control Input", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, total_days)
    ax2.set_ylim(0, 1)

    plt.tight_layout()

    # Save
    output_dir = REPO_ROOT / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "fig4_recourse.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"\nFigure saved: {fig_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
