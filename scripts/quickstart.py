#!/usr/bin/env python3
"""
Quickstart script for SIDTHE model simulation.

Simulates the epidemic over 350 days with no control (u=0),
generates a figure showing % ICU over time, and prints diagnostics.

Usage (from repository root):
    python3 scripts/quickstart.py
"""
import sys
from pathlib import Path

# Add src/ to path for imports when running from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt

from sidthe.params import x0, theta_nom, T_MAX, DT
from sidthe.integrators import simulate_days


def main() -> None:
    """Run SIDTHE simulation and generate output figure."""
    # Simulation parameters
    n_days = 350
    u_seq = np.zeros(n_days, dtype=np.float64)  # No control (u=0)

    # Run simulation
    ts, xs = simulate_days(x0, theta_nom, u_seq, dt=DT)

    # Extract state T (ICU occupancy fraction) - index 3
    T_traj = xs[:, 3]

    # Diagnostics
    sum_x0 = x0.sum()  # Initial mass (may differ from 1 per paper values)
    mass = xs.sum(axis=1)  # Sum of all compartments at each time
    mass_drift = np.abs(mass - sum_x0)  # Drift from initial mass
    max_mass_drift = mass_drift.max()
    deviation_from_1 = np.abs(mass - 1.0).max()  # Deviation from 1
    min_state = xs.min()

    print("=" * 60)
    print("SIDTHE Model Quickstart - Simulation Results")
    print("=" * 60)
    print(f"Simulation: {n_days} days, dt = {DT} day, u = 0 (no control)")
    print("-" * 60)
    print(f"sum(x0) = {sum_x0:.6f}")
    print(f"Max mass drift from x0: |sum(x) - sum(x0)| = {max_mass_drift:.2e}")
    print(f"Max deviation from 1: |sum(x) - 1| = {deviation_from_1:.2e}")
    print(f"Min state value over trajectory: min(x) = {min_state:.2e}")
    print("-" * 60)

    # Create output directory
    output_dir = REPO_ROOT / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "quickstart_T.png"

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot % ICU = 100 * T
    ax.plot(ts, 100 * T_traj, "b-", linewidth=2, label="% ICU (100 × T)")

    # Plot threshold line at 100 * T_MAX = 0.2 %
    ax.axhline(
        y=100 * T_MAX,
        color="r",
        linestyle="--",
        linewidth=1.5,
        label=f"ICU threshold (100 × T_max = {100 * T_MAX:.1f}%)",
    )

    ax.set_xlabel("Time [days]", fontsize=12)
    ax.set_ylabel("% ICU", fontsize=12)
    ax.set_title("SIDTHE Model: ICU Occupancy (u = 0, no control)", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_days)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

    print(f"Figure saved: {fig_path}")
    print(f"Figure saved: {fig_path.with_suffix('.pdf')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
