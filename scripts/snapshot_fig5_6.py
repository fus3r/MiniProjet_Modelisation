#!/usr/bin/env python3
"""
Snapshot for Figures 5 and 6 (MPC solution snapshots).

Note: Figures 5 and 6 in the paper show single MPC snapshot solutions,
which are internal solver outputs. This script generates comparable
diagnostics showing the MPC prediction horizon at a selected time step.

Usage (from repository root):
    python scripts/snapshot_fig5_6.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt

from sidthe.params import x0, theta_nom, T_MAX, T_NPI, DT, generate_scenarios
from sidthe.integrators import simulate_days
from sidthe.mpc import build_controller, MPCConfig


def main() -> int:
    print("=" * 60)
    print("MPC Snapshot (Figures 5/6 style)")
    print("=" * 60)
    
    # Setup
    config = MPCConfig(
        horizon_days=84,
        npi_days=T_NPI,
        enforce_icu_daily=True,
    )
    
    # Generate scenarios
    thetas_all, probs_all = generate_scenarios(theta_nom, rel=0.05)
    rng = np.random.default_rng(123)
    n_scenarios = 10
    idx = rng.choice(len(thetas_all), n_scenarios, replace=False)
    thetas = thetas_all[idx]
    probs = probs_all[idx]
    probs /= probs.sum()
    
    # Build controller
    solve = build_controller("robust", thetas, probs, config)
    
    # Solve at initial state
    result = solve(x0)
    
    if not result["status"]:
        print("MPC infeasible at x0!")
        return 1
    
    print(f"MPC solved successfully")
    print(f"  u0 = {result['u0_applied']:.4f}")
    print(f"  u_blocks = {result['u_blocks']}")
    print(f"  objective = {result['objective']:.6f}")
    
    # Simulate predicted trajectories for each scenario
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    u_blocks = result["u_blocks"]
    n_blocks = len(u_blocks)
    
    for s_idx in range(n_scenarios):
        theta_s = thetas[s_idx]
        from sidthe.params import SIDTHEParams
        theta_params = SIDTHEParams.from_array(theta_s)
        
        # Build control sequence (14 days per block)
        u_seq = np.repeat(u_blocks, T_NPI)
        
        # Simulate
        ts, xs = simulate_days(x0, theta_params, u_seq, dt=DT)
        
        # Plot ICU
        axes[0].plot(ts, xs[:, 3] * 100, alpha=0.5, linewidth=1)
    
    # ICU threshold
    axes[0].axhline(y=100 * T_MAX, color="r", linestyle="--", linewidth=1.5,
                    label=f"ICU threshold ({100*T_MAX:.1f}%)")
    axes[0].set_ylabel("% ICU", fontsize=11)
    axes[0].set_title("MPC Prediction Horizon (Robust, 10 scenarios)", fontsize=12)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)
    
    # Plot control blocks - extend last step to horizon end
    # u_blocks: array of planned controls for each block (length n)
    # T_NPI: block duration in days
    # horizon_days: end of horizon (e.g., 84)
    u_blocks = np.asarray(u_blocks).reshape(-1)
    t0 = np.arange(len(u_blocks)) * T_NPI              # [0, 14, 28, 42, 56, 70]
    t_step = np.r_[t0, config.horizon_days]            # [0, 14, 28, 42, 56, 70, 84] (len=n+1)
    u_step = np.r_[u_blocks, u_blocks[-1]]             # [u0, u1, ..., u5, u5]       (len=n+1)
    
    # Sanity checks to prevent regressions
    assert t_step[0] == 0, "t_step must start at 0"
    assert t_step[-1] == config.horizon_days, "t_step must end at horizon_days"
    assert len(t_step) == len(u_step), "t_step and u_step must have same length"
    
    # With where="post", y[i] is drawn from x[i] to x[i+1]
    # So y[5] (last block value) is drawn from x[5]=70 to x[6]=84
    # The last value y[6] is never drawn (no x[7]), which is fine since it's just a duplicate
    axes[1].fill_between(t_step, 0, u_step, step='post', alpha=0.3, color="steelblue")
    axes[1].step(t_step, u_step, where="post", linewidth=2, color="steelblue")
    axes[1].set_xlabel("Time [days]", fontsize=11)
    axes[1].set_ylabel("α reduction (u)", fontsize=11)
    axes[1].set_title("MPC Planned Control Sequence", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, config.horizon_days)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save
    output_dir = REPO_ROOT / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "snapshot_mpc_horizon.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    
    print(f"\nFigure saved: {fig_path}")
    print(f"Figure saved: {fig_path.with_suffix('.pdf')}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
