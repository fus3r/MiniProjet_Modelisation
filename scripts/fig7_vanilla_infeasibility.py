#!/usr/bin/env python3
"""
Figure 7 : Démonstration de l'infaisabilité du MPC vanilla.

Montre que le MPC nominal (utilisant uniquement les paramètres nominaux)
peut devenir infaisable quand les vrais paramètres diffèrent du nominal.

Référence : Figure 7 de l'article, page 7.

Usage (depuis la racine du dépôt) :
    python3 scripts/fig7_vanilla_infeasibility.py
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
    """Simule le MPC vanilla avec plusieurs scénarios réels."""
    print("=" * 60)
    print("Figure 7 : Démonstration infaisabilité MPC vanilla")
    print("=" * 60)

    # Configuration
    n_true_scenarios = 25
    total_days = 350
    config = MPCConfig(horizon_days=84, npi_days=T_NPI)

    # Génération des scénarios "vrais"
    thetas_all, _ = generate_scenarios(theta_nom, rel=0.05)
    rng = np.random.default_rng(42)  # Seed moderne pour reproductibilité
    true_indices = rng.choice(len(thetas_all), n_true_scenarios, replace=False)

    # Construction du contrôleur vanilla (paramètres nominaux seulement)
    nominal_theta = theta_nom.to_array().reshape(1, 6)
    nominal_prob = np.array([1.0])
    solve_mpc = build_controller("vanilla", nominal_theta, nominal_prob, config)

    print(f"Simulation de {n_true_scenarios} scénarios réels...")
    print(f"Contrôleur : MPC vanilla avec paramètres nominaux")
    print(f"Horizon : {config.horizon_days} jours, bloc NPI : {T_NPI} jours")
    print("-" * 60)

    # Stockage des trajectoires
    trajectories = []  # Liste de (ts, xs, us, infeasible_day)

    for idx, true_idx in enumerate(true_indices):
        true_theta = SIDTHEParams.from_array(thetas_all[true_idx])

        # Initialisation
        x_current = x0.copy()
        xs_traj = [x_current.copy()]
        us_traj = []
        ts_traj = [0.0]
        infeasible_day = None

        day = 0
        while day < total_days:
            # Décision MPC au début de chaque bloc NPI
            if day % T_NPI == 0:
                result = solve_mpc(x_current)

                if not result["status"]:
                    # Infaisable - marquer et arrêter
                    infeasible_day = day
                    break

                u_applied = result["u0_applied"]
            else:
                # Continue avec le même contrôle
                pass

            # Simule un jour avec les VRAIS paramètres
            u_seq = np.array([u_applied])
            _, xs_step = simulate_days(x_current, true_theta, u_seq, dt=DT)
            x_current = xs_step[-1]

            xs_traj.append(x_current.copy())
            us_traj.append(u_applied)
            ts_traj.append(day + 1)
            day += 1

        trajectories.append({
            "ts": np.array(ts_traj),
            "xs": np.array(xs_traj),
            "us": np.array(us_traj) if us_traj else np.array([]),
            "infeasible_day": infeasible_day,
            "true_idx": true_idx,
        })

        status = "INFEASIBLE at day " + str(infeasible_day) if infeasible_day else "OK"
        print(f"  Scenario {idx+1:2d}: {status}")

    # Comptage faisable/infaisable
    n_infeasible = sum(1 for t in trajectories if t["infeasible_day"] is not None)
    n_feasible = n_true_scenarios - n_infeasible
    print("-" * 60)
    print(f"Résultats : {n_feasible} faisables, {n_infeasible} infaisables")

    # Création de la figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Haut : % réa
    ax1 = axes[0]
    for traj in trajectories:
        ts = traj["ts"]
        T_vals = traj["xs"][:, 3] * 100  # % ICU

        if traj["infeasible_day"] is not None:
            # Plot trajectory until infeasibility
            ax1.plot(ts, T_vals, "gray", alpha=0.5, linewidth=0.8)
            # Mark infeasibility point
            inf_idx = traj["infeasible_day"]
            if inf_idx < len(T_vals):
                ax1.scatter([inf_idx], [T_vals[inf_idx]], color="red", marker="x", s=50, zorder=5)
        else:
            ax1.plot(ts, T_vals, "gray", alpha=0.5, linewidth=0.8)

    # ICU threshold
    ax1.axhline(y=100 * T_MAX, color="r", linestyle="--", linewidth=1.5,
                label=f"ICU threshold ({100*T_MAX:.1f}%)")
    ax1.set_ylabel("% ICU", fontsize=12)
    ax1.set_title("Vanilla MPC: ICU Occupancy (× = infeasibility)", fontsize=14)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Bottom: Alpha reduction (u)
    ax2 = axes[1]
    for traj in trajectories:
        if len(traj["us"]) > 0:
            ts_u = traj["ts"][:-1]  # Control at each day
            us = traj["us"]

            if traj["infeasible_day"] is not None:
                ax2.plot(ts_u, us, "gray", alpha=0.5, linewidth=0.8)
                # Mark infeasibility
                inf_idx = traj["infeasible_day"]
                if inf_idx < len(us):
                    ax2.scatter([inf_idx], [us[inf_idx]], color="red", marker="x", s=50, zorder=5)
            else:
                ax2.plot(ts_u, us, "gray", alpha=0.5, linewidth=0.8)

    ax2.set_xlabel("Time [days]", fontsize=12)
    ax2.set_ylabel("α reduction (u)", fontsize=12)
    ax2.set_title("Vanilla MPC: Control Input", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, total_days)
    ax2.set_ylim(0, 1)

    plt.tight_layout()

    # Save figure (PNG + PDF)
    output_dir = REPO_ROOT / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "fig7_vanilla_infeasibility.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

    print(f"\nFigure saved: {fig_path}")
    print(f"Figure saved: {fig_path.with_suffix('.pdf')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
