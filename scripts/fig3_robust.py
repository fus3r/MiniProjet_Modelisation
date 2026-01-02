#!/usr/bin/env python3
"""
Figure 3 : Simulation SNMPC robuste.

Démontre le SNMPC robuste basé scénarios (Eq 14) où la séquence
de contrôle est partagée entre tous les scénarios.

Référence : Figure 3 de l'article, page 6.

Usage (depuis la racine du dépôt) :
    python3 scripts/fig3_robust.py
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
    """Lance la simulation SNMPC robuste."""
    print("=" * 60)
    print("Figure 3 : Simulation SNMPC robuste")
    print("=" * 60)

    # Configuration
    n_sim_scenarios = 25  # Nombre de scénarios à simuler
    total_days = 350
    
    # Sélection EXPLICITE des scénarios pour le MPC (pas de sous-échantillonnage caché)
    n_mpc_scenarios = 20  # Nombre de scénarios pour le MPC robuste
    scenario_seed = 123   # Pour reproductibilité
    
    # Config : pas de réduction interne (max_scenarios_robust=None)
    config = MPCConfig(
        horizon_days=84,
        npi_days=T_NPI,
        enforce_icu_daily=True,
        max_scenarios_robust=None,  # Utilise exactement ce qu'on passe
    )

    # Génère les 729 scénarios
    thetas_all, probs_all = generate_scenarios(theta_nom, rel=0.05)

    # EXPLICITE : sélection d'un sous-ensemble pour le MPC robuste
    rng = np.random.default_rng(scenario_seed)
    mpc_indices = rng.choice(len(thetas_all), n_mpc_scenarios, replace=False)
    thetas_mpc = thetas_all[mpc_indices]
    probs_mpc = probs_all[mpc_indices]
    probs_mpc = probs_mpc / probs_mpc.sum()

    print(f"Construction du contrôleur robuste...")
    print(f"  - n_mpc_scenarios : {n_mpc_scenarios}")
    print(f"  - scenario_seed : {scenario_seed}")
    print(f"  - enforce_icu_daily : {config.enforce_icu_daily}")
    solve_mpc = build_controller("robust", thetas_mpc, probs_mpc, config)
    
    # Vérifie le nombre réel de scénarios utilisés
    test_result = solve_mpc(x0)
    print(f"  - n_scenarios_used (effectif) : {test_result['n_scenarios_used']}")

    # Sélection des scénarios pour la simulation
    sim_rng = np.random.default_rng(42)
    sim_indices = sim_rng.choice(len(thetas_all), n_sim_scenarios, replace=False)

    print(f"Simulation de {n_sim_scenarios} scénarios réels...")
    print(f"Horizon : {config.horizon_days} jours, bloc NPI : {T_NPI} jours")
    print("-" * 60)

    # Stockage
    trajectories = []

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
                    # Repli sur contrôle max si infaisable
                    u_applied = config.u_max
                    print(f"  Scénario {idx+1} : infaisable au jour {day}, utilise u_max")

            # Simule un jour
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

        # Progression
        if (idx + 1) % 5 == 0:
            print(f"  Terminé {idx+1}/{n_sim_scenarios} scénarios")

    # Rapport violation max réa
    max_T_all = max(traj["xs"][:, 3].max() for traj in trajectories)
    max_violation = max_T_all - T_MAX
    print("-" * 60)
    print(f"max(T - T_MAX) = {max_violation:.6e}")
    if max_violation > 1e-6:
        print(f"  ATTENTION : Seuil réa dépassé !")

    # Création de la figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Haut : % réa
    ax1 = axes[0]
    for traj in trajectories:
        ts = traj["ts"]
        T_vals = traj["xs"][:, 3] * 100
        ax1.plot(ts, T_vals, "steelblue", alpha=0.4, linewidth=0.8)

    ax1.axhline(y=100 * T_MAX, color="r", linestyle="--", linewidth=1.5,
                label=f"ICU threshold ({100*T_MAX:.1f}%)")
    ax1.set_ylabel("% ICU", fontsize=12)
    ax1.set_title("Robust SNMPC (Eq 14): ICU Occupancy", fontsize=14)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Bottom: Control
    ax2 = axes[1]
    for traj in trajectories:
        ts_u = traj["ts"][:-1]
        ax2.plot(ts_u, traj["us"], "steelblue", alpha=0.4, linewidth=0.8)

    ax2.set_xlabel("Time [days]", fontsize=12)
    ax2.set_ylabel("α reduction (u)", fontsize=12)
    ax2.set_title("Robust SNMPC: Control Input", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, total_days)
    ax2.set_ylim(0, 1)

    plt.tight_layout()

    # Save (PNG + PDF)
    output_dir = REPO_ROOT / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "fig3_robust.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

    print(f"\nFigure saved: {fig_path}")
    print(f"Figure saved: {fig_path.with_suffix('.pdf')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
