#!/usr/bin/env python3
"""
Snapshot pour les Figures 5 et 6 (instantanés de solution MPC).

Note : Les figures 5 et 6 de l'article montrent des solutions instantanées du MPC,
qui sont des sorties internes du solveur. Ce script génère des diagnostics
comparables montrant l'horizon de prédiction MPC à un instant donné.

Usage (depuis la racine du dépôt) :
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
    print("Snapshot MPC (style Figures 5/6)")
    print("=" * 60)
    
    # Configuration
    config = MPCConfig(
        horizon_days=84,
        npi_days=T_NPI,
        enforce_icu_daily=True,
    )
    
    # Génération des scénarios
    thetas_all, probs_all = generate_scenarios(theta_nom, rel=0.05)
    rng = np.random.default_rng(123)
    n_scenarios = 10
    idx = rng.choice(len(thetas_all), n_scenarios, replace=False)
    thetas = thetas_all[idx]
    probs = probs_all[idx]
    probs /= probs.sum()
    
    # Construction du contrôleur
    solve = build_controller("robust", thetas, probs, config)
    
    # Résolution à l'état initial
    result = solve(x0)
    
    if not result["status"]:
        print("MPC infaisable en x0 !")
        return 1
    
    print(f"MPC résolu avec succès")
    print(f"  u0 = {result['u0_applied']:.4f}")
    print(f"  u_blocks = {result['u_blocks']}")
    print(f"  objective = {result['objective']:.6f}")
    
    # Simule les trajectoires prédites pour chaque scénario
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    u_blocks = result["u_blocks"]
    n_blocks = len(u_blocks)
    
    for s_idx in range(n_scenarios):
        theta_s = thetas[s_idx]
        from sidthe.params import SIDTHEParams
        theta_params = SIDTHEParams.from_array(theta_s)
        
        # Construit la séquence de contrôle (14 jours par bloc)
        u_seq = np.repeat(u_blocks, T_NPI)
        
        # Simule
        ts, xs = simulate_days(x0, theta_params, u_seq, dt=DT)
        
        # Trace réa
        axes[0].plot(ts, xs[:, 3] * 100, alpha=0.5, linewidth=1)
    
    # Seuil réa
    axes[0].axhline(y=100 * T_MAX, color="r", linestyle="--", linewidth=1.5,
                    label=f"ICU threshold ({100*T_MAX:.1f}%)")
    axes[0].set_ylabel("% ICU", fontsize=11)
    axes[0].set_title("MPC Prediction Horizon (Robust, 10 scenarios)", fontsize=12)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)
    
    # Tracé des blocs de contrôle - prolonge le dernier jusqu'à la fin de l'horizon
    # u_blocks : contrôles planifiés pour chaque bloc (longueur n)
    # T_NPI : durée d'un bloc en jours
    # horizon_days : fin de l'horizon (ex: 84)
    u_blocks = np.asarray(u_blocks).reshape(-1)
    t0 = np.arange(len(u_blocks)) * T_NPI              # [0, 14, 28, 42, 56, 70]
    t_step = np.r_[t0, config.horizon_days]            # [0, 14, 28, 42, 56, 70, 84] (len=n+1)
    u_step = np.r_[u_blocks, u_blocks[-1]]             # [u0, u1, ..., u5, u5]       (len=n+1)
    
    # Vérifications pour éviter les régressions
    assert t_step[0] == 0, "t_step doit commencer à 0"
    assert t_step[-1] == config.horizon_days, "t_step doit finir à horizon_days"
    assert len(t_step) == len(u_step), "t_step et u_step doivent avoir la même longueur"
    
    # Avec where="post", y[i] est tracé de x[i] à x[i+1]
    # Donc y[5] (dernière valeur de bloc) est tracé de x[5]=70 à x[6]=84
    # La dernière valeur y[6] n'est jamais tracée (pas de x[7]), ce qui est ok car c'est un doublon
    axes[1].fill_between(t_step, 0, u_step, step='post', alpha=0.3, color="steelblue")
    axes[1].step(t_step, u_step, where="post", linewidth=2, color="steelblue")
    axes[1].set_xlabel("Time [days]", fontsize=11)
    axes[1].set_ylabel("α reduction (u)", fontsize=11)
    axes[1].set_title("MPC Planned Control Sequence", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, config.horizon_days)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Sauvegarde
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
