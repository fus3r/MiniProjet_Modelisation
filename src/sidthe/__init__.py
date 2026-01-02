"""
Package du modèle épidémique SIDTHE.

Fournit la dynamique EDO, l'intégration RK4, la gestion des paramètres,
le calcul d'ensemble sûr, et les contrôleurs MPC pour le modèle
compartimental SIDTHE avec entrées de contrôle.

Référence : Eq (4a)-(4f) de l'article pour la dynamique, page 3.
"""
from .params import SIDTHEParams, x0, theta_nom, DT, U_MAX, T_MAX, T_NPI
from .params import generate_scenarios
from .dynamics import rhs
from .integrators import rk4_step, simulate_days
from .safe_set import invariant_bounds, safe_set_intersection
from .mpc import MPCConfig, build_controller

__all__ = [
    "SIDTHEParams",
    "x0",
    "theta_nom",
    "DT",
    "U_MAX",
    "T_MAX",
    "T_NPI",
    "generate_scenarios",
    "rhs",
    "rk4_step",
    "simulate_days",
    "invariant_bounds",
    "safe_set_intersection",
    "MPCConfig",
    "build_controller",
]
