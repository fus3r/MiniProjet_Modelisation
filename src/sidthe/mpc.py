"""
Contrôle prédictif (MPC) pour le modèle SIDTHE.

Références :
    - MPC nominal (vanilla) : Eq (12), page 4
    - SNMPC avec recourse : Eq (13), page 5
    - SNMPC robuste : Eq (14), page 5

Utilise CasADi pour la formulation NLP résolue par IPOPT.
Formulation single-shooting (pas de variables d'état dans le NLP).
"""
from dataclasses import dataclass, field
from typing import Callable

import casadi as ca
import numpy as np

from .params import SIDTHEParams, U_MAX, T_MAX, T_NPI, DT
from .safe_set import safe_set_intersection


@dataclass
class MPCConfig:
    """
    Configuration du MPC.

    horizon_days : horizon de prédiction [jours] (défaut 84 = 6 blocs de 14 jours)
    npi_days : durée d'un bloc NPI (contrôle constant par morceaux)
    u_max, t_max : bornes sur contrôle et occupation réa
    enforce_icu_daily : contrainte T ≤ t_max chaque jour (sinon fin de bloc)
    max_scenarios_* : réduction du nombre de scénarios si spécifié
    """

    horizon_days: int = 84
    dt: float = DT
    npi_days: int = T_NPI
    u_max: float = U_MAX
    t_max: float = T_MAX
    enforce_icu_daily: bool = True
    max_scenarios_robust: int | None = None
    max_scenarios_recourse: int | None = None
    scenario_seed: int = 123

    @property
    def n_blocks(self) -> int:
        """Nombre de blocs de contrôle sur l'horizon."""
        return self.horizon_days // self.npi_days


def _compute_constraint_violation(g: np.ndarray, lbg: list, ubg: list) -> float:
    """
    Calcule la violation max des contraintes (positif = violé).
    """
    g_arr = np.array(g).flatten()
    lbg_arr = np.array(lbg, dtype=float)
    ubg_arr = np.array(ubg, dtype=float)
    
    # Violation borne inf : lbg - g (positif si violé)
    lb_viol = np.where(np.isfinite(lbg_arr), lbg_arr - g_arr, -np.inf)
    # Violation borne sup : g - ubg (positif si violé)
    ub_viol = np.where(np.isfinite(ubg_arr), g_arr - ubg_arr, -np.inf)
    
    max_viol = max(np.max(lb_viol), np.max(ub_viol), 0.0)
    return float(max_viol)


def _build_casadi_rhs() -> ca.Function:
    """
    Construit la fonction CasADi du membre de droite des EDO SIDTHE.

    Retourne rhs(x, u, theta) -> dxdt (symbolique).
    """
    # Variables symboliques
    x = ca.SX.sym("x", 6)
    u = ca.SX.sym("u")
    theta = ca.SX.sym("theta", 6)

    # État : [S, I, D, T, H, E]
    S, I, D, T, H, E = x[0], x[1], x[2], x[3], x[4], x[5]

    # Paramètres : [alpha, gamma, lam, delta, sigma, tau]
    alpha = theta[0]
    gamma = theta[1]
    lam = theta[2]
    delta = theta[3]
    sigma = theta[4]
    tau = theta[5]

    # Termes récurrents (epsilon pour éviter div/0)
    eps = 1e-12
    lam_plus_gamma = lam + gamma + eps
    infection_rate = alpha * (1.0 - u) * S * I

    # Eq (4a)-(4f)
    Sdot = -infection_rate
    Idot = infection_rate - gamma * (1.0 + lam / lam_plus_gamma) * I
    Ddot = gamma * I - (delta + lam) * D
    Tdot = delta * D - (sigma + tau) * T
    Hdot = sigma * T + lam * D + lam * (gamma / lam_plus_gamma) * I
    Edot = tau * T

    dxdt = ca.vertcat(Sdot, Idot, Ddot, Tdot, Hdot, Edot)

    return ca.Function("rhs", [x, u, theta], [dxdt], ["x", "u", "theta"], ["dxdt"])


def _build_rk4_step(rhs_fn: ca.Function, dt: float) -> ca.Function:
    """
    Construit la fonction CasADi pour un pas RK4.

    Retourne step(x, u, theta) -> x_next.
    """
    x = ca.SX.sym("x", 6)
    u = ca.SX.sym("u")
    theta = ca.SX.sym("theta", 6)

    k1 = rhs_fn(x, u, theta)
    k2 = rhs_fn(x + 0.5 * dt * k1, u, theta)
    k3 = rhs_fn(x + 0.5 * dt * k2, u, theta)
    k4 = rhs_fn(x + dt * k3, u, theta)

    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return ca.Function("rk4_step", [x, u, theta], [x_next], ["x", "u", "theta"], ["x_next"])


def _reduce_scenarios(
    thetas_array: np.ndarray,
    probs: np.ndarray,
    max_scenarios: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Réduit le nombre de scénarios si max_scenarios est spécifié.

    Retourne (thetas_utilisés, probs_utilisées, n_utilisés).
    """
    n_scenarios = thetas_array.shape[0]
    if max_scenarios is None or n_scenarios <= max_scenarios:
        return thetas_array, probs, n_scenarios

    # Réduction déterministe (seed fixe pour reproductibilité)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_scenarios, max_scenarios, replace=False)
    indices = np.sort(indices)

    thetas_used = thetas_array[indices]
    probs_used = probs[indices]
    probs_used = probs_used / probs_used.sum()

    return thetas_used, probs_used, max_scenarios


def build_controller(
    mode: str,
    thetas_array: np.ndarray,
    probs: np.ndarray,
    config: MPCConfig | None = None,
    safe_set: bool = False,
) -> Callable[[np.ndarray], dict]:
    """
    Construit un contrôleur MPC pour le modèle SIDTHE.

    Modes disponibles :
        - "vanilla" : MPC nominal (un seul scénario), Eq (12)
        - "robust" : SNMPC robuste (contrôles partagés), Eq (14)
        - "recourse" : SNMPC avec recourse (non-anticipativité sur u0), Eq (13)

    Retourne une fonction solve(x0) -> dict avec les clés :
        'status', 'u0_applied', 'u_blocks', 'objective', 'iterations',
        'constraint_violation', 'n_scenarios_used'
    """
    if config is None:
        config = MPCConfig()

    # Construction de la dynamique CasADi
    rhs_fn = _build_casadi_rhs()
    step_fn = _build_rk4_step(rhs_fn, config.dt)

    # Bornes ensemble sûr (intersection sur les scénarios)
    if safe_set:
        safe_bounds = safe_set_intersection(thetas_array)
    else:
        safe_bounds = None

    # Construction du NLP selon le mode
    if mode == "vanilla":
        return _build_vanilla_controller(
            step_fn, thetas_array[0], config, safe_bounds
        )
    elif mode == "robust":
        return _build_robust_controller(
            step_fn, thetas_array, probs, config, safe_bounds
        )
    elif mode == "recourse":
        return _build_recourse_controller(
            step_fn, thetas_array, probs, config, safe_bounds
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'vanilla', 'robust', or 'recourse'.")


def _build_vanilla_controller(
    step_fn: ca.Function,
    theta: np.ndarray,
    config: MPCConfig,
    safe_bounds: dict | None,
) -> Callable[[np.ndarray], dict]:
    """
    Construit le MPC nominal (vanilla), Eq (12).

    Utilise un seul jeu de paramètres (nominal) pour la prédiction.
    """
    n_blocks = config.n_blocks
    npi_days = config.npi_days
    u_max = config.u_max
    t_max = config.t_max

    # Variables de décision : u_blocks (n_blocks,)
    u_blocks = ca.SX.sym("u_blocks", n_blocks)

    # Paramètre : état initial x0 (6,)
    x0_param = ca.SX.sym("x0", 6)

    # Conversion theta en CasADi
    theta_ca = ca.DM(theta)

    # Déroulement single-shooting
    x = x0_param
    cost = 0.0
    constraints = []
    lbg = []
    ubg = []

    for block in range(n_blocks):
        u_block = u_blocks[block]
        for day in range(npi_days):
            # Coût : l(u) = u² (page 6)
            cost += u_block ** 2

            # Pas RK4
            x = step_fn(x, u_block, theta_ca)

            # Contrainte réa : T ≤ t_max (quotidien)
            constraints.append(x[3])
            lbg.append(-ca.inf)
            ubg.append(t_max)

            # Non-négativité : x ≥ 0
            for i in range(6):
                constraints.append(x[i])
                lbg.append(0.0)
                ubg.append(ca.inf)

    # Contraintes terminales ensemble sûr (si activé)
    if safe_bounds is not None:
        constraints.append(x[0])  # S_N ≤ Smax
        lbg.append(-ca.inf)
        ubg.append(safe_bounds["Smax"])
        constraints.append(x[1])  # I_N <= Imax
        lbg.append(-ca.inf)
        ubg.append(safe_bounds["Imax"])
        constraints.append(x[2])  # D_N <= Dmax
        lbg.append(-ca.inf)
        ubg.append(safe_bounds["Dmax"])
        constraints.append(x[3])  # T_N <= Tmax
        lbg.append(-ca.inf)
        ubg.append(safe_bounds["Tmax"])

    # Formulation NLP
    nlp = {
        "x": u_blocks,
        "f": cost,
        "g": ca.vertcat(*constraints),
        "p": x0_param,
    }

    # Options solveur (sortie silencieuse)
    opts = {
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "print_time": 0,
        "ipopt.max_iter": 500,
        "ipopt.tol": 1e-6,
    }

    solver = ca.nlpsol("mpc_vanilla", "ipopt", nlp, opts)

    # Bornes sur les variables de décision
    lbx = [0.0] * n_blocks
    ubx = [u_max] * n_blocks

    def solve(x0: np.ndarray) -> dict:
        """Résout le MPC pour l'état initial donné."""
        try:
            sol = solver(
                x0=np.zeros(n_blocks),
                lbx=lbx,
                ubx=ubx,
                lbg=lbg,
                ubg=ubg,
                p=x0,
            )
            u_opt = np.array(sol["x"]).flatten()
            stats = solver.stats()

            return {
                "status": stats["success"],
                "u0_applied": float(u_opt[0]),
                "u_blocks": u_opt,
                "objective": float(sol["f"]),
                "iterations": stats.get("iter_count", 0),
                "constraint_violation": _compute_constraint_violation(sol["g"], lbg, ubg),
                "n_scenarios_used": 1,
            }
        except Exception as e:
            return {
                "status": False,
                "u0_applied": 0.0,
                "u_blocks": np.zeros(n_blocks),
                "objective": np.inf,
                "iterations": 0,
                "constraint_violation": np.inf,
                "n_scenarios_used": 1,
                "error": str(e),
            }

    return solve


def _build_robust_controller(
    step_fn: ca.Function,
    thetas_array: np.ndarray,
    probs: np.ndarray,
    config: MPCConfig,
    safe_bounds: dict | None,
) -> Callable[[np.ndarray], dict]:
    """
    Construit le SNMPC robuste, Eq (14).

    Les contrôles sont partagés entre TOUS les scénarios (pas de recourse).
    Les contraintes doivent être satisfaites pour chaque scénario.
    """
    n_blocks = config.n_blocks
    npi_days = config.npi_days
    u_max = config.u_max
    t_max = config.t_max
    enforce_icu_daily = config.enforce_icu_daily

    # Utilise exactement les scénarios fournis (pas de réduction cachée)
    thetas_used, probs_used, n_scenarios_used = _reduce_scenarios(
        thetas_array, probs,
        config.max_scenarios_robust,
        config.scenario_seed
    )

    # Variables de décision : u_blocks partagés pour tous les scénarios
    u_blocks = ca.SX.sym("u_blocks", n_blocks)

    # Paramètre : état initial x0 (6,)
    x0_param = ca.SX.sym("x0", 6)

    # Coût : u partagé ⇒ E[u²] = u², pas besoin de pondérer par scénario
    cost = 0.0
    for block in range(n_blocks):
        cost += npi_days * (u_blocks[block] ** 2)

    # Construction des contraintes sur tous les scénarios
    constraints = []
    lbg = []
    ubg = []

    for i in range(n_scenarios_used):
        theta_i = ca.DM(thetas_used[i])

        # Déroulement pour le scénario i
        x = x0_param

        for block in range(n_blocks):
            u_block = u_blocks[block]
            for day in range(npi_days):
                # Pas RK4
                x = step_fn(x, u_block, theta_i)

                # Contrainte réa quotidienne : T ≤ t_max
                if enforce_icu_daily:
                    constraints.append(x[3])
                    lbg.append(-ca.inf)
                    ubg.append(t_max)

            # Contraintes fin de bloc (non-négativité sur états clés)
            if not enforce_icu_daily:
                # Réa uniquement en fin de bloc
                constraints.append(x[3])
                lbg.append(-ca.inf)
                ubg.append(t_max)

            # Non-negativity at block end
            constraints.append(x[0])  # S >= 0
            lbg.append(0.0)
            ubg.append(ca.inf)
            constraints.append(x[1])  # I >= 0
            lbg.append(0.0)
            ubg.append(ca.inf)
            constraints.append(x[3])  # T >= 0
            lbg.append(0.0)
            ubg.append(ca.inf)

        # Terminal safe set constraints for this scenario (if enabled)
        if safe_bounds is not None:
            constraints.append(x[0])  # S_N <= Smax
            lbg.append(-ca.inf)
            ubg.append(safe_bounds["Smax"])
            constraints.append(x[1])  # I_N <= Imax
            lbg.append(-ca.inf)
            ubg.append(safe_bounds["Imax"])
            constraints.append(x[2])  # D_N <= Dmax
            lbg.append(-ca.inf)
            ubg.append(safe_bounds["Dmax"])
            constraints.append(x[3])  # T_N <= Tmax
            lbg.append(-ca.inf)
            ubg.append(safe_bounds["Tmax"])

    # NLP formulation
    nlp = {
        "x": u_blocks,
        "f": cost,
        "g": ca.vertcat(*constraints),
        "p": x0_param,
    }

    opts = {
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "print_time": 0,
        "ipopt.max_iter": 1000,
        "ipopt.tol": 1e-6,
    }

    solver = ca.nlpsol("mpc_robust", "ipopt", nlp, opts)

    lbx = [0.0] * n_blocks
    ubx = [u_max] * n_blocks

    def solve(x0: np.ndarray) -> dict:
        try:
            sol = solver(
                x0=np.ones(n_blocks) * u_max * 0.5,
                lbx=lbx,
                ubx=ubx,
                lbg=lbg,
                ubg=ubg,
                p=x0,
            )
            u_opt = np.array(sol["x"]).flatten()
            stats = solver.stats()

            return {
                "status": stats["success"],
                "u0_applied": float(u_opt[0]),
                "u_blocks": u_opt,
                "objective": float(sol["f"]),
                "iterations": stats.get("iter_count", 0),
                "constraint_violation": _compute_constraint_violation(sol["g"], lbg, ubg),
                "n_scenarios_used": n_scenarios_used,
            }
        except Exception as e:
            return {
                "status": False,
                "u0_applied": 0.0,
                "u_blocks": np.zeros(n_blocks),
                "objective": np.inf,
                "iterations": 0,
                "constraint_violation": np.inf,
                "n_scenarios_used": n_scenarios_used,
                "error": str(e),
            }

    return solve


def _build_recourse_controller(
    step_fn: ca.Function,
    thetas_array: np.ndarray,
    probs: np.ndarray,
    config: MPCConfig,
    safe_bounds: dict | None,
) -> Callable[[np.ndarray], dict]:
    """
    Construit le SNMPC avec recourse (non-anticipativité), Eq (13).

    Contrainte (13e) : seul u_0 est partagé, les blocs suivants peuvent
    différer selon le scénario.
    """
    n_blocks = config.n_blocks
    npi_days = config.npi_days
    u_max = config.u_max
    t_max = config.t_max
    enforce_icu_daily = config.enforce_icu_daily

    # Utilise exactement les scénarios fournis (pas de réduction cachée)
    thetas_used, probs_used, n_scenarios_used = _reduce_scenarios(
        thetas_array, probs,
        config.max_scenarios_recourse,
        config.scenario_seed
    )

    # Variables de décision :
    # - u0_shared : premier bloc, partagé (non-anticipativité Eq 13e)
    # - u_rest[i] : blocs 1 à n_blocks-1 pour le scénario i
    u0_shared = ca.SX.sym("u0_shared")

    u_rest_vars = []
    for i in range(n_scenarios_used):
        u_rest_i = ca.SX.sym(f"u_rest_{i}", n_blocks - 1)
        u_rest_vars.append(u_rest_i)

    # Paramètre : état initial x0 (6,)
    x0_param = ca.SX.sym("x0", 6)

    # Construction coût et contraintes
    total_cost = 0.0
    constraints = []
    lbg = []
    ubg = []

    for idx in range(n_scenarios_used):
        theta_i = ca.DM(thetas_used[idx])
        p_i = probs_used[idx]

        # Déroulement pour le scénario idx
        x = x0_param
        scenario_cost = 0.0

        for block in range(n_blocks):
            if block == 0:
                u_block = u0_shared  # Partagé (non-anticipativité)
            else:
                u_block = u_rest_vars[idx][block - 1]  # Spécifique au scénario

            for day in range(npi_days):
                scenario_cost += u_block ** 2
                x = step_fn(x, u_block, theta_i)

                # Contrainte réa quotidienne : T ≤ t_max
                if enforce_icu_daily:
                    constraints.append(x[3])
                    lbg.append(-ca.inf)
                    ubg.append(t_max)

            # Contraintes fin de bloc
            if not enforce_icu_daily:
                constraints.append(x[3])
                lbg.append(-ca.inf)
                ubg.append(t_max)

            # Non-negativity at block end
            constraints.append(x[0])  # S >= 0
            lbg.append(0.0)
            ubg.append(ca.inf)
            constraints.append(x[1])  # I >= 0
            lbg.append(0.0)
            ubg.append(ca.inf)
            constraints.append(x[3])  # T >= 0
            lbg.append(0.0)
            ubg.append(ca.inf)

        # Terminal safe set constraints for this scenario (if enabled)
        if safe_bounds is not None:
            constraints.append(x[0])  # S_N <= Smax
            lbg.append(-ca.inf)
            ubg.append(safe_bounds["Smax"])
            constraints.append(x[1])  # I_N <= Imax
            lbg.append(-ca.inf)
            ubg.append(safe_bounds["Imax"])
            constraints.append(x[2])  # D_N <= Dmax
            lbg.append(-ca.inf)
            ubg.append(safe_bounds["Dmax"])
            constraints.append(x[3])  # T_N <= Tmax
            lbg.append(-ca.inf)
            ubg.append(safe_bounds["Tmax"])

        # Coût espéré (Eq 13)
        total_cost += p_i * scenario_cost

    # Concaténation des variables de décision
    all_vars = [u0_shared] + u_rest_vars
    x_nlp = ca.vertcat(*all_vars)

    # NLP
    nlp = {
        "x": x_nlp,
        "f": total_cost,
        "g": ca.vertcat(*constraints),
        "p": x0_param,
    }

    opts = {
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "print_time": 0,
        "ipopt.max_iter": 1000,
        "ipopt.tol": 1e-6,
    }

    solver = ca.nlpsol("mpc_recourse", "ipopt", nlp, opts)

    # Bornes
    n_vars = 1 + n_scenarios_used * (n_blocks - 1)
    lbx = [0.0] * n_vars
    ubx = [u_max] * n_vars

    def solve(x0: np.ndarray) -> dict:
        try:
            sol = solver(
                x0=np.ones(n_vars) * u_max * 0.5,
                lbx=lbx,
                ubx=ubx,
                lbg=lbg,
                ubg=ubg,
                p=x0,
            )
            x_opt = np.array(sol["x"]).flatten()
            u0_opt = x_opt[0]
            stats = solver.stats()

            # Moyenne des u_blocks pour le reporting
            u_blocks_avg = np.zeros(n_blocks)
            u_blocks_avg[0] = u0_opt
            for block in range(1, n_blocks):
                block_vals = []
                for idx in range(n_scenarios_used):
                    offset = 1 + idx * (n_blocks - 1)
                    block_vals.append(x_opt[offset + block - 1])
                u_blocks_avg[block] = np.mean(block_vals)

            return {
                "status": stats["success"],
                "u0_applied": float(u0_opt),
                "u_blocks": u_blocks_avg,
                "objective": float(sol["f"]),
                "iterations": stats.get("iter_count", 0),
                "constraint_violation": _compute_constraint_violation(sol["g"], lbg, ubg),
                "n_scenarios_used": n_scenarios_used,
            }
        except Exception as e:
            return {
                "status": False,
                "u0_applied": 0.0,
                "u_blocks": np.zeros(n_blocks),
                "objective": np.inf,
                "iterations": 0,
                "constraint_violation": np.inf,
                "n_scenarios_used": n_scenarios_used,
                "error": str(e),
            }

    return solve
