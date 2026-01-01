"""
Model Predictive Control (MPC) for SIDTHE epidemic model.

Reference:
    - Vanilla MPC: Eq (12), page 4
    - Scenario-based SNMPC with recourse: Eq (13), page 5
    - Robust SNMPC: Eq (14), page 5

Uses CasADi for NLP formulation with IPOPT solver.
Single-shooting formulation (no state variables in NLP).
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
    MPC configuration parameters.

    Attributes
    ----------
    horizon_days : int
        Prediction horizon in days (default 84 = 6 blocks of 14 days).
    dt : float
        Integration time step [days] (default 1.0).
    npi_days : int
        NPI block duration [days] (default 14, piecewise-constant control).
    u_max : float
        Maximum control intensity (default 0.75).
    t_max : float
        ICU capacity constraint on T state (default 0.002).
    enforce_icu_daily : bool
        If True, enforce T <= t_max every day. If False, only at block ends.
    max_scenarios_robust : int | None
        Max scenarios for robust MPC. None = use all provided scenarios.
    max_scenarios_recourse : int | None
        Max scenarios for recourse MPC. None = use all provided scenarios.
    scenario_seed : int
        Seed for deterministic scenario reduction (if max_scenarios_* is set).
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
        """Number of control blocks in horizon."""
        return self.horizon_days // self.npi_days


def _compute_constraint_violation(g: np.ndarray, lbg: list, ubg: list) -> float:
    """
    Compute max constraint violation accounting for bounds.
    
    violation = max(max(lbg - g), max(g - ubg), 0)
    """
    g_arr = np.array(g).flatten()
    lbg_arr = np.array(lbg, dtype=float)
    ubg_arr = np.array(ubg, dtype=float)
    
    # Lower bound violation: lbg - g (positive if violated)
    lb_viol = np.where(np.isfinite(lbg_arr), lbg_arr - g_arr, -np.inf)
    # Upper bound violation: g - ubg (positive if violated)
    ub_viol = np.where(np.isfinite(ubg_arr), g_arr - ubg_arr, -np.inf)
    
    max_viol = max(np.max(lb_viol), np.max(ub_viol), 0.0)
    return float(max_viol)


def _build_casadi_rhs() -> ca.Function:
    """
    Build CasADi function for SIDTHE ODE right-hand side.

    Implements Eq (4a)-(4f) from paper, page 3.

    Returns
    -------
    rhs_fn : ca.Function
        CasADi function: rhs(x, u, theta) -> dxdt
        where x is (6,), u is scalar, theta is (6,).
    """
    # Symbolic variables
    x = ca.SX.sym("x", 6)
    u = ca.SX.sym("u")
    theta = ca.SX.sym("theta", 6)

    # Unpack state: [S, I, D, T, H, E]
    S, I, D, T, H, E = x[0], x[1], x[2], x[3], x[4], x[5]

    # Unpack parameters: [alpha, gamma, lam, delta, sigma, tau]
    alpha = theta[0]
    gamma = theta[1]
    lam = theta[2]
    delta = theta[3]
    sigma = theta[4]
    tau = theta[5]

    # Common terms (avoid division by zero with small epsilon)
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
    Build CasADi function for one RK4 integration step.

    Parameters
    ----------
    rhs_fn : ca.Function
        Right-hand side function from _build_casadi_rhs().
    dt : float
        Time step size.

    Returns
    -------
    step_fn : ca.Function
        CasADi function: step(x, u, theta) -> x_next
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
    Reduce scenarios if max_scenarios is specified.

    Returns (thetas_used, probs_used, n_used).
    """
    n_scenarios = thetas_array.shape[0]
    if max_scenarios is None or n_scenarios <= max_scenarios:
        return thetas_array, probs, n_scenarios

    # Deterministic reduction using seed
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
    Build MPC controller for SIDTHE model.

    Parameters
    ----------
    mode : str
        Controller mode:
        - "vanilla": Nominal MPC using first scenario only, Eq (12)
        - "robust": Robust SNMPC with shared controls, Eq (14)
        - "recourse": SNMPC with non-anticipativity on first block, Eq (13)
    thetas_array : np.ndarray
        Parameter scenarios, shape (n_scenarios, 6).
    probs : np.ndarray
        Scenario probabilities, shape (n_scenarios,).
    config : MPCConfig, optional
        MPC configuration (default: MPCConfig()).
    safe_set : bool, optional
        Whether to enforce terminal safe set constraints (default False).

    Returns
    -------
    solve : Callable
        Function solve(x0) -> dict with keys:
        - 'status': bool, True if solved successfully
        - 'u0_applied': float, control to apply now
        - 'u_blocks': np.ndarray, full control sequence
        - 'objective': float, optimal cost
        - 'iterations': int, solver iterations
        - 'constraint_violation': float, max constraint violation
        - 'n_scenarios_used': int, actual number of scenarios used
    """
    if config is None:
        config = MPCConfig()

    # Build CasADi dynamics
    rhs_fn = _build_casadi_rhs()
    step_fn = _build_rk4_step(rhs_fn, config.dt)

    # Safe set bounds (intersection over scenarios)
    if safe_set:
        safe_bounds = safe_set_intersection(thetas_array)
    else:
        safe_bounds = None

    # Build NLP based on mode
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
    Build vanilla (nominal) MPC controller.

    Implements Eq (12) from paper, page 4.
    Uses single nominal parameter set for prediction.
    """
    n_blocks = config.n_blocks
    npi_days = config.npi_days
    u_max = config.u_max
    t_max = config.t_max

    # Decision variables: u_blocks (n_blocks,)
    u_blocks = ca.SX.sym("u_blocks", n_blocks)

    # Parameter: initial state x0 (6,)
    x0_param = ca.SX.sym("x0", 6)

    # Convert theta to CasADi
    theta_ca = ca.DM(theta)

    # Single-shooting rollout
    x = x0_param
    cost = 0.0
    constraints = []
    lbg = []
    ubg = []

    for block in range(n_blocks):
        u_block = u_blocks[block]
        for day in range(npi_days):
            # Cost: l(u) = u^2 (page 6)
            cost += u_block ** 2

            # RK4 step
            x = step_fn(x, u_block, theta_ca)

            # ICU constraint: T <= t_max (daily)
            constraints.append(x[3])
            lbg.append(-ca.inf)
            ubg.append(t_max)

            # Non-negativity: x >= 0
            for i in range(6):
                constraints.append(x[i])
                lbg.append(0.0)
                ubg.append(ca.inf)

    # Terminal safe set constraints (if enabled)
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

    # Solver options
    opts = {
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "print_time": 0,
        "ipopt.max_iter": 500,
        "ipopt.tol": 1e-6,
    }

    solver = ca.nlpsol("mpc_vanilla", "ipopt", nlp, opts)

    # Bounds on decision variables
    lbx = [0.0] * n_blocks
    ubx = [u_max] * n_blocks

    def solve(x0: np.ndarray) -> dict:
        """Solve MPC for given initial state."""
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
    Build robust SNMPC controller.

    Implements Eq (14) from paper, page 5.
    Control sequence is shared across ALL scenarios (no recourse).
    Constraints must be satisfied for ALL scenarios.

    Cost is simplified: since u is shared, E[u^2] = u^2.
    Constraints are enforced for every scenario.
    """
    n_blocks = config.n_blocks
    npi_days = config.npi_days
    u_max = config.u_max
    t_max = config.t_max
    enforce_icu_daily = config.enforce_icu_daily

    # Use exactly the scenarios provided (no hidden reduction)
    thetas_used, probs_used, n_scenarios_used = _reduce_scenarios(
        thetas_array, probs,
        config.max_scenarios_robust,
        config.scenario_seed
    )

    # Decision variables: u_blocks shared for all scenarios
    u_blocks = ca.SX.sym("u_blocks", n_blocks)

    # Parameter: initial state x0 (6,)
    x0_param = ca.SX.sym("x0", 6)

    # Cost: since u is shared across scenarios, cost = sum_k (npi_days * u_k^2)
    # (No need to compute per-scenario, it's the same for all)
    cost = 0.0
    for block in range(n_blocks):
        cost += npi_days * (u_blocks[block] ** 2)

    # Build constraints over all scenarios
    constraints = []
    lbg = []
    ubg = []

    for i in range(n_scenarios_used):
        theta_i = ca.DM(thetas_used[i])

        # Rollout for scenario i
        x = x0_param

        for block in range(n_blocks):
            u_block = u_blocks[block]
            for day in range(npi_days):
                # RK4 step
                x = step_fn(x, u_block, theta_i)

                # Daily ICU constraint: T <= t_max
                if enforce_icu_daily:
                    constraints.append(x[3])
                    lbg.append(-ca.inf)
                    ubg.append(t_max)

            # End-of-block constraints (non-negativity for key states)
            if not enforce_icu_daily:
                # ICU only at block end
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
    Build SNMPC controller with recourse (non-anticipativity).

    Implements Eq (13) from paper, page 5.
    Non-anticipativity constraint (13e): first control u_0 is shared,
    subsequent controls can differ per scenario.
    """
    n_blocks = config.n_blocks
    npi_days = config.npi_days
    u_max = config.u_max
    t_max = config.t_max
    enforce_icu_daily = config.enforce_icu_daily

    # Use exactly the scenarios provided (no hidden reduction)
    thetas_used, probs_used, n_scenarios_used = _reduce_scenarios(
        thetas_array, probs,
        config.max_scenarios_recourse,
        config.scenario_seed
    )

    # Decision variables:
    # - u0_shared: first block control, shared (non-anticipativity Eq 13e)
    # - u_rest[i]: blocks 1 to n_blocks-1 for scenario i
    u0_shared = ca.SX.sym("u0_shared")

    u_rest_vars = []
    for i in range(n_scenarios_used):
        u_rest_i = ca.SX.sym(f"u_rest_{i}", n_blocks - 1)
        u_rest_vars.append(u_rest_i)

    # Parameter: initial state x0 (6,)
    x0_param = ca.SX.sym("x0", 6)

    # Build cost and constraints
    total_cost = 0.0
    constraints = []
    lbg = []
    ubg = []

    for idx in range(n_scenarios_used):
        theta_i = ca.DM(thetas_used[idx])
        p_i = probs_used[idx]

        # Rollout for scenario idx
        x = x0_param
        scenario_cost = 0.0

        for block in range(n_blocks):
            if block == 0:
                u_block = u0_shared  # Shared (non-anticipativity)
            else:
                u_block = u_rest_vars[idx][block - 1]  # Scenario-specific

            for day in range(npi_days):
                scenario_cost += u_block ** 2
                x = step_fn(x, u_block, theta_i)

                # Daily ICU constraint: T <= t_max
                if enforce_icu_daily:
                    constraints.append(x[3])
                    lbg.append(-ca.inf)
                    ubg.append(t_max)

            # End-of-block constraints
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

        # Expected cost (Eq 13)
        total_cost += p_i * scenario_cost

    # Concatenate all decision variables
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

    # Bounds
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

            # Extract average u_blocks for reporting
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
