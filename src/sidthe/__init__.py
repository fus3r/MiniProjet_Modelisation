"""
SIDTHE epidemic model package.

Provides ODE dynamics, RK4 integration, parameter handling,
safe set computation, and MPC controllers for the SIDTHE
compartmental model with control inputs.

Reference: Paper Eq (4a)-(4f) for dynamics, page 3.
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
