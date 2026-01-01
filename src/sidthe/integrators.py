"""
Numerical integrators for the SIDTHE model.

Implements RK4 (Runge-Kutta 4th order) with dt=1 day as specified on page 6.
"""
from typing import Callable

import numpy as np

from .params import SIDTHEParams


def rk4_step(
    rhs_fn: Callable[[float, np.ndarray, float, SIDTHEParams], np.ndarray],
    t: float,
    x: np.ndarray,
    u: float,
    params: SIDTHEParams,
    dt: float,
) -> np.ndarray:
    """
    Perform one RK4 (Runge-Kutta 4th order) integration step.

    Parameters
    ----------
    rhs_fn : Callable
        Right-hand side function with signature rhs(t, x, u, params) -> dxdt.
    t : float
        Current time.
    x : np.ndarray
        Current state vector, shape (6,).
    u : float
        Control input (constant over the step).
    params : SIDTHEParams
        Model parameters.
    dt : float
        Time step size.

    Returns
    -------
    x_next : np.ndarray
        State at time t + dt, shape (6,).
    """
    k1 = rhs_fn(t, x, u, params)
    k2 = rhs_fn(t + 0.5 * dt, x + 0.5 * dt * k1, u, params)
    k3 = rhs_fn(t + 0.5 * dt, x + 0.5 * dt * k2, u, params)
    k4 = rhs_fn(t + dt, x + dt * k3, u, params)

    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x_next


def simulate_days(
    x0: np.ndarray,
    params: SIDTHEParams,
    u_seq: np.ndarray,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate the SIDTHE model over a sequence of days.

    Uses RK4 integration with the specified time step (default dt=1 day).

    Parameters
    ----------
    x0 : np.ndarray
        Initial state vector, shape (6,).
    params : SIDTHEParams
        Model parameters.
    u_seq : np.ndarray
        Control sequence, shape (N,). u_seq[k] is the control applied on day k.
    dt : float, optional
        Time step for RK4 integration (default 1.0 day).

    Returns
    -------
    ts : np.ndarray
        Time points, shape (N+1,). ts[0] = 0, ts[k] = k * dt.
    xs : np.ndarray
        State trajectory, shape (N+1, 6). xs[k] is the state at time ts[k].
    """
    # Import here to avoid circular imports
    from .dynamics import rhs

    N = len(u_seq)
    xs = np.zeros((N + 1, 6), dtype=np.float64)
    ts = np.arange(N + 1, dtype=np.float64) * dt

    xs[0] = x0.copy()

    for k in range(N):
        xs[k + 1] = rk4_step(rhs, ts[k], xs[k], u_seq[k], params, dt)

    return ts, xs
