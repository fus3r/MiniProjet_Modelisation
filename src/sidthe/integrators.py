"""
Intégrateurs numériques pour le modèle SIDTHE.

Runge-Kutta d'ordre 4 avec pas dt=1 jour (cf. article page 6).
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
    Un pas d'intégration RK4 (Runge-Kutta ordre 4).

    Le contrôle u est supposé constant sur le pas.
    Retourne l'état à t+dt.
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
    Simule le modèle SIDTHE sur plusieurs jours.

    Applique u_seq[k] le jour k et intègre par RK4.
    Retourne (ts, xs) avec xs[k] = état au temps ts[k].
    """
    # Import local pour éviter la circularité
    from .dynamics import rhs

    N = len(u_seq)
    xs = np.zeros((N + 1, 6), dtype=np.float64)
    ts = np.arange(N + 1, dtype=np.float64) * dt

    xs[0] = x0.copy()

    for k in range(N):
        xs[k + 1] = rk4_step(rhs, ts[k], xs[k], u_seq[k], params, dt)

    return ts, xs
