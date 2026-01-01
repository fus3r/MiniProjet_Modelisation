"""
SIDTHE model dynamics.

Reference: ODE system Eq (4a)-(4f), page 3.
"""
import numpy as np

from .params import SIDTHEParams, U_MAX


def rhs(
    t: float,
    x: np.ndarray,
    u: float,
    params: SIDTHEParams,
) -> np.ndarray:
    """
    Right-hand side of the SIDTHE ODE system.

    Implements Eq (4a)-(4f) from the paper, page 3:
        Sdot = -α(1-u) S I                                   (4a)
        Idot = α(1-u) S I - γ*(1 + λ/(λ+γ))*I                (4b)
        Ddot = γ I - (δ+λ) D                                 (4c)
        Tdot = δ D - (σ+τ) T                                 (4d)
        Hdot = σ T + λ D + λ*(γ/(λ+γ))*I                     (4e)
        Edot = τ T                                           (4f)

    States are population fractions: x = [S, I, D, T, H, E].

    Parameters
    ----------
    t : float
        Current time (unused in autonomous system, kept for interface).
    x : np.ndarray
        State vector of shape (6,): [S, I, D, T, H, E].
    u : float
        Control input in [0, U_MAX]. Represents intervention intensity.
        (Paper: u in [0, u_max] with u_max = 0.75)
    params : SIDTHEParams
        Model parameters (α, γ, λ, δ, σ, τ).

    Returns
    -------
    dxdt : np.ndarray
        Time derivative of state, shape (6,).
    """
    # Assertions for safety
    assert x.shape == (6,), f"State x must have shape (6,), got {x.shape}"
    u = float(np.clip(u, 0.0, U_MAX))  # Clamp control to [0, U_MAX]

    # Unpack state
    S, I, D, T, H, E = x

    # Unpack parameters
    alpha = params.alpha
    gamma = params.gamma
    lam = params.lam
    delta = params.delta
    sigma = params.sigma
    tau = params.tau

    # Precompute common terms
    lam_plus_gamma = lam + gamma
    assert lam_plus_gamma > 0, "λ + γ must be > 0 to avoid division by zero"
    infection_rate = alpha * (1.0 - u) * S * I

    # Eq (4a)-(4f)
    Sdot = -infection_rate
    Idot = infection_rate - gamma * (1.0 + lam / lam_plus_gamma) * I
    Ddot = gamma * I - (delta + lam) * D
    Tdot = delta * D - (sigma + tau) * T
    Hdot = sigma * T + lam * D + lam * (gamma / lam_plus_gamma) * I
    Edot = tau * T

    return np.array([Sdot, Idot, Ddot, Tdot, Hdot, Edot], dtype=np.float64)
