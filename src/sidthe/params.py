"""
SIDTHE model parameters and constants.

Reference: Paper Eq (5), page 3 for parameter order θ=[α,γ,λ,δ,σ,τ]^T.
Numerical values from page 6.
"""
from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass
class SIDTHEParams:
    """
    SIDTHE model epidemiological parameters.

    Parameter order follows Eq (5), page 3: θ = [α, γ, λ, δ, σ, τ]^T.

    Attributes
    ----------
    alpha : float
        Transmission rate (α).
    gamma : float
        Rate of detection/diagnosis (γ).
    lam : float
        Recovery rate for diagnosed (λ). Named 'lam' to avoid Python keyword.
    delta : float
        Rate of worsening to critical (δ).
    sigma : float
        Recovery rate from critical (σ).
    tau : float
        Death rate from critical (τ).
    """

    alpha: float
    gamma: float
    lam: float
    delta: float
    sigma: float
    tau: float

    def to_array(self) -> np.ndarray:
        """Convert parameters to numpy array, shape (6,)."""
        return np.array(
            [self.alpha, self.gamma, self.lam, self.delta, self.sigma, self.tau],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "SIDTHEParams":
        """
        Create SIDTHEParams from numpy array.

        Parameters
        ----------
        arr : np.ndarray
            Array of shape (6,) with parameters [α, γ, λ, δ, σ, τ].

        Returns
        -------
        SIDTHEParams
            Instance with the given parameter values.
        """
        assert arr.shape == (6,), f"Expected shape (6,), got {arr.shape}"
        return cls(
            alpha=float(arr[0]),
            gamma=float(arr[1]),
            lam=float(arr[2]),
            delta=float(arr[3]),
            sigma=float(arr[4]),
            tau=float(arr[5]),
        )


# ---------------------------------------------------------------------------
# Simulation constants (page 6)
# ---------------------------------------------------------------------------

DT: float = 1.0  # Time step [days] for RK4 integration
U_MAX: float = 0.75  # Maximum control intensity
T_MAX: float = 0.002  # ICU capacity threshold (fraction)
T_NPI: int = 14  # Non-pharmaceutical intervention delay [days]


# ---------------------------------------------------------------------------
# Initial condition x0 (page 6)
# Order: [S, I, D, T, H, E]
# ---------------------------------------------------------------------------

x0: np.ndarray = np.array(
    [0.99, 0.008, 1.9e-4, 1e-4, 0.0, 0.0],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Nominal parameters θ_nom (page 6)
# ---------------------------------------------------------------------------

theta_nom: SIDTHEParams = SIDTHEParams(
    alpha=0.35,
    gamma=0.1,
    lam=0.09,
    delta=2e-3,
    sigma=0.015,
    tau=0.01,
)


def generate_scenarios(
    theta: SIDTHEParams,
    rel: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate uncertainty scenarios with ±rel% variation on each parameter.

    Each of the 6 parameters takes 3 values: (1-rel), 1.0, (1+rel) times nominal.
    This yields 3^6 = 729 scenarios.

    Parameters
    ----------
    theta : SIDTHEParams
        Nominal parameter set.
    rel : float, optional
        Relative perturbation (default 0.05 for ±5%).

    Returns
    -------
    thetas_array : np.ndarray
        Array of shape (729, 6) with all scenario parameter sets.
    probs : np.ndarray
        Uniform probability weights, shape (729,), summing to 1.
    """
    multipliers = [1.0 - rel, 1.0, 1.0 + rel]
    nominal = theta.to_array()

    # Cartesian product of 6 parameters, each with 3 values
    scenarios = []
    for combo in product(multipliers, repeat=6):
        scenarios.append(nominal * np.array(combo))

    thetas_array = np.array(scenarios, dtype=np.float64)  # shape (729, 6)
    n_scenarios = thetas_array.shape[0]
    probs = np.ones(n_scenarios, dtype=np.float64) / n_scenarios

    return thetas_array, probs
