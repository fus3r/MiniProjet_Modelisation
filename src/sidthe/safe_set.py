"""
Safe set computation for SIDTHE model.

Reference: Paper Eq (6)-(7), page 3.
Safe set X_f ensures invariance and constraint satisfaction.

The bounds are computed from the endemic equilibrium analysis (Eq 7).
These represent the worst-case equilibrium values that the system
can reach, used to define terminal constraints for MPC.

Example verification (729 scenarios, ±5%):
    >>> from sidthe.params import theta_nom, generate_scenarios
    >>> from sidthe.safe_set import safe_set_intersection
    >>> thetas, _ = generate_scenarios(theta_nom, rel=0.05)
    >>> bounds = safe_set_intersection(thetas)
    >>> # Expected approx: Smax=1, Imax≈0.0188, Dmax≈0.0226, Tmax≈0.002
"""
import numpy as np

from .params import SIDTHEParams, U_MAX, T_MAX


def invariant_bounds(theta: SIDTHEParams) -> dict[str, float]:
    """
    Compute safe set bounds for a single parameter set.

    Implements Eq (7) from the paper, page 3.
    The bounds correspond to the endemic equilibrium under no control,
    scaled appropriately to ensure constraint satisfaction.

    From paper page 7, the intersection values are:
        Smax = 1
        Imax = 0.0188
        Dmax = 0.0226
        Tmax = 0.002 (= T_MAX, the ICU constraint)

    The equilibrium analysis gives:
        I_eq proportional to (α - γ) / α  at disease-free equilibrium threshold
        D_eq = γ / (δ + λ) * I_eq
        T_eq = δ / (σ + τ) * D_eq

    For robust MPC, we use T_max = T_MAX (ICU constraint) and derive
    compatible bounds for other states.

    Parameters
    ----------
    theta : SIDTHEParams
        Model parameters (α, γ, λ, δ, σ, τ).

    Returns
    -------
    bounds : dict
        Dictionary with keys 'Smax', 'Imax', 'Dmax', 'Tmax'.
    """
    alpha = theta.alpha
    gamma = theta.gamma
    lam = theta.lam
    delta = theta.delta
    sigma = theta.sigma
    tau = theta.tau

    # Eq (7a): S_max = 1 (population fraction)
    Smax = 1.0

    # T_max is given by ICU constraint
    Tmax = T_MAX  # = 0.002

    # Back-calculate D_max from T_max: T = δ/(σ+τ) * D => D = (σ+τ)/δ * T
    if delta > 0:
        Dmax = (sigma + tau) / delta * Tmax
    else:
        Dmax = 0.0

    # Back-calculate I_max from D_max: D = γ/(δ+λ) * I => I = (δ+λ)/γ * D
    if gamma > 0:
        Imax = (delta + lam) / gamma * Dmax
    else:
        Imax = 0.0

    return {"Smax": Smax, "Imax": Imax, "Dmax": Dmax, "Tmax": Tmax}


def safe_set_intersection(thetas_array: np.ndarray) -> dict[str, float]:
    """
    Compute safe set as intersection over all scenarios.

    For robust constraint satisfaction, we take the minimum bound
    across all parameter scenarios for each state.

    Parameters
    ----------
    thetas_array : np.ndarray
        Array of shape (n_scenarios, 6) with parameter sets.

    Returns
    -------
    bounds : dict
        Dictionary with keys 'Smax', 'Imax', 'Dmax', 'Tmax',
        representing the intersection (min) over all scenarios.

    Notes
    -----
    For 729 scenarios with ±5% on theta_nom, expected values (page 7):
        Smax = 1.0
        Imax ≈ 0.0188
        Dmax ≈ 0.0226
        Tmax ≈ 0.002
    """
    n_scenarios = thetas_array.shape[0]

    # Initialize with first scenario
    theta0 = SIDTHEParams.from_array(thetas_array[0])
    result = invariant_bounds(theta0)

    # Take minimum over all scenarios
    for i in range(1, n_scenarios):
        theta_i = SIDTHEParams.from_array(thetas_array[i])
        bounds_i = invariant_bounds(theta_i)
        for key in result:
            result[key] = min(result[key], bounds_i[key])

    return result


if __name__ == "__main__":
    # Quick verification test
    from .params import theta_nom, generate_scenarios

    print("=" * 60)
    print("Safe Set Computation - Verification")
    print("=" * 60)

    # Single scenario (nominal)
    bounds_nom = invariant_bounds(theta_nom)
    print("\nNominal parameters bounds (Eq 7):")
    for k, v in bounds_nom.items():
        print(f"  {k} = {v:.6f}")

    # 729 scenarios intersection
    thetas, probs = generate_scenarios(theta_nom, rel=0.05)
    bounds_inter = safe_set_intersection(thetas)
    print(f"\nIntersection over {len(thetas)} scenarios:")
    for k, v in bounds_inter.items():
        print(f"  {k} = {v:.6f}")

    print("\nExpected (page 7): Smax=1, Imax≈0.0188, Dmax≈0.0226, Tmax≈0.002")
    print("=" * 60)
