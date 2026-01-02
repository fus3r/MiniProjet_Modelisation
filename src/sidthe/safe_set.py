"""
Calcul de l'ensemble sûr pour le modèle SIDTHE.

Référence : Eq (6)-(7), page 3.
L'ensemble sûr X_f garantit l'invariance et le respect des contraintes.

Les bornes sont calculées depuis l'analyse de l'équilibre endémique (Eq 7).
Elles représentent les valeurs d'équilibre pire cas, utilisées
comme contraintes terminales pour le MPC.

Vérification (729 scénarios, ±5%):
    >>> from sidthe.params import theta_nom, generate_scenarios
    >>> from sidthe.safe_set import safe_set_intersection
    >>> thetas, _ = generate_scenarios(theta_nom, rel=0.05)
    >>> bounds = safe_set_intersection(thetas)
    >>> # Attendu environ: Smax=1, Imax≈0.0188, Dmax≈0.0226, Tmax≈0.002
"""
import numpy as np

from .params import SIDTHEParams, U_MAX, T_MAX


def invariant_bounds(theta: SIDTHEParams) -> dict[str, float]:
    """
    Calcule les bornes de l'ensemble sûr pour un jeu de paramètres.

    Implémente Eq (7). On fixe T_max (contrainte réa) puis on remonte
    les bornes compatibles pour D et I.

    Valeurs de référence (page 7): Smax=1, Imax=0.0188, Dmax=0.0226, Tmax=0.002
    """
    alpha = theta.alpha
    gamma = theta.gamma
    lam = theta.lam
    delta = theta.delta
    sigma = theta.sigma
    tau = theta.tau

    # Eq (7a): S_max = 1 (fraction de population)
    Smax = 1.0

    # T_max donné par la contrainte de capacité réa
    Tmax = T_MAX  # = 0.002

    # Calcul inverse de D_max depuis T_max : T = δ/(σ+τ) × D => D = (σ+τ)/δ × T
    if delta > 0:
        Dmax = (sigma + tau) / delta * Tmax
    else:
        Dmax = 0.0

    # Calcul inverse de I_max depuis D_max : D = γ/(δ+λ) × I => I = (δ+λ)/γ × D
    if gamma > 0:
        Imax = (delta + lam) / gamma * Dmax
    else:
        Imax = 0.0

    return {"Smax": Smax, "Imax": Imax, "Dmax": Dmax, "Tmax": Tmax}


def safe_set_intersection(thetas_array: np.ndarray) -> dict[str, float]:
    """
    Calcule l'ensemble sûr comme intersection sur tous les scénarios.

    Pour la robustesse, on prend le minimum des bornes sur tous les scénarios.

    Pour 729 scenarios avec +/-5% sur theta_nom, valeurs attendues (page 7):
        Smax = 1.0
        Imax ~ 0.0188
        Dmax ~ 0.0226
        Tmax ~ 0.002
    """
    n_scenarios = thetas_array.shape[0]

    # Initialise avec le premier scénario
    theta0 = SIDTHEParams.from_array(thetas_array[0])
    result = invariant_bounds(theta0)

    # Prend le min sur tous les scénarios
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
