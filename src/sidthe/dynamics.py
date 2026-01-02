"""
Dynamique du modèle SIDTHE.

Implémente le système d'EDO (4a)-(4f), cf. article page 3.
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
    Membre de droite du système d'EDO SIDTHE.

    Calcule dx/dt selon les équations (4a)-(4f) de l'article :
        Sdot = -α(1-u) S I
        Idot = α(1-u) S I - γ*(1 + λ/(λ+γ))*I
        Ddot = γ I - (δ+λ) D
        Tdot = δ D - (σ+τ) T
        Hdot = σ T + λ D + λ*(γ/(λ+γ))*I
        Edot = τ T

    L'état x = [S, I, D, T, H, E] représente des fractions de population.
    Le contrôle u ∈ [0, 0.75] modélise l'intensité des interventions.
    """
    # Vérification de cohérence
    assert x.shape == (6,), f"État x doit être de dim 6, reçu {x.shape}"
    u = float(np.clip(u, 0.0, U_MAX))  # Borne le contrôle à [0, U_MAX]

    # Extraction de l'état
    S, I, D, T, H, E = x

    # Extraction des paramètres épidémiologiques
    alpha = params.alpha
    gamma = params.gamma
    lam = params.lam
    delta = params.delta
    sigma = params.sigma
    tau = params.tau

    # Termes récurrents (évite la redondance)
    lam_plus_gamma = lam + gamma
    assert lam_plus_gamma > 0, "λ + γ doit être > 0 pour éviter la division par zéro"
    infection_rate = alpha * (1.0 - u) * S * I

    # Équations (4a)-(4f)
    Sdot = -infection_rate
    Idot = infection_rate - gamma * (1.0 + lam / lam_plus_gamma) * I
    Ddot = gamma * I - (delta + lam) * D
    Tdot = delta * D - (sigma + tau) * T
    Hdot = sigma * T + lam * D + lam * (gamma / lam_plus_gamma) * I
    Edot = tau * T

    return np.array([Sdot, Idot, Ddot, Tdot, Hdot, Edot], dtype=np.float64)
