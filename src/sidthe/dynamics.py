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


def get_jacobian_matrix(x, u, p):
    """
    Calcule la matrice Jacobienne A = df/dx au point d'état x.
    Requis pour l'analyse de stabilité locale (Cours 01-1 / TD1).
    """
    # Récupération des variables d'état
    S, I, D, T, H, E = x

    # Termes auxiliaires (dépendants des paramètres p)
    # Vérifie que p possède bien ces attributs (ex: p.gamma)
    term_gamma_eff = p.gamma * (1 + p.lam / (p.lam + p.gamma))
    term_lambda_rec = p.lam * (p.gamma / (p.lam + p.gamma))

    # Initialisation de la matrice 6x6
    J = np.zeros((6, 6))

    # Ligne 1: dS/dt = -alpha(1-u)SI
    J[0, 0] = -p.alpha * (1 - u) * I  # dS/dS
    J[0, 1] = -p.alpha * (1 - u) * S  # dS/dI

    # Ligne 2: dI/dt (Dynamique critique de l'infection)
    J[1, 0] = p.alpha * (1 - u) * I           # dI/dS
    J[1, 1] = p.alpha * (1 - u) * S - term_gamma_eff  # dI/dI

    # Ligne 3: dD/dt
    J[2, 1] = p.gamma
    J[2, 2] = -(p.delta + p.lam)

    # Ligne 4: dT/dt
    J[3, 2] = p.delta
    J[3, 3] = -(p.sigma + p.tau)

    # Ligne 5: dH/dt
    J[4, 1] = term_lambda_rec
    J[4, 2] = p.lam
    J[4, 3] = p.sigma

    # Ligne 6: dE/dt
    J[5, 3] = p.tau

    return J

def check_stability_eigenvalues(x_eq, u_eq, p):
    """
    Retourne les valeurs propres de la Jacobienne.
    Critère TD1 : Stable si Max(Re(eigenvalues)) < 0.
    """
    J = get_jacobian_matrix(x_eq, u_eq, p)
    eigvals = np.linalg.eigvals(J)
    return eigvals