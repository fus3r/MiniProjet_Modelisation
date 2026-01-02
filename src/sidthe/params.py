"""
Paramètres et constantes du modèle SIDTHE.

Ordre des paramètres : θ = [α, γ, λ, δ, σ, τ]^T (Eq 5, page 3).
Valeurs numériques tirées de l'article, page 6.
"""
from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass
class SIDTHEParams:
    """
    Paramètres épidémiologiques du modèle SIDTHE.

    α : taux de transmission
    γ : taux de détection
    λ : taux de guérison des diagnostiqués (nom 'lam' pour éviter le mot-clé Python)
    δ : taux d'aggravation vers soins critiques
    σ : taux de guérison des cas critiques
    τ : taux de décès
    """

    alpha: float
    gamma: float
    lam: float
    delta: float
    sigma: float
    tau: float

    def to_array(self) -> np.ndarray:
        """Convertit les paramètres en array numpy (6,)."""
        return np.array(
            [self.alpha, self.gamma, self.lam, self.delta, self.sigma, self.tau],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "SIDTHEParams":
        """Crée un SIDTHEParams depuis un array [α, γ, λ, δ, σ, τ]."""
        assert arr.shape == (6,), f"Attendu (6,), reçu {arr.shape}"
        return cls(
            alpha=float(arr[0]),
            gamma=float(arr[1]),
            lam=float(arr[2]),
            delta=float(arr[3]),
            sigma=float(arr[4]),
            tau=float(arr[5]),
        )


# ---------------------------------------------------------------------------
# Constantes de simulation (page 6)
# ---------------------------------------------------------------------------

DT: float = 1.0  # Pas de temps [jours] pour RK4
U_MAX: float = 0.75  # Intensité max du contrôle
T_MAX: float = 0.002  # Seuil de capacité en réanimation (fraction)
T_NPI: int = 14  # Durée d'un bloc NPI [jours]


# ---------------------------------------------------------------------------
# Condition initiale x0 (page 6)
# Ordre : [S, I, D, T, H, E]
# Note : sum(x0) = 0.99829, légèrement < 1 (valeurs de l'article).
# La masse totale est conservée pendant l'intégration.
# ---------------------------------------------------------------------------

x0: np.ndarray = np.array(
    [0.99, 0.008, 1.9e-4, 1e-4, 0.0, 0.0],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Paramètres nominaux θ_nom (page 6)
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
    Génère les scénarios d'incertitude avec variation ±rel sur chaque paramètre.

    Chaque paramètre prend 3 valeurs : (1-rel), 1.0, (1+rel) fois la valeur nominale.
    Produit 3^6 = 729 scénarios avec probabilités uniformes.
    """
    multipliers = [1.0 - rel, 1.0, 1.0 + rel]
    nominal = theta.to_array()

    # Produit cartésien des 6 paramètres, chacun avec 3 valeurs
    scenarios = []
    for combo in product(multipliers, repeat=6):
        scenarios.append(nominal * np.array(combo))

    thetas_array = np.array(scenarios, dtype=np.float64)  # shape (729, 6)
    n_scenarios = thetas_array.shape[0]
    probs = np.ones(n_scenarios, dtype=np.float64) / n_scenarios

    return thetas_array, probs
