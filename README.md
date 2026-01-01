# Simulateur SIDTHE – Sujet 3, Groupe D

**Auteurs** : Darwish Riad, Nadjar Benjamin, Sihalov Volodymyr  
**Cours** : Modélisation – Représentations et analyse des modèles

---

## Installation

```bash
pip install -r requirements.txt
```

## Expérience typique (one-shot)

```bash
python scripts/run_experiment.py
```

Cette commande génère :
- Simulation quickstart (dynamique SIDTHE sans contrôle)
- Figure 3 : MPC robuste
- Figure 4 : MPC avec recourse
- Figure 7 : Infaisabilité MPC vanilla

## Générer les figures principales

```bash
python scripts/fig3_robust.py      # → outputs/figures/fig3_robust.png/.pdf
python scripts/fig4_recourse.py    # → outputs/figures/fig4_recourse.png/.pdf
python scripts/fig7_vanilla_infeasibility.py  # → outputs/figures/fig7_vanilla_infeasibility.png/.pdf
```

## Smoke test (validation rapide < 30s)

```bash
python scripts/smoke_test.py
```

## Structure des sorties

```
outputs/
└── figures/
    ├── fig3_robust.png/.pdf
    ├── fig4_recourse.png/.pdf
    ├── fig7_vanilla_infeasibility.png/.pdf
    └── quickstart_T.png/.pdf
```

## Tests

```bash
python scripts/sanity_scenarios.py   # Vérifie génération 729 scénarios
python scripts/check_icu_violations.py  # Vérifie contraintes ICU
```

## Bonus : Simulateur à événements discrets (Partie II)

```bash
python scripts/des_simulator.py
```

Génère `outputs/figures/des_trace.png` : trace des modes épidémiques.
