# Modélisation Épidémique SIDTHE & Contrôle MPC

**Sujet 3 – Groupe D**
**Cours** : Modélisation – Représentations et analyse des modèles (2025-2026)

**Auteurs** :
* Darwish Riad
* Nadjar Benjamin
* Sihalov Volodymyr

---

## Description du Projet

Ce projet implémente le modèle épidémiologique **SIDTHE** (*Susceptible, Infected, Diagnosed, Threatened, Healed, Expired*) pour simuler et contrôler la saturation des unités de soins intensifs (ICU).

Le simulateur est structuré pour répondre aux deux parties du sujet :
1.  **Partie I (Continu)** : Modèle d'état (EDO) et contrôle optimal (MPC) pour gérer le confinement et respecter les capacités hospitalières.
2.  **Partie II (Discret)** : Modèle stochastique par Réseau de Petri Temporisé (Algorithme de Gillespie) pour simuler la gestion de crise par seuils (Automate Hybride).

---

## Installation

Le projet nécessite Python 3.8+.

1.  **Installation des dépendances**
    ```bash
    pip install -r requirements.txt
    ```
    *(Inclut `numpy`, `scipy`, `matplotlib` et `casadi`).*

---

## 1. Validation Académique (Mathématiques)

Ce script est **fondamental**. Il valide les propriétés théoriques du modèle (Chapitre 1) avant toute simulation. Il prouve la cohérence scientifique du simulateur.

```bash
python run_academic_validation.py
```
---

## 2. Partie I : Modèle Continu & Contrôle MPC

Cette partie simule les équations différentielles et l'optimisation (MPC).

### Lancement Rapide (Quickstart)
Pour visualiser la dynamique naturelle (explosion de l'épidémie sans contrôle) :
```bash
python scripts/quickstart.py
```
Expérience Complète (One-Shot)
Pour générer l'ensemble des résultats de la Partie I en une seule commande :
```bash
python scripts/run_experiment.py
```
Scénarios Spécifiques
Vous pouvez relancer chaque figure individuellement :
```
MPC Robuste (Fig 3) : Contrôle avec incertitude paramétrique.
```bash
python scripts/fig3_robust.py
```
MPC avec Recourse (Fig 4) : Stratégie adaptative.
```bash
python scripts/fig4_recourse.py
```
Infaisabilité (Fig 7) : Limites du MPC standard.
```bash
python scripts/fig7_vanilla_infeasibility.py
```
---

## 🎲 3. Partie II : Événements Discrets

Cette partie utilise un **Réseau de Petri Temporisé** pour simuler la stochasticité et les transitions de modes (Normal/Alerte/Confinement).

```bash
python scripts/des_simulator.py
```
---

## ✅ 4. Tests et Qualité du Code

Pour garantir la fiabilité du simulateur, nous avons inclus une suite de tests automatisés :

* **Smoke Test** (Validation rapide < 30s de l'environnement) :
    ```bash
    python scripts/smoke_test.py
    ```
* **Vérification des contraintes ICU** (Analyse post-simulation) :
    ```bash
    python scripts/check_icu_violations.py
    ```
* **Validation des scénarios** (Génération de l'arbre des possibles) :
    ```bash
    python scripts/sanity_scenarios.py
    ```
