import numpy as np
import sys
import os

# Ajout du chemin courant
sys.path.append(os.getcwd())

try:
    from src.sidthe.dynamics import *
    # On essaie d'importer params
    try:
        import src.sidthe.params as p_module
    except ImportError:
        p_module = None
except ImportError as e:
    print(f"ERREUR CRITIQUE : {e}")
    sys.exit(1)

def run_validation():
    print("==================================================")
    print("   VALIDATION ACADÉMIQUE (Groupe 3)")
    print("   Chapitre 1 (État/Stabilité) & Chapitre 2 (DES)")
    print("==================================================\n")

    # 1. Définition des paramètres nominaux
    class Params: pass
    par = Params()
    # Valeurs par défaut
    par.alpha = 0.35; par.gamma = 0.1; par.lam = 0.09; 
    par.delta = 2e-3; par.sigma = 0.015; par.tau = 0.01;
    par.u_max = 0.75

    # Récupération des vrais paramètres si possible
    if p_module and hasattr(p_module, 'NominalParams'):
        try:
            temp = p_module.NominalParams()
            par.alpha = temp.alpha
            # etc...
        except: pass

    # ---------------------------------------------------------
    # PARTIE I : STABILITÉ & VALEURS PROPRES (CHAPITRE 1 / TD1)
    # ---------------------------------------------------------
    print("--- [CHAPITRE 1] ANALYSE DE STABILITÉ (JACOBIENNE) ---")
    
    # Équilibre DFE (Disease Free Equilibrium) : I=0
    x_dfe = np.array([0.99, 0, 0, 0, 0, 0]) 
    u_ctrl = 0.0 

    try:
        # Calcul via la nouvelle fonction ajoutée dans dynamics.py
        eigvals = check_stability_eigenvalues(x_dfe, u_ctrl, par)
        max_re_eig = np.max(np.real(eigvals))

        print(f"Point d'analyse (Équilibre) : S={x_dfe[0]}, I=0")
        print(f"Valeurs propres de la matrice A :\n {np.round(eigvals, 4)}")
        print(f"Max(Re(lambda)) : {max_re_eig:.4f}")
        
        if max_re_eig < 0:
            print("=> RÉSULTAT : SYSTÈME STABLE (Retour à l'équilibre)")
        else:
            print("=> RÉSULTAT : SYSTÈME INSTABLE (Démarrage épidémique)")

        # Vérification avec R0 théorique
        term_gamma = par.gamma * (1 + par.lam / (par.lam + par.gamma))
        R0 = par.alpha / term_gamma
        print(f"R0 calculé : {R0:.4f} (Instable si > 1)\n")
        
    except Exception as e:
        print(f"Erreur calcul stabilité : {e}")

if __name__ == "__main__":
    run_validation()