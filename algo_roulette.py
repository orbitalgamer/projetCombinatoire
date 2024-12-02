import random
import numpy as np
from sklearn.cluster import KMeans

from opti_combi_projet_pythoncode_texte import fobj,compareP1betterthanP2,matrices1_ledm,matrices2_slackngon
from utils import LEDM,lire_fichier,random_matrix
import utils

def selection_par_roulette_multi(M, population, fobj, n_parents, tol=1e-14):
    # Étape 1 : Calcul des rangs et des petites valeurs singulières
    evaluation = [fobj(M, individu, tol) for individu in population]
    rangs = np.array([eval[0] for eval in evaluation])
    val_singulieres = np.array([eval[1] for eval in evaluation])
    
    # Étape 2 : Première roulette basée sur les rangs
    rangs_inverses = 1 / (1 + rangs)  # Probabilité inverse du rang (plus bas est mieux)
    proba_rangs = rangs_inverses / rangs_inverses.sum()  # Normaliser
    parents = []
    
    for _ in range(n_parents):
        # Sélectionner un rang en fonction des probabilités
        rang_choisi = np.random.choice(np.unique(rangs), p=np.bincount(rangs)[np.unique(rangs)] / len(rangs))
        
       # Étape 3 : Filtrer les individus ayant ce rang
        indices_candidats = [i for i, rang in enumerate(rangs) if rang == rang_choisi]
        candidats = [population[i] for i in indices_candidats]
        val_sing_candidats = val_singulieres[indices_candidats]

        # Étape 4 : Roulette sur les valeurs singulières parmi les candidats
        inverses_vals = 1 / (1 + val_sing_candidats)  # Plus petite valeur singulière est meilleure
        proba_vals = inverses_vals / inverses_vals.sum()  # Normalisation des probabilités
        index_choisi = np.random.choice(len(candidats), p=proba_vals)
        
        # Ajouter le parent sélectionné
        parents.append(candidats[index_choisi])
    
    return parents

