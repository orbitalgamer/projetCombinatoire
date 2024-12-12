import random
import numpy as np
from sklearn.cluster import KMeans

from opti_combi_projet_pythoncode_texte import fobj,compareP1betterthanP2,matrices1_ledm,matrices2_slackngon
from utils import LEDM,lire_fichier,random_matrix
import utils


def selection_par_roulette_multi(M, population, fobj, n_parents):
    # Étape 1 : Calcul des rangs et des petites valeurs singulières
    evaluation = np.array([fobj(M, individu) for individu in population])
   
    rangs = evaluation[:, 0]  # Extraire les rangs
    val_singulieres = evaluation[:, 1]  # Extraire les valeurs singulières
    
    # Étape 2 : Première roulette basée sur les rangs
    rangs_inverses = 1 / (1 + rangs)  # Probabilité inverse du rang (plus bas est mieux)
    proba_rangs = rangs_inverses / rangs_inverses.sum()  # Normaliser
    parents = []
    
    for _ in range(n_parents):
        # Sélectionner un rang en fonction des probabilités
        rang_choisi = np.random.choice(rangs, p=proba_rangs)
        
       # Étape 3 : Filtrer les individus ayant ce rang
        indices_candidats = np.where(rangs == rang_choisi)[0]
        candidats = [population[i] for i in indices_candidats]
        val_sing_candidats = val_singulieres[indices_candidats]

        # Étape 4 : Roulette sur les valeurs singulières parmi les candidats
        inverses_vals = 1 / (1 + val_sing_candidats)  # Plus petite valeur singulière est meilleure
        proba_vals = inverses_vals / inverses_vals.sum()  # Normalisation des probabilités
        index_choisi = np.random.choice(len(candidats), p=proba_vals)
        
        # Ajouter le parent sélectionné
        parents.append(candidats[index_choisi])
    
    return parents
