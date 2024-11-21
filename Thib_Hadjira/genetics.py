import time
import random
import numpy as np  # Pour manipuler les matrices
import Thib_enfant  # On suppose que ta fonction de croisement fonctionne avec des matrices

def cost_function(matrice):
    """Calcule le coût de la solution représentée par une matrice."""
    # Exemple simple : Somme des éléments de la matrice
    return np.sum(matrice)

def evolutionnaire(sol, voisinage, temps_max=2, memetique=True):
    a = time.time()
    sol = np.array(sol)  # Transformer sol en matrice si ce n'est pas déjà le cas
    parents = []
    sujet_max =10
    taux_mutation = 1 / 4
    parents.append(sol)
    kmax = 10000
    best_cost = float('inf')
    best_sol = None

    # Génération de solutions initiales
    for i in range(sujet_max - 1):
        new_parent = np.random.choice([-1, 1], size=sol.shape)
        if not any(np.array_equal(new_parent, parent) for parent in parents):
            parents.append(new_parent)

    while time.time() - a < temps_max:

        # Génération des enfants
        enfants = []
        for i in range(0, len(parents) - 1, 2):
            enfant1, enfant2 = Thib_enfant.croisement__elem_1_2(parents[i], parents[i + 1])
            enfants.append(enfant1)
            enfants.append(enfant2)

        # Sélection des enfants à muter
        random_list = random.sample(range(len(enfants)), int(sujet_max * taux_mutation))

        # if memetique:
        #     pass
        #     # Exemple de recherche locale sur matrice (non implémentée ici)
        # else:
        #     # Mutation aléatoire sur les matrices
        #     for idx in random_list:
        #         i, j = np.random.randint(0, enfants[idx].shape[0]), np.random.randint(0, enfants[idx].shape[1])
        #         enfants[idx][i, j] = np.random.random_sample([-1,1])  # Exemple : mutation en remplaçant un élément par une valeur aléatoire

        # Fusion enfants + parents
        parents += enfants

        # Sélection des meilleurs sujets
        parents = [(cost_function(parent), parent) for parent in parents]
        parents = sorted(parents, key=lambda x: x[0])[:len(parents) // 2]
        if parents[0][0] < best_cost:
            best_cost, best_sol = parents[0]
        parents = [x[1] for x in parents]

        # Remélanger les sujets
        random.shuffle(parents)
        kmax -= 1

    # Compter les occurrences des solutions
    count_dict = {}
    for matrice in parents:
        key = tuple(matrice.flatten())
        count_dict[key] = count_dict.get(key, 0) + 1

    # Afficher les occurrences (optionnel)
    print(count_dict.values())
    return best_sol, best_cost, time.time() - a


sol_initiale = [[1, -1, 1], [-1, 1, -1], [1, -1, 1]]
voisinage = None  # Non utilisé ici
best_sol, best_cost, duree = evolutionnaire(sol_initiale, voisinage)
print("Meilleure solution :\n", best_sol)
print("Coût : ", best_cost)
print("Durée : ", duree, "secondes")

