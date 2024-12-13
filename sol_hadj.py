import numpy as np
import random
from utils import LEDM,lire_fichier,random_matrix,optimal_k,clustering_lines
reel_matrix = LEDM(20,120)
 
def fobj(M, P):
    sing_values = np.linalg.svd(P * np.sqrt(M), compute_uv=False)  # Calcul des valeurs singulières
    tol = max(M.shape) * sing_values[0] * np.finfo(float).eps  # Tolérance
    ind_nonzero = np.where(sing_values > tol)[0]  # Valeurs > tolérance
    return len(ind_nonzero), sing_values[ind_nonzero[-1]]  # Objectifs : rang et plus petite valeur singulière

def compareP1betterthanP2(M, P1, P2):
    r1, s1 = fobj(M, P1)  # Récupère les deux objectifs pour P1
    r2, s2 = fobj(M, P2)  # Récupère les deux objectifs pour P2
    if r1 != r2:
        return r1 < r2  # On compare d'abord le rang
    return s1 < s2  # Si égalité, on compare la plus petite valeur singulière

def generate_population(size, shape):
    """Génère une population initiale de matrices aléatoires {+1, -1}."""
    return [np.random.choice([-1, 1], size=shape) for _ in range(size)]

def evaluate_population(population, M):
    """Évalue la population en calculant les objectifs."""
    return [(P, fobj(M, P)) for P in population]

def tournament_selection(population, M, k=3):
    """Sélectionne un individu via un tournoi de taille k."""
    tournament = random.sample(population, k)
    best_individual = min(tournament, key=lambda x: (x[1][0], x[1][1]))  # Choisit le meilleur selon les objectifs
    return best_individual[0]

def crossover(parent1, parent2, crossover_rate=0.7):
    """Effectue un croisement mono-point avec un taux donné."""
    child = parent1.copy()
    if np.random.rand() < crossover_rate:
        point = np.random.randint(0, parent1.size)  # Choisit un point de croisement
        child.ravel()[point:] = parent2.ravel()[point:]  # Échange la partie après ce point
    return child

def mutate(individual, mutation_rate=0.1):
    """Effectue une mutation en inversant un bit aléatoire."""
    if np.random.rand() < mutation_rate:
        i, j = np.random.randint(0, individual.shape[0]), np.random.randint(0, individual.shape[1])
        individual[i, j] *= -1
    return individual

def genetique(M, max_k, voisinage, list_methode_cross, mutation_rate, memetique, time, max_depth, n_parents, parent_init=None, method_next_gen="Best"):
    # Initialiser la population avec la recherche locale ou greedy si un parent_init est donné
    if parent_init is None:
        n_clusters = optimal_k(M, max_k)  # Trouver optimal_k pour les clusters
        line_labels = clustering_lines(M, n_clusters)
        col_labels = clustering_columns(M, n_clusters)

        # Générer une population initiale aléatoire si aucun parent_init n'est donné
        parents = [generate_initial_P(M, n_clusters) for _ in range(n_parents)]
    else:
        parents = [parent_init]  # Parent initial fourni
    
    best_matrice = parents[0]

    for t in range(time):
        random.shuffle(parents)
        # Création des enfants
        methode_cross = list_methode_cross[t % len(list_methode_cross)]
        enfants = methode_cross(parents)

        enfants, enfants_mute = enfants[:int(len(enfants) * mutation_rate)], enfants[int(len(enfants) * mutation_rate):]

        if memetique:
            for i in range(len(enfants_mute) // 2):
                enfants_mute[i] = full_local_search(M, enfants_mute[i], voisinage, max_depth)
            for i in range(len(enfants_mute) // 2, len(enfants_mute)):
                n1, n2 = random.randint(0, len(M) - 1), random.randint(0, len(M[0]) - 1)
                enfants_mute[i][n1][n2] = -enfants_mute[i][n1][n2]
        else:
            for i in range(len(enfants_mute)):
                n1, n2 = random.randint(0, len(M) - 1), random.randint(0, len(M[1]) - 1)
                enfants_mute[i][n1][n2] = -enfants_mute[i][n1][n2]

        enfants += enfants_mute
        parents += enfants

        # Sélection des meilleurs parents
        if method_next_gen == "Best":
            parents = [(fobj(M, parent), parent) for parent in parents]
            parents = [x[1] for x in sorted(parents, key=lambda x: (x[0][0], x[0][1]))[:len(parents) // 2]]
        elif method_next_gen == "Tournament":
            random.shuffle(parents)
            new_parents = []
            for i in range(0, len(parents), 2):
                if compareP1betterthanP2(M, parents[i], parents[i+1]):
                    new_parents.append(parents[i])
                else:
                    new_parents.append(parents[i+1])
            parents = new_parents.copy()

        print(f"{t} sur {time}")
        if compareP1betterthanP2(M, parents[0], best_matrice):
            best_matrice = parents[0].copy()
            print(f"improve")

    count_dict = {}
    for matrice in parents:
        key = tuple(matrice.flatten())
        count_dict[key] = count_dict.get(key, 0) + 1

    print(count_dict.values())
    return best_matrice

def one_point_crossover(parents):
    enfants=[]
    for i in range(0, len(parents) - 1, 2):
        parent1 = parents[i]
        parent2 = parents[i + 1]
        point = random.randint(0, len(parent1) - 1)
        enfant1 = np.vstack((parent1[:point], parent2[point:]))
        enfant2= np.vstack((parent2[:point], parent1[point:]))
        enfants.append(enfant1)
        enfants.append(enfant2)
    return enfants

def uniform_crossover(parents):
    # Liste pour stocker les enfants
    enfants = []
    
    # Itérer sur les paires de parents dans la liste
    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        parent2 = parents[i + 1]
        
        # Vérifier que les parents ont la même forme
        if parent1.shape != parent2.shape:
            raise ValueError("Les dimensions des deux parents doivent être identiques.")
        
        # Générer un masque binaire aléatoire de la même forme que les parents
        mask = np.random.randint(2, size=parent1.shape)
        
        # Créer les enfants en fonction du masque
        enfant1 = np.where(mask == 1, parent1, parent2)
        enfant2 = np.where(mask == 0, parent1, parent2)
        
        # Ajouter les enfants à la liste des enfants
        enfants.append(enfant1)
        enfants.append(enfant2)
    
    return enfants
    # Exemple de paramètres pour tester
n_parents = 10  # Taille de la population
time = 100      # Nombre de générations
mutation_rate = 0.2  # Taux de mutation
max_k = 5  # Nombre maximum de clusters pour les méthodes de clustering
max_depth = 5  # Profondeur maximale pour la recherche locale

list_methode_cross = [uniform_crossover, one_point_crossover]  # Liste de méthodes de croisement

# Tester l'algorithme génétique
best_solution = genetique(reel_matrix, max_k, 10, list_methode_cross, mutation_rate, True, time, max_depth, n_parents)






