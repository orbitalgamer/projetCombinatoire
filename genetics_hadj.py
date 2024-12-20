import random
import numpy as np
from sklearn.cluster import KMeans
from algo_roulette import selection_par_roulette_multi
from opti_combi_projet_pythoncode_texte import fobj,compareP1betterthanP2,matrices1_ledm,matrices2_slackngon
from utils import LEDM,lire_fichier,random_matrix
import utils

#reel_matrix= utils.lire_fichier("data/exempleslide_matrice (1).txt")
#reel_matrix= utils.lire_fichier("data/ledm6_matrice (1).txt")
#reel_matrix= utils.lire_fichier("data/correl5_matrice.txt")
reel_matrix = LEDM(20,20)
#reel_matrix = reel_matrix.transpose()
#reel_matrix = matrices2_slackngon(7)
#reel_matrix = random_matrix(25,25,5)
M = reel_matrix


def divide_matrix(matrix):
    m, n = matrix.shape
    m2, n2 = m // 2, n // 2
    A11 = matrix[:m2, :n2]
    A12 = matrix[:m2, n2:]
    A21 = matrix[m2:, :n2]
    A22 = matrix[m2:, n2:]
    return A11, A12, A21, A22

# Fonction pour réassembler 4 sous-matrices
def reassemble_matrix(M_list):
    A11, A12, A21, A22 = M_list
    top = np.hstack((A11, A12))
    bottom = np.hstack((A21, A22))
    return np.vstack((top, bottom))

def local_search_random_unique(M,matrice_init, voisinage, max_attempts=100):
    """
    Recherche locale aléatoire avec évitement des répétitions.

    Args:
        matrice_init: La matrice initiale.
        voisinage: (int) Type de voisinage (actuellement non utilisé mais peut être étendu).
        max_attempts: Nombre maximal de modifications à tester.

    Returns:
        best_matrice: La meilleure matrice trouvée.
    """
    new_matrice = matrice_init.copy()
    best_matrice = matrice_init.copy()
    # current_best_score = fobj(M, best_matrice)  # Score de la matrice initiale

    # Liste des positions déjà visitées
    visited_positions = set()

    for _ in range(max_attempts):
        # Générer une position non encore visitée
        while True:
            i = random.randint(0, len(matrice_init) - 1)
            j = random.randint(0, len(matrice_init[0]) - 1)
            if (i, j) not in visited_positions:  # Vérifier si on a déjà visité cette position
                visited_positions.add((i, j))
                break  # Sortir une fois qu'on a une position unique

        # Modifier la valeur à cette position
        new_matrice[i][j] = -new_matrice[i][j]

        # Comparer la nouvelle matrice à la meilleure trouvée jusqu'à présent
        if compareP1betterthanP2(M, new_matrice, best_matrice):
            best_matrice = new_matrice.copy()
            #current_best_score = fobj(M, best_matrice)
        else:
            # Revenir en arrière si la modification n'est pas bénéfique
            new_matrice[i][j] = -new_matrice[i][j]

        # Arrêter si toutes les positions possibles ont été visitées
        if len(visited_positions) >= len(matrice_init) * len(matrice_init[0]):
            break

    return best_matrice


def local_search(M,matrice_init,voisinage):
    new_matrice = matrice_init.copy()
    best_matrice = new_matrice.copy()
    if voisinage == 0:
        for i in range(len(matrice_init)):
            for j in range(len(matrice_init[0])):
                new_matrice[i][j] = -new_matrice[i][j]
                if compareP1betterthanP2(M,new_matrice,matrice_init):
                    best_matrice = new_matrice.copy()
                new_matrice[i][j] = -new_matrice[i][j]
    return best_matrice

def full_local_search(M,matrice_init,voisinage,max_depth = 10):
    best_matrice = matrice_init.copy()
    for i in range(max_depth):
        new_matrice = local_search_random_unique(M,best_matrice,voisinage)
        #new_matrice = local_search(M,best_matrice,voisinage)
        if compareP1betterthanP2(M,new_matrice,best_matrice):
            best_matrice = new_matrice.copy()
            #print("improve in full")
        else:

            break
    return best_matrice

def _random_matrix(n):
    list_matrix = list()
    for _ in range(n):
        list_matrix.append(np.random.choice([-1, 1], size=M.shape))
    return list_matrix


def clustering_lines(M, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(M)
    return labels

def clustering_columns(M, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(M.T)
    return labels

def generate_initial_P(M, line_labels, col_labels,noise_prob):
    P = np.zeros_like(M)
    # unique_line_labels = np.unique(line_labels)
    # unique_col_labels = np.unique(col_labels)
    
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            P[i, j] = 1 if (line_labels[i] + col_labels[j]) % 2 == 0 else -1
            if np.random.rand() < noise_prob:  # Avec une certaine probabilité, changer le cluster
                P[i,j] = -P[i,j]
    return P

def genetique(M,n_clusters,voisinage,list_methode_cross,mutation_rate,memetique,time,max_depth,n_parents, parent_init = None,method_next_gen = "Best"):
    if parent_init is None:
        # Clustering des lignes et des colonnes
        line_labels = clustering_lines(M, n_clusters)
        col_labels = clustering_columns(M, n_clusters)
        
        # Générer la population initiale
        parents = [ generate_initial_P(M, line_labels, col_labels,noise_prob=0.00) for i in range(n_parents)]
        #parents += [generate_initial_P(M, line_labels, col_labels,noise_prob=0.05) for _ in range(50)]
    elif parent_init[0] is None:
        parents = parent_init[1]
    else:
        parents = [parent_init.copy() for i in range(n_parents)]
    # return parents[0]
    best_matrice = parents[0]
    for t in range(time):
        if len(parents) != 100:
            pass
        random.shuffle(parents)
        #Creation enfants
        methode_cross = list_methode_cross[t%len(list_methode_cross)]

        enfants : list = methode_cross(parents)

        enfants,enfants_mute = enfants[:int(len(enfants)*mutation_rate)],enfants[int(len(enfants)*mutation_rate):]
        
        
        if memetique:
            for i in range(len(enfants_mute)//2):
                enfants_mute[i] = full_local_search(M,enfants_mute[i],voisinage,max_depth)
            for i in range(len(enfants_mute)//2, len(enfants_mute)):
                n1,n2 = random.randint(0, len(M)-1),random.randint(0, len(M[0])-1)
                enfants_mute[i][n1][n2] = -enfants_mute[i][n1][n2]
        else:
            for i in range(len(enfants_mute)):
                n1,n2 = random.randint(0, len(M)-1),random.randint(0, len(M[1])-1)
                enfants_mute[i][n1][n2] = -enfants_mute[i][n1][n2]
        enfants += enfants_mute
        parents += enfants

        # Sélection des meilleurs sujets
        
        if method_next_gen == "Best":
            parents = [(fobj(M, parent), parent) for parent in parents]
            parents = [x[1] for x in sorted(parents, key=lambda x: (x[0][0], x[0][1]))[:len(parents) // 2]]
        # elif method_next_gen =="Weels":
        #     for i,parent in enumerate (parents):
        #         if parent[0][0] != parents[0][0][0]:
        #             end_of_best = i
        #             break
        #     pass
        elif method_next_gen == "Tournament":
            random.shuffle(parents)
            new_parents = []
            for i in range(0,len(parents),2):
                if compareP1betterthanP2(M, parents[i], parents[i+1]):
                    new_parents.append(parents[i])
                else:
                    new_parents.append(parents[i+1])
            parents = new_parents.copy()
        elif method_next_gen == "roulette":
            selection_par_roulette_multi(M, parents, fobj, n_parents)

                



        print(f"{t} sur {time}")
        if compareP1betterthanP2(M,parents[0],best_matrice):
            best_matrice = parents[0].copy()
            print(methode_cross.__name__)
            print(f"improve")
    count_dict = {}
    for matrice in parents:
        key = tuple(matrice.flatten())
        count_dict[key] = count_dict.get(key, 0) + 1

    # Afficher les occurrences (optionnel)
    print(count_dict.values())
    return best_matrice

import matplotlib.pyplot as plt
def optimal_k(M, max_k=10):
    inertias = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(M)
        inertias.append(kmeans.inertia_)
    
    # Tracer la courbe de l'inertie
    plt.plot(range(1, max_k+1), inertias, marker='o')
    plt.title('Méthode du coude')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Inertie')
    plt.show()

# Appliquer cette méthode pour déterminer optimal k


def cross_by_half_split(parents):
    enfants = []
    # Couple de parents
    for i in range(0, len(parents) - 1, 2):
        parent1, parent2 = parents[i], parents[i + 1]
        
        # Initialisation des enfants
        enfant1 = np.zeros_like(parent1)
        enfant2 = np.zeros_like(parent2)
        
        # Taille de la matrice
        mid_row = parent1.shape[0] // 2  # Trouver la moitié de la hauteur
        
        # Construire enfant1 et enfant2
        enfant1[:mid_row, :] = parent1[:mid_row, :]  # Haut de parent1
        enfant1[mid_row:, :] = parent2[mid_row:, :]  # Bas de parent2
        
        enfant2[:mid_row, :] = parent2[:mid_row, :]  # Haut de parent2
        enfant2[mid_row:, :] = parent1[mid_row:, :]  # Bas de parent1
        
        # Ajouter les enfants à la liste
        enfants.append(enfant1)
        enfants.append(enfant2)
    
    # Mélanger les enfants pour plus de diversité
    random.shuffle(enfants)
    return enfants

def cross_by_vertical_split(parents):
    enfants = []
    # Couple de parents
    for i in range(0, len(parents) - 1, 2):
        parent1, parent2 = parents[i], parents[i + 1]
        
        # Initialisation des enfants
        enfant1 = np.zeros_like(parent1)
        enfant2 = np.zeros_like(parent2)
        
        # Taille de la matrice
        mid_col = parent1.shape[1] // 2  # Trouver la moitié des colonnes
        
        # Construire enfant1 et enfant2
        enfant1[:, :mid_col] = parent1[:, :mid_col]  # Gauche de parent1
        enfant1[:, mid_col:] = parent2[:, mid_col:]  # Droite de parent2
        
        enfant2[:, :mid_col] = parent2[:, :mid_col]  # Gauche de parent2
        enfant2[:, mid_col:] = parent1[:, mid_col:]  # Droite de parent1
        
        # Ajouter les enfants à la liste
        enfants.append(enfant1)
        enfants.append(enfant2)
    
    # Mélanger les enfants pour plus de diversité
    random.shuffle(enfants)
    return enfants

def cross_by_blocks(parents):
    enfants = []
    # Couple de parents
    for i in range(0, len(parents) - 1, 2):
        parent1, parent2 = parents[i], parents[i + 1]
        
        # Initialisation des enfants
        enfant1 = np.zeros_like(parent1)
        enfant2 = np.zeros_like(parent2)
        
        # Dimensions de la matrice
        mid_row = parent1.shape[0] // 2
        mid_col = parent1.shape[1] // 2
        
        # Enfant 1 : mélange de blocs
        enfant1[:mid_row, :mid_col] = parent1[:mid_row, :mid_col]  # Haut-gauche de parent1
        enfant1[mid_row:, mid_col:] = parent2[mid_row:, mid_col:]  # Bas-droite de parent2
        enfant1[:mid_row, mid_col:] = parent2[:mid_row, mid_col:]  # Haut-droite de parent2
        enfant1[mid_row:, :mid_col] = parent1[mid_row:, :mid_col]  # Bas-gauche de parent1
        
        # Enfant 2 : inverse des blocs
        enfant2[:mid_row, :mid_col] = parent2[:mid_row, :mid_col]  # Haut-gauche de parent2
        enfant2[mid_row:, mid_col:] = parent1[mid_row:, mid_col:]  # Bas-droite de parent1
        enfant2[:mid_row, mid_col:] = parent1[:mid_row, mid_col:]  # Haut-droite de parent1
        enfant2[mid_row:, :mid_col] = parent2[mid_row:, :mid_col]  # Bas-gauche de parent2
        
        # Ajouter les enfants à la liste
        enfants.append(enfant1)
        enfants.append(enfant2)
    
    # Mélanger les enfants pour plus de diversité
    random.shuffle(enfants)
    return enfants

def cross_by_alternating_rows(parents):
    enfants = []
    # Couple de parents
    for i in range(0, len(parents) - 1, 2):
        parent1, parent2 = parents[i], parents[i + 1]
        
        # Initialisation des enfants
        enfant1 = np.zeros_like(parent1)
        enfant2 = np.zeros_like(parent2)
        
        # Alterner les lignes
        for j in range(parent1.shape[0]):
            if j % 2 == 0:
                enfant1[j, :] = parent1[j, :]
                enfant2[j, :] = parent2[j, :]
            else:
                enfant1[j, :] = parent2[j, :]
                enfant2[j, :] = parent1[j, :]
        
        # Ajouter les enfants à la liste
        enfants.append(enfant1)
        enfants.append(enfant2)
    
    # Mélanger les enfants pour plus de diversité
    random.shuffle(enfants)
    return enfants

def cross_by_alternating_line(parents):
    enfants = []
    # Couple de parents
    for i in range(0, len(parents) - 1, 2):
        parent1, parent2 = parents[i], parents[i + 1]
        
        # Initialisation des enfants
        enfant1 = np.zeros_like(parent1)
        enfant2 = np.zeros_like(parent2)
        
        # Alterner les lignes
        for j in range(parent1.shape[1]):
            if j % 2 == 0:
                enfant1[:, j] = parent1[:, j]
                enfant2[:, j] = parent2[:, j]
            else:
                enfant1[:, j] = parent2[:, j]
                enfant2[:, j] = parent1[:, j]
        
        # Ajouter les enfants à la liste
        enfants.append(enfant1)
        enfants.append(enfant2)
    
    # Mélanger les enfants pour plus de diversité
    random.shuffle(enfants)
    return enfants

def cross_by_elem_1_2(parents):
    enfants = []
    #Couple de parents
    for i in range(0, len(parents) - 1, 2):
        parent1,parent2 = parents[i],parents[i+1]
        enfant1 = np.zeros_like(parent1)
        enfant2 = np.zeros_like(parent2)
         # Remplir les enfants en alternant les éléments
        for j in range(parent1.shape[0]):
            for k in range(parent1.shape[1]):
                if (j + k) % 2 == 0:
                    # Enfant1 prend l'élément de parent1, Enfant2 de parent2
                    enfant1[j][k] = parent1[j][k]
                    enfant2[j][k] = parent2[j][k]
                else:
                    # Enfant1 prend l'élément de parent2, Enfant2 de parent1
                    enfant1[j][k] = parent2[j][k]
                    enfant2[j][k] = parent1[j][k]
        enfants.append(enfant1)
        enfants.append(enfant2)
    random.shuffle(enfants)
    return enfants

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
def VNS(M,n_clusters,voisinage,kmax,max_depth = 10, init = None):
    if init is None:
        line_labels = clustering_lines(M, n_clusters)
        col_labels = clustering_columns(M, n_clusters)
        
        # Générer la population initiale
        init_matrix = generate_initial_P(M, line_labels, col_labels,noise_prob=0)
    else:
        init_matrix = init.copy()
    voisinage_index = 0
    best_matrix = init_matrix.copy()
    for i in range(kmax):
        if voisinage_index == 0:
            n1,n2 = random.randint(0, len(M)-1),random.randint(0, len(M[1])-1)
            init_matrix[n1,n2] = -init_matrix[n1,n2]
        elif voisinage_index == 1:
            n1,n2 = random.randint(0, len(M)-1),random.randint(0, len(M[1])-1)
            init_matrix[n1,n2] = -init_matrix[n1,n2]
            init_matrix[-n1,-n2] = -init_matrix[-n1,-n2]
        elif voisinage_index == 2:
            n1,n2 = random.randint(0, len(M)-1),random.randint(0, len(M[1])-1)
            init_matrix[n1,n2] = -init_matrix[n1,n2]
            init_matrix[-n1,n2] = -init_matrix[-n1,n2]
        elif voisinage_index == 3:
            n1,n2 = random.randint(0, len(M)-1),random.randint(0, len(M[1])-1)
            init_matrix[n1,n2] = -init_matrix[n1,n2]
            init_matrix[n1,-n2] = -init_matrix[n1,-n2]
        elif voisinage_index == 4:
            l = random.randint(0,init_matrix.shape[1]-1)
            init_matrix[l,:] = -init_matrix[l,:]
        elif voisinage_index == 5:
            c = random.randint(0,init_matrix.shape[0]-1)
            init_matrix[:,c] = -init_matrix[:,c]
        
        
        init_matrix = full_local_search(M,init_matrix, voisinage,max_depth)
        # if compareP1betterthanP2(M, init_matrix_2, init_matrix):
        #     pass
        # else:
        #     pass
        
        print(f"iteration {i}")
        if compareP1betterthanP2(M, init_matrix, best_matrix):
            best_matrix = init_matrix.copy()
            print(f"improve by voisinage {voisinage_index}")
        else:
            init_matrix = best_matrix.copy()
        voisinage_index = (voisinage_index+1)%6
    return best_matrix
            
def recherche_kmins(M,n_clusters,max_iter):
    line_labels = clustering_lines(M, n_clusters)
    col_labels = clustering_columns(M, n_clusters)
    matrice = generate_initial_P(M, line_labels, col_labels,noise_prob=0.0)
    matrice = full_local_search(M, matrice,0,100)
    best_sol = matrice.copy()
    for i in range(max_iter):
        matrice = generate_initial_P(M, line_labels, col_labels,noise_prob=0.1)
        matrice = full_local_search(M, matrice,0,100)
        if compareP1betterthanP2(M, matrice, best_sol):
            best_sol = matrice.copy()
            print(f"iteration {i} improve")
    return best_sol
   
    
#optimal_k(M)

# o_matrix = np.random.choice([-1, 1], size=M.shape)
# print(fobj(M,full_local_search(o_matrix,0)))
list_cross = [uniform_crossover]#[#cross_by_half_split,cross_by_vertical_split,cross_by_elem_1_2,cross_by_alternating_rows,cross_by_alternating_line,cross_by_blocks,uniform_crossover,one_point_crossover]
#M_list = divide_matrix(M)
list_sol = list()

# for elem in M_list:
#     genetique_matrix = genetique(elem,2,0,list_cross,0.20,True,50,10,n_parents = 50,parent_init = None)
#     list_sol.append(genetique_matrix)
# sol =  reassemble_matrix(list_sol)
# print(fobj(reel_matrix,sol))
# genetique_matrix = genetique(M,2,0,list_cross,0.20,True,50,10   ,n_parents = 100, parent_init = sol)
# liste_parent = [None]
# for i in range(1):
# #     liste_parent.append(_random_matrix(100))
genetique_matrix = genetique(M,2,0,list_cross,0.20,False,100,max_depth=10,n_parents = 100,parent_init=None,method_next_gen="Best")
print(fobj(M,genetique_matrix))

# print(fobj(reel_matrix,sol))

# VNS_matrix = VNS(M,2,0,1000,max_depth = 10,init = genetique_matrix)
# print(fobj(M,VNS_matrix))
# VNS_matrix = VNS(M,2,0,2000,max_depth = 10)
# print(fobj(M,VNS_matrix))

# k_mins_search = recherche_kmins(M, 2, 1000)
# print(fobj(M,k_mins_search))

# dict_values([1, 1, 1, 27, 1, 6, 5, 7, 2, 46, 1, 2])
# (12, 0.718034780966193)
#reel_matrix = matrices1_ledm(25)
#2 means, 100 parents
# [[ 1. -1. -1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1. -1. -1. -1. -1. -1.
#   -1. -1. -1. -1. -1. -1. -1.]
#  [ 1.  1. -1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1. -1. -1. -1. -1. -1.
#   -1. -1. -1. -1. -1. -1. -1.]
#  [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1. -1. -1. -1. -1. -1.
#   -1. -1. -1. -1. -1. -1. -1.]
#  [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1. -1. -1. -1. -1. -1.
#   -1. -1. -1. -1. -1. -1. -1.]
#  [ 1.  1.  1. -1.  1.  1.  1.  1.  1.  1.  1.  1.  1. -1. -1. -1. -1. -1.
#   -1. -1. -1. -1. -1. -1. -1.]
#  [ 1.  1.  1. -1. -1.  1.  1.  1.  1.  1.  1.  1.  1. -1. -1. -1. -1. -1.
#   -1. -1. -1. -1. -1. -1. -1.]
#  [ 1.  1.  1.  1.  1.  1.  1. -1.  1.  1.  1.  1.  1. -1. -1. -1. -1. -1.
#   -1. -1. -1. -1. -1. -1. -1.]
#  [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1. -1. -1. -1. -1. -1.
#   -1. -1. -1. -1. -1. -1. -1.]
#  [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1. -1. -1. -1. -1. -1.
#   -1. -1. -1. -1. -1. -1. -1.]
#  [ 1.  1.  1.  1.  1.  1.  1.  1. -1.  1.  1.  1.  1. -1. -1. -1. -1. -1.
#   -1. -1. -1. -1. -1. -1. -1.]
#  [-1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1. -1. -1. -1. -1. -1. -1.
#   -1. -1. -1.  1. -1. -1. -1.]
#  [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1. -1. -1. -1. -1. -1. -1.
#   -1. -1. -1. -1. -1. -1. -1.]
#  [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1. -1. -1. -1. -1. -1.
#   -1. -1. -1. -1. -1. -1. -1.]
#  [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.  1.  1.  1.  1.  1.
#    1.  1.  1.  1.  1.  1.  1.]
#  [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.  1.  1.  1.  1.
#    1.  1.  1.  1.  1.  1.  1.]
#  [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.  1.  1.  1.
#    1.  1.  1.  1.  1.  1.  1.]
#  [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.  1.  1.  1.  1. -1.
#    1.  1.  1.  1.  1.  1.  1.]
#  [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.  1.  1.  1.  1.  1.
#    1.  1.  1.  1.  1.  1.  1.]
#  [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.  1.  1.  1.  1.  1.
#    1.  1.  1.  1.  1.  1.  1.]
#  [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.  1.  1.  1.  1.  1.
#   -1.  1.  1.  1.  1.  1.  1.]
#  [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.  1.  1.  1.  1.  1.
#    1.  1.  1.  1. -1.  1.  1.]
#  [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.  1.  1.  1.  1.  1.
#    1.  1.  1.  1. -1.  1.  1.]
#  [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.  1.  1.  1.  1.  1.
#    1.  1.  1.  1.  1.  1.  1.]
#  [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.  1.  1.  1.  1.  1.
#    1.  1.  1.  1.  1.  1.  1.]
#  [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.  1.  1.  1.  1.  1.
#    1.  1.  1.  1.  1. -1.  1.]]
