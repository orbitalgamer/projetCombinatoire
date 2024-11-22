import random
import numpy as np
from sklearn.cluster import KMeans

from opti_combi_projet_pythoncode_texte import fobj,compareP1betterthanP2,matrices1_ledm
import utils

#reel_matrix= utils.lire_fichier("data/exempleslide_matrice (1).txt")
reel_matrix = matrices1_ledm(25)
M = reel_matrix

def local_search(matrice_init,voisinage):
    new_matrice = matrice_init
    best_matrice = new_matrice
    if voisinage == 0:
        for i in range(len(matrice_init)):
            for j in range(len(matrice_init[0])):
                new_matrice[i][j] = -new_matrice[i][j]
                if compareP1betterthanP2(reel_matrix,new_matrice,matrice_init):
                    best_matrice = new_matrice
                new_matrice[i][j] = -new_matrice[i][j]
    return best_matrice

def full_local_search(matrice_init,voisinage):
    best_matrice = matrice_init
    while(True):
        new_matrice = local_search(best_matrice,voisinage)
        if compareP1betterthanP2(reel_matrix,new_matrice,best_matrice):
            best_matrice = new_matrice
        else:
            break
    return best_matrice

def random_matrix(n):
    list_matrix = list()
    for _ in range(n):
        list_matrix.append(np.random.choice([-1, 1], size=reel_matrix.shape))
    return list_matrix


def clustering_lines(M, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(M)
    return labels

def clustering_columns(M, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(M.T)
    return labels

def generate_initial_P(M, line_labels, col_labels):
    P = np.zeros_like(M)
    unique_line_labels = np.unique(line_labels)
    unique_col_labels = np.unique(col_labels)
    
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            P[i, j] = 1 if (line_labels[i] + col_labels[j]) % 2 == 0 else -1
    return P

def genetique(M,n_clusters,voisinage,list_methode_cross,mutation_rate,memetique,time):

    # Clustering des lignes et des colonnes
    line_labels = clustering_lines(M, n_clusters)
    col_labels = clustering_columns(M, n_clusters)
    
    # Générer la population initiale
    parents = [generate_initial_P(M, line_labels, col_labels) for _ in range(100)]

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
                enfants_mute[i] = full_local_search(enfants_mute[i],voisinage)
            for i in range(len(enfants_mute)//2, len(enfants_mute)):
                n1,n2 = random.randint(0, len(reel_matrix)-1),random.randint(0, len(reel_matrix[0])-1)
                enfants_mute[i][n1][n2] = -enfants_mute[i][n1][n2]
        else:
            for i in range(len(enfants_mute)):
                n1,n2 = random.randint(0, len(reel_matrix)-1),random.randint(0, len(reel_matrix[1])-1)
                enfants_mute[i][n1][n2] = -enfants_mute[i][n1][n2]
        enfants += enfants_mute

        parents += enfants

        # Sélection des meilleurs sujets
        parents = [(fobj(reel_matrix, parent), parent) for parent in parents]
        parents = [x[1] for x in sorted(parents, key=lambda x: (x[0][0], x[0][1]))[:len(parents) // 2]]
        print(f"{t} sur {time}")
        if compareP1betterthanP2(reel_matrix,parents[0],best_matrice):
            best_matrice = parents[0]
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
#optimal_k(M)

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


# o_matrix = np.random.choice([-1, 1], size=reel_matrix.shape)
# print(fobj(reel_matrix,full_local_search(o_matrix,0)))
list_cross = [cross_by_half_split,cross_by_vertical_split,cross_by_elem_1_2,cross_by_alternating_rows,cross_by_alternating_line,cross_by_blocks]
genetique_matrix = genetique(M,2,0,list_cross,0.20,True,100)
print(fobj(reel_matrix,genetique_matrix))


# dict_values([2, 5, 1, 77, 12, 1, 2])
# (13, 0.9108630313464734)
#reel_matrix = matrices1_ledm(25)
#2 means, 100 parents
# [[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1. -1. -1. -1. -1. -1.
#   -1. -1. -1. -1. -1. -1. -1.]
#  [-1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1. -1. -1. -1. -1. -1.
#   -1. -1. -1. -1. -1. -1. -1.]
#  [ 1.  1.  1. -1. -1.  1.  1.  1.  1.  1.  1.  1.  1. -1. -1. -1. -1. -1.
#   -1. -1. -1. -1. -1. -1. -1.]
#  [ 1.  1.  1. -1. -1.  1.  1.  1.  1.  1.  1.  1.  1. -1. -1. -1. -1. -1.
#   -1. -1. -1. -1. -1. -1. -1.]
#  [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1. -1. -1. -1. -1. -1.
#   -1. -1. -1. -1. -1. -1. -1.]
#  [ 1.  1.  1.  1.  1. -1.  1.  1.  1.  1.  1.  1.  1. -1. -1. -1. -1. -1.
#   -1. -1. -1. -1. -1. -1. -1.]
#  [ 1.  1.  1.  1.  1. -1.  1.  1.  1.  1.  1.  1.  1. -1. -1. -1. -1. -1.
#   -1. -1. -1. -1. -1. -1. -1.]
#  [ 1.  1.  1.  1.  1.  1.  1.  1. -1.  1.  1.  1.  1. -1. -1. -1. -1. -1.
#   -1. -1. -1. -1. -1. -1. -1.]
#  [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1. -1. -1. -1. -1. -1.
#   -1. -1. -1. -1. -1. -1. -1.]
#  [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1. -1. -1. -1. -1. -1.
#   -1. -1. -1. -1. -1. -1. -1.]
#  [ 1.  1.  1.  1.  1.  1.  1.  1.  1. -1.  1.  1.  1. -1. -1. -1. -1. -1.
#   -1. -1. -1. -1. -1. -1. -1.]
#  [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1. -1. -1. -1. -1. -1. -1.
#   -1. -1. -1. -1. -1. -1. -1.]
#  [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1. -1. -1. -1. -1. -1.
#   -1. -1. -1. -1. -1. -1. -1.]
#  [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.  1.  1.  1.  1.  1.
#    1.  1.  1.  1.  1.  1.  1.]
#  [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.  1.  1.  1.  1.
#    1.  1.  1.  1.  1.  1.  1.]
#  [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.  1.  1.  1.  1.  1.
#   -1. -1. -1.  1. -1.  1.  1.]
#  [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.  1.  1.  1.  1.  1.
#    1.  1.  1. -1.  1.  1.  1.]
#  [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.  1.  1.  1.  1.  1.
#   -1.  1.  1.  1.  1.  1.  1.]
#  [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.  1.  1.  1.  1.  1.
#    1.  1.  1.  1.  1.  1.  1.]
#  [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.  1.  1.  1.  1.  1.
#    1.  1.  1.  1.  1.  1.  1.]
#  [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.  1.  1.  1.  1.  1.
#    1. -1.  1.  1.  1.  1.  1.]
#  [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.  1.  1.  1.  1.  1.
#    1.  1.  1.  1.  1.  1.  1.]
#  [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.  1.  1.  1.  1.  1.
#    1.  1.  1.  1.  1. -1. -1.]
#  [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.  1.  1.  1.  1.  1.
#    1.  1.  1.  1.  1.  1. -1.]
#  [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.  1.  1.  1.  1.  1.
#    1.  1.  1.  1.  1.  1.  1.]]




