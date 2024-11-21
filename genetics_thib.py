import random
import numpy as np

from opti_combi_projet_pythoncode_texte import fobj,compareP1betterthanP2
import utils

reel_matrix= utils.lire_fichier("data/exempleslide_matrice (1).txt")

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

def genetique(parents,voisinage,list_methode_cross,mutation_rate,memetique,time):
    best_matrice = parents[0]
    for i in range(time):
        if len(parents) != 100:
            pass
        random.shuffle(parents)
        #Creation enfants
        methode_cross = list_methode_cross[i%2]
        enfants : list = methode_cross(parents)

        enfants,enfants_mute = enfants[:int(len(enfants)*mutation_rate)],enfants[int(len(enfants)*mutation_rate):]
        
        
        if memetique:
            for i in range(len(enfants_mute)//2):
                enfants_mute[i] = full_local_search(enfants_mute[i],voisinage)
            for i in range(len(enfants_mute)//2, len(enfants_mute)):
                n1,n2 = random.randint(0, len(reel_matrix)-1),random.randint(0, len(reel_matrix[0])-1)
                enfants_mute[i][n1][n2] = -enfants_mute[i][n1][n2]
        else:
            for i in range(enfants_mute):
                n1,n2 = random.randint(0, len(reel_matrix)-1),random.randint(0, len(reel_matrix[1])-1)
                enfants_mute[i][n1][n2] = -enfants_mute[i][n1][n2]
        enfants += enfants_mute

        parents += enfants

        # Sélection des meilleurs sujets
        parents = [(fobj(reel_matrix, parent), parent) for parent in parents]
        parents = [x[1] for x in sorted(parents, key=lambda x: (x[0][0], x[0][1]))[:len(parents) // 2]]

        if compareP1betterthanP2(reel_matrix,parents[0],best_matrice):
            best_matrice = parents[0]

    count_dict = {}
    for matrice in parents:
        key = tuple(matrice.flatten())
        count_dict[key] = count_dict.get(key, 0) + 1

    # Afficher les occurrences (optionnel)
    print(count_dict.values())
    return best_matrice


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
list_cross = [cross_by_half_split,cross_by_vertical_split]
print(fobj(reel_matrix,genetique(random_matrix(100),0,list_cross,0.25,True,1000)))





