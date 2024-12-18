import copy
import itertools
import random
import time
from joblib import Parallel, delayed
import numpy as np
from sklearn.cluster import KMeans

from opti_combi_projet_pythoncode_texte import fobj,compareP1betterthanP2,matrices1_ledm,matrices2_slackngon
from utils import LEDM,lire_fichier,random_matrix,pat_ledm, get_pc_name
import utils
import matplotlib.pyplot as plt

best_sol_list = []
#%%
#reel_matrix= utils.lire_fichier("data/exempleslide_matrice (1).txt")
#reel_matrix= utils.lire_fichier("data/ledm6_matrice (1).txt")
#reel_matrix= utils.lire_fichier("data/correl5_matrice.txt")

#reel_matrix = LEDM(120,120)
#reel_matrix = reel_matrix.transpose()
#reel_matrix = matrices2_slackngon(16)
#reel_matrix = random_matrix(50,50,3)



def local_search_random_unique(M,matrice_init, voisinage, max_attempts=200):
    new_matrice = matrice_init.copy()
    best_matrice = matrice_init.copy()
    visited_positions = set()

    for _ in range(max_attempts):
        while True:
            i = random.randint(0, len(matrice_init) - 1)
            j = random.randint(0, len(matrice_init[0]) - 1)
            if (i, j) not in visited_positions:
                visited_positions.add((i, j))
                break


        new_matrice[i][j] = -new_matrice[i][j]

        if compareP1betterthanP2(M, new_matrice, best_matrice ):
            best_matrice = new_matrice.copy()
        else:

            new_matrice[i][j] = -new_matrice[i][j]

        # Arrêter si toutes les positions possibles ont été visitées
        if len(visited_positions) >= len(matrice_init) * len(matrice_init[0]):
            break

    return best_matrice


def local_search(M,matrice_init,voisinage):
    new_matrice = matrice_init.copy()
    best_matrice = new_matrice.copy()
    stop = False
    if voisinage == 0:
        for i in range(len(matrice_init)):
            if stop == True:
                break
            for j in range(len(matrice_init[0])):
                if stop == True:
                    break
                new_matrice[i][j] = -new_matrice[i][j]
                if compareP1betterthanP2(M,new_matrice,matrice_init ):
                    best_matrice = new_matrice.copy()
                    print(fobj(M,best_matrice))
                    stop = True
                    break
                new_matrice[i][j] = -new_matrice[i][j]
    return best_matrice

def full_local_search(M,matrice_init,voisinage,max_depth = 10):
    best_matrice = matrice_init.copy()
    for i in range(max_depth):
        new_matrice = local_search_random_unique(M,best_matrice,voisinage)
        if compareP1betterthanP2(M,new_matrice,best_matrice ):
            best_matrice = new_matrice.copy()
        else:

            break
    return best_matrice


def clustering_lines(M, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(M)
    return labels

def clustering_columns(M, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(M.T)
    return labels

def generate_initial_P(M, line_labels, col_labels,noise_prob = 0):
    P = np.zeros_like(M)
    
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            P[i, j] = 1 if (line_labels[i] + col_labels[j]) % 2 == 0 else -1
            if np.random.rand() < noise_prob: 
                P[i,j] = -P[i,j]
    return P

def genetique(M,n_clusters,voisinage,list_methode_cross,mutation_rate,memetique,time,max_depth,n_parents, parent_init = None,method_next_gen = "Best"):
    diffusion =0
    choose_cross = 0
    if parent_init is None:
        # Clustering des lignes et des colonnes
        line_labels = clustering_lines(M, n_clusters)
        col_labels = clustering_columns(M, n_clusters)
        
        # Générer la population initiale
        parents = [ generate_initial_P(M, line_labels, col_labels,noise_prob=0.00) for i in range(n_parents)]
        #parents += [generate_initial_P(M, line_labels, col_labels,noise_prob=0.05) for _ in range(50)]
    elif parent_init[0] is None:
        parents = parent_init[1]
        diffusion =4
    else:
        parents = [parent_init.copy() for i in range(n_parents)]

    best_matrice = parents[0]
    for t in range(time):
        if len(parents) != 100:
            pass
        random.shuffle(parents)

        #Creation enfants
        methode_cross = list_methode_cross[choose_cross%len(list_methode_cross)]
        enfants = methode_cross(parents)
        random.shuffle(enfants)
        enfants,enfants_mute = enfants[:int(len(enfants)*mutation_rate)],enfants[int(len(enfants)*mutation_rate):]
        
        if memetique:
            for i in range(2):
                enfants_mute[i] = full_local_search(M,enfants_mute[i],voisinage,max_depth)
            for i in range(2,len(enfants_mute)):
                n1,n2 = random.randint(0, len(M)-1),random.randint(0, len(M[0])-1)
                enfants_mute[i][n1][n2] = -enfants_mute[i][n1][n2]
        else:
            for i in range(len(enfants_mute)):
                n1,n2 = random.randint(0, len(M)-1),random.randint(0, len(M[1])-1)
                enfants_mute[i][n1][n2] = -enfants_mute[i][n1][n2]
        enfants += enfants_mute

        parents += enfants

        # Sélection
        if diffusion > 0:
            diffusion -= 1
        else : 
            if method_next_gen == "Best":
                if len(parents) >2* n_parents:
                    parents = [(fobj(M, parent ), parent) for parent in parents]
                    parents = [x[1] for x in sorted(parents, key=lambda x: (x[0][0], x[0][1]))[:len(parents)//4]]
                else:
                    parents = [(fobj(M, parent ), parent) for parent in parents]
                    parents = [x[1] for x in sorted(parents, key=lambda x: (x[0][0], x[0][1]))[:len(parents)//2]]

            elif method_next_gen == "Tournament":
                random.shuffle(parents)
                new_parents = []
                if len(parents) >2* n_parents:
                    for i in range(0,len(parents),2):
                        if compareP1betterthanP2(M, parents[i], parents[i+1] ):
                            new_parents.append(parents[i])
                        else:
                            new_parents.append(parents[i+1])
                    parents = new_parents.copy()
                new_parents = []
                for i in range(0,len(parents),2):
                    if compareP1betterthanP2(M, parents[i], parents[i+1] ):
                        new_parents.append(parents[i])
                    else:
                        new_parents.append(parents[i+1])
                parents = new_parents.copy()
            elif method_next_gen == "Tournament_pro":
                while len(parents) > n_parents:
                    parents = [(fobj(M, parent ), parent) for parent in parents]
                    challenger = [x[1] for x in sorted(parents, key=lambda x: (x[0][0], x[0][1]))[len(parents) // 4:3*len(parents) // 4]]
                    parents = [x[1] for x in sorted(parents, key=lambda x: (x[0][0], x[0][1]))[:len(parents) // 4]]
                random.shuffle(challenger)
                new_challenger = []
                for i in range(0,len(challenger),2):
                    if compareP1betterthanP2(M, challenger[i], challenger[i+1] ):
                        new_challenger.append(challenger[i])
                    else:
                        new_challenger.append(challenger[i+1])
                parents += new_challenger 

        print(f"{t} sur {time}")
        if compareP1betterthanP2(M,parents[0],best_matrice ):
            best_matrice = parents[0].copy()
            
            print(fobj(M,best_matrice))
            print(methode_cross.__name__)
            print(f"improve")
        else:
            choose_cross += 1

    count_dict = {}
    for matrice in parents:
        key = tuple(matrice.flatten())
        count_dict[key] = count_dict.get(key, 0) + 1

    #Show occurence
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



def perm(type: int,mat:np.ndarray,index: int,index2=None):
    mat_tmp=copy.deepcopy(mat)
    if type==0: #Voisinnage multiplie un terme par -1
        x=index//mat.shape[1]
        y=index%mat.shape[1]
        mat_tmp[x][y]*=-1
    elif type==1:
        mat_tmp[index,:]*=-1
    elif type==2:
        mat_tmp[:,index]*=-1
    elif type==3:
        mat_tmp[index,:],mat_tmp[index2,:]=(mat_tmp[index2,:],mat_tmp[index,:])
    elif type==4:
        mat_tmp[:,index],mat_tmp[:,index2]=(mat_tmp[:,index2],mat_tmp[:,index])
    return mat_tmp

def greedy(matrix,pattern,setup_break,la_totale,verbose=False):
    if matrix.size==1 and matrix[0][0]==0:
        return pattern
    counter=0
    while counter<1:
        counter+=1
        for i in range(matrix.shape[0]*matrix.shape[1]):   
            pattern_tmp=perm(0,pattern,i)
            if compareP1betterthanP2(matrix,pattern_tmp,pattern):
                pattern=copy.deepcopy(pattern_tmp)
                if verbose:
                    print(f"0 rank: {fobj(matrix,pattern)[0]}, valeur min: {fobj(matrix,pattern)[1]}")
                counter=0
                if setup_break==1 or setup_break==3:break
        if la_totale:
            for i in range(matrix.shape[0]):   
                pattern_tmp=perm(1,pattern,i)
                if compareP1betterthanP2(matrix,pattern_tmp,pattern):
                    pattern=copy.deepcopy(pattern_tmp)
                    if verbose:
                        print(f"1 rank: {fobj(matrix,pattern)[0]}, valeur min: {fobj(matrix,pattern)[1]}")
                    counter=0
                    if setup_break==1 or setup_break==3:break
            for i in range(matrix.shape[1]):   
                pattern_tmp=perm(2,pattern,i)
                if compareP1betterthanP2(matrix,pattern_tmp,pattern):
                    pattern=copy.deepcopy(pattern_tmp)
                    if verbose:
                        print(f"2 rank: {fobj(matrix,pattern)[0]}, valeur min: {fobj(matrix,pattern)[1]}")
                    counter=0
                    if setup_break==1 or setup_break==3:break
            for i in range(matrix.shape[0]):
                for j in range(i,matrix.shape[0]):    
                    pattern_tmp=perm(3,pattern,i,j)
                    if compareP1betterthanP2(matrix,pattern_tmp,pattern):
                        pattern=copy.deepcopy(pattern_tmp)
                        if verbose:
                            print(f"3 rank: {fobj(matrix,pattern)[0]}, valeur min: {fobj(matrix,pattern)[1]}")
                        counter=0
                        if setup_break==1 or setup_break==3:break
                else:
                    continue
                if setup_break==2 or setup_break==3:break
            for i in range(matrix.shape[1]):
                for j in range(i,matrix.shape[1]):    
                    pattern_tmp=perm(4,pattern,i,j)
                    if compareP1betterthanP2(matrix,pattern_tmp,pattern):
                        pattern=copy.deepcopy(pattern_tmp)
                        if verbose:
                            print(f"4 rank: {fobj(matrix,pattern)[0]}, valeur min: {fobj(matrix,pattern)[1]}")
                        counter=0
                        if setup_break==1 or setup_break==3:break
                else:
                    continue
                if setup_break==2 or setup_break==3:break
    return pattern

def subdivise_mat(mat,size):

    list_mat=[]
    for i in range(mat.shape[0]//size+1):
        for j in range(mat.shape[1]//size+1):
            tmp=mat[i*size:(i+1)*size,j*size:(j+1)*size]
            if tmp.size!=0:
                list_mat.append(tmp)
    return list_mat

def reassemble_mat(mat,size,list_mat):
    x=mat.shape[0]//size
    if mat.shape[0]%size:
        x+=1
    y=mat.shape[1]//size
    if mat.shape[1]%size:
        y+=1
    list_math=[]
    for i in range(x):
        list_math.append(np.hstack(list_mat[i*y:i*y+y]))
    matrix=np.vstack(list_math)
    return matrix

def Resolve_metaheuristic(funct,matrix,pattern,param,verbose=False):
        print(f"Testing for size={param[0]}, param2={param[1]} and param3={param[2]}")
        list_mat=subdivise_mat(matrix,param[0])
        list_pat=subdivise_mat(pattern,param[0])
        for i in range(len(list_pat)):
            list_pat[i]=funct(list_mat[i], list_pat[i],param[1],param[2],verbose)
        pattern_tmp=reassemble_mat(pattern,param[0],list_pat)
        pattern_tmp=funct(matrix,pattern_tmp,param[1],param[2],verbose)
        return (pattern_tmp,param)

def VNS(M,n_clusters,voisinage, kmax,matrix_name=None, max_depth = 10, init = None):
    #initialise à 0
    dico_ameliration = dict()
    for i in range(6):
        dico_ameliration[i]=0
    
    if init is None:
        line_labels = clustering_lines(M, n_clusters)
        col_labels = clustering_columns(M, n_clusters)
        
        # Générer la population initiale
        init_matrix = generate_initial_P(M, line_labels, col_labels,noise_prob=0)
    else:
        init_matrix = init.copy()
    voisinage_index = 0
    best_matrix = init_matrix.copy()
    if mat_name != None:
        utils.ecrire_fichier(matrix_name,M,best_matrix)
    else:
        utils.ecrire_fichier("solution.txt",M,best_matrix)
    n_not_best = 1
    for i in range(kmax):
        if voisinage_index == 0:
            for _ in range(int(n_not_best**0.33)):
                n1,n2 = random.randint(0, len(M)-1),random.randint(0, len(M[1])-1)
                init_matrix[n1,n2] = -init_matrix[n1,n2]
        elif voisinage_index == 1:
            for _ in range(int(n_not_best**0.33)):
                n1,n2 = random.randint(0, len(M)-1),random.randint(0, len(M[1])-1)
                init_matrix[n1,n2] = -init_matrix[n1,n2]
                init_matrix[-n1,-n2] = -init_matrix[-n1,-n2]
        elif voisinage_index == 2:
            for _ in range(int(n_not_best**0.33)):
                n1,n2 = random.randint(0, len(M)-1),random.randint(0, len(M[1])-1)
                init_matrix[n1,n2] = -init_matrix[n1,n2]
                init_matrix[-n1,n2] = -init_matrix[-n1,n2]
        elif voisinage_index == 3:
            for _ in range(int(n_not_best**0.33)):
                n1,n2 = random.randint(0, len(M)-1),random.randint(0, len(M[1])-1)
                init_matrix[n1,n2] = -init_matrix[n1,n2]
                init_matrix[n1,-n2] = -init_matrix[n1,-n2]
        elif voisinage_index == 4:
            for _ in range(int(n_not_best**0.33)):
                l = random.randint(0,init_matrix.shape[0]-1)
                init_matrix[l,:] = -init_matrix[l,:]
        elif voisinage_index == 5:
            for _ in range(int(n_not_best**0.33)):
                c = random.randint(0,init_matrix.shape[1]-1)
                init_matrix[:,c] = -init_matrix[:,c]
        
        
        init_matrix = greedy(M,init_matrix,0,False)

        
        print(f"iteration {i}")
        if compareP1betterthanP2(M, init_matrix, best_matrix):
            best_matrix = init_matrix.copy()
            n_not_best = 1
            print(f"improve by voisinage {voisinage_index}")
            utils.ecrire_fichier("solution.txt",M,best_matrix)
            dico_ameliration[voisinage_index]+=1
        else:
            init_matrix = best_matrix.copy()
            if n_clusters < 150:
                n_not_best += 1
        voisinage_index = (voisinage_index+1)%6
    return best_matrix, dico_ameliration
            
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
   
def Bruiteur_matrix(matrice,noise_prob):
    P = matrice.copy()
    for i in range(matrice.shape[0]):
        for j in range(matrice.shape[1]):
            if np.random.rand() < noise_prob:  # Avec une certaine probabilité, changer le cluster
                P[i,j] = -P[i,j]
    return P

def GenerationParents(M,nombre,methods):
    nombre_par_groupe = nombre//len(methods)
    n_good = 2
    liste_parent = []
    bruit = 0
    for init_method in methods:
        for i in range(n_good):
            liste_parent.append(init_method)
        for i in range(nombre_par_groupe-n_good):
            bruit += 0.002
            bruit *= 1.01
            liste_parent.append(Bruiteur_matrix(init_method,bruit))
    if len(liste_parent) < nombre:
        liste_parent.append(np.ones(M))
    elif len(liste_parent) > nombre:
        while(len(liste_parent) > nombre):
            liste_parent.pop()
    return liste_parent

def Johanmethod(matrix, debug = True,best_param = False,pattern = None, random_search=True):
    debug=debug
    best_param=best_param
    metah=0 #0 for greedy, 1 for tabu, 2 for local search
    matrix = matrix
    # if pattern == None:
    #     pattern=np.ones(matrix.shape)
    #     #pattern=np.random.choice([-1,1],size=matrix.shape)
    # else:
    pattern = pattern
    if best_param:
        #determination meilleur parametre
        start_time=time.time()
        pattern_best=copy.deepcopy(pattern)
        if metah==0:
            if not random_search:
                la_totale=[False,True]
                setup_break=range(4)
                size=range(2,max(matrix.shape)+1)
                param=itertools.product(la_totale,setup_break,size)
                data=Parallel(n_jobs=-1)(delayed(Resolve_metaheuristic)(greedy,matrix,pattern,(i[2],i[1],i[0])) for i in param)
                for (pattern_tmp,p) in data:
                    if compareP1betterthanP2(matrix,pattern_tmp,pattern_best):
                        pattern_best=copy.deepcopy(pattern_tmp)
                        size_best=p[0]
                        setup_break_best=p[1]
                        la_totale_best=p[2]
                        print(f"for param size={size_best}, setup_break={setup_break_best} and la_totale={la_totale_best} rank: {fobj(matrix,pattern_best)[0]}, valeur min: {fobj(matrix,pattern_best)[1]}")
                        
            else:
                #recherche aléatoire pour limiter nombre de test qu'on fait
                la_totale=[False,True]
                setup_break=range(1)
                size=range(2,35)
                param=itertools.product(la_totale,setup_break,size)

                nombreDeTest = 50 if 50>len(la_totale)*len(setup_break)*len(size) else len(la_totale)*len(setup_break)*len(size)

                indTest = set(np.random.choice(len(la_totale)*len(setup_break)*len(size), 50, replace=False))
                testsur =[]
                for i,j in enumerate(param):
                    if i in indTest:
                        testsur.append(j)


                data=Parallel(n_jobs=-1)(delayed(Resolve_metaheuristic)(greedy,matrix,pattern,(i[2],i[1],i[0])) for i in testsur)
                for (pattern_tmp,p) in data:
                    if compareP1betterthanP2(matrix,pattern_tmp,pattern_best):
                        pattern_best=copy.deepcopy(pattern_tmp)
                        size_best=p[0]
                        setup_break_best=p[1]
                        la_totale_best=p[2]
                        print(f"for param size={size_best}, setup_break={setup_break_best} and la_totale={la_totale_best} rank: {fobj(matrix,pattern_best)[0]}, valeur min: {fobj(matrix,pattern_best)[1]}")
            
            print(f"param opti: size={size_best}, setup_break={setup_break_best} and la_totale={la_totale_best}")
        elif metah==1:
            queue=range(1,11)
            size=range(2,max(matrix.shape)+1)
            param=itertools.product(queue,size)
            data=Parallel(n_jobs=-1)(delayed(Resolve_metaheuristic)(tabu,matrix,pattern,(i[1],i[0])) for i in param)
            for (pattern_tmp,p) in data:
                if compareP1betterthanP2(matrix,pattern_tmp,pattern_best):
                    pattern_best=copy.deepcopy(pattern_tmp)
                    size_best=p[0]
                    queue_best=p[1]
                    print(f"for param size={size_best} and queue={queue_best} rank: {fobj(matrix,pattern_best)[0]}, valeur min: {fobj(matrix,pattern_best)[1]}")
            
            print(f"param opti: size={size_best} and queue={queue_best}")
        elif metah==2:
            la_totale=[False,True]
            size=range(2,max(matrix.shape)+1)
            param=itertools.product(la_totale,size)
            data=Parallel(n_jobs=-1)(delayed(Resolve_metaheuristic)(recherche_locale,matrix,pattern,(i[1],i[0])) for i in param)
            for (pattern_tmp,p) in data:
                if compareP1betterthanP2(matrix,pattern_tmp,pattern_best):
                    pattern_best=copy.deepcopy(pattern_tmp)
                    size_best=p[0]
                    la_totale_best=p[1]
                    print(f"for param size={size_best} and la_totale={la_totale_best} rank: {fobj(matrix,pattern_best)[0]}, valeur min: {fobj(matrix,pattern_best)[1]}")
            print(f"param opti: size={size_best} and la_totale={la_totale_best}")

        print(fobj(matrix,pattern_best))

        end_time=time.time()
        print(f"temps de calcul pour trouve param opti= {end_time-start_time}s")

    if debug:
        start_time=time.time()
        if not best_param:
            size_best=7
            setup_break_best=0 #0,1,2 or 3
            la_totale_best=True #True or False
        if metah==0:
            (pattern_tmp,p)=Resolve_metaheuristic(greedy,matrix,pattern,(size_best,setup_break_best,la_totale_best),verbose=True)
        end_time=time.time()
        print(fobj(matrix,pattern_tmp))
        print(f"temps de calcul pour calculer solution= {end_time-start_time}s")

    
    utils.ecrire_fichier("solution.txt",matrix,pattern_tmp)
    return pattern_tmp
def Clustermethod(M, n_clusters):
    line_labels = clustering_lines(M, n_clusters)
    col_labels = clustering_columns(M, n_clusters)

    cluster = generate_initial_P(M, line_labels, col_labels)
    return cluster

if __name__=="__main__":
    mat_name = "LEDM 32 x 25"
    save_name = f"{mat_name}-{get_pc_name()}-{int(time.time()*1000)}.txt"
    print(f"va etre sauvegarder sous le nom {save_name}")
    import os
    print(os.getcwd())
    # M = utils.lire_fichier(f"./data/{mat_name}.txt")
    M = LEDM(30,25)

    #param johan
    best_Param = True #pour calculer best_param
    random_parm = False #pour rehcercher les param par random bien pout >30 en taille car ainsi reste calculable
    #choix matrice de base
    pat_johan_init=np.ones(M.shape)
    #pat_johan_init=np.random.choice([-1,1],size=M.shape)

    #param VNS
    nbr_cluster = 2 #nombre de cluster pour quand lance init kmeans mais osef si init avec johan
    voisinage  = 0 #je sais pas ce que représente
    maxIteration = 1000 #nombre d'itération que fait
    max_depth = 10 #aucune idée à quoi correspond
    
    
    list_cross = [cross_by_half_split,cross_by_vertical_split,cross_by_alternating_rows,cross_by_alternating_line,cross_by_blocks]
    evolution_rank = np.zeros((3,2)) #taille initial, init johan, génétique 1, génétique 2, VNS
    methode_utilisee = ["initial", "initialisation johan", "VNS"]
    evolution_rank[0] = [np.min(M.shape),-1]

    import time
    a = time.time()
    liste_method = []
    print("Start Johan")
    johan_method = Johanmethod(best_param=True, matrix=M, pattern=pat_johan_init, random_search=random_parm)
    evolution_rank[1] = fobj(M, johan_method)
    print(fobj(M,johan_method))



    VNS_matrix, dico = VNS(M,nbr_cluster,voisinage,maxIteration,max_depth = max_depth,init = johan_method.copy(), matrix_name=save_name)
    print(f"sauvegarder sous le nom {save_name}")

    if save_name != None:
        utils.ecrire_fichier(save_name,M,VNS_matrix)
    else:
        utils.ecrire_fichier("solution.txt",M,VNS_matrix)

    evolution_rank[2] = fobj(M, VNS_matrix)

    #affichage
    #en gros va map l'interval [min; max] dans [-0.5;0.5] pour que soit un minimum visible
    minValue = 99999
    maxValue = 1e-18
    for i,j in evolution_rank:
        if j>0:
            minValue = j if minValue>j else minValue
            maxValue = j if maxValue<j else maxValue

    print(minValue, maxValue)
    score = np.zeros(evolution_rank.shape[0])
    c= 0
    if minValue != maxValue:
        for i,j in evolution_rank:
            if j>0:
                score[c] = i + (-0.5 + (np.log10(j)-np.log10(minValue))/(np.log10(maxValue)-np.log10(minValue)))
            else:
                score[c] = i+0.5 #si pas de valeur définir comme init met au max
            c+=1
    else:
        score=evolution_rank[:,0]

    print(evolution_rank)
    print(score)
    print(fobj(M,VNS_matrix))
    print(f"And Johan was {fobj(M,johan_method)}")
    
    
    
    plt.figure(figsize=(12, 8))
    
    plt.bar(methode_utilisee, score)
    plt.xticks(rotation=90)
    plt.xlabel("methode utilisée")
    plt.ylabel("rang")
    plt.title(f"Evolution du rang obtenu en fonction des étapes de l'heuristique pour la matrice {mat_name}")
    print(dico)
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f"plot {save_name}", format='png')

    plt.figure(figsize=(12, 8))
    plt.bar(dico.keys(), dico.values())
    plt.xlabel("voisinage exploré")
    plt.ylabel("utilisations")
    plt.title(f"Voisinage exploré lors de l'exécution du VNS sur la matrice {mat_name}")
    plt.show()
    plt.savefig(f"plot VNS {save_name}", format='png')




def run(Mat):
    list_cross = [cross_by_half_split,cross_by_vertical_split,cross_by_alternating_rows,cross_by_alternating_line,cross_by_blocks]

    liste_method = []
    print("Start Johan")
    johan_method = Johanmethod(best_param=True, matrix=Mat)
    print(fobj(Mat,johan_method))
    print("Start Genetique")
    genetique_method = genetique(Mat,2,0,list_cross,0.20,False,500,max_depth=5  ,n_parents = 100,parent_init=None,method_next_gen="Tournament_pro")
    print(fobj(Mat,genetique_method))
    cluster_method = Clustermethod(Mat,n_clusters=2)
    print(fobj(Mat,cluster_method))
    liste_method.append(johan_method)
    liste_method.append(genetique_method)
    liste_method.append(cluster_method)
    liste_method.append(np.ones(Mat.shape))
    liste_method.append(np.ones(Mat.shape)*(-1))

    parents = GenerationParents(Mat,100,liste_method)
    temp = [None]
    temp.append(parents)
    parents = temp
    genetique_matrix = genetique(Mat,2,0,list_cross,0.20,False,1000,max_depth=5  ,n_parents = 100,parent_init=parents,method_next_gen="Tournament")
    print(fobj(Mat,genetique_matrix ))


    VNS_matrix = VNS(Mat,2,0,1000,max_depth = 10,init = genetique_matrix, matrix_name=mat_name)
    print(fobj(Mat,VNS_matrix))
    print(f"And Johan was {fobj(Mat,johan_method)}")
    return VNS_matrix #return la meilleur matrice

#lancement génétique thibaut si nécessaire
  # print("Start Genetique")
    # genetique_method = genetique(M,2,0,list_cross,0.20,False,1000,max_depth=5  ,n_parents = 100,parent_init=None,method_next_gen="Tournament_pro")
    # evolution_rank[2] = fobj(M, genetique_method)

    # print(fobj(M,genetique_method))

    # cluster_method = Clustermethod(M,n_clusters=2)
    # print(fobj(M,cluster_method))
    # liste_method.append(johan_method)
    # liste_method.append(genetique_method)
    # liste_method.append(cluster_method)
    # liste_method.append(np.ones(M.shape))
    # liste_method.append(np.ones(M.shape)*(-1))

    # parents = GenerationParents(M,100,liste_method)
    # temp = [None]
    # temp.append(parents)
    # parents = temp
    # genetique_matrix = genetique(M,2,0,list_cross,0.20,False,1000,max_depth=5  ,n_parents = 100,parent_init=parents,method_next_gen="Tournament")
    # print(fobj(M,genetique_matrix ))

    # evolution_rank[3] = fobj(M, genetique_matrix)