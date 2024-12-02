import numpy as np
import random
from opti_combi_projet_pythoncode_texte import fobj
from collections import deque
from utils import random_matrix
from numba import njit
from sklearn.cluster import KMeans


def clustering_lines(M, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(M)
    return labels

def clustering_columns(M, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(M.T)
    return labels

def generate_initial_P(M, n_cluster):
    line_labels = clustering_lines(M, n_clusters=n_cluster)
    col_labels = clustering_columns(M, n_clusters=n_cluster)
    P = np.zeros_like(M)
    
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            P[i, j] = 1 if (line_labels[i] + col_labels[j]) % 2 == 0 else -1
    return P


def tabou_search(matrice):
    def compare(better, actual):
        if better[0]<actual[0]:
            return True
        elif better[0]==actual[0]:
            return better[1]<actual[1]
        else:
            return False

    def get_hash(m):
        return hash(m.tobytes())

    P = np.random.choice([1,-1], size=matrice.shape) #init P
    P = generate_initial_P(matrice, 2)
    bestP= P.copy()

    bestValue= (990,1e38)

    longueur = 100
    listTaboue = deque(maxlen=10)
    mouventTaboue = set()
    max_iter=50000
    voisinage_size = 100

    for a in range(max_iter):
        if not a % 1000:
            print(f"currently at {a}")
        mouventTaboue.clear() #reset mouvement
        
        for e in range(voisinage_size): #génère les 100 voisins non taboue
            test_p = P.copy()
            #création des permutation non taboue
            i = random.randrange(test_p.shape[0])
            j = random.randrange(test_p.shape[1])
            test_p[i,j] *= -1
            if((i,j) in mouventTaboue and get_hash(test_p) in listTaboue): #check si taboue
                e-=1
                continue
            mouventTaboue.add((i,j)) #sauvegarde
            
            
            

        #recherche le meilleur résultat parmis les 100
        best_sol_voisinage = (9999,1e18)
        best_permut = (0,0)
        for c, (i,j) in enumerate(mouventTaboue):
            test_p = P.copy()
            test_p[i,j]*=-1 #recharge la parmutation
            obj = fobj(matrice, test_p)
            if(obj, best_sol_voisinage):
                best_sol_voisinage=obj
                best_permut = (i,j) #stock la permutation



        if compare(best_sol_voisinage, bestValue):
            bestP[i,j] *=-1 #fait la permutation
            bestValue=best_sol_voisinage
            listTaboue.append(get_hash(matrice))
            print(f"better sol found, best_rank={bestValue[0]}, singulare value = {bestValue[1]}")
            # if :
            #     retour = fobj(matrice, test_p)
            #     if compare(retour, bestValue):
            #         bestValue=retour
            #         listTaboue.append(test_p) #stock
            #         P=test_p.copy() #mest à jour
            #         
    return bestP, bestValue

M = random_matrix(7,7,3)
P, bestValue = tabou_search(M)

print(f"best_rank={bestValue[0]}, singulare value = {bestValue[1]}")
