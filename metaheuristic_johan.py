from opti_combi_projet_pythoncode_texte import fobj,compareP1betterthanP2
import utils
import numpy as np
import random
import copy

def perm(type: int,mat:np.ndarray,index):
    mat_tmp=copy.deepcopy(mat)
    if type==0: #Voisinnage multiplie un terme par -1
        x=i//mat.shape[1]
        y=i%mat.shape[1]
        mat_tmp[x][y]*=-1
    return mat_tmp



# matrix=utils.lire_fichier("data/exempleslide_matrice (1).txt")
matrix=utils.lire_fichier("data/ledm6_matrice (1).txt")
print(matrix)
pattern=np.ones(matrix.shape)
print(fobj(matrix,pattern))
counter=0
while counter<10:
    for i in range(matrix.shape[0]*matrix.shape[1]):   
        pattern_tmp=perm(0,pattern,i)
        if compareP1betterthanP2(matrix,pattern_tmp,pattern):
            pattern=copy.deepcopy(pattern_tmp)
            print(fobj(matrix,pattern))
            counter=0
    counter+=1

utils.ecrire_fichier("solution.txt",matrix,pattern)

    