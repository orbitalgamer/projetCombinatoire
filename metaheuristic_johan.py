from opti_combi_projet_pythoncode_texte import fobj,compareP1betterthanP2
import utils
import numpy as np
import random
import copy

def perm(type: int,mat:np.ndarray,index: int):
    mat_tmp=copy.deepcopy(mat)
    if type==0: #Voisinnage multiplie un terme par -1
        x=index//mat.shape[1]
        y=index%mat.shape[1]
        mat_tmp[x][y]*=-1
    elif type==1:
        mat_tmp[index,:]*=-1
    elif type==2:
        mat_tmp[:,index]*=-1
    return mat_tmp

def recherche_locale(matrix,pattern,la_totale=False):
    counter=0
    while counter<10:
        for i in range(matrix.shape[0]*matrix.shape[1]):   
            pattern_tmp=perm(0,pattern,i)
            if compareP1betterthanP2(matrix,pattern_tmp,pattern):
                pattern=copy.deepcopy(pattern_tmp)
                print(f"rank: {fobj(matrix,pattern)[0]}, valeur min: {fobj(matrix,pattern)[1]}")
                counter=0
        if la_totale:
            for i in range(matrix.shape[0]):   
                pattern_tmp=perm(1,pattern,i)
                if compareP1betterthanP2(matrix,pattern_tmp,pattern):
                    pattern=copy.deepcopy(pattern_tmp)
                    print(f"rank: {fobj(matrix,pattern)[0]}, valeur min: {fobj(matrix,pattern)[1]}")
                    counter=0
            for i in range(matrix.shape[1]):   
                pattern_tmp=perm(2,pattern,i)
                if compareP1betterthanP2(matrix,pattern_tmp,pattern):
                    pattern=copy.deepcopy(pattern_tmp)
                    print(f"rank: {fobj(matrix,pattern)[0]}, valeur min: {fobj(matrix,pattern)[1]}")
                    counter=0
        counter+=1
    return pattern

# matrix=utils.lire_fichier("data/exempleslide_matrice (1).txt")
# matrix=utils.lire_fichier("data/ledm6_matrice (1).txt")
matrix=utils.LEDM (25,25)
print(matrix)

# pattern=np.random.choice([-1,1],size=matrix.shape)
pattern=np.ones(matrix.shape)

print(fobj(matrix,pattern))
pattern=recherche_locale(matrix, pattern,la_totale=True)
print(fobj(matrix,pattern))


utils.ecrire_fichier("solution.txt",matrix,pattern)

    