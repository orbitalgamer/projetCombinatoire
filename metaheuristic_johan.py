from opti_combi_projet_pythoncode_texte import fobj,compareP1betterthanP2
import utils
import numpy as np
import random
import copy

def perm(type: int,mat:np.ndarray):
    mat_tmp=copy.deepcopy(mat)
    if type==0: #Voisinnage multiplie un terme par -1
        x=random.randrange(mat.shape[0])
        y=random.randrange(mat.shape[1])
        mat_tmp[x][y]*=-1
    return mat_tmp



matrix=utils.lire_fichier("data/exempleslide_matrice (1).txt")
print(matrix)
pattern=np.ones(matrix.shape)
while True:
    print(pattern)
    pattern=perm(0,pattern)
    