from opti_combi_projet_pythoncode_texte import fobj,compareP1betterthanP2,matrices2_slackngon
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
    while counter<1:
        counter+=1
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
        list_math.append(np.hstack(list_mat[i*y:i*y+x]))
    matrix=np.vstack(list_math)
    return matrix

def tabu(matrix,pattern,file,max_attemp):
    #init liste
    list_tabu=[]
    for _ in range(file):
        list_tabu.append(pattern)
    pattern_best=copy.deepcopy(pattern)

    counter=0
    attemp=0
    while attemp<=max_attemp:
        pattern_tmp_best=perm(0,pattern,0)
        for i in range(matrix.shape[0]*matrix.shape[1]):   
                pattern_tmp=perm(0,pattern,i)
                if compareP1betterthanP2(matrix,pattern_tmp,pattern_tmp_best) and not any(np.array_equal(pattern_tmp,i) for i in list_tabu):
                    pattern_tmp_best=copy.deepcopy(pattern_tmp)
        list_tabu[counter]=pattern_tmp_best
        counter=(counter+1)%file
        attemp+=1
        if compareP1betterthanP2(matrix,pattern_tmp_best,pattern_best):
            pattern_best=copy.deepcopy(pattern_tmp_best)
            print(f"rank: {fobj(matrix,pattern_best)[0]}, valeur min: {fobj(matrix,pattern_best)[1]}")
            attemp=0
    return pattern_best



# matrix=utils.lire_fichier("data/exempleslide_matrice (1).txt")
# matrix=utils.lire_fichier("data/ledm6_matrice (1).txt")
# matrix=matrices2_slackngon(7)
matrix=utils.LEDM (120,120)
# matrix=utils.random_matrix(25,25,5)
print(matrix)

# pattern=np.random.choice([-1,1],size=matrix.shape)
pattern=-np.ones(matrix.shape)
print(fobj(matrix,pattern))

size=12
locale=False
if locale:
    list_mat=subdivise_mat(matrix,size)
    list_pat=subdivise_mat(pattern,size)

    for i in range(len(list_pat)):
        print(f"sub matrice nbr: {i}")
        list_pat[i]=recherche_locale(list_mat[i], list_pat[i],la_totale=True)
    pattern=reassemble_mat(pattern,size,list_pat)
    print("Complete matrix")
    pattern=recherche_locale(matrix,pattern,la_totale=True)
else:
    list_mat=subdivise_mat(matrix,size)
    list_pat=subdivise_mat(pattern,size)

    for i in range(len(list_pat)):
        print(f"sub matrice nbr: {i}")
        list_pat[i]=tabu(list_mat[i], list_pat[i],10,100)
    pattern=reassemble_mat(pattern,size,list_pat)
    print("Complete matrix")
    pattern=tabu(matrix,pattern,10,1000)

print(fobj(matrix,pattern))




utils.ecrire_fichier("solution.txt",matrix,pattern)

    