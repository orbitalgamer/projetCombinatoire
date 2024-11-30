from opti_combi_projet_pythoncode_texte import fobj,compareP1betterthanP2,matrices2_slackngon
import utils
import numpy as np
import random
import copy

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

def recherche_locale(matrix,pattern,la_totale=False,verbose=False):
    counter=0
    while counter<1:
        counter+=1
        pattern_best=copy.deepcopy(pattern)
        for i in range(matrix.shape[0]*matrix.shape[1]):   
            pattern_tmp=perm(0,pattern,i)
            if compareP1betterthanP2(matrix,pattern_tmp,pattern_best):
                pattern_best=copy.deepcopy(pattern_tmp)
                if verbose:
                    print(f"0 rank: {fobj(matrix,pattern_best)[0]}, valeur min: {fobj(matrix,pattern_best)[1]}")
                counter=0
        if la_totale:
            for i in range(matrix.shape[0]):   
                pattern_tmp=perm(1,pattern,i)
                if compareP1betterthanP2(matrix,pattern_tmp,pattern_best):
                    pattern_best=copy.deepcopy(pattern_tmp)
                    if verbose:
                        print(f"1 rank: {fobj(matrix,pattern_best)[0]}, valeur min: {fobj(matrix,pattern_best)[1]}")
                    counter=0
            for i in range(matrix.shape[1]):   
                pattern_tmp=perm(2,pattern,i)
                if compareP1betterthanP2(matrix,pattern_tmp,pattern_best):
                    pattern_best=copy.deepcopy(pattern_tmp)
                    if verbose:
                        print(f"2 rank: {fobj(matrix,pattern_best)[0]}, valeur min: {fobj(matrix,pattern_best)[1]}")
                    counter=0
            for i in range(matrix.shape[0]):
                for j in range(i,matrix.shape[0]):    
                    pattern_tmp=perm(3,pattern,i,j)
                    if compareP1betterthanP2(matrix,pattern_tmp,pattern_best):
                        pattern_best=copy.deepcopy(pattern_tmp)
                        if verbose:
                            print(f"3 rank: {fobj(matrix,pattern_best)[0]}, valeur min: {fobj(matrix,pattern_best)[1]}")
                        counter=0
            for i in range(matrix.shape[1]):
                for j in range(i,matrix.shape[1]):    
                    pattern_tmp=perm(4,pattern,i,j)
                    if compareP1betterthanP2(matrix,pattern_tmp,pattern_best):
                        pattern_best=copy.deepcopy(pattern_tmp)
                        if verbose:
                            print(f"4 rank: {fobj(matrix,pattern_best)[0]}, valeur min: {fobj(matrix,pattern_best)[1]}")
                        counter=0
        pattern=copy.deepcopy(pattern_best)
    return pattern

def greedy(matrix,pattern,setup_break,la_totale=False,verbose=False):
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
                    # print(f"rank: {fobj(matrix,pattern_tmp_best)[0]}, valeur min: {fobj(matrix,pattern_tmp_best)[1]}")
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
# matrix=utils.lire_fichier("data/correl5_matrice.txt")
# matrix=utils.lire_fichier("data/synthetic_matrice.txt")
# matrix=matrices2_slackngon(7)
matrix=utils.LEDM (100,100)
# matrix=utils.random_matrix(7,7,3)

# pattern=np.random.choice([-1,1],size=matrix.shape)
pattern=np.ones(matrix.shape)
# pattern=utils.generate_initial_P(matrix,2,2)
print(fobj(matrix,pattern))

metah=0 #0 for greedy, 1 for tabu, 2 for local search

#determination meilleur parametre
pattern_best=copy.deepcopy(pattern)
if metah==0:
    for setup_break in range(4):
        for size in range(2,max(matrix.shape)+1):
            print(f"Testing for size={size} and setup_break={setup_break}")
            list_mat=subdivise_mat(matrix,size)
            list_pat=subdivise_mat(pattern,size)

            for i in range(len(list_pat)):
                list_pat[i]=greedy(list_mat[i], list_pat[i],setup_break,la_totale=True,verbose=False)
            pattern_tmp=reassemble_mat(pattern,size,list_pat)
            pattern_tmp=greedy(matrix,pattern_tmp,setup_break,la_totale=True,verbose=False)
            if compareP1betterthanP2(matrix,pattern_tmp,pattern_best):
                pattern_best=copy.deepcopy(pattern_tmp)
                size_best=size
                setup_break_best=setup_break
                print(f"for param size={size} and setup_break={setup_break} rank: {fobj(matrix,pattern_best)[0]}, valeur min: {fobj(matrix,pattern_best)[1]}")
    print(f"param opti: size={size_best} and setup_break={setup_break_best}")
elif metah==1:
    for queue in range(20):
        for size in range(2,max(matrix.shape)+1):
            list_mat=subdivise_mat(matrix,size)
            list_pat=subdivise_mat(pattern,size)

            for i in range(len(list_pat)):
                list_pat[i]=tabu(list_mat[i], list_pat[i],queue,100)
            pattern_tmp=reassemble_mat(pattern,size,list_pat)
            pattern_tmp=tabu(matrix,pattern,queue,100)
            if compareP1betterthanP2(matrix,pattern_tmp,pattern_best):
                pattern_best=copy.deepcopy(pattern_tmp)
                print(f"for param size={size} and queue={queue} rank: {fobj(matrix,pattern_best)[0]}, valeur min: {fobj(matrix,pattern_best)[1]}")

elif metah==2:
    for size in range(2,max(matrix.shape)+1):
        list_mat=subdivise_mat(matrix,size)
        list_pat=subdivise_mat(pattern,size)

        for i in range(len(list_pat)):
            list_pat[i]=recherche_locale(list_mat[i], list_pat[i],la_totale=True,verbose=False)
        pattern_tmp=reassemble_mat(pattern,size,list_pat)
        pattern_tmp=recherche_locale(matrix,pattern,la_totale=True,verbose=False)
        if compareP1betterthanP2(matrix,pattern_tmp,pattern_best):
                pattern_best=copy.deepcopy(pattern_tmp)
                print(f"for param size={size} rank: {fobj(matrix,pattern_best)[0]}, valeur min: {fobj(matrix,pattern_best)[1]}")


print(fobj(matrix,pattern_best))




utils.ecrire_fichier("solution.txt",matrix,pattern_best)

    