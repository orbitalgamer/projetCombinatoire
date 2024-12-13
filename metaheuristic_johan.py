from opti_combi_projet_pythoncode_texte import fobj,compareP1betterthanP2,matrices2_slackngon
import utils
import numpy as np
import random
import copy
import time
from joblib import Parallel,delayed
import itertools

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

def recherche_locale(matrix,pattern,param,la_totale,verbose=False):
    if matrix.size==1 and matrix[0][0]==0:
        return pattern
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

def tabu(matrix,pattern,file,param,verbose=False,max_attemp=100):
    if matrix.size==1 and matrix[0][0]==0:
        return pattern
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
                    if verbose:
                        print(f"rank: {fobj(matrix,pattern_tmp_best)[0]}, valeur min: {fobj(matrix,pattern_tmp_best)[1]}")
                    pattern_tmp_best=copy.deepcopy(pattern_tmp)
        list_tabu[counter]=pattern_tmp_best
        counter=(counter+1)%file
        attemp+=1
        if compareP1betterthanP2(matrix,pattern_tmp_best,pattern_best):
            pattern_best=copy.deepcopy(pattern_tmp_best)
            if verbose:
                print(f"rank: {fobj(matrix,pattern_best)[0]}, valeur min: {fobj(matrix,pattern_best)[1]}")
            attemp=0
    return pattern_best

def Resolve_metaheuristic(funct,matrix,pattern,param,verbose=False):
        print(f"Testing for size={param[0]}, param2={param[1]} and param3={param[2]}")
        list_mat=subdivise_mat(matrix,param[0])
        list_pat=subdivise_mat(pattern,param[0])
        list_pat=Parallel(n_jobs=-1)(delayed(funct)(list_mat[i],list_pat[i],param[1],param[2],verbose) for i in range(len(list_pat)))
        pattern_tmp=reassemble_mat(pattern,param[0],list_pat)
        pattern_tmp=funct(matrix,pattern_tmp,param[1],param[2],verbose)
        return (pattern_tmp,param)

# matrix=utils.lire_fichier("data/exempleslide_matrice (1).txt")
# matrix=utils.lire_fichier("data/ledm6_matrice (1).txt")
matrix=utils.lire_fichier("data/correl5_matrice.txt")
# matrix=utils.lire_fichier("data/synthetic_matrice.txt")
# matrix=matrices2_slackngon(5)
# matrix=utils.LEDM (120,120)
# matrix=utils.random_matrix(120,120,2)

# pattern=np.random.choice([-1,1],size=matrix.shape)
pattern=np.ones(matrix.shape)
# pattern=utils.generate_initial_P(matrix,2,2)
# pattern=utils.pat_ledm(matrix) #best solution for LEDM
print(fobj(matrix,pattern))

debug=True
best_param=True
metah=0 #0 for greedy, 1 for tabu, 2 for local search


if best_param:
    #determination meilleur parametre
    start_time=time.time()
    pattern_best=copy.deepcopy(pattern)
    if metah==0:
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
                
        print(f"param opti: size={size_best}, setup_break={setup_break_best} and la_totale={la_totale_best}")
    elif metah==1:
        queue=range(1,11)
        size=range(2,max(matrix.shape)+1)
        param=itertools.product(queue,size)
        data=Parallel(n_jobs=-1)(delayed(Resolve_metaheuristic)(tabu,matrix,pattern,(i[1],i[0],'/')) for i in param)
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
        data=Parallel(n_jobs=-1)(delayed(Resolve_metaheuristic)(recherche_locale,matrix,pattern,(i[1],'/',i[0])) for i in param)
        for (pattern_tmp,p) in data:
            if compareP1betterthanP2(matrix,pattern_tmp,pattern_best):
                pattern_best=copy.deepcopy(pattern_tmp)
                size_best=p[0]
                la_totale_best=p[2]
                print(f"for param size={size_best} and la_totale={la_totale_best} rank: {fobj(matrix,pattern_best)[0]}, valeur min: {fobj(matrix,pattern_best)[1]}")
        print(f"param opti: size={size_best} and la_totale={la_totale_best}")

    print(fobj(matrix,pattern_best))
    print(f"temps de calcul pour trouve param opti= {time.time()-start_time}s")

if debug:
    start_time=time.time()
    if not best_param:
        size_best=15
        setup_break_best=0 #0,1,2 or 3
        la_totale_best=True #True or False
    if metah==0:
        (pattern_tmp,p)=Resolve_metaheuristic(greedy,matrix,pattern,(size_best,setup_break_best,la_totale_best),verbose=True)
    elif metah==1:
        (pattern_tmp,p)=Resolve_metaheuristic(tabu,matrix,pattern,(size_best,queue_best,'/'),verbose=True)
    elif metah==2:
        (pattern_tmp,p)=Resolve_metaheuristic(recherche_locale,matrix,pattern,(size_best,'/',la_totale_best),verbose=True)
    end_time=time.time()
    print(fobj(matrix,pattern_tmp))
    print(f"temps de calcul pour calculer solution= {end_time-start_time}s")
    print(f"param size={size_best} setup_break={setup_break_best} and la_totale={la_totale_best}")

utils.ecrire_fichier("solution.txt",matrix,pattern_tmp)

    