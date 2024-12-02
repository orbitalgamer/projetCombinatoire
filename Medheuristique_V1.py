import numpy as np
from functools import cmp_to_key
import random
import time
from utils import LEDM
from utils import ecrire_fichier

def compareP1betterthanP2(M,P1,P2):
  r1, s1 = fobj(M,P1)
  r2, s2 = fobj(M,P2)
  if r1 != r2:
      return r1 < r2 
  return s1 < s2      

def fobj(M,P,tol=1e-14):
  sing_values = np.linalg.svd(P*np.sqrt(M), compute_uv=False) # Calcul des valeurs singulières de la matrice P.*sqrt(M)
  ind_nonzero = np.where(sing_values > tol)[0]                # indices des valeurs > tolérance donnée
  return len(ind_nonzero), sing_values[ind_nonzero[-1]]       # on retourne objectif1=rang et objectif2=plus petite val sing. non-nulle

def mut (X, n_mut):
    sh = X.shape
    
    for iter in range(n_mut):
        i = random.randint(0, sh[0]-1)
        j = random.randint(0, sh[1]-1)
        X[i,j]=-1* X[i,j]
    
    
    
    
def Algo(M,max_iter):
    max_time=50
    def genere_enfants(P1,P2,nb_enfants):
        Sh=P1.shape
        if Sh!= P2.shape:
            print(f"erreur, les parents n'ont pas la meme taille !!!")
            exit
        Enfants=[]
        for w in range(nb_enfants):
            enfant=np.ones(Sh)
            for i in range(Sh[0]):
                for j in range(Sh[1]):
                    #tirage de la variable aléatoire:
                    v_a=np.random.rand()
                    if v_a>0.5:
                        enfant[i,j]=parent_1[i,j]
                    else:
                        enfant[i,j]=parent_1[i,j]
            Enfants.append(enfant.copy())
        return Enfants
    
    def Compare_M (P1,P2):
        return compareP1betterthanP2(M,P1,P2)
    
    def meilleurs_de_la_gen(P1,P2,Enfants,alpha,best,ite):
        Enfants.append(P1.copy())
        Enfants.append(P2.copy())
        Meilleurs_Enfants = sorted(Enfants,key=cmp_to_key(Compare_M),reverse=True)
    
    
        x=int(np.floor(alpha*(len(Meilleurs_Enfants)-1)))
        Meilleurs = Meilleurs_Enfants[:x]
        
        if Compare_M(Meilleurs[0],best)== True:
            best=Meilleurs[0].copy()
            ite+=1

        le_futur = random.sample(Meilleurs, 2)
        
        return le_futur
    
    # phase d'initialisation : premiers parents
    parent_1=np.ones(M.shape)
    parent_2=-1*np.ones(M.shape)
    best=parent_1.copy()
    start_time = time.time()
    ite=0
    iter=0
    while (time.time()-start_time<max_time):
        Childrens=genere_enfants(parent_1, parent_2, 5)
        Neo_parents=meilleurs_de_la_gen(parent_1, parent_2, Childrens, 0.6, best,ite)
        parent_1=Neo_parents[0].copy()
        parent_2=Neo_parents[1].copy()
        print(f"Itération n°{iter}, t = {time.time()-start_time} s")
        iter+=1
        
        if iter%100 == 0 :
            mut (parent_1,1)
            mut (parent_2,1)
            mut (best, 2)
        if iter > max_iter:
            break
    return best
       
M=LEDM(7,7)
X=Algo(M, 500000)
print(X)
print(fobj(M, X))