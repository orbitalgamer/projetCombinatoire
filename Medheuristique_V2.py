import numpy as np
import random
import time
from utils import LEDM
from utils import ecrire_fichier
from scipy.linalg import circulant
import matplotlib.pyplot as plt

def genereparents_Random(X,nb_parents):
    Parents=[]
    for i in range (nb_parents):
        matrice = np.random.choice([-1, 1], size=(X.shape[0], X.shape[1]))
        Parents.append(matrice)
    return Parents

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
def reduire (L,M):
    taille=len(L)
    néo=[]
    for i in range (int(np.floor(taille/2))):
        if compareP1betterthanP2(M, L[2*i], L[2*i+1]) == True:
            néo.append(L[2*i])
        else:
            néo.append(L[2*i+1])
    return néo
        
        
        
def tournament (L,M):
    for i in range(3):
        random.shuffle(L)
        L=reduire(L, M)
    return L
        
def genére_enfents (Parents,n_enfants):
    enfants=[]
    for iter in range (n_enfants):
        E=np.ones(Parents[0].shape)
        
        for i in range (Parents[0].shape[0]):
            for j in range (Parents[0].shape[1]):
                nombre = random.randint(0, len(Parents)-1)
                E[i,j]=Parents[nombre][i,j]
        enfants.append(E)
    return enfants



def Medheurist (M,maxtime,maxiter):
    param=25
    def Compare_M (P1,P2):
        return compareP1betterthanP2(M,P1,P2)
    # état initial
    st=time.time()
    results=[]
    Parents = genereparents_Random(M, param*4)
    best=Parents[0]
    for i in range (1,param*4):
        if Compare_M(Parents[i], best) == True:
            best=Parents[i]
    
    for k in range (maxiter):
        print (f"Itération n°{k+1}, temps écoulé :{time.time()-st} s")
        Childrens=genére_enfents(Parents, param*28)
        concurents = Childrens + Parents
        if len(concurents)!=param*32:
            print(f"problèmes de concurents")
        Parents=tournament(concurents, M)
        if len(concurents)!=param*32:
            print(f"problèmes dans le tournoi")
        for i in range (param*4):
            if Compare_M(Parents[i], best) == True:
                best=Parents[i]
        x,y=fobj(M, best)
        results.append(x)
        if time.time()-st > maxtime :
            print(f"Temps moyen d'une itération : {(time.time()-st)/k}")
            break
        
    
    iterations = list(range( len(results)))


    plt.plot(iterations, results, marker='o', label="Valeur par itération")
    plt.xlabel("Itération")
    plt.ylabel("Rank")
    plt.title(f"Évolution du rank par itération pour une matrice de taille {M.shape} \n Algo génétique a {param*4} parents et {param*28} enfants")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(best)
    print(fobj(M, best))
    return best




M=LEDM(7,7)
X=Medheurist(M, 100,600)

    
        
    