import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
from scipy.sparse import random, csr_matrix

def lire_fichier(file):
    with open(file, 'r') as f:
        first_line = f.readline().strip()
        n_rows, n_cols = map(int, first_line.split())  
    
    # Charger le reste du fichier dans un tableau NumPy
    matrix = np.loadtxt(file, skiprows=1)
    
    # Vérification des dimensions
    if matrix.shape != (n_rows, n_cols):
        raise ValueError("Les dimensions de la matrice ne correspondent pas à la première ligne.")
    
    return matrix

def ecrire_fichier(file,matrix,P):
    def fobj2(M,P,tol=1e-14):
      sing_values = np.linalg.svd(P*np.sqrt(M), compute_uv=False)
      tol         = max(M.shape)*sing_values[0]*np.finfo(float).eps  # Calcul de la tolérance à utiliser pour la matrice P*sqrt(M)
      ind_nonzero = np.where(sing_values > tol)[0]                  
      return sing_values[ind_nonzero]
  
    
    with open(file, "w") as f:
        Sh=matrix.shape
        for i in range (Sh[0]):
            for j in range (Sh[1]):
                f.write(f"{int(P[i,j])}")
                if j!= Sh[1]-1:
                    f.write(f" ")
            f.write(f"\n")
                
        S=fobj2(matrix, P)
        
        for i in range(len(S)):
            f.write(f"{S[i]}\n")

def LEDM (n,m):
    M=np.zeros((n,m))
    
    for i in range(n):
        for j in range(m):
            M[i,j]=(i-j)**2
    return M


def random_matrix(m ,n, r):
    matrix_mr = np.random.rand(m, r)
    matrix_rn = np.random.rand(r, n)

    # Generate random masks with values between -1 and 1
    mask_mr = np.random.uniform(-1, 1, (m, r))
    mask_rn = np.random.uniform(-1, 1, (r, n))

    # Apply the masks
    masked_mr = matrix_mr * mask_mr
    masked_rn = matrix_rn * mask_rn

    # Perform the matrix multiplication and element-wise power
    base_line = (masked_mr @ masked_rn) ** 2
    return base_line



def sparse_random_matrix(m:int ,n: int, r:int, density : float):
    U = random(m, r, density=density, format='csr', data_rvs=np.random.rand)
    V = random(r, n, density=density, format='csr', data_rvs=np.random.rand)

    Mat = (U @ V)
    return (10*Mat.toarray())**2


def optimal_k(M, max_k=5):
    inertias = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(M)
        inertias.append(kmeans.inertia_)
    
    # Tracer la courbe de l'inertie
    plt.plot(range(1, max_k+1), inertias, marker='o')
    plt.title('Méthode du coude')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Inertie')
    plt.show()

def clustering_lines(M, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(M)
    return labels

def clustering_columns(M, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(M.T)
    return labels


def generate_initial_P(M, n_clusters_line, n_clusters_columns,noise_prob=0):
    P = np.zeros_like(M)
    line_labels=clustering_lines(M,n_clusters_line)
    col_labels=clustering_columns(M,n_clusters_columns)
    
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            P[i, j] = 1 if (line_labels[i] + col_labels[j]) % 2 == 0 else -1
            if np.random.rand() < noise_prob:  # Avec une certaine probabilité, changer le cluster
                P[i,j] = -P[i,j]
    return P

def pat_ledm(M):
    n,m=M.shape
    pat=np.ones(M.shape)
    for i in range(n):
        middle=False
        for j in range(m):
            if M[i][j]==0:
                middle=True
            if middle:
                pat[i][j]*=-1
    return pat

import os

def get_pc_name(): #pour mettre dans le nom que écrit
    nom_pc = os.environ.get('COMPUTERNAME') or os.environ.get('HOSTNAME')
    if not nom_pc and hasattr(os, 'uname'):
        nom_pc = os.uname().nodename
    return nom_pc
