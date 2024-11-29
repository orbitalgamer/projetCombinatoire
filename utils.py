import numpy as np
from sklearn.cluster import KMeans

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


def random_matrix(m:int ,n: int, r:int):
    return ((np.random.rand(m,r)*10)@(np.random.rand(r,n)*10))**2

def clustering_lines(M, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(M)
    return labels

def clustering_columns(M, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(M.T)
    return labels

def generate_initial_P(M, line_labels, col_labels,noise_prob):
    P = np.zeros_like(M)

def generate_initial_P(M, line_labels, col_labels,noise_prob):
    P = np.zeros_like(M)
    # unique_line_labels = np.unique(line_labels)
    # unique_col_labels = np.unique(col_labels)
    
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            P[i, j] = 1 if (line_labels[i] + col_labels[j]) % 2 == 0 else -1
            if np.random.rand() < noise_prob:  # Avec une certaine probabilité, changer le cluster
                P[i,j] = -P[i,j]
    return P

#Exemple call genrate_initial_p
#line_labels = clustering_lines(M, n_clusters)
#col_labels = clustering_columns(M, n_clusters)
#generate_initial_P(M, line_labels, col_labels,noise_prob=0.05)