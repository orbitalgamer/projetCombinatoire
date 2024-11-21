import numpy as np


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
      return sing_values[0:ind_nonzero]  
  
    
    with open(file, "w") as f:
        Sh=matrix.shape
        for i in range (Sh[0]):
            for j in range (Sh[1]):
                f.write(f"{int(matrix[i,j])}")
                if j!= Sh[1]-1:
                    f.write(f" ")
            f.write(f"\n")
                
        S=fobj2(matrix, P)
        
        for i in range(len(S)):
            f.write(f"{S[i]}\n")

