import numpy as np

def croisement__elem_1_2(parent1, parent2):
    """
    Croise deux matrices (parent1 et parent2) pour produire deux enfants.
    Les deux matrices doivent avoir la même forme.
    
    Arguments :
        parent1 : np.ndarray
        parent2 : np.ndarray
        
    Retourne :
        (np.ndarray, np.ndarray) : Les deux enfants générés.
    """
    if parent1.shape != parent2.shape:
        raise ValueError("Les deux matrices doivent avoir la même forme.")
    
    # Créer les matrices enfants avec la même taille
    enfant1 = np.zeros_like(parent1)
    enfant2 = np.zeros_like(parent2)
    
    # Remplir les enfants en alternant les éléments
    for i in range(parent1.shape[0]):
        for j in range(parent1.shape[1]):
            if (i + j) % 2 == 0:
                # Enfant1 prend l'élément de parent1, Enfant2 de parent2
                enfant1[i, j] = parent1[i, j]
                enfant2[i, j] = parent2[i, j]
            else:
                # Enfant1 prend l'élément de parent2, Enfant2 de parent1
                enfant1[i, j] = parent2[i, j]
                enfant2[i, j] = parent1[i, j]
    
    return enfant1, enfant2
