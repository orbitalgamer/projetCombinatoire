import random
def rand_transfo(matrice):
    taille = matrice.shape
    while(True):
        n,n_2 = random.randint(0, taille[0]-1),random.randint(0, taille[1]-1)
        