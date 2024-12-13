import genetics_thib_v3 as model

from utils import LEDM, lire_fichier, ecrire_fichier
from joblib import Parallel, delayed


num_parallel = 4
Matrice = LEDM(32,32) #définit la matrice
#lancement du modèle




output = Parallel(n_jobs=-1)(delayed(model.run)(Matrice) for _ in range(num_parallel))



for i in output:
    print(fobj(Matrice, i))
