import genetics_thib_v3 as model

from utils import LEDM, lire_fichier, ecrire_fichier, get_pc_name, number_launched
from joblib import Parallel, delayed
from opti_combi_projet_pythoncode_texte import fobj


num_parallel = 4
Matrice = LEDM(32,32) #définit la matrice
#lancement du modèle

print(get_pc_name())
print(number_launched())



# output = Parallel(n_jobs=-1)(delayed(model.run)(Matrice) for _ in range(num_parallel))



# for i in output:
#     print(fobj(Matrice, i))
