import random 
import numpy as np
def generer_enfant(parent1,parent2):
    row,col=parent1.shape
    enfant=np.zeros((row,col),dtype=parent1.dtype)
    for i range(row):
        if i%2 == 0:
            enfant[i]=parent1[i]
        else:
            enfant[i]=parent2[i]

parent1=np.random.randint(0, 10, size=(3, 5))
parent2=np.random.randint(0, 10, size=(3, 5))
enf=generer_enfant(parent1,parent2)
print(enf)