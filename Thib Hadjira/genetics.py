import time
import random
def evolutionnaire(sol, voisinage,temps_max = 60, memetique = True):
    a = time.time()
    sol = sol[:]
    parents = []
    sujet_max = 200
    taux_mutation = 1/4
    parents.append(sol)
    kmax = 10000
    best_cost = float('inf')
    best_sol = None
    
    #Generation de solution initiales
    for i in range(sujet_max-1):
        new_parent = random.sample(sol, len(sol))
        if new_parent not in parents:
            parents.append(new_parent)
    
    while(time.time()-a < temps_max):
        
        #generation enfants
        enfants = []
        for i in range(0, len(parents)-1, 2):
            enfant1, enfant2 = faire_enfant(parents[i], parents[i + 1])
            enfants.append(enfant1)
            enfants.append(enfant2)
        
        #Choix des enfants qui seront muté
        random_list = list()
        while(len(random_list) < sujet_max*taux_mutation):
            random_n = random.randint(0, len(enfants)-1)
            if random_n in random_list:
                pass
            else:
                random_list.append(random_n)
        #Si memetique, split des enfants a muter        
        if memetique:
            pass
            # memetique_list = random_list[:len(random_list)//2]
            # non_memetique_list = random_list[len(random_list)//2:]
            
            # #Mutation par recherche locale
            # for elem in memetique_list:
            #     enfants[elem] = recherche_locale_full(enfants[elem], voisinage)[0]
            
            # #Mutation aleatoire par permutation unique
            # for elem in non_memetique_list:
            #     random_n_1 = random.randint(0, len(enfants[0])-1)
            #     random_n_2 = random.randint(0, len(enfants[0])-1)
            #     enfants[elem] = swap(enfants[elem], random_n_1, random_n_2)
        else:
            
            #Mutation aleatoire par permutation unique
            for elem in random_list:
                random_n_1 = random.randint(0, len(enfants[0])-1)
                random_n_2 = random.randint(0, len(enfants[0])-1)
                enfants[elem] = swap(enfants[elem], random_n_1, random_n_2)
        
        #Fusion enfants parents
        parents += enfants 
        
        #Concours des meilleurs sujets
        for i in range(len(parents)):
            parents[i] = cost_function(parents[i]),parents[i]
        parents = sorted(parents, key=lambda x: x[0])[:len(parents)//2]
        if parents[0][0] < best_cost:
            best_cost,best_sol = parents[0]
        parents = [x[1] for x in parents]
        
        #Remelanger les sujets
        random.shuffle(parents)
        kmax -= 1
    # Dictionnaire classique pour compter les occurrences des listes
    count_dict = {}
    
    # Remplir le dictionnaire
    for lst in parents:
        # Utiliser un tuple pour que la liste soit une clé valide
        tuple_lst = tuple(lst)
        
        if tuple_lst in count_dict:
            count_dict[tuple_lst] += 1
        else:
            count_dict[tuple_lst] = 1
    
    # # Convertir à nouveau les clés en listes pour afficher le résultat final
    # result = {list(k): v for k, v in count_dict.items()}
    
    # Affichage du résultat
    print(count_dict.values())
    return best_sol,best_cost,time.time()-a

def faire_enfant(parent1, parent2):
    n = len(parent1)
    start, end = sorted(random.sample(range(n), 2))
    
    if parent1 == parent2:
        #print("consanguin")
        enfant1 = random.sample(parent1,len(parent1))
        enfant2 = random.sample(parent1,len(parent1))
    else:
        # Copier le segment de parent1
        enfant1 = [None] * n
        enfant1[start:end] = parent1[start:end]
        reste = list()
        
        
        
        #Copie de l'anti-segment du parent2 en checkant les duplications
        current_position = 0
        for elem in parent2:
            if elem not in enfant1:
                if current_position >= n:
                    current_position = 0
                if enfant1[current_position] is None:     
                    enfant1[current_position] = elem
                else:
                    reste.append(elem)
                current_position += 1
            else:
                current_position += 1
    
        #Remplissage des espaces manquant a partir de donnee du parents2
        current_position = 0
        for i,elem in enumerate(enfant1):
            if elem is None:  
                enfant1[i] = reste[current_position]
                current_position += 1
                
        reste = [] 
        # Répéter pour l'autre enfant
        
        # Copier le segment de parent2
        enfant2 = [None] * n
        enfant2[start:end] = parent2[start:end]
        
        #Copie de l'anti-segment du parent1 en checkant les duplications
        current_position = 0
        for elem in parent1:
            if elem not in enfant2:
                if current_position >= n:
                    current_position = 0
                if enfant2[current_position] is None:     
                    enfant2[current_position] = elem
                else:
                    reste.append(elem)
                current_position += 1
            else:
                current_position += 1
        
        #Remplissage des espaces manquant a partir de donnee du parents1
        current_position = 0
        for i,elem in enumerate(enfant2):
            if elem is None:  
                enfant2[i] = reste[current_position]
                current_position += 1
        
    return enfant1, enfant2