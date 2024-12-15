#!/bin/bash

# Demander à l'utilisateur de saisir un nombre
read -p "Combien de fois voulez-vous exécuter le programme Python ? " n

# Vérifier si l'entrée est un nombre valide
if [[ "$n" =~ ^[0-9]+$ ]]; then
    # Boucle pour exécuter le programme Python le nombre de fois demandé
    for ((i = 1; i <= n; i++)); do
        # Ouvrir un nouveau terminal pour chaque exécution
        echo "Exécution #$i"
        
        # Utiliser osascript pour ouvrir un nouveau terminal et exécuter Python
        osascript -e "tell application \"Terminal\" to do script \"python3 'genetics_thib_v3.py'\""
        
        # Si tu utilises iTerm2 au lieu de Terminal.app, tu peux utiliser la commande suivante
        # osascript -e "tell application \"iTerm\" to create window with default profile"
        # osascript -e "tell current session of current window of application \"iTerm\" to write text \"python3 '/path/to/votre_programme.py'\""
        
    done
else
    echo "Veuillez entrer un nombre valide."
fi
