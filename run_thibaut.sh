#!/bin/bash

# Demander à l'utilisateur de saisir un nombre
read -p "Combien de fois voulez-vous exécuter le programme Python ? " n

# Vérifier si l'entrée est un nombre valide
if [[ "$n" =~ ^[0-9]+$ ]]; then
    # Chemin vers le répertoire contenant le script Python
    script_directory="/Users/medhi/Documents/test graph/projetCombinatoire"

    # Boucle pour exécuter le programme Python le nombre de fois demandé
    for ((i = 1; i <= n; i++)); do
        echo "Exécution #$i"

        # Utiliser osascript pour ouvrir un nouveau terminal, changer de dossier et exécuter Python
        osascript -e "tell application \"Terminal\" to do script \"cd '$script_directory' && python3 'genetics_thib_v3.py'\""

        # Si tu utilises iTerm2 au lieu de Terminal.app
        # osascript -e "tell application \"iTerm\" to create window with default profile"
        # osascript -e "tell current session of current window of application \"iTerm\" to write text \"cd '$script_directory' && python3 'genetics_thib_v3.py'\""
    done
else
    echo "Veuillez entrer un nombre valide."
fi

