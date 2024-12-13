using LinearAlgebra
using Clustering
using Random

function fobj(M, P)
    sing_values = svdvals(P .* sqrt.(M))  # Calcul des valeurs singulières de la matrice P.*sqrt(M)
    tol = maximum(size(M)) * sing_values[1] * eps(Float64)  # Calcul de la tolérance
    ind_nonzero = findall(sing_values .> tol)  # indices des valeurs > tolérance
    return length(ind_nonzero), sing_values[ind_nonzero[end]]
end

function compareP1betterthanP2(M, P1, P2)
    r1, s1 = fobj(M, P1)  # on récupère les deux objectifs pour le pattern P1
    r2, s2 = fobj(M, P2)  # on récupère les deux objectifs pour le pattern P2
    if r1 != r2  # on traite les objectifs de façon lexicographique
        return r1 < r2  # d'abord les valeurs du rang
    end
    return s1 < s2  # si les rangs sont égaux, on prend en compte la valeur de la plus petite valeur singulière
end


function LEDM(n,m)
    mat = zeros(n,m)
    for i in 1:n
        for j in 1:m
            mat[i,j]=(i-j)^2
        end
    end
    return mat
end

function random_matrix(m,n,r)
    return ((10*rand(m,r)) * (10*rand(r,n)))^2
end

function clustering_lines(M, n_clusters)
    kmeans = KMeans(M, n_clusters)  # Applique KMeans sur les lignes
    return kmeans.assignments  # Retourne les étiquettes des lignes
end

# Fonction de clustering des colonnes
function clustering_columns(M, n_clusters)
    kmeans = KMeans(M', n_clusters)  # Applique KMeans sur les colonnes (M.T en Python)
    return kmeans.assignments  # Retourne les étiquettes des colonnes
end

# Fonction de génération de la matrice initiale P
function generate_initial_P(M, n_clusters_line, n_clusters_columns, noise_prob=0)
    P = zeros(Int, size(M))  # Crée une matrice de zéros de la même taille que M
    
    line_labels = clustering_lines(M, n_clusters_line)  # Clustering des lignes
    col_labels = clustering_columns(M, n_clusters_columns)  # Clustering des colonnes
    
    for i in 1:size(M, 1)  # Pour chaque ligne
        for j in 1:size(M, 2)  # Pour chaque colonne
            P[i, j] = (line_labels[i] + col_labels[j]) % 2 == 0 ? 1 : -1  # Valeur initiale basée sur les étiquettes
            if rand() < noise_prob  # Si un bruit est ajouté
                P[i, j] = -P[i, j]  # Inverse la valeur
            end
        end
    end
    return P
end

function lire_fichier(file)
    # Ouvrir le fichier et lire la première ligne
    openfile = open(file, "r")
    first_line = readline(openfile)
    close(openfile)
    
    # Extraire le nombre de lignes et de colonnes
    n_rows, n_cols = parse.(Int, split(first_line))  # On utilise parse pour convertir en entier
    
    # Charger le reste du fichier dans une matrice
    matrix = readdlm(file, skipstart=1)  # Relecture du fichier en sautant la première ligne
    
    # Vérification des dimensions
    if size(matrix) != (n_rows, n_cols)
        throw(ErrorException("Les dimensions de la matrice ne correspondent pas à la première ligne."))
    end
    
    return matrix
end

function ecrire_fichier(file, matrix, P)
  
    # Ouvrir le fichier pour l'écriture
    openfile = open(file, "w")
    Sh = size(matrix)
    
    # Écriture de la matrice P dans le fichier
    for i in 1:Sh[1]
        for j in 1:Sh[2]
            write(openfile, "$(P[i,j])")  # Écrire les valeurs de P
            if j != Sh[2]
                write(openfile, " ")
            end
        end
        write(openfile, "\n")
    end

    # Calcul des valeurs singulières
    S = svdvals(P .* sqrt.(matrix))
    
    # Écriture des valeurs singulières dans le fichier
    for s in S
        write(openfile, "$s\n")
    end
    
    close(openfile)  # Fermer le fichier après l'écriture
end

using LinearAlgebra

function circulant(v)
    n = length(v)
    M = Matrix{Float64}(undef, n, n)
    for i in 1:n
        M[i, :] = circshift(v, i-1)
    end
    return M
end


function matrices2_slackngon(n)
    # Créer un vecteur d'entrées pour la matrice circulante
    v = cos(π/n) .- cos.(π/n .+ 2π * (0:n-1) / n)
    
    # Créer la matrice circulante
    M = circulant(v)
    
    # Normaliser M en divisant par M[1, 3]
    M ./= M[1, 3]
    
    # Remplacer les éléments négatifs par 0
    M .*= (M .> 0)
    
    # Modification de la matrice selon les règles spécifiées
    for i in 1:n
        M[i, i] = 0
        if i < n
            M[i, i + 1] = 0
        else
            M[i, 1] = 0
        end
    end
    
    return M
end
