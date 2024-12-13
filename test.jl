println("helloworld")
include("utils.jl")
using Base.Threads
using Dates
ENV["JULIA_NUM_THREADS"] = 24

function perm(type, mat, index, index2=69)
    tmp = copy(mat)
    if type==0
        x = div(index-1, size(mat, 2))  # div() pour la division entière (équivalent de // en Python)
        y = ((index-1) % size(mat, 2))  # % pour obtenir le reste de la division
        tmp[x+1, y+1] *= -1  # Modification de l'élément (x, y) de mat_tmp
    elseif type==1
        tmp[index, :]*=-1
    elseif type==2
        tmp[:,index]*=-1
    elseif type==3
        tmp[index,:], tmp[index2,:]=tmp[index2,:], tmp[index,:]
    elseif type==4
        tmp[:,index], tmp[:,index2]=tmp[:,index2], tmp[:,index]
    end
    return tmp
end

function recherche_locale(matrix, pattern, param, la_totale, verbose=false)
    if size(matrix) == (1,1) && matrix[1,1] == 0 #vérifie que pas du 1x1 et si élém pas null pou quand écoupe
        return pattern
    end
    counter = 0
    while counter<1
        counter+=1
        pattern_best= copy(pattern) #backup pour le modifier comme on veut
        for i in 1:size(matrix, 1)*size(matrix,2) #multiplie dimension pour nombre itération pour cahque échange d'une valeur à l'opposé
            pattern_tmp = perm(0, pattern, i) #explore 1er voisiange au complet
            if compareP1betterthanP2(matrix, pattern_tmp, pattern_best) 
                pattern_best=copy(pattern_tmp)
                if verbose
                    println("0 rank: $(fobj(matrix, pattern_best)[1]), valeur min: $(fobj(matrix,pattern_best)[2])")
                end
                counter=0 #explore tant que n'amliore plus
            end
        end
        if la_totale #dit si veut tous les voisanges ou pas
            for i in 1:size(matrix, 1) #explore tous le voisinage 1 au complet donc éhcange de signe d'une colonne
                pattern_tmp = perm(1, pattern, i)
                if compareP1betterthanP2(matrix, pattern_tmp, pattern_best)
                    pattern_best=copy(pattern_tmp)
                    if verbose
                        println("1 rank: $(fobj(matrix, pattern_best)[1]), valeur min: $(fobj(matrix,pattern_best)[2])")
                    end
                    counter=0
                end
            end
            for i in 1:size(matrix, 2)
                pattern_tmp=perm(2, pattern,i)
                if compareP1betterthanP2(matrix, pattern_tmp, pattern_best)
                    pattern_best=copy(pattern_tmp)
                    if verbose
                        println("2 rank: $(fobj(matrix, pattern_best)[1]), valeur min: $(fobj(matrix,pattern_best)[2])")
                    end
                    counter=0
                end
            end
            for i in 1:size(matrix, 1)
                for j in i:size(matrix,1)
                    pattern_tmp = perm(3, pattern, i,j) #explore le switch de 2 Random mais sur ligne
                    if compareP1betterthanP2(matrix, pattern_tmp, pattern_best)
                        pattern_best=copy(pattern_tmp)
                        if verbose
                            println("3 rank: $(fobj(matrix, pattern_best)[1]), valeur min: $(fobj(matrix,pattern_best)[2])")
                        end
                        counter=0
                    end
                end
            end
            for i in 1:size(matrix, 2)
                for j in i:size(matrix,2)
                    pattern_tmp = perm(4, pattern, i,j) #explore le switch de 2 Random mais sur colone
                    if compareP1betterthanP2(matrix, pattern_tmp, pattern_best)
                        pattern_best=copy(pattern_tmp)
                        if verbose
                            println("4 rank: $(fobj(matrix, pattern_best)[1]), valeur min: $(fobj(matrix,pattern_best)[2])")
                        end
                        counter=0
                    end
                end
            end
        end
        pattern = copy(pattern_best)
    end
    return pattern
end


function greedy(matrix, pattern, setup_break, la_totale, verbose=false)
    if size(matrix) == (1,1) && matrix[1,1] == 0 #vérifie que pas du 1x1 et si élém pas null pou quand écoupe
       return pattern
    end
    counter = 0
    while counter < 1
        counter+=1
        for i in 1:(size(matrix, 1)*size(matrix, 2))
            pattern_tmp = perm(0, pattern, i)
            if compareP1betterthanP2(matrix, pattern_tmp, pattern) 
                pattern=copy(pattern_tmp)
                if verbose
                    println("0 rank: $(fobj(matrix, pattern)[1]), valeur min: $(fobj(matrix,pattern)[2])")
                end
                counter=0 #explore tant que n'amliore plus
                if setup_break==1 || setup_break == 3
                    break
                end
            end         
        end
        if la_totale #dit si veut tous les voisanges ou pas
            for i in 1:size(matrix, 1) #explore tous le voisinage 1 au complet donc éhcange de signe d'une colonne
                pattern_tmp = perm(1, pattern, i)
                if compareP1betterthanP2(matrix, pattern_tmp, pattern)
                    pattern=copy(pattern_tmp)
                    if verbose
                        println("1 rank: $(fobj(matrix, pattern)[1]), valeur min: $(fobj(matrix,pattern)[2])")
                    end
                    counter=0
                    if setup_break==1 || setup_break == 3
                        break
                    end
                end
            end
            for i in 1:size(matrix, 2)
                pattern_tmp=perm(2, pattern,i)
                if compareP1betterthanP2(matrix, pattern_tmp, pattern)
                    pattern=copy(pattern_tmp)
                    if verbose
                        println("2 rank: $(fobj(matrix, pattern)[1]), valeur min: $(fobj(matrix,pattern)[2])")
                    end
                    counter=0
                    if setup_break==1 || setup_break == 3
                        break
                    end
                end
            end
            for i in 1:size(matrix, 1)
                has_break = false
                for j in i:size(matrix,1)
                    pattern_tmp = perm(3, pattern, i,j) #explore le switch de 2 Random mais sur ligne
                    if compareP1betterthanP2(matrix, pattern_tmp, pattern)
                        pattern=copy(pattern_tmp)
                        if verbose
                            println("3 rank: $(fobj(matrix, pattern)[1]), valeur min: $(fobj(matrix,pattern)[2])")
                        end
                        counter=0
                        if setup_break==1 || setup_break == 3
                            has_break = true
                            break
                        end
                    end
                    if has_break
                        if setup_break==2 || setup_break==3
                            break
                        end
                    end
                end
                for i in 1:size(matrix, 2)
                    for j in i:size(matrix,2)
                        has_break = false
                        pattern_tmp = perm(4, pattern, i,j) #explore le switch de 2 Random mais sur colone
                        if compareP1betterthanP2(matrix, pattern_tmp, pattern)
                            pattern=copy(pattern_tmp)
                            if verbose
                                println("4 rank: $(fobj(matrix, pattern)[1]), valeur min: $(fobj(matrix,pattern)[2])")
                            end
                            counter=0
                            if setup_break==1 || setup_break == 3
                                has_break=true
                                break
                            end
                        end
                    end
                    if has_break
                        if setup_break==2 || setup_break==3
                            break
                        end
                    end
                end
            end
        end
    end
    return pattern
end

#subdivise en sous matrice
function subdivise_mat(mat, s)
    x=div(size(mat,1),s)
    if size(mat,1)%s!=0
        x+=1
    end
    y=div(size(mat,2),s)
    if size(mat,2)%s!=0
        y+=1
    end
    list_mat = []
    for i in 0:x-1
        for j in 0:y-1
            push!(list_mat,mat[i*s+1:min(size(mat,1),(i+1)*s), j*s+1:min(size(mat,2),(j+1)*s)]) #remarque dernière tranche inclu de base en julia
        end
    end
    return list_mat
end

function reassemble_mat(matrix, s, list_mat)
    x = div(size(matrix,1),s)
    if size(matrix,1) % s != 0
        x+=1
    end
    y = div(size(matrix,2),s)
    if size(matrix,2) % s !=0
        y+=1
    end

    liste = []

    for i in 0:(x-1)
        push!(liste, hcat(list_mat[i*y+1:i*y+y]...))
    end
    mat = vcat(liste...)
    return mat
end


#tabu osef

#lancement
function Resolve_metaheuristic(funct, matrix, pattern, param, verbose=false)
    println("testing for size = $(param[1]), param2=$(param[2]), and param3=$(param[3])")
    list_mat = subdivise_mat(matrix, param[1]) #subdivise
    list_pat = subdivise_mat(pattern, param[1]) #pareil pour pattern
    @threads for i in eachindex(list_pat) #parallèle
        list_pat[i] = funct(list_mat[i], list_pat[i], param[2], param[3], verbose)
    end
    pattern_tmp = reassemble_mat(pattern, param[1], list_pat)
    pattern_tmp = funct(matrix, pattern_tmp, param[2], param[3], verbose)
    return (pattern_tmp,param)
end

function main()
    matrix = LEDM(32,32)

    pattern = ones(size(matrix))

    println(fobj(matrix, pattern))

    debug = true
    best_param = true
    metah = 0

    if best_param
        start_time = time()
        pattern_best = copy(pattern)
        if metah == 0
            data = []
            @threads for i in CartesianIndices((2:maximum(size(matrix)),0:3,0:1))
                if i[3]==1
                    push!(data, Resolve_metaheuristic(greedy, matrix, pattern, (i[1], i[2], true)))
                else
                    push!(data, Resolve_metaheuristic(greedy, matrix, pattern, (i[1], i[2], false)))
                end
            end

            for (patter_tmp, p) in data
                if compareP1betterthanP2(matrix, patter_tmp, pattern_best)
                    pattern_best = copy(patter_tmp)
                    size_best=p[1]
                    setup_break_best = p[2]
                    la_totale_best = p[3]
                    println("for param size=$size_best, setup_break=$setup_break_best and la_totale=$la_totale_best rank : $(fobj(matrix,pattern_best)[1]), valeur min = $(fobj(matrix, pattern_best)[2])")
                end
            end
            println("param opti size=$size_best, setup_break=$setup_break_best and la_totale=$la_totale_best")
        elseif metah == 2
            data = []
            @threads for i in CartesianIndices((2:maximum(size(matrix)),0:1))
                if i[2]==1
                    push!(data, Resolve_metaheuristic(greedy, matrix, pattern, (i[1],'/', true)))
                else
                    push!(data, Resolve_metaheuristic(greedy, matrix, pattern, (i[1],'/', false)))
                end
            end
            
            for (patter_tmp, p) in data
                if compareP1betterthanP2(matrix, patter_tmp, pattern_best)
                    pattern_best = copy(patter_tmp)
                    size_best=p[1]
                    setup_break_best = p[2]
                    la_totale_best = p[3]
                    println("fo param size=$size_best, setup_break=$setup_break_best and la_totale=$la_totale_best rank : $(fobj(matrix,pattern_best)[1]), valeur min = $(fobj(matrix, pattern_best)[2])")
                end
            end
            println("fo param size=$size_best, setup_break=$setup_break_best and la_totale=$la_totale_best")
        end
        println(fobj(matrix, pattern_best))
        println("took to optimize $(time()-start_time)")
    end

    if debug
        start_time = time()
        if !best_param
            size_best=6
            setup_break_best=0
            la_totale_best=false
        end
        if metah==0
            (patter_tmp, p) = Resolve_metaheuristic(greedy, matrix, pattern, (size_best, setup_break_best, la_totale_best), true)
        elseif metah==2
            (patter_tmp, p) = Resolve_metaheuristic(recherche_locale, matrix, pattern, (size_best, setup_break_best, la_totale_best), true)
        end
        println(fobj(matrix, patter_tmp))
        println("took $(time()-start_time)s")
    end
    ecrire_fichier("solution_julia.txt",matrix,patter_tmp)
end

main()


