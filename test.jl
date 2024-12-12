print("helloworld")
include("utils.jl")
using Base.Threads
using Dates
ENV["JULIA_NUM_THREADS"] = 24

function perm(type, mat, index, index2=69)
    tmp = copy(mat)
    if type==0
        x = div(index, size(mat, 2))  # div() pour la division entière (équivalent de // en Python)
        if x == 0
            x+=1
        end
        y = (index % size(mat, 2))  # % pour obtenir le reste de la division
        if y == 0
            y+=1
        end
        tmp[x, y] *= -1  # Modification de l'élément (x, y) de mat_tmp
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
    #if size(matrix) == (1,1) && matrix[1,1] == 0 #vérifie que pas du 1x1 et si élém pas null pou quand écoupe
    #    return pattern
    #end
    counter = 0
    while counter < 1
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
                if !has_break
                    continue
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
                if !has_break
                    continue
                end
            end
        end
    end
    return pattern
end
end

#subdivise en sous matrice
function subdivise_mat(mat, s)
    list_mat = []
    for i in 0:(div(size(mat,1),s)-1)
        for j in 0:(div(size(mat,1), s)-1)
            tmp = mat[i*s+1:(i+1)*s, j*s+1:(j+1)*s] #remarque dernière tranche inclu de base en julia
            if !isempty(tmp)
                push!(list_mat, tmp)
            end
        end
    end
    return list_mat
end

function reassemble_mat(matrix, s, list_mat)
    println("size s = $s")
    x = div(size(matrix,1),s)
    if size(matrix,1) % s != 0
        x+=1
    end
    y = div(size(matrix,2),s)
    if size(matrix,2) % s !=0
        y+=1
    end

    mat = zeros(size(matrix)) #recrée matrice
    println("x : $x")
    println("y : $y")

    #for i in 0:(div(size(mat,1),s)-1)
    #    for j in 0:(div(size(mat,1), s)-1)
    #        mat[i*s+1:(i+1)*s, j*s+1:(j+1)*s] = list_mat[i*y+1:i*y+y] #remarque dernière tranche inclu de base en julia
            
    #    end
    #end

    l=0
    c=0

    liste = []

    for i in 0:(x-1)
        println(i)
        println("took $(i*y+1) : $(i*y+y)")
        #mat[c*y+1:(c*y +y), (l*x+1):(l*x+x)]=list_mat[i*y+1:i*y+y]
        #c+=1
        #if (c*y +y)> size(matrix,2)
        #    c=0
        #    l+=1
        #end
        push!(liste, hcat(list_mat[i*y+1:i*y+y]...))
    end
    mat = vcat(liste...)



    return mat
end


#tabu osef

#lancement
function Resolve_metaheuristic(funct, matrix, pattern, param, verbose=false)
    print("testing for size = $(param[1]), param2=$(param[2]), and param3=$param[3]")
    list_mat = subdivise_mat(matrix, param[1]) #subdivise
    list_pat = subdivise_mat(pattern, param[1]) #pareil pour pattern
    for i in 1:length(list_pat) #parallèle
        list_pat[i] = funct(list_mat[i], list_pat[i], param[2], param[3], verbose)
    end
    pattern_tmp = reassemble_mat(pattern, param[1], list_pat)
    pattern_tmp = funct(matrix, pattern_tmp, param[2], param[3], verbose)
    return pattern_tmp
end

function main()
    matrix = LEDM(120,120)

    pattern = ones(size(matrix))

    print(fobj(matrix, pattern))

    debug = true
    best_param = false
    metah = 0

    if best_param
        start_time = time()
        pattern_best = copy(pattern)
        if metah == 0
            #la_total = [false, true]
            #setup_break = 0:3 
            #size = 2:max(size(matrix))
            data = []
            @threads for i in [true, false]
                for j in 0:3
                    for k in 2:max(size(matrix))
                        push!(matrix, Resolve_metaheuristic(greedy, matrix, pattern, (k, j, i)))
                    end
                end
            end
            
            for (patter_tmp, p) in data
                if compareP1betterthanP2(matrix, patter_tmp, pattern_best)
                    patter_best = copy(patter_tmp)
                    size_best=p[1]
                    setup_break_best = p[2]
                    la_totale_best = p[3]
                    print("fo param size=$size_best, setup_break=$setup_break_best and la_totale=$la_totale_best rank : $(fobj(matrix,patter_best)[1]), valeur min = $(fob(matrix, patter_best)[2])")
                end
            end
            print("fo param size=$size_best, setup_break=$setup_break_best and la_totale=$la_totale_best")
        elseif metah == 2
            la_total = [false, true]
            #setup_break = 0:3 
            #size = 2:max(size(matrix))
            data = []
            @threads for i in [true, false]
                for j in 0:3
                    for k in 2:max(size(matrix))
                        push!(matrix, Resolve_metaheuristic(recherche_locale, matrix, pattern, (k, j, i)))
                    end
                end
            end
            
            for (patter_tmp, p) in data
                if compareP1betterthanP2(matrix, patter_tmp, pattern_best)
                    patter_best = copy(patter_tmp)
                    size_best=p[1]
                    setup_break_best = p[2]
                    la_totale_best = p[3]
                    print("fo param size=$size_best, setup_break=$setup_break_best and la_totale=$la_totale_best rank : $(fobj(matrix,patter_best)[1]), valeur min = $(fob(matrix, patter_best)[2])")
                end
            end
            print("fo param size=$size_best, setup_break=$setup_break_best and la_totale=$la_totale_best")
        end
        print(fobj(matrix, patter_best))
        print("took to optimize $(time()-start_time)")
    end

    println(debug)
    

    if debug
        start_time = time()
        println(best_param)
        if !best_param
            size_best=12
            setup_break_best=0
            la_totale_best=true
        end
        if metah==0
            (patter_tmp, p) = Resolve_metaheuristic(greedy, matrix, pattern, (size_best, setup_break_best, la_totale_best), true)
        elseif metah==2
            (patter_tmp, p) = Resolve_metaheuristic(recherche_locale, matrix, pattern, (size_best, setup_break_best, la_totale_best), true)
        end
        print(fobj(matrix, patter_tmp))
        print("took $(time()-start_time)")
    end
end

main()


