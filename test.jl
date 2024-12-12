include("utils.jl")

function perm(type, mat, index, index2)
    tmp = copy(mat)
    if type==0
        x = div(index, size(mat, 2))  # div() pour la division entière (équivalent de // en Python)
        y = index % size(mat, 2)  # % pour obtenir le reste de la division
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
    if size(matrix) == (1, 1) && matrix[1, 1] == 0
        return pattern
    end
    
    counter = 0
    while counter < 1
        counter += 1
        pattern_best = copy(pattern)  # Equivalent de deepcopy en Julia

        for i in 1:(size(matrix, 1) * size(matrix, 2))
            pattern_tmp = perm(0, pattern, i)
            if compareP1betterthanP2(matrix, pattern_tmp, pattern_best)
                pattern_best = copy(pattern_tmp)
                if verbose
                    println("0 rank: ", fobj(matrix, pattern_best)[1], ", valeur min: ", fobj(matrix, pattern_best)[2])
                end
                counter = 0
            end
        end

        if la_totale
            for i in 1:size(matrix, 1)
                pattern_tmp = perm(1, pattern, i)
                if compareP1betterthanP2(matrix, pattern_tmp, pattern_best)
                    pattern_best = deepcopy(pattern_tmp)
                    if verbose
                        println("1 rank: ", fobj(matrix, pattern_best)[1], ", valeur min: ", fobj(matrix, pattern_best)[2])
                    end
                    counter = 0
                end
            end

            for i in 1:size(matrix, 2)
                pattern_tmp = perm(2, pattern, i)
                if compareP1betterthanP2(matrix, pattern_tmp, pattern_best)
                    pattern_best = deepcopy(pattern_tmp)
                    if verbose
                        println("2 rank: ", fobj(matrix, pattern_best)[1], ", valeur min: ", fobj(matrix, pattern_best)[2])
                    end
                    counter = 0
                end
            end

            for i in 1:size(matrix, 1)
                for j in i:size(matrix, 1)
                    pattern_tmp = perm(3, pattern, i, j)
                    if compareP1betterthanP2(matrix, pattern_tmp, pattern_best)
                        pattern_best = deepcopy(pattern_tmp)
                        if verbose
                            println("3 rank: ", fobj(matrix, pattern_best)[1], ", valeur min: ", fobj(matrix, pattern_best)[2])
                        end
                        counter = 0
                    end
                end
            end

            for i in 1:size(matrix, 2)
                for j in i:size(matrix, 2)
                    pattern_tmp = perm(4, pattern, i, j)
                    if compareP1betterthanP2(matrix, pattern_tmp, pattern_best)
                        pattern_best = deepcopy(pattern_tmp)
                        if verbose
                            println("4 rank: ", fobj(matrix, pattern_best)[1], ", valeur min: ", fobj(matrix, pattern_best)[2])
                        end
                        counter = 0
                    end
                end
            end
        end

        pattern = copy(pattern_best)
    end

    return pattern
end

function greedy(matrix, pattern, setup_break, la_totale, verbose=false)
    if size(matrix) == (1, 1) && matrix[1, 1] == 0
        return pattern
    end
    
    counter = 0
    while counter < 1
        counter += 1
        for i in 1:(size(matrix, 1) * size(matrix, 2))
            pattern_tmp = perm(0, pattern, i)
            if compareP1betterthanP2(matrix, pattern_tmp, pattern)
                pattern = deepcopy(pattern_tmp)
                if verbose
                    println("0 rank: ", fobj(matrix, pattern)[1], ", valeur min: ", fobj(matrix, pattern)[2])
                end
                counter = 0
                if setup_break == 1 || setup_break == 3
                    break
                end
            end
        end

        if la_totale
            for i in 1:size(matrix, 1)
                pattern_tmp = perm(1, pattern, i)
                if compareP1betterthanP2(matrix, pattern_tmp, pattern)
                    pattern = deepcopy(pattern_tmp)
                    if verbose
                        println("1 rank: ", fobj(matrix, pattern)[1], ", valeur min: ", fobj(matrix, pattern)[2])
                    end
                    counter = 0
                    if setup_break == 1 || setup_break == 3
                        break
                    end
                end
            end

            for i in 1:size(matrix, 2)
                pattern_tmp = perm(2, pattern, i)
                if compareP1betterthanP2(matrix, pattern_tmp, pattern)
                    pattern = deepcopy(pattern_tmp)
                    if verbose
                        println("2 rank: ", fobj(matrix, pattern)[1], ", valeur min: ", fobj(matrix, pattern)[2])
                    end
                    counter = 0
                    if setup_break == 1 || setup_break == 3
                        break
                    end
                end
            end

            for i in 1:size(matrix, 1)
                for j in i:size(matrix, 1)
                    pattern_tmp = perm(3, pattern, i, j)
                    if compareP1betterthanP2(matrix, pattern_tmp, pattern)
                        pattern = deepcopy(pattern_tmp)
                        if verbose
                            println("3 rank: ", fobj(matrix, pattern)[1], ", valeur min: ", fobj(matrix, pattern)[2])
                        end
                        counter = 0
                        if setup_break == 1 || setup_break == 3
                            break
                        end
                    end
                end
                if setup_break == 2 || setup_break == 3
                    break
                end
            end

            for i in 1:size(matrix, 2)
                for j in i:size(matrix, 2)
                    pattern_tmp = perm(4, pattern, i, j)
                    if compareP1betterthanP2(matrix, pattern_tmp, pattern)
                        pattern = deepcopy(pattern_tmp)
                        if verbose
                            println("4 rank: ", fobj(matrix, pattern)[1], ", valeur min: ", fobj(matrix, pattern)[2])
                        end
                        counter = 0
                        if setup_break == 1 || setup_break == 3
                            break
                        end
                    end
                end
                if setup_break == 2 || setup_break == 3
                    break
                end
            end
        end
    end

    return pattern
end


