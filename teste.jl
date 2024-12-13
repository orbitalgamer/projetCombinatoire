using LinearAlgebra
using Random
using Distributed
using Statistics
include("utils.jl")

# Fonction permutée (équivalent à perm en Python)
function perm(type::Int, mat::Array{Float64,2}, index::Int, index2::Union{Int, Nothing}=nothing)
    mat_tmp = copy(mat)
    if type == 0
        x = div(index, size(mat, 2))
        y = mod(index, size(mat, 2))
        mat_tmp[x+1, y+1] *= -1
    elseif type == 1
        mat_tmp[index+1, :] .*= -1
    elseif type == 2
        mat_tmp[:, index+1] .*= -1
    elseif type == 3 && index2 !== nothing
        mat_tmp[index+1, :], mat_tmp[index2+1, :] = mat_tmp[index2+1, :], mat_tmp[index+1, :]
    elseif type == 4 && index2 !== nothing
        mat_tmp[:, index+1], mat_tmp[:, index2+1] = mat_tmp[:, index2+1], mat_tmp[:, index+1]
    end
    return mat_tmp
end

# Recherche locale
function recherche_locale(matrix, pattern, param, la_totale, verbose=false)
    if size(matrix) == (1, 1) && matrix[1, 1] == 0
        return pattern
    end

    counter = 0
    while counter < 1
        counter += 1
        pattern_best = copy(pattern)
        
        for i in 1:(size(matrix, 1) * size(matrix, 2))
            pattern_tmp = perm(0, pattern, i-1)
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
                pattern_tmp = perm(1, pattern, i-1)
                if compareP1betterthanP2(matrix, pattern_tmp, pattern_best)
                    pattern_best = copy(pattern_tmp)
                    if verbose
                        println("1 rank: ", fobj(matrix, pattern_best)[1], ", valeur min: ", fobj(matrix, pattern_best)[2])
                    end
                    counter = 0
                end
            end

            for i in 1:size(matrix, 2)
                pattern_tmp = perm(2, pattern, i-1)
                if compareP1betterthanP2(matrix, pattern_tmp, pattern_best)
                    pattern_best = copy(pattern_tmp)
                    if verbose
                        println("2 rank: ", fobj(matrix, pattern_best)[1], ", valeur min: ", fobj(matrix, pattern_best)[2])
                    end
                    counter = 0
                end
            end
        end

        pattern = copy(pattern_best)
    end

    return pattern
end

# Fonction subdivise_mat
function subdivise_mat(mat, size)
    list_mat = []
    for i in 0:(div(size(mat, 1), size))
        for j in 0:(div(size(mat, 2), size))
            tmp = mat[i*size+1:min((i+1)*size, size(mat, 1)), j*size+1:min((j+1)*size, size(mat, 2))]
            if !isempty(tmp)
                push!(list_mat, tmp)
            end
        end
    end
    return list_mat
end

# Fonction reassemble_mat
function reassemble_mat(mat, size, list_mat)
    x = div(size(mat, 1), size)
    if mod(size(mat, 1), size) != 0
        x += 1
    end
    y = div(size(mat, 2), size)
    if mod(size(mat, 2), size) != 0
        y += 1
    end

    list_math = []
    for i in 1:x
        push!(list_math, hcat(list_mat[(i-1)*y+1:i*y]...))
    end

    return vcat(list_math...)
end

# Fonction Resolve_metaheuristic
function Resolve_metaheuristic(funct, matrix, pattern, param, verbose=false)
    println("Testing for size=", param[1], ", param2=", param[2], " and param3=", param[3])
    list_mat = subdivise_mat(matrix, param[1])
    list_pat = subdivise_mat(pattern, param[1])
    list_pat = pmap(i -> funct(list_mat[i], list_pat[i], param[2], param[3], verbose), 1:length(list_pat))
    pattern_tmp = reassemble_mat(pattern, param[1], list_pat)
    pattern_tmp = funct(matrix, pattern_tmp, param[2], param[3], verbose)
    return (pattern_tmp, param)
end

# Fonction tabu
function tabu(matrix, pattern, file, param, verbose=false, max_attempt=100)
    if size(matrix) == (1, 1) && matrix[1, 1] == 0
        return pattern
    end

    list_tabu = [copy(pattern) for _ in 1:file]
    pattern_best = copy(pattern)
    counter = 0
    attempt = 0

    while attempt <= max_attempt
        pattern_tmp_best = perm(0, pattern, 0)
        for i in 1:(size(matrix, 1) * size(matrix, 2))
            pattern_tmp = perm(0, pattern, i-1)
            if compareP1betterthanP2(matrix, pattern_tmp, pattern_tmp_best) && !any(equal(pattern_tmp, i) for i in list_tabu)
                if verbose
                    println("rank: ", fobj(matrix, pattern_tmp_best)[1], ", valeur min: ", fobj(matrix, pattern_tmp_best)[2])
                end
                pattern_tmp_best = copy(pattern_tmp)
            end
        end
        list_tabu[counter+1] = pattern_tmp_best
        counter = mod(counter + 1, file)
        attempt += 1
        if compareP1betterthanP2(matrix, pattern_tmp_best, pattern_best)
            pattern_best = copy(pattern_tmp_best)
            if verbose
                println("rank: ", fobj(matrix, pattern_best)[1], ", valeur min: ", fobj(matrix, pattern_best)[2])
            end
            attempt = 0
        end
    end
    return pattern_best
end

# Fonction greedy
function greedy(matrix, pattern, setup_break, la_totale, verbose=false)
    if size(matrix) == (1, 1) && matrix[1, 1] == 0
        return pattern
    end

    counter = 0
    while counter < 1
        counter += 1
        for i in 1:(size(matrix, 1) * size(matrix, 2))
            pattern_tmp = perm(0, pattern, i-1)
            if compareP1betterthanP2(matrix, pattern_tmp, pattern)
                pattern = copy(pattern_tmp)
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
                pattern_tmp = perm(1, pattern, i-1)
                if compareP1betterthanP2(matrix, pattern_tmp, pattern)
                    pattern = copy(pattern_tmp)
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
                pattern_tmp = perm(2, pattern, i-1)
                if compareP1betterthanP2(matrix, pattern_tmp, pattern)
                    pattern = copy(pattern_tmp)
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
                    pattern_tmp = perm(3, pattern, i-1, j-1)
                    if compareP1betterthanP2(matrix, pattern_tmp, pattern)
                        pattern = copy(pattern_tmp)
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
                    pattern_tmp = perm(4, pattern, i-1, j-1)
                    if compareP1betterthanP2(matrix, pattern_tmp, pattern)
                        pattern = copy(pattern_tmp)
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

# Bloc principal traduit
function main()
    matrix = LEDM(120, 120)
    pattern = ones(Float64, size(matrix))
    println(fobj(matrix, pattern))

    debug = true
    best_param = false
    metah = 0 # 0 for greedy, 1 for tabu, 2 for local search

    if best_param
        start_time = time()
        pattern_best = copy(pattern)

        if metah == 0
            la_totale = [false, true]
            setup_break = 0:3
            size_range = 2:(maximum(size(matrix)) + 1)

            params = [(l, s, z) for l in la_totale, s in setup_break, z in size_range]
            results = pmap(param -> Resolve_metaheuristic(greedy, matrix, pattern, param), params)

            for (pattern_tmp, p) in results
                if compareP1betterthanP2(matrix, pattern_tmp, pattern_best)
                    pattern_best = copy(pattern_tmp)
                    println("for param size=", p[3], ", setup_break=", p[2], ", la_totale=", p[1], ", rank=", fobj(matrix, pattern_best)[1], ", min=", fobj(matrix, pattern_best)[2])
                end
            end
        elseif metah == 1
            queue_range = 1:10
            size_range = 2:(maximum(size(matrix)) + 1)

            params = [(q, z) for q in queue_range, z in size_range]
            results = pmap(param -> Resolve_metaheuristic(tabu, matrix, pattern, param), params)

            for (pattern_tmp, p) in results
                if compareP1betterthanP2(matrix, pattern_tmp, pattern_best)
                    pattern_best = copy(pattern_tmp)
                    println("for param size=", p[2], ", queue=", p[1], ", rank=", fobj(matrix, pattern_best)[1], ", min=", fobj(matrix, pattern_best)[2])
                end
            end
        elseif metah == 2
            la_totale = [false, true]
            size_range = 2:(maximum(size(matrix)) + 1)

            params = [(l, z) for l in la_totale, z in size_range]
            results = pmap(param -> Resolve_metaheuristic(recherche_locale, matrix, pattern, param), params)

            for (pattern_tmp, p) in results
                if compareP1betterthanP2(matrix, pattern_tmp, pattern_best)
                    pattern_best = copy(pattern_tmp)
                    println("for param size=", p[2], ", la_totale=", p[1], ", rank=", fobj(matrix, pattern_best)[1], ", min=", fobj(matrix, pattern_best)[2])
                end
            end
        end

        println(fobj(matrix, pattern_best))
        println("Optimization time: ", time() - start_time, " seconds")
    end

    if debug
        start_time = time()

        if !best_param
            size_best = 12
            setup_break_best = 0
            la_totale_best = true
        end

        if metah == 0
            pattern_tmp, _ = Resolve_metaheuristic(greedy, matrix, pattern, (size_best, setup_break_best, la_totale_best))
        elseif metah == 1
            pattern_tmp, _ = Resolve_metaheuristic(tabu, matrix, pattern, (size_best, 10))
        elseif metah == 2
            pattern_tmp, _ = Resolve_metaheuristic(recherche_locale, matrix, pattern, (size_best, la_totale_best))
        end

        println(fobj(matrix, pattern_tmp))
        println("Solution calculation time: ", time() - start_time, " seconds")
    end
end

main()

