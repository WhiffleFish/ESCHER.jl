function value_traverse(sol::ESCHERSolver, h, p)
    (;game, ϵ) = sol
    current_player = player(game, h)

    if isterminal(game, h)
        return utility(game, 1, h) # trained on p1 utilities (assuming zero sum)
    elseif iszero(current_player)
        A = chance_actions(game, h)
        a = rand(sol.rng, A)

        h′= next_hist(game, h, a)
        v̂ = value_traverse(sol, h′, p)

        h_vec = vectorized_hist(game, h)
        push!(sol.value_buffer, h_vec, Float32(v̂))

        return v̂
    end

    I = vectorized_info(game, infokey(game, h))
    A = actions(game, h)

    if current_player == p
        π̃ = sol.sample_policy(I)
        σ = regret_match_strategy(sol, p, I)
        # a_idx = weighted_sample(sol.rng, π̃)
        a_idx = weighted_sample(sol.rng, σ)
        h′ = next_hist(game, h, A[a_idx])
        v̂ = value_traverse(sol, h′, p)#*(σ[a_idx]/π̃[a_idx])

        h_vec = vectorized_hist(game, h)

        push!(sol.value_buffer, h_vec, Float32(v̂))

        # bootstrap maybe?
        return v̂
    else
        π_ni = regret_match_strategy(sol, current_player, I)
        π̃ = ϵ .* π_ni .+ (1 - ϵ)*inv(length(A))
        # a_idx = weighted_sample(sol.rng, π̃)
        a_idx = weighted_sample(sol.rng, π_ni)
        h′ = next_hist(game, h, A[a_idx])
        v̂ = value_traverse(sol, h′, p)#*(π_ni[a_idx] / π̃[a_idx])
        h_vec = vectorized_hist(game, h)
        push!(sol.value_buffer, h_vec, Float32(v̂))
        return v̂
    end
end


function regret_traverse(sol::ESCHERSolver, h, p)
    (;game) = sol
    current_player = player(game, h)

    if isterminal(game, h)
        return utility(game, p, h) # trained on p1 utilities (assuming zero sum)
    elseif iszero(current_player)
        A = chance_actions(game, h)
        a = rand(sol.rng, A)
        h′ = next_hist(game, h, a)
        return regret_traverse(sol, h′, p)
    end

    I = vectorized_info(game, infokey(game, h))
    A = actions(game, h)

    if current_player == p
        π̃ = sol.sample_policy(I)
        σ = regret_match_strategy(sol, p, I)
        a_idx = weighted_sample(sol.rng, π̃)
        v = 0.0
        r̂ = zero(σ)
        for i in eachindex(σ)
            q = child_value(sol, p, h, A[i])
            r̂[i] = q
            v += σ[i]*q
        end
        r̂ .-= v
        h′ = next_hist(game, h, A[a_idx])
        buffer_regret!(sol, p, I, r̂)
        buffer_strategy!(sol, I, σ)
        return regret_traverse(sol, h′, p)
    else
        π_ni = regret_match_strategy(sol, p, I)
        a_idx = weighted_sample(sol.rng, π_ni)
        h′ = next_hist(game, h, A[a_idx])
        return regret_traverse(sol, h′, p)
    end
end


function child_value(sol::ESCHERSolver, p, h, a)
    h′ = next_hist(sol.game, h, a)
    return if isterminal(sol.game, h′)
        utility(sol.game, p, h′)
    else
        value(sol, p, vectorized_hist(sol.game,h′))
    end
end

function regret_match_strategy(sol::ESCHERSolver, p, I)
    return regret_match!(regret(sol, p, I))
end

function regret_match!(r::AbstractVector{T}) where T
    s = zero(T)
    for i ∈ eachindex(r)
        if r[i] > zero(T)
            s += r[i]
        else
            r[i] = zero(T)
        end
    end
    return s > zero(T) ? (r ./= s) : fill!(r,inv(length(r)))
end

function weighted_sample(rng::Random.AbstractRNG, σ::AbstractVector)
    t = rand(rng)
    i = 1
    cw = σ[1]
    while cw < t && i < length(σ)
        i += 1
        cw += σ[i]
    end
    return i
end
