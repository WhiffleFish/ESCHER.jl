struct TabularInfoState <: CFR.AbstractInfoState
    regret::Vector{Float64}
    strategy::Vector{Float64}
end
TabularInfoState(n::Int) = TabularInfoState(zeros(n), zeros(n))

struct UniformSampler end

(::UniformSampler)(::Any, n::Integer) = CFR.FillVec(inv(n), n)

struct TabularESCHERSolver{K,G,SP,H,RNG} <: CFR.AbstractCFRSolver{K, G, TabularInfoState}
    game::G
    sample_policy::SP
    I::Dict{K, TabularInfoState}
    value::Dict{H, Float64}
    rng::RNG
end

function TabularESCHERSolver(game::Game{H,K}) where {H,K}
    return TabularESCHERSolver(
        game,
        UniformSampler(),
        Dict{K, TabularInfoState}(),
        Dict{H, Float64}(),
        Random.default_rng()
    )
end

function CFR.strategy(sol::TabularESCHERSolver{K}, I::K) where K
    infostate = get(sol.I, I, nothing)
    if isnothing(infostate) # FIXME: terrible idea - assumes constant size action space
        L = length(first(values(sol.I)).strategy)
        return fill(inv(L), L)
    else
        σ = copy(infostate.strategy)
        return σ ./= sum(σ)
    end
end

function regret_match_strategy(sol::TabularESCHERSolver, p, I::TabularInfoState)
    return regret_match!(copy(I.regret))
end

function regret_match_strategy(sol::TabularESCHERSolver, p, I)
    return regret_match_strategy(sol, p, sol.I[I])
end

function infostate(sol::TabularESCHERSolver, h, n)
    return get!(sol.I, infokey(sol.game, h)) do
        TabularInfoState(n)
    end
end

function CFR.train!(sol::TabularESCHERSolver, T::Int; show_progress=true, cb=()->())
    prog = Progress(T; enabled=show_progress)
    h0 = initialhist(sol.game)
    for t ∈ 1:T
        fill_value!(sol, h0)
        for p ∈ 1:2
            traverse(sol, h0, p)
        end
        cb()
        next!(prog)
    end
end

function traverse(sol::TabularESCHERSolver, h, p)
    (;game) = sol
    current_player = player(game, h)

    if isterminal(game, h)
        return utility(game, p, h) # trained on p1 utilities (assuming zero sum)
    elseif iszero(current_player)
        A = chance_actions(game, h)
        a = rand(sol.rng, A)
        h′ = next_hist(game, h, a)
        return traverse(sol, h′, p)
    end

    A = actions(game, h)
    I = infostate(sol, h, length(A))

    if current_player == p

        π_sample = sol.sample_policy(I, length(A))
        σ = regret_match_strategy(sol, p, I)
        a_idx = weighted_sample(sol.rng, π_sample)
        v = 0.0
        r̂ = child_values(sol, h)
        for i in eachindex(σ,A)
            q = child_value(sol, p, h, A[i])
            r̂[i] = q
            v += σ[i]*q
        end
        r̂ .-= v
        I.regret .+= r̂
        I.strategy .+= σ
        h′ = next_hist(game, h, A[a_idx])
        return traverse(sol, h′, p)
    else
        π_ni = regret_match_strategy(sol, p, I)
        a_idx = weighted_sample(sol.rng, π_ni)
        h′ = next_hist(game, h, A[a_idx])
        return traverse(sol, h′, p)
    end
end

function child_value(sol, p, h, a)
    h′ = next_hist(sol.game, h, a)
    return isone(p) ? sol.value[h′] : -sol.value[h′]
end

function child_values(sol::TabularESCHERSolver, h)
    A = actions(sol.game, h)
    q = zeros(length(A))
    for i in eachindex(A)
        h′ = next_hist(sol.game, h, A[i])
        q[i] = sol.value[h′]
    end
    return q
end

function fill_value!(sol::TabularESCHERSolver, h)
    game = sol.game
    p = player(game, h)
    if isterminal(game, h)
        u = utility(game, 1, h)
        sol.value[h] = u
        return u
    elseif iszero(p)
        A = chance_actions(game, h)
        v = 0.0
        for a in A
            v += fill_value!(sol, next_hist(game, h, a))
        end
        u = v/length(A)
        sol.value[h] = u
        return u
    else
        CFR.infoset(sol, h)
        I = infokey(game, h)
        σ = regret_match_strategy(sol, p, I)
        A = actions(game, h)
        v = 0.0
        for i in eachindex(σ, A)
            v += σ[i]*fill_value!(sol, next_hist(game, h, A[i]))
        end
        sol.value[h] = v
        return v
    end
end
