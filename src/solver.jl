struct NullInfoState <: CounterfactualRegret.AbstractInfoState end
Base.@kwdef struct ESCHERSolver{G,V,R,S,VB,RB,SB,SP,OPT,RNG<:AbstractRNG} <: CFR.AbstractCFRSolver{Nothing,G,NullInfoState}
    game::G
    trajectories::Int = 1_000
    value_batch_size::Int = 256
    value_batches::Int = 100
    value_buffer_size::Int
    regret_batch_size::Int = 256
    regret_batches::Int = 100
    regret_buffer_size::Int
    strategy_batch_size::Int = 256
    strategy_batches::Int = 100
    strategy_buffer_size::Int
    value::V
    regret::R
    strategy::S
    value_buffer::VB
    regret_buffer::RB
    strategy_buffer::SB
    sample_policy::SP
    optimizer::OPT = Adam(1e-3)
    ϵ::Float64 = 0.3
    rng::RNG = Random.default_rng()
    exact_value::Bool = false
end

function ESCHERSolver(game::Game{H,K};
    value = nothing,
    regret = nothing,
    strategy = nothing,
    value_buffer_size::Int = 100_000,
    regret_buffer_size::Int = 100_000,
    strategy_buffer_size::Int = 100_000,
    kwargs...
    ) where {H,K}

    h0 = initialhist(game)
    I0 = vectorized_info(game,infokey(game,h0))
    vh = vectorized_hist(game,h0)
    VH = typeof(vh)
    VK = typeof(I0)
    @assert VK <: AbstractVector "`vectorized_info(::Game{H,K}, ::K)` should return vector"
    @assert VH <: AbstractVector "`vectorized_hist(::Game{H,K}, ::H)` should return vector"

    in_size, out_size = in_out_sizes(game)
    sample_policy = UniformPolicy(out_size)

    value = if isnothing(value)
        Chain(
            Dense(in_size, 32, sigmoid),
            Dense(32, 32, sigmoid),
            Dense(32, 1)
        )
    else
        value
    end

    val_ret = typeof(value(vh))
    value_buffer = MemBuffer{VH,Float32}(value_buffer_size)

    regret = if isnothing(regret)
        regret_net = Chain(
            Dense(in_size, 32, sigmoid),
            Dense(32, 32, sigmoid),
            Dense(32, out_size)
        )
        (regret_net, deepcopy(regret_net))
    elseif !isa(regret, Tuple)
        (regret, deepcopy(regret))
    else
        regret
    end

    regret_ret = typeof(regret[1](I0))
    regret_buffer = (MemBuffer{VK,regret_ret}(regret_buffer_size), MemBuffer{VK,regret_ret}(regret_buffer_size))

    strategy = if isnothing(strategy)
        Chain(
            Dense(in_size, 32, sigmoid),
            Dense(32, 32, sigmoid),
            Dense(32, out_size),
            softmax)
    else
        strategy
    end

    strat_ret = first(Base.return_types(strategy, (VK,)))
    strategy_buffer = MemBuffer{VK,strat_ret}(strategy_buffer_size)

    return ESCHERSolver(;
        game, value, regret, strategy,
        sample_policy,
        value_buffer_size,
        regret_buffer_size,
        strategy_buffer_size,
        value_buffer,
        regret_buffer, strategy_buffer, kwargs...)
end

value(sol::ESCHERSolver, i, x) = isone(i) ? only(sol.value(x)) : -only(sol.value(x))
regret(sol::ESCHERSolver, p, x) = sol.regret[p](x)
CFR.strategy(sol::ESCHERSolver, x) = sol.strategy(vectorized_info(sol.game, x))

function buffer_regret!(sol, p, s, r̂)
    push!(sol.regret_buffer[p], s, r̂)
end

function buffer_strategy!(sol, s, a)
    push!(sol.strategy_buffer, s, a)
end

function in_out_sizes(game::Game)
    h0 = initialhist(game)
    k = vectorized_info(game, infokey(game, h0))
    A = actions(game, h0)
    in_size = length(k)
    out_size = length(A)
    return in_size, out_size
end
