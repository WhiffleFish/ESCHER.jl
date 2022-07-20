using ESCHER
using Flux
using CounterfactualRegret
using CounterfactualRegret.Games
const CFR = CounterfactualRegret
using StaticArrays
using Plots

function ESCHER.vectorized(game::Kuhn, I::Tuple)
    p, pc, hist = I
    h = convert(SVector{3,Float32}, hist)
    SA[Float32(p), Float32(pc), h...]
end

function ESCHER.vectorized(game::Kuhn, h::Games.KuhnHist)
    (;cards, action_hist) = h
    c = convert(SVector{2,Float32}, cards)
    a = convert(SVector{3,Float32}, action_hist)
    SA[c..., a...]
end

game = Kuhn()
sol = ESCHERSolver(game;
    trajectories = 1000,
    value_buffer_size = 1_000_000,
    regret_buffer_size = 1_000_000,
    strategy_buffer_size = 1_000_000,
    value = Chain(Dense(5,32,relu), Dense(32,32,relu), Dense(32,1)),
    regret = Chain(Dense(5,32,relu), Dense(32,32,relu), Dense(32,2)),
    strategy = Chain(Dense(5,32,relu), Dense(32,32,relu), Dense(32,2), softmax),
    optimizer = Adam(1e-2)
)
cb = ESCHER.ExploitabilityCallback(sol, 10)
train!(sol, 1000, cb = cb)
plot(cb)

net = Chain(Dense(5,32,tanh), Dense(32,32,tanh), Dense(32,1))
t = ESCHER.training_run(net, sol.value_buffer, 128, 200, Adam(1e-2))
last(t.loss) / t.lower_limit - 1.
plot(t)
