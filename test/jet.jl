using JET
using ESCHER
using Flux
using CounterfactualRegret
using CounterfactualRegret.Games
const CFR = CounterfactualRegret
using StaticArrays
using Plots

ESCHER.vectorized(::MatrixGame, I) = SA[Float32(I)]

function ESCHER.vectorized(game::Kuhn, I)
    p, pc, hist = I
    h = convert(SVector{3,Float32}, hist)
    SA[Float32(p), Float32(pc), h...]
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
train!(sol, 10)
plot(cb;)
JET.@report_call train!(sol, 10, cb = cb)

length(sol.regret_buffer[1])

net = Chain(Dense(5,32,relu), Dense(32,32,relu), Dense(32,2))

ESCHER.initialize!(sol.value)
hist = ESCHER.train_net_tracked!(sol.value, sol.value_buffer, sol.value_batch_size, sol.value_batches, Adam(1e-2))
ll = ESCHER.lower_limit_loss(sol.value_buffer)
plot(hist)
Plots.abline!(0,ll, ls=:dash)

p = 1
ESCHER.initialize!(sol.regret[p])
hist = ESCHER.train_net_tracked!(sol.regret[p], sol.regret_buffer[p], sol.regret_batch_size, sol.regret_batches, Adam(1e-2))
ll = ESCHER.lower_limit_loss(sol.regret_buffer[p])
plot(hist)
Plots.abline!(0,ll, ls=:dash)

ESCHER.initialize!(sol.strategy)
hist = ESCHER.train_net_tracked!(sol.strategy, sol.strategy_buffer, sol.strategy_batch_size, sol.strategy_batches, Adam(1e-2))
ll = ESCHER.lower_limit_loss(sol.strategy_buffer)
plot(hist)
Plots.abline!(0,ll, ls=:dash)

buff = sol.regret_buffer[1]
x = first(buff.x)
y_ = buff.y[map(==(x), buff.x)]
sol.regret[1](x)

sol.regret[1]

sum(y_) / length(y_)

sol.value_buffer

using Plots
plot(cb)


sol.value



net = Chain(Dense(5,32,relu),Dense(32,32,relu),Dense(32,2))
hist = ESCHER.train_net_tracked!(net, sol.value_buffer, 126, 1000, Adam(1e-3))
ll = ESCHER.lower_limit_loss(sol.value_buffer)
plot(hist;yscale=:log10)
Plots.abline!(0,ll, ls=:dash)
sol.regret


ESCHER.exact_value(sol, initialhist(game))


##
t = ESCHER.training_run(sol.value, sol.value_buffer, sol.value_batch_size, 100, Adam(1e-2))
plot(t; lw=2)
