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

ESCHER.traverse_value!(sol)
ESCHER.train_value!(sol)

h0 = initialhist(game)
h = next_hist(game, h0, rand(chance_actions(game, h0)))
hv = ESCHER.vectorized(game, h)
N = 100_000
v̂ = sum(ESCHER.value_traverse(sol, h, 1) for _ in 1:N)/N
v = ESCHER.exact_value(sol, h)
only(sol.value(hv))

I = ESCHER.vectorized(game,infokey(game, h))
sol.regret[1](I)
ESCHER.regret_match_strategy(sol, 1, @SVector(rand(5)))
