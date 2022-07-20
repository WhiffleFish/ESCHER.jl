@testset "value estimation" begin
    N = 100_000
    game = Kuhn()
    sol = ESCHERSolver(game)
    h0 = initialhist(game)
    for a in chance_actions(game, h0)
        h = next_hist(game, h0, a)
        v̂ = sum(ESCHER.value_traverse(sol, h, 1) for _ in 1:N)/N
        v = ESCHER.exact_value(sol, h)
        @test isapprox(v̂,v, atol=3e-2)
    end
end
