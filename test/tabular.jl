@testset "tabular" begin
    game = Kuhn()
    sol = TabularESCHERSolver(game)
    cb = CFR.ExploitabilityCallback(sol, 100)
    train!(sol, 100_000; cb=cb)
    @test last(cb.hist.y) < 1e-2
end
