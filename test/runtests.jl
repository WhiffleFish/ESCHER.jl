using ESCHER
using Flux
using CounterfactualRegret
using CounterfactualRegret.Games
const CFR = CounterfactualRegret
using StaticArrays
using Test


ESCHER.vectorized(::MatrixGame, I::Int) = SA[Float32(I)]
ESCHER.vectorized(::MatrixGame, h) = h

function ESCHER.vectorized(game::Kuhn, I)
    p, pc, hist = I
    h = convert(SVector{3,Float32}, hist)
    SA[Float32(p), Float32(pc), h...]
end

function ESCHER.vectorized(game::Kuhn, h::KuhnHist)
    p, pc, hist = I
    h = convert(SVector{3,Float32}, hist)
    SA[Float32(p), Float32(pc), h...]
end

@testset "smoke" begin
    game = Kuhn()
    sol = ESCHERSolver(game)
    train!(sol, 10)
    vb = sol.value_buffer
    @test length(vb.x) == length(vb.y) == length(vb)
    @test length(vb) > 0

    rb1 = first(sol.regret_buffer)
    rb2 = last(sol.regret_buffer)
    @test length(rb1.x) == length(rb1.y) == length(rb1)
    @test length(rb1) > 0
    @test length(rb2.x) == length(rb2.y) == length(rb2)
    @test length(rb2) > 0

    @test length(sol.strategy_buffer.x) == length(sol.strategy_buffer.y)
    @test length(sol.strategy_buffer.x) > 0
end
