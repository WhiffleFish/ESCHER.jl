using ESCHER
using Flux
using CounterfactualRegret
using CounterfactualRegret.Games
const CFR = CounterfactualRegret
using StaticArrays
using Test

include("smoke.jl")
include("value.jl")
include("tabular.jl")
