module ESCHER

using CounterfactualRegret
using CUDA
using RecipesBase
const CFR = CounterfactualRegret
using ProgressMeter
using Random
using Flux
using StaticArrays

export ESCHERSolver, TabularESCHERSolver

include("solver.jl")
include("buffer.jl")
include("traverse.jl")
include("train.jl")
include("callback.jl")
include("fitting.jl")
include("value.jl")
include("tabular.jl")
include("recurrent.jl")

end # module
