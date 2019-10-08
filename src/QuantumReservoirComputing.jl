module QuantumReservoirComputing

using DifferentialEquations, LightGraphs, LinearAlgebra

include("input_handler.jl")
export InputSignal

include("ginzburg_landau.jl")
export GLLayer

end # module
