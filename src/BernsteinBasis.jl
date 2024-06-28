module BernsteinBasis

using BenchmarkTools
using SparseArrays
using StaticArrays
using LinearAlgebra
using SparseMatricesCSR

include("utility.jl")

include("ElevationMatrix.jl")
include("ordering.jl")

include("constants.jl")

include("BernsteinDerivativeMatrix_3D.jl")
include("BernsteinLift.jl")

export ElevationMatrix
export BernsteinDerivativeMatrix_3D_r, BernsteinDerivativeMatrix_3D_s, 
    BernsteinDerivativeMatrix_3D_t, mul!
export BernsteinLift, mul!

end
