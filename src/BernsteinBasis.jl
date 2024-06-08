module BernsteinBasis

using LinearAlgebra

include("BernsteinDerivativeMatrix_3D.jl")
# Write your package code here.
export BernsteinDerivativeMatrix_3D_r, BernsteinDerivativeMatrix_3D_s, 
    BernsteinDerivativeMatrix_3D_t, mul!
    
end
