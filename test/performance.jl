using BenchmarkTools
using BernsteinBasis
using LinearAlgebra


println("Lift matrix-vector multiplication for N = 7")
@btime mul!($(zeros(120)), $(BernsteinLift(7)), $(rand(Float64, 36)))
@btime mul!($(zeros(120)), $(rand(Float64, 120, 36)), $(rand(Float64, 36)))

println("Lift matrix-matrix multiplication for N = 7")
@btime mul!($(zeros(120, 120)), $(BernsteinLift(7)), $(rand(Float64, 36, 120))) 
@btime mul!($(zeros(120, 120)), $(rand(Float64, 120, 36)), $(rand(Float64, 36, 120))) 

println("Derivative matrix-matrix multiplication for N = 7")
@btime mul!($(zeros(120, 120)), $(BernsteinDerivativeMatrix_3D_r(7)), $(rand(Float64, 120, 120))) 
@btime mul!($(zeros(120, 120)), $(Matrix(BernsteinDerivativeMatrix_3D_r(7))), $(rand(Float64, 120, 120))) 

println("Derivative matrix-vector multiplication for N = 7")
@btime mul!($(zeros(120)), $(BernsteinDerivativeMatrix_3D_r(7)), $(rand(Float64, 120))) 
@btime mul!($(zeros(120)), $(Matrix(BernsteinDerivativeMatrix_3D_r(7))), $(rand(Float64, 120))) 
