using StaticArrays
using BenchmarkTools
using LinearAlgebra

function mymul!(C, A, B)
    for j = axes(C, 2), i = axes(C, 1)
        C[i, j] = sum(A[i, k] * B[k, j] for k=axes(A, 2))
    end
end

const Dr = rand(Float64, 10, 10)
const A = rand(SVector{3, Float64}, 10, 20)
const B = zeros(SVector{3, Float64}, 10, 20)

@btime mul!($(B), $(Dr), $(A))
@btime mymul!($(B), $(Dr), $(A))