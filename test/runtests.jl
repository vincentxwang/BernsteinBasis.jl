using BernsteinBasis
using Test

@testset "3D derivative matrix fast multiply" begin
    for N in 1:12
        Np = div((N + 1) * (N + 2) * (N + 3), 6)
        x = rand(Float64, Np)
        out = zeros(Float64, Np)
        # Converts LHS to Matrix so it doesn't use mul!.
        @test collect(BernsteinDerivativeMatrix_3D_r(N)) * x ≈ mul!(out, BernsteinDerivativeMatrix_3D_r(N), x)
        @test collect(BernsteinDerivativeMatrix_3D_s(N)) * x ≈ mul!(out, BernsteinDerivativeMatrix_3D_s(N), x)
        @test collect(BernsteinDerivativeMatrix_3D_t(N)) * x ≈ mul!(out, BernsteinDerivativeMatrix_3D_t(N), x)
    end
end
