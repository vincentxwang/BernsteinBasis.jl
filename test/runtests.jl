using BernsteinBasis
using Test
using LinearAlgebra

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

@testset "Lift matrix multiplication" begin
    # Alternate construction of lift matrix
    for N in 1:9
        L0 = (N + 1)^2/2 * transpose(ElevationMatrix{N+1}()) * ElevationMatrix{N+1}()
        Lf = L0
        E = I
        for i in 1:N
            E = E * ElevationMatrix{N+1-i}() 
            Lf = vcat(Lf, (-1)^i * binomial(N, i) / (1 + i) * transpose(E) * L0)
        end
        Np = div((N + 1) * (N + 2), 2)
        x = rand(Float64, Np)

        result = Lf * x
        out = similar(result)
        @test result ≈ mul!(out, BernsteinLift(N), x)
    end
end