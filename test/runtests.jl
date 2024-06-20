using BernsteinBasis
using Test
using LinearAlgebra

@testset "3D derivative matrix-vector multiplication" begin
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

@testset "3D derivative matrix-matrix multiplication" begin
    for N in 1:6
        Np = div((N + 1) * (N + 2) * (N + 3), 6)
        x = rand(Float64, Np, Np)
        out = zeros(Float64, Np, Np)
        # Converts LHS to Matrix so it doesn't use mul!.
        @test collect(BernsteinDerivativeMatrix_3D_r(N)) * x ≈ mul!(out, BernsteinDerivativeMatrix_3D_r(N), x)
        @test collect(BernsteinDerivativeMatrix_3D_s(N)) * x ≈ mul!(out, BernsteinDerivativeMatrix_3D_s(N), x)
        @test collect(BernsteinDerivativeMatrix_3D_t(N)) * x ≈ mul!(out, BernsteinDerivativeMatrix_3D_t(N), x)
    end
end

@testset "Lift matrix-vector multiplication" begin
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


@testset "2D index ordering" begin
    for N in 1:20
        index = 1
        tri_offsets = BernsteinBasis.tri_offsets(N)
        for j in 0:N
            for i in 0:N-j
                k = N-i-j
                @test index == BernsteinBasis.ij_to_linear(i, j, tri_offsets)
                index += 1
            end
        end
    end
end

@testset "3D index ordering" begin
    for N in 1:20
        index = 1
        tri_offsets = BernsteinBasis.tri_offsets(N)
        tet_offsets = BernsteinBasis.tet_offsets(N)
        for k in 0:N
            for j in 0:N-k
                for i in 0:N-j-k
                    l = N-i-j-k
                    @test index == BernsteinBasis.ijk_to_linear(i, j, k, tri_offsets, tet_offsets)
                    index += 1
                end
            end
        end
    end
end