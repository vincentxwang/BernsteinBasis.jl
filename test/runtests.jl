using BernsteinBasis
using Test
using LinearAlgebra
using SparseArrays
using StartUpDG

@testset "Matrix-vector mul! of BernsteinDerivativeMatrix_3D" begin
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

@testset "Matrix-matrix mul! of BernsteinDerivativeMatrix_3D" begin
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

@testset "face_lift_multiply vs. a more naive algorithm" begin
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
        @test result ≈ BernsteinBasis.face_lift_multiply!(out, BernsteinLift(N), x)
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

@testset "Elevation multiply vs. ElevationMatrix" begin
    for N in 1:20
        Np = div((N + 1) * (N + 2), 2)
        Np_out = div((N + 2) * (N + 3), 2)
        x = rand(Float64, Np)
        out = zeros(Float64, Np_out)
        @test ElevationMatrix{N+1}() * x ≈ BernsteinBasis.elevation_multiply!(out, N, x, BernsteinBasis.tri_offset_table[N])
    end
end

@testset "Elevation/reduction multiply vs. ElevationMatrix by constructing L0" begin
    for N in 1:9
        L0 = (N + 1)^2/2 * sparse(transpose(ElevationMatrix{N+1}()) * ElevationMatrix{N+1}())

        Np = div((N + 1) * (N + 2), 2)
        x = rand(Float64, Np)
        Np_out = div((N + 2) * (N + 3), 2)
        E = zeros(Float64, Np_out)
        BernsteinBasis.elevation_multiply!(E, N, x, BernsteinBasis.tri_offset_table[N])
        BernsteinBasis.reduction_multiply!(E, N + 1, E, BernsteinBasis.tri_offset_table[N+1])
        E .*= (N + 1)^2/2

        @test L0 * x ≈ E[1:Np]
    end
end

@testset "Overloaded lift matrix mul! vs. get_bernstein_lift" begin
    for N in 1:8
        Np2 = div((N + 1) * (N + 2), 2)
        Np3 = div((N + 1) * (N + 2) * (N + 3), 6)
        X = rand(Float64, 4 * Np2)
        Y = rand(Float64, 4 * Np2, 2)
        @test BernsteinBasis.get_bernstein_lift(N) * X ≈ mul!(zeros(Np3), BernsteinLift(N), X)
        @test BernsteinBasis.get_bernstein_lift(N) * Y ≈ mul!(zeros(Np3, 2), BernsteinLift(N), Y) 
    end
end

@testset "Lift matrix threaded_mul! vs. get_bernstein_lift" begin
    for N in 1:9
        Np2 = div((N + 1) * (N + 2), 2)
        Np3 = div((N + 1) * (N + 2) * (N + 3), 6)
        X = rand(Float64, 4 * Np2)
        @test BernsteinBasis.get_bernstein_lift(N) * X ≈ threaded_mul!(zeros(Np3), BernsteinBasis.MultithreadedBernsteinLift(N, 4), X, 3)
    end
end

@testset "3D derivative matrices vs. StartUpDG" begin
    for N in 1:9
        rd = RefElemData(Tet(), N)
        (; r, s, Fmask) = rd
        rf, sf = rd.r[Fmask[:,1]], rd.t[Fmask[:,1]]
        rd = RefElemData(Tet(), N; quad_rule_face = (rf, sf, ones(length(rf))))

        bernstein_vandermonde, _ = bernstein_basis(Tet(), N, rd.r, rd.s, rd.t)
   
        @test bernstein_vandermonde \ rd.Dr * bernstein_vandermonde ≈ BernsteinDerivativeMatrix_3D_r(N)
        @test bernstein_vandermonde \ rd.Ds * bernstein_vandermonde ≈ BernsteinDerivativeMatrix_3D_s(N)
        @test bernstein_vandermonde \ rd.Dt * bernstein_vandermonde ≈ BernsteinDerivativeMatrix_3D_t(N)
    end
end

@testset "Lift matrix vs. StartUpDG" begin
    for N in 1:10
        rd = RefElemData(Tet(), N)

        rtri, stri = nodes(Tri(), N)
        rfq, sfq, wfq = quad_nodes(Tri(), rd.N)
        Vq_face = vandermonde(Tri(), rd.N, rfq, sfq) / vandermonde(Tri(), rd.N, rtri, stri)

        nodal_LIFT = rd.LIFT * kron(I(4), Vq_face)

        # create interp matrix from Fmask node ordering to quadrature node ordering
        (; r, s, Fmask) = rd
        Fmask = reshape(Fmask, :, 4)
        rf, sf = rd.r[Fmask[:,1]], rd.t[Fmask[:,1]]
        rd = RefElemData(Tet(), N; quad_rule_face = (rf, sf, ones(length(rf))))

        rf, sf, tf, wf = reshape.((rd.rf, rd.sf, rd.tf, rd.wf), :, 4) 

        VBf2, _ = bernstein_basis(Tri(), N, rf[:,1], tf[:,1])
        VBf3, _ = bernstein_basis(Tri(), N, sf[:,2], tf[:,2])
        VBf4, _ = bernstein_basis(Tri(), N, sf[:,3], tf[:,3])
        VBf1, _ = bernstein_basis(Tri(), N, rf[:,4], sf[:,4])
        
        (; r, s, t) = rd

        vande, _ = bernstein_basis(Tet(), N, r, s, t)

        @test BernsteinBasis.get_bernstein_lift(N) ≈ inv(vande) * nodal_LIFT * blockdiag(sparse.((VBf1, VBf2, VBf3, VBf4))...)     
    end
end