"""
    BernsteinLift

Lift matrix for a single face on a standard tetrahedron.

- Chan, Jesse and Tim Warburton (2017)
  GPU-accelerated Bernstein-Bezier discontinuous Galerkin methods for wave problems
  [DOI: 10.48550/arXiv.1512.06025](https://doi.org/10.48550/arXiv.1512.06025)
"""
mutable struct BernsteinLift 
    N::Int
    E::Vector{Float64}
    F::Vector{Float64}
end

function BernsteinLift(N)
    Np1 = div((N+1) *(N+2), 2)
    # F will store this many entries because we multiply by the elevation matrix.
    Np2 = div((N+2) * (N+3), 2)
    BernsteinLift(N, zeros(Np1), zeros(Np2))
end

"""
    reduction_multiply!(out, N, x, offset)

Multiplies `x` by the 2D Bernstein reduction matrix that maps from degree ``N`` polynomials to 
``N - 1``.

`tri_offsets(N)` should be passed as the precomputed `offset` value.
"""
function reduction_multiply!(out, N, x, offset)
    row = 1
    @inbounds for j in 0:N-1
        @simd for i in 0:N-1-j
            # k = N-1-i-j

            linear_index1 = ij_to_linear(i+1,j,offset)
            linear_index2 = ij_to_linear(i,j+1,offset)
            linear_index3 = ij_to_linear(i,j,offset) # k+1
            
            # Closed form formulas for above in `i,j,N`
            # linear_index1 = (i+1) + div(j * (2N + 3 - j), 2) + 1
            # linear_index2 = i + div((j+1) * (2N + 3 - (j+1)), 2) + 1
            # linear_index3 = linear_index3 - 1

            val = muladd((i+1), x[linear_index1], 
            muladd((j+1), x[linear_index2],
            (N-i-j) * x[linear_index3])) # This is really k * x[linear_index3]
            
            out[row] = val/N
            row += 1
        end
    end
    return out
end

"""
    elevation_multiply!(out, N, x, offset)

Multiplies `x` by the 2D Bernstein elevation matrix that maps from degree ``N`` polynomials to 
``N + 1``.

`tri_offsets(N)` should be passed as the precomputed `offset` value.
"""
function elevation_multiply!(out, N, x, offset)
    row = 1
    @inbounds for j in 0:N+1
        @simd for i in 0:N+1-j
            k = N+1-i-j

            val = zero(eltype(x))

            if i > 0
                val = i * x[ij_to_linear(i-1,j,offset)]
            end

            if j > 0
                val = muladd(j, x[ij_to_linear(i,j-1,offset)], val)
            end

            if k > 0
                val = muladd(k, x[ij_to_linear(i,j,offset)], val)
            end

            out[row] = val/(N+1)
            row += 1
        end
    end
    return out
end

#stuff
# N = 7
# Np = div((N + 1) * (N + 2), 2)
# x = rand(Float64, Np)
# Np_out = div((N + 2) * (N + 3), 2)
# E = zeros(Float64, Np_out)
# @btime BernsteinBasis.elevation_multiply!($E, $N, $x, $(BernsteinBasis.tri_offset_table[N]))
# @btime mul!($E, $(sparse(ElevationMatrix{N+1}())), $x)
# @btime mul!($x, $(L0_table[7]), $x)
# @btime $E .*= ($N + 1)^2/2
# @btime BernsteinBasis.reduction_multiply!($E, $(N + 1), $E, $(BernsteinBasis.tri_offset_table[N+1]))

# ElevationMatrix{7}()

# spy(ElevationMatrix{7}(), ms = 2)

"""
    fast_lift_multiply!(out, N, L0, x, offset, l_j, E)

Multiplies `x` by the "nice" lift matrix face (``rs``-plane).

# Arguments
``L0::SparseMatrixCSR{Float64, Int64}``: Precomputed matrix as defined in (Chan 2017)
``x::AbstractVector``: Input vector
``tri_offset_table::NTuple{20, NTuple{21, Int}}``: Precomputed vector of offset tuples
``l_j_table::NTuple{20, Float64}``: Precomputed coefficients given by `l_j(N)`
``E::Vector{Float64}`` Dummy vector (with same dimensions as `x`) to reduce redundant memory allocation.

- Chan, Jesse and Tim Warburton (2017)
  GPU-accelerated Bernstein-Bezier discontinuous Galerkin methods for wave problems
  [DOI: 10.48550/arXiv.1512.06025](https://doi.org/10.48550/arXiv.1512.06025)
"""
function fast_lift_multiply!(out, N, x, E, F)
    @inbounds begin 
        elevation_multiply!(F, N, x, tri_offset_table[N])
        reduction_multiply!(E, N + 1, F, tri_offset_table[N+1])
        E .*= (N + 1)^2/2
        index1 = div((N + 1) * (N + 2), 2)
        view(out,1:index1) .= E
        reduction_multiply!(E, N, E, tri_offset_table[N])
        for j in 1:N
            diff = div((N + 1 - j) * (N + 2 - j), 2)

            # Assign the next (N+1-j)(N+2-j)/2 entries as l_j * (E^N_{N_j})^T u^f
            for i in 1:diff
                out[index1 + i] = l_j_table[N][j] * E[i]
            end

            # view(out,(index1+1):(index1+diff)) .= l_j_table[N][j] .* @view E[1:diff]

            if j < N
                index1 = index1 + diff
                reduction_multiply!(E, N-j, E, tri_offset_table[N-j])
            end
        end
    end
    return out
end

function LinearAlgebra.mul!(out::AbstractVector{T}, L::BernsteinLift, x::AbstractVector{T}) where T<:Real
    fast_lift_multiply!(out, L.N, x, L.E, L.F)
end

function LinearAlgebra.mul!(out::AbstractMatrix{T}, D::BernsteinLift, x::AbstractMatrix{T}) where T<:Real
    @simd for n in axes(x,2)
        @inbounds LinearAlgebra.mul!(view(out,:,n), D, view(x,:,n))
    end
    return out
end
