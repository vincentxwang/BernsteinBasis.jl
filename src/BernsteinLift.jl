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
end

function l_j(N)
    return ntuple(j -> (j <= N) ? ((isodd(j) ? -1.0 : 1.0) * binomial(N, j) / (1.0 + j)) : 0.0, 20) 
end

# TODO: Procompiling L0_table takes a very long time...
const L0_table = tuple([SparseMatrixCSR(transpose((N + 1)^2/2 * sparse(transpose(ElevationMatrix{N+1}()) * ElevationMatrix{N+1}()))) for N in 1:10]...)
const tri_offset_table = tuple([tri_offsets(N) for N in 1:20]...)
const l_j_table = tuple([l_j(N) for N in 1:20]...)

function BernsteinLift(N)
    Np = div((N + 1) * (N + 2), 2)
    BernsteinLift(N, zeros(Np))
end

"""
    reduction_multiply!(out, N, x, offset)

Multiplies `x` by the Bernstein reduction matrix that maps from degree ``N`` polynomials to 
``N - 1``.

`tri_offsets(N)` should be passed as the precomputed `offset` value.
"""
function reduction_multiply!(out, N, x, offset)
    row = 1
    @inbounds for j in 0:N-1
        @simd for i in 0:N-1-j
            k = N-1-i-j

            # Below are derived from `ij_to_lienar``
            linear_index1 = i + 1 + offset[j+1] + 1
            linear_index2 = i + offset[j+2] + 1
            linear_index3 = i + offset[j+1] + 1

            val = muladd((i+1), x[linear_index1], 
            muladd((j+1), x[linear_index2],
            muladd((k+1), x[linear_index3], 0.0)))
            out[row] = val/N
            row += 1
        end
    end
    return out
end

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
function fast_lift_multiply!(out, N, L0, x, tri_offset_table, l_j_table, E)
    @inbounds begin 
        mul!(E, L0, x)
        index1 = div((N + 1) * (N + 2), 2)
        out[1:index1] .= E
        reduction_multiply!(E, N, E, tri_offset_table[N])
        for j in 1:N
            diff = div((N + 1 - j) * (N + 2 - j), 2)

            # Assign the next (N+1-j)(N+2-j)/2 entries as l_j * (E^N_{N_j})^T u^f
            @views @simd for i in 1:diff
                out[index1 + i] = l_j_table[j] * E[i]
            end

            if j < N
                index1 = index1 + diff
                reduction_multiply!(E, N-j, E, tri_offset_table[N-j])
            end
        end
    end
    return out
end

function LinearAlgebra.mul!(out::AbstractVector{T}, L::BernsteinLift, x::AbstractVector{T}) where T<:Real
    fast_lift_multiply!(out, L.N, L0_table[L.N], x, tri_offset_table, l_j_table[L.N], L.E)
end

function LinearAlgebra.mul!(out::AbstractMatrix{T}, D::BernsteinLift, x::AbstractMatrix{T}) where T<:Real
    @simd for n in axes(x,2)
        @inbounds LinearAlgebra.mul!(view(out,:,n), D, view(x,:,n))
    end
    return out
end