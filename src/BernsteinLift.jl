"""
    BernsteinLift

Lift matrix for a single face on a standard tetrahedron.

- Chan, Jesse and Tim Warburton (2017)
  GPU-accelerated Bernstein-Bezier discontinuous Galerkin methods for wave problems
  [DOI: 10.48550/arXiv.1512.06025](https://doi.org/10.48550/arXiv.1512.06025)
"""
mutable struct BernsteinLift 
    N::Int
    L0::SparseMatrixCSC{Float64, Int64}
    tri_offset_table::NTuple{20, NTuple{21, Int}}
    l_j_table::NTuple{20, Float64}
    tet_offset::NTuple{21, Int}
    E::Vector{Float64}
end

function l_j(N)
    return ntuple(j -> (j <= N) ? ((isodd(j) ? -1.0 : 1.0) * binomial(N, j) / (1.0 + j)) : 0.0, 20) 
end

function BernsteinLift(N)
    Np = div((N + 1) * (N + 2), 2)
    L0 = (N + 1)^2/2 * sparse(transpose(ElevationMatrix{N+1}()) * ElevationMatrix{N+1}())
    tri_offset_table = tuple([tri_offsets(N) for N in 1:20]...)
    BernsteinLift(N, L0, tri_offset_table, l_j(N), tet_offsets(N), zeros(Np))
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
        for i in 0:N-1-j
            k = N-1-i-j
            val = muladd((i+1), x[ij_to_linear(i+1, j, offset)], 0)
            val = muladd((j+1), x[ij_to_linear(i, j+1, offset)], val)
            val = muladd((k+1), x[ij_to_linear(i, j, offset)], val) # k + 1
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
``L0::SparseMatrixCSC{Float64, Int64}``: Precomputed matrix as defined in (Chan 2017)
``x::AbstractVector``: Input vector
``tri_offset_table::NTuple{20, NTuple{21, Int}}``: Precomputed vector of offset tuples
``l_j_table::NTuple{20, Float64}``: Precomputed coefficients given by `l_j(N)`
``E::Vector{Float64}`` Dummy vector (with same dimensions as `x`) to reduce redundant memory allocation.

- Chan, Jesse and Tim Warburton (2017)
  GPU-accelerated Bernstein-Bezier discontinuous Galerkin methods for wave problems
  [DOI: 10.48550/arXiv.1512.06025](https://doi.org/10.48550/arXiv.1512.06025)
"""
function fast_lift_multiply!(out, N, L0, x, tri_offset_table, l_j_table, E)
    mul!(E, L0, x)
    index1 = div((N + 1) * (N + 2), 2)
    out[1:index1] .= E
    reduction_multiply!(E, N, E, tri_offset_table[N])
    @inbounds for j in 1:N
        diff = div((N + 1 - j) * (N + 2 - j), 2)
        index2 = index1 + diff
        # Assign the next (N+1-j)(N+2-j)/2 entries as l_j * (E^N_{N_j})^T u^f
        out[(index1 + 1): index2] .= l_j_table[j] .* @view E[1:diff]
        if j < N
            index1 = index2
            reduction_multiply!(E, N-j, E, tri_offset_table[N-j])
        end
    end
    return out
end

function LinearAlgebra.mul!(out, L::BernsteinLift, x)
    fast_lift_multiply!(out, L.N, L.L0, x, L.tri_offset_table, L.l_j_table, L.E)
end