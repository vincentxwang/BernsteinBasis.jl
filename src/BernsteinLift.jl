using Test
using BenchmarkTools
using LinearAlgebra
using SparseArrays

"""======= stuff to get L0 and tests to work ========="""

struct ElevationMatrix{N} <: AbstractMatrix{Float64} end

function Base.size(::ElevationMatrix{N}) where {N}
    return (div((N + 1) * (N + 2), 2), div(N * (N + 1), 2))
end

"""
    bernstein_2d_scalar_multiindex_lookup(N)

Returns a vector that maps scalar indices -> multi-indices.
"""
function bernstein_2d_scalar_multiindex_lookup(N)
    scalar_to_multiindex = [(i,j,N-i-j) for j in 0:N for i in 0:N-j]
    multiindex_to_scalar = Dict(zip(scalar_to_multiindex, collect(1:length(scalar_to_multiindex))))
    return scalar_to_multiindex, multiindex_to_scalar
end

function Base.getindex(::ElevationMatrix{N}, m, n) where {N}
    (i1,j1,k1) = bernstein_2d_scalar_multiindex_lookup(N)[1][m]
    (i2,j2,k2) = bernstein_2d_scalar_multiindex_lookup(N-1)[1][n]
    if ((i1, j1, k1) == (i2 + 1,j2,k2)) return (i2 + 1)/N
    elseif ((i1, j1, k1) == (i2,j2 + 1,k2)) return (j2 + 1)/N
    elseif ((i1, j1, k1) == (i2,j2,k2 + 1)) return (k2 + 1)/N
    else return 0.0 end
end

"""end of elevation matrix"""

mutable struct BernsteinLift 
    N::Int
    L0::SparseMatrixCSC{Float64, Int64}
    tri_offset_table::NTuple{20, NTuple{21, Int}}
    l_j_table::NTuple{20, Float64}
    tet_offset::NTuple{21, Int}
    E::Vector{Float64}
end

# assumes i,j >= 0
ij_to_linear(i,j,offset) = i + offset[j+1] + 1 

function tri_offsets(N)
    tup = [0]
    count = 0
    for i in 1:20
        count += N + 2 - i
        push!(tup, count)
    end
    return tuple(tup...)
end

function tet_offsets(N)
    tup = [0]
    count = 0
    for i in 1:20
        count += div((N + 2 - i) * (N + 3 - i), 2)
        push!(tup, count)
    end
    return tuple(tup...)
end

function ijk_to_linear(i,j,k, tri_offsets, tet_offsets)
    return i + tri_offsets[j+1] + 1 + tet_offsets[k+1] - j * k
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

L = BernsteinLift(7)

# pass in offsets(N).
"""
    reduction_multiply!(out, N, x, offset)

multiplies x by the bernstein reduction matrix: N -> N - 1

pass in tri_offsets(N) as offset
"""
function reduction_multiply!(out, N, x, offset)
    row = 1
    @inbounds for j in 0:N-1
        for i in 0:N-1-j
            k = N-1-i-j
            val = muladd((i+1), x[ij_to_linear(i+1, j, offset)], 0)
            val = muladd((j+1), x[ij_to_linear(i, j+1, offset)], val)
            val = muladd((k+1), x[ij_to_linear(i, j, offset)], val) # k + 1
            out[row] = val/N # not sure why putting / N above is faster?
            row += 1
        end
    end
    return out 
end

"""
    fast_lift_multiply!(out, N, L0, x, offset, l_j, E)

multiply x by nice lift matrix face

L0 - as defined in paper
x - input vector
offset - precomputed vector of offset tuples generated by offsets(Tri(), N)
l_j - precomputed coefficients given by l_j(N)
E - pass to reduce redundant memory allocation. vector with same dimensions as x
"""
function fast_lift_multiply!(out, N, L0, x, tri_offset_table, l_j, E)
    mul!(E, L0, x)
    index1 = div((N + 1) * (N + 2), 2)
    out[1:index1] .= E
    reduction_multiply!(E, N, E, tri_offset_table[N])
    @inbounds for j in 1:N
        diff = div((N + 1 - j) * (N + 2 - j), 2)
        index2 = index1 + diff
        # assign the next (N+1-j)(N+2-j)/2 entries as l_j * (E^N_{N_j})^T u^f
        out[(index1 + 1): index2] .= l_j[j] .* @view E[1:diff]
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

@btime mul!($(zeros(120)), $(BernsteinLift(7)), $(rand(Float64, 36))) # 339.804 ns (0 allocations: 0 bytes)