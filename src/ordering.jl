"""
    ij_to_linear(i,j,offset)

Returns the scalar index of the ``(i,j,k)``-th 2D Bernstein basis.
"""
ij_to_linear(i,j, tri_offset) = i + tri_offset[j+1] + 1 

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

"""
    ijk_to_linear(i,j,offset)

Returns the scalar index of the ``(i,j,k,l)``-th 3D Bernstein basis.
"""
function ijk_to_linear(i,j,k, tri_offsets, tet_offsets)
    return i + tri_offsets[j+1] + 1 + tet_offsets[k+1] - j * k
end

function linear_to_ijkl_lookup(N)
    return [(i, j, k, N - i - j - k) for k in 0:N for j in 0:N - k for i in 0:N - j - k]
end

"""
    bernstein_2d_scalar_multiindex_lookup(N)

Returns a vector that maps scalar indices to multi-indices of the ``N``-th degree Bernstein
polynomials.
"""
function bernstein_2d_scalar_multiindex_lookup(N)
    scalar_to_multiindex = [(i,j,N-i-j) for j in 0:N for i in 0:N-j]
    multiindex_to_scalar = Dict(zip(scalar_to_multiindex, collect(1:length(scalar_to_multiindex))))
    return scalar_to_multiindex, multiindex_to_scalar
end