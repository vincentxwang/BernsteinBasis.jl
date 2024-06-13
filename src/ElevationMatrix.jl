"""
    ElevationMatrix{N} <: AbstractMatrix{Float64}

Two-dimensional degree elevation operator for Bernstein polynomials. Expresses polynomials
of degree ``N - 1`` as polynomials of degree ``N``.

- Kirby, Robert C. (2016)
  Fast inversion of the simplicial Bernstein mass matrix
  [DOI: 10.1007/s00211-016-0795-0]https://doi.org/10.1007/s00211-016-0795-0
"""
struct ElevationMatrix{N} <: AbstractMatrix{Float64} end

function Base.size(::ElevationMatrix{N}) where {N}
    return (div((N + 1) * (N + 2), 2), div(N * (N + 1), 2))
end

function Base.getindex(::ElevationMatrix{N}, m, n) where {N}
    (i1,j1,k1) = bernstein_2d_scalar_multiindex_lookup(N)[1][m]
    (i2,j2,k2) = bernstein_2d_scalar_multiindex_lookup(N-1)[1][n]
    if ((i1, j1, k1) == (i2 + 1,j2,k2)) return (i2 + 1)/N
    elseif ((i1, j1, k1) == (i2,j2 + 1,k2)) return (j2 + 1)/N
    elseif ((i1, j1, k1) == (i2,j2,k2 + 1)) return (k2 + 1)/N
    else return 0.0 end
end