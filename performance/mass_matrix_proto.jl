using BernsteinBasis

struct BernsteinMass <: AbstractMatrix{Int64} 
    N::Int
end

function Base.size(M::BernsteinMass)
    N = M.N
    Np = div((N + 1) * (N + 2) * (N + 3), 6)
    return (Np, Np)
end

function Base.getindex(M::BernsteinMass, m, n)
    N = M.N
    linear_to_ijkl = BernsteinBasis.linear_to_ijkl_lookup(N)
    (i1, j1, k1, l1) = linear_to_ijkl[m]
    (i2, j2, k2, l2) = linear_to_ijkl[n]
    x = (factorial(N) / (factorial(i1) * factorial(j1) * factorial(k1) * factorial(l1)))
    y = (factorial(N) / (factorial(i2) * factorial(j2) * factorial(k2) * factorial(l2)))
    z = factorial(i1 + i2) * factorial(j1 + j2) * factorial(k1 + k2) * factorial(l1 + l2)/factorial(big(2N  + 3))
    return x * y * z * factorial(big(2N + 3)) * (factorial(i2) * factorial(j2) * factorial(k2) * factorial(l2)) * (factorial(i1) * factorial(j1) * factorial(k1) * factorial(l1))
end

struct Bernstein2Mass <: AbstractMatrix{Int64} 
    N::Int
end

function Base.size(M::Bernstein2Mass)
    N = M.N
    Np = div((N + 1) * (N + 2), 2)
    return (Np, Np)
end

function Base.getindex(M::Bernstein2Mass, m, n)
    N = M.N
    (i1, j1, k1) = BernsteinBasis.bernstein_2d_scalar_multiindex_lookup(N)[1][m]
    (i2, j2, k2) = BernsteinBasis.bernstein_2d_scalar_multiindex_lookup(N)[1][n]
    x = (factorial(N) / (factorial(i1) * factorial(j1) * factorial(k1)))
    y = (factorial(N) / (factorial(i2) * factorial(j2) * factorial(k2)))
    z = factorial(i1 + i2) * factorial(j1 + j2) * factorial(k1 + k2)/factorial(big(2N  + 2))
    return x * y * z * factorial(big(2N + 2))
end



Bernstein2Mass(2)