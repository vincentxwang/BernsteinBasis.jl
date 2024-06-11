"""
    BernsteinDerivativeMatrix_3D_r <: AbstractMatrix{Float64}

Derivative matrix with respect to the first Cartesian coordinate r in the 3D Bernstein basis.

# Fields
- `N::Int`: Order of Bernstein polynomials. Supports up to N = 20.
"""
struct BernsteinDerivativeMatrix_3D_r <: AbstractMatrix{Float64} 
    N::Int
    tri_offsets::NTuple{21, Int}
    tet_offsets::NTuple{21, Int}
    BernsteinDerivativeMatrix_3D_r(N) = new(N, tri_offsets(N), tet_offsets(N))
end

"""
    BernsteinDerivativeMatrix_3D_s <: AbstractMatrix{Float64}

Derivative matrix with respect to the second Cartesian coordinate s in the 3D Bernstein basis.

# Fields
- `N::Int`: Order of Bernstein polynomials. Supports up to N = 20.
"""
struct BernsteinDerivativeMatrix_3D_s <: AbstractMatrix{Float64}
    N::Int
    tri_offsets::NTuple{21, Int}
    tet_offsets::NTuple{21, Int}
    BernsteinDerivativeMatrix_3D_s(N) = new(N, tri_offsets(N), tet_offsets(N))
end

"""
    BernsteinDerivativeMatrix_3D_t <: AbstractMatrix{Float64}

Derivative matrix with respect to the third Cartesian coordinate t in the 3D Bernstein basis.

# Fields
- `N::Int`: Order of Bernstein polynomials. Supports up to N = 20.
"""
struct BernsteinDerivativeMatrix_3D_t <: AbstractMatrix{Float64}
    N::Int
    tri_offsets::NTuple{21, Int}
    tet_offsets::NTuple{21, Int}
    BernsteinDerivativeMatrix_3D_t(N) = new(N, tri_offsets(N), tet_offsets(N))
end

function Base.size(Dr::BernsteinDerivativeMatrix_3D_r)
    N = Dr.N
    Np = div((N + 1) * (N + 2) * (N + 3), 6)
    return (Np, Np)
end

Base.size(Ds::BernsteinDerivativeMatrix_3D_s) = size(BernsteinDerivativeMatrix_3D_r(Ds.N))
Base.size(Dt::BernsteinDerivativeMatrix_3D_t) = size(BernsteinDerivativeMatrix_3D_r(Dt.N))

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

"""
    get_coeff(i1, j1, k1, l1, i2, j2, k2, l2)

Returns the value of the `(i1, j1, k1, l1), (i2, j2, k2, l2)`-th entry of the 3D Bernstein derivative 
matrix with respect to `i` (first barycentric coordinate).
"""
function get_coeff(i1, j1, k1, l1, i2, j2, k2, l2)
    if (i1, j1, k1, l1) == (i2, j2, k2, l2) return i1
    elseif (i1 + 1, j1 - 1, k1, l1) == (i2, j2, k2, l2) return j1
    elseif (i1 + 1, j1, k1 - 1, l1) == (i2, j2, k2, l2) return k1
    elseif (i1 + 1, j1, k1, l1 - 1) == (i2, j2, k2, l2) return l1
    else return 0 end
end

function linear_to_ijkl_lookup(N)
    return [(i, j, k, N - i - j - k) for k in 0:N for j in 0:N - k for i in 0:N - j - k]
end

function Base.getindex(Dr::BernsteinDerivativeMatrix_3D_r, m, n)
    N = Dr.N
    linear_to_ijkl = linear_to_ijkl_lookup(N)
    (i1, j1, k1, l1) = linear_to_ijkl[m]
    (i2, j2, k2, l2) = linear_to_ijkl[n]
    # du/dr = 1/2 du/di - 1/2 du/dl
    return 0.5 * (get_coeff(i1, j1, k1, l1, i2, j2, k2, l2) - get_coeff(l1, j1, k1, i1, l2, j2, k2, i2))
end


function Base.getindex(Ds::BernsteinDerivativeMatrix_3D_s, m, n)
    N = Ds.N
    linear_to_ijkl = linear_to_ijkl_lookup(N)
    (i1, j1, k1, l1) = linear_to_ijkl[m]
    (i2, j2, k2, l2) = linear_to_ijkl[n]
    # du/ds = 1/2 du/dj - 1/2 du/dl
    return 0.5 * (get_coeff(j1, i1, k1, l1, j2, i2, k2, l2) - get_coeff(l1, j1, k1, i1, l2, j2, k2, i2))
end

function Base.getindex(Dt::BernsteinDerivativeMatrix_3D_t, m, n)
    N = Dt.N
    linear_to_ijkl = linear_to_ijkl_lookup(N)
    (i1, j1, k1, l1) = linear_to_ijkl[m]
    (i2, j2, k2, l2) = linear_to_ijkl[n]
    # du/dt = 1/2 du/dk - 1/2 du/dl
    return 0.5 * (get_coeff(k1, j1, i1, l1, k2, j2, i2, l2) - get_coeff(l1, j1, k1, i1, l2, j2, k2, i2))
end

function fast_Dr_multiply!(out, N, x, tri_offset, tet_offset)
    row = 1
    @inbounds for k in 0:N
        for j in 0:N - k
            for i in 0:N - j - k
                l = N - i - j - k
                val = 0.0
                x_row = x[row]

                x1 = x_row
                x4 = (i > 0) ? x[ijk_to_linear(i - 1, j, k, tri_offset, tet_offset)] : 0.0
                @fastmath val += i * (x1 - x4)

                if j > 0
                    x1 = x[ijk_to_linear(i + 1, j - 1, k, tri_offset, tet_offset)]
                    x4 = x[ijk_to_linear(i, j - 1, k, tri_offset, tet_offset)]
                    @fastmath val += j * (x1 - x4)
                end

                if k > 0
                    x1 = x[ijk_to_linear(i + 1, j, k - 1, tri_offset, tet_offset)]
                    x4 = x[ijk_to_linear(i, j, k - 1, tri_offset, tet_offset)]
                    @fastmath val += k * (x1 - x4)
                end

                x1 = (l > 0) ? x[ijk_to_linear(i + 1, j, k, tri_offset, tet_offset)] : 0.0
                x4 = x_row
                @fastmath val += l * (x1 - x4)

                @fastmath out[row] = 0.5 * val

                row += 1
            end
        end
    end
    return out
end

function LinearAlgebra.mul!(out, Dr::BernsteinDerivativeMatrix_3D_r, x)
    return fast_Dr_multiply!(out, Dr.N, x, Dr.tri_offsets, Dr.tet_offsets)
end

function fast_Ds_multiply!(out, N, x, tri_offset, tet_offset)
    row = 1
    @inbounds for k in 0:N
        for j in 0:N - k
            for i in 0:N - j - k
                l = N - i - j - k
                val = 0.0
                x_row = x[row]

                if i > 0
                    x2 = x[ijk_to_linear(i - 1, j + 1, k, tri_offset, tet_offset)]
                    x4 = x[ijk_to_linear(i - 1, j, k, tri_offset, tet_offset)]
                    @fastmath val += i * (x2 - x4)
                end

                x2 = x_row
                x4 = (j > 0) ? x[ijk_to_linear(i, j - 1, k, tri_offset, tet_offset)] : 0.0
                @fastmath val += j * (x2 - x4)

                if k > 0
                    x2 = x[ijk_to_linear(i, j + 1, k - 1, tri_offset, tet_offset)]
                    x4 = x[ijk_to_linear(i, j, k - 1, tri_offset, tet_offset)]
                    @fastmath val += k * (x2 - x4)
                end

                x2 = (l > 0) ? x[ijk_to_linear(i, j + 1, k, tri_offset, tet_offset)] : 0.0
                x4 = x_row
                @fastmath val += l * (x2 - x4)

                @fastmath out[row] = 0.5 * val

                row += 1
            end
        end
    end
    return out
end

function LinearAlgebra.mul!(out, Ds::BernsteinDerivativeMatrix_3D_s, x)
    return fast_Ds_multiply!(out, Ds.N, x, Ds.tri_offsets, Ds.tet_offsets)
end

function fast_Dt_multiply!(out, N, x, tri_offset, tet_offset)
    row = 1
    @inbounds for k in 0:N
        for j in 0:N - k
            for i in 0:N - j - k
                l = N - i - j - k
                val = 0.0
                x_row = x[row]

                if i > 0
                    x3 = x[ijk_to_linear(i - 1, j, k + 1, tri_offset, tet_offset)]
                    x4 = x[ijk_to_linear(i - 1, j, k, tri_offset, tet_offset)]
                    @fastmath val += i * (x3 - x4)
                end

                if j > 0
                    x3 = x[ijk_to_linear(i, j - 1, k + 1, tri_offset, tet_offset)]
                    x4 = x[ijk_to_linear(i, j - 1, k, tri_offset, tet_offset)]
                    @fastmath val += j * (x3 - x4)
                end

                x3 = x_row
                x4 = (k > 0) ? x[ijk_to_linear(i, j, k - 1, tri_offset, tet_offset)] : 0.0
                @fastmath val += k * (x3 - x4)

                x3 = (l > 0) ? x[ijk_to_linear(i, j, k + 1, tri_offset, tet_offset)] : 0.0
                x4 = x_row
                @fastmath val += l * (x3 - x4)

                @fastmath out[row] = 0.5 * val

                row += 1
            end
        end
    end
    return out
end

function LinearAlgebra.mul!(out, Dt::BernsteinDerivativeMatrix_3D_t, x)
    return fast_Dt_multiply!(out, Dt.N, x, Dt.tri_offsets, Dt.tet_offsets)
end