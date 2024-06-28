"""
    BernsteinLift

Lift matrix for a single face on a standard tetrahedron. Supports `mul!` operations, but
should optimally be used for matrix-vector multiplications rather than matrix-matrix
multiplications.

- Chan, Jesse and Tim Warburton (2017)
  GPU-accelerated Bernstein-Bezier discontinuous Galerkin methods for wave problems
  [DOI: 10.48550/arXiv.1512.06025](https://doi.org/10.48550/arXiv.1512.06025)
"""
struct BernsteinLift 
    N::Int
    # Preallocated vectors for intermediate computations
    E::Vector{Float64}
    F::Vector{Float64}
    G1::Vector{Float64}
    G2::Vector{Float64}
    G3::Vector{Float64}
end

function BernsteinLift(N)
    Np1 = div((N+1) * (N+2), 2)
    # F will store this many entries because we multiply by the elevation matrix.
    Np2 = div((N+2) * (N+3), 2)
    Np3 = div((N+1) * (N+2) * (N+3), 6)
    BernsteinLift(N, zeros(Np1), zeros(Np2), zeros(Np3), zeros(Np3), zeros(Np3))
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

"""
    fast_lift_multiply!(out, N, L0, x, offset, l_j, E)

Multiplies `x` by the "nice" lift matrix face (``rs``-plane) of the lift matrix.

# Arguments
``x::AbstractVector``: Input vector

- Chan, Jesse and Tim Warburton (2017)
  GPU-accelerated Bernstein-Bezier discontinuous Galerkin methods for wave problems
  [DOI: 10.48550/arXiv.1512.06025](https://doi.org/10.48550/arXiv.1512.06025)
"""
function face_lift_multiply!(out, N, x, E, F)
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

function face_lift_multiply!(out::AbstractVector{T}, L::BernsteinLift, x::AbstractVector{T}) where T<:Real
    face_lift_multiply!(out, L.N, x, L.E, L.F)
end

function face_lift_multiply!(out::AbstractMatrix{T}, D::BernsteinLift, x::AbstractMatrix{T}) where T<:Real
    @simd for n in axes(x,2)
        @inbounds face_lift_multiply!(view(out,:,n), D, view(x,:,n))
    end
    return out
end

"""
    LinearAlgebra.mul!(out::AbstractVector, L::BernsteinLift, x::AbstractVector)

Multiplies `x` by the Bernstein lift matrix `L`, as defined in documentation.

# Arguments
``out::AbstractVector``: Length ``\frac{(N+1)(N+2)(N+3)}{6}`` vector
``x::AbstractVector``: Length ``2 \\cdot \frac{(N+1)(N+2)}{2}`` vector
"""
function LinearAlgebra.mul!(out::AbstractVector, L::BernsteinLift, x::AbstractVector)
    N = L.N
    tri_offset = tri_offset_table[N]
    tet_offset = tet_offset_table[N]

    Np = div((N + 1) * (N + 2), 2)
    o1 = Np
    o2 = 2 * Np
    o3 = 3 * Np
    o4 = 4 * Np

    face_lift_multiply!(out, L, @view x[o3+1:o4])
    face_lift_multiply!(L.G1, L, @view x[1:o1])
    face_lift_multiply!(L.G2, L, @view x[o1+1:o2])
    face_lift_multiply!(L.G3, L, @view x[o2+1:o3])

    row = 1
    @inbounds for k in 0:N
        for j in 0:N-k
            for i in 0:N-k-j
                l = N-i-j-k
                out[row] += L.G1[ijk_to_linear(i,k,j, tri_offset, tet_offset)] 
                out[row] += L.G2[ijk_to_linear(j,k,l, tri_offset, tet_offset)] 
                out[row] += L.G3[ijk_to_linear(l,j,i, tri_offset, tet_offset)] 
                row += 1
            end
        end
    end
    return out
end

function LinearAlgebra.mul!(out::AbstractMatrix{T}, D::BernsteinLift, x::AbstractMatrix{T}) where T<:Real
    @simd for n in axes(x,2)
        @inbounds LinearAlgebra.mul!(view(out,:,n), D, view(x,:,n))
    end
    return out
end
