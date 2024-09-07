"""
    cartesian_to_barycentric(elem::Union{Line, Tri, Tet}, coords)
    cartesian_to_barycentric(elem::Union{Line, Tri, Tet}, coords...)

Converts a `DIMS` by ``N_p`` matrix of cartesian coordinates into a matrix of barycentric coordinates.
Alternatively, also takes in `DIMS` number of vectors of coordinates.
"""
function cartesian_to_barycentric(::Line, coords::AbstractMatrix)
    bary = hcat([[(col[1] + 1) / 2,
                 (1 - col[1]) / 2, 
                  ] for col in eachcol(coords)]...)
    return (bary[i,:] for i in 1:2)    
end

function cartesian_to_barycentric(::Tri, coords::AbstractMatrix)
    bary = hcat([[(col[1] + 1) / 2,
                  (col[2] + 1) / 2, 
                 -(col[1] + col[2]) / 2, 
                  ] for col in eachcol(coords)]...)
    return (bary[i,:] for i in 1:3)    
end

function cartesian_to_barycentric(::Tet, coords::AbstractMatrix)
    bary = hcat([[(1 + col[1]) / 2, 
                  (1 + col[2]) / 2, 
                  (1 + col[3]) / 2,
                 -(1 + col[1] + col[2] + col[3]) / 2] for col in eachcol(coords)]...)
    return (bary[i,:] for i in 1:4)
end

cartesian_to_barycentric(elem, coords...) = 
    cartesian_to_barycentric(elem, vcat(permutedims.(coords)...))

"""
    bernstein_basis(::Line, N, r)
    bernstein_basis(elem::Tri, N, r, s)
    bernstein_basis(elem::Tet, N, r, s, t)

Wrapper for `bernstein_basis_from_barycentric` that takes in ``rst``-space coordinates, each in 
``[-1, 1]``, instead of barycentric coordinates.

Returns the ``N_p x N_p`` generalized Vandermonde matrix ``\\mathcal{V}``, where ``\\mathcal{V}_{mn}`` = 
the ``n``-th Bernstein basis evaluated at the ``m``-th point, followed by its derivative matrices.

Use `::Line` for 1D, `::Tri` for 2D, and `::Tet` for 3D. 

We order the Bernstein basis according to exponents in a "reverse-dictionary" order of the first DIM-1
coordinates. Does not work for ``N > 20`` because of `factorial()` limitations.

# Arguments
- `N::Int`: Bernstein basis degree
- `r,s,t::AbstractArray{T,1}`: ``N_p``-sized vectors of distinct 2D *barycentric* coordinates to be 
used as interpolatory points
"""
function bernstein_basis(elem::Line, N, r)
    bernstein_basis_from_barycentric(elem, N, cartesian_to_barycentric(elem, r)...)
end

bernstein_basis(elem::Tri, N, r, s) = 
    bernstein_basis_from_barycentric(elem, N, cartesian_to_barycentric(elem, r, s)...)

bernstein_basis(elem::Tet, N, r, s, t) = 
    bernstein_basis_from_barycentric(elem, N, cartesian_to_barycentric(elem, r, s, t)...)

"""
    bernstein_basis_from_barycentric(::Line, N, r, s)
    bernstein_basis_from_barycentric(::Tri, N, r, s, t)
    bernstein_basis_from_barycentric(::Tet, N, r, s, t, u)

Returns the ``N_p x N_p`` generalized Vandermonde matrix ``\\mathcal{V}``, where ``\\mathcal{V}_{mn}`` = 
the ``n``-th Bernstein basis evaluated at the ``m``-th point, followed by its derivative matrices.

Use `::Line` for 1D, `::Tri` for 2D, and `::Tet` for 3D. 
    
We order the Bernstein basis according to exponents in a "reverse-dictionary" order of the first DIM-1
coordinates. Does not work for ``N > 20`` because of `factorial()` limitations.

# Arguments
- `N::Int`: Bernstein basis degree
- `r,s,t::AbstractArray{T,1}`: ``N_p``-sized vectors of distinct 2D *barycentric* coordinates to be 
used as interpolatory points
"""
function bernstein_basis_from_barycentric(::Line, N, r, s)
    V =  hcat(@. [(factorial(N)/(factorial(i) * factorial(N - i))) * r^i * s^(N - i) for i in 0:N]...)
    return V, nothing
end

function bernstein_basis_from_barycentric(::Tri, N, r, s, t)
    V =  hcat(@. [(factorial(N)/(factorial(i) * factorial(j) * factorial(N - i - j))) * r^i * s^j * t^(N - i - j) for j in 0:N for i in 0:N-j]...)
    # V =  hcat(@. [(factorial(N)/(factorial(i) * factorial(j) * factorial(N - i - j))) * r^i * s^j * t^(N - i - j) for i in 0:N for j in 0:N-i]...)
    Vi = hcat([evaluate_bernstein_derivatives(Tri(), N, r, s, t, i, j, N - i - j)[1] for j in 0:N for i in 0:N-j]...)
    Vj = hcat([evaluate_bernstein_derivatives(Tri(), N, r, s, t, i, j, N - i - j)[2] for j in 0:N for i in 0:N-j]...)
    Vk = hcat([evaluate_bernstein_derivatives(Tri(), N, r, s, t, i, j, N - i - j)[3] for j in 0:N for i in 0:N-j]...)
    return V, Vi, Vj, Vk
end

function bernstein_basis_from_barycentric(::Tet, N, r, s, t, u)
    V = hcat(@. [(factorial(N)/(factorial(i) * factorial(j) * factorial(k) * factorial(N - i - j - k))) *
        r^i * s^j * t^k * u^(N - i - j - k) for k in 0:N for j in 0:N-k for i in 0:N-k-j]...)
    # V = hcat(@. [(factorial(N)/(factorial(i) * factorial(j) * factorial(k) * factorial(N - i - j - k))) *
    #     r^i * s^j * t^k * u^(N - i - j - k) for j in 0:N for k in 0:N-j for i in 0:N-j-k]...)
    Vi = hcat([evaluate_bernstein_derivatives(Tet(), N, r, s, t, u, i, j, k, N - i - j - k)[1] for k in 0:N for j in 0:N-k for i in 0:N-k-j]...)
    Vj = hcat([evaluate_bernstein_derivatives(Tet(), N, r, s, t, u, i, j, k, N - i - j - k)[2] for k in 0:N for j in 0:N-k for i in 0:N-k-j]...)
    Vk = hcat([evaluate_bernstein_derivatives(Tet(), N, r, s, t, u, i, j, k, N - i - j - k)[3] for k in 0:N for j in 0:N-k for i in 0:N-k-j]...)
    Vl = hcat([evaluate_bernstein_derivatives(Tet(), N, r, s, t, u, i, j, k, N - i - j - k)[4] for k in 0:N for j in 0:N-k for i in 0:N-k-j]...)
    return V, Vi, Vj, Vk, Vl
end
 
function evaluate_bernstein_derivatives(::Tet, N, r, s, t, u, i, j, k, l)
    if i + j + k + l != N
        throw(DomainError([i,j,k,l],"Barycentric coordinates do not sum to total degree."))
    end
    if i > 0
        vi = @. (factorial(N)/(factorial(i-1) * factorial(j) * factorial(k) * factorial(l))) * r^(i-1) * s^j * t^k * u^l
    else 
        # vector with all zeros
        vi = 0 .* r
    end
    if j > 0
        vj = @. (factorial(N)/(factorial(i) * factorial(j-1) * factorial(k) * factorial(l))) * r^i * s^(j-1) * t^k * u^l
    else 
        # vector with all zeros
        vj = 0 .* s
    end
    if k > 0
        vk = @. (factorial(N)/(factorial(i) * factorial(j) * factorial(k-1) * factorial(l))) * r^i * s^j * t^(k-1) * u^l
    else 
        # vector with all zeros
        vk = 0 .* t
    end
    if l > 0
        vl = @. (factorial(N)/(factorial(i) * factorial(j) * factorial(k) * factorial(l-1))) * r^i * s^j * t^k * u^(l-1)
    else 
        # vector with all zeros
        vl = 0 .* u
    end
    return vi, vj, vk, vl
end

function evaluate_bernstein_derivatives(::Tri, N, r, s, t, i, j, k)
    # Since we are working with a triangle, set u = 0 and l = 0
    u = zeros(length(r))
    l = 0

    vi, vj, vk, _ = evaluate_bernstein_derivatives(Tet(), N, r, s, t, u, i, j, k, l)
    return vi, vj, vk
end

"""
    get_bernstein_lift(N)

Returns an ``N``-degree 3D Bernstein lift matrix as a `Matrix` type, in contrast to the 
more optimized `BernsteinLift(N)`.

This function is derived from StartUpDG's `Tet()` struct.
"""
function get_bernstein_lift(N)
    rd = RefElemData(Tet(), N)
    # LIFT: quad points on faces -> polynomial in volume
    # ---> define Vq: polynomials on faces -> quad points on faces

    rf = rd.rf
    sf = rd.sf
    tf = rd.tf
    wf = rd.wf
    rf, sf, tf, wf = reshape.((rf, sf, tf, wf), :, 4) 

    Vq, _ = bernstein_basis(Tet(), N, rd.rstq...)
    MB = Vq' * diagm(rd.wq) * Vq
    Vf, _ = bernstein_basis(Tet(), N, rd.rstf...)
    VBf2, _ = bernstein_basis(Tri(), N, rf[:,1], tf[:,1])
    VBf3, _ = bernstein_basis(Tri(), N, sf[:,2], tf[:,2])
    VBf4, _ = bernstein_basis(Tri(), N, sf[:,3], tf[:,3])
    VBf1, _ = bernstein_basis(Tri(), N, rf[:,4], sf[:,4])
    MBf = Vf' * diagm(rd.wf) * blockdiag(sparse.((VBf1, VBf2, VBf3, VBf4))...)

    bernstein_LIFT = MB \ MBf # bernstein_to_nodal_volume \ (nodal_LIFT * bernstein_to_nodal_face)
    bernstein_LIFT = Matrix(droptol!(sparse(bernstein_LIFT), 3 * N^2 * length(bernstein_LIFT) * eps()))
    return bernstein_LIFT
end