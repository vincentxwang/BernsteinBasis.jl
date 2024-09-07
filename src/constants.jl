"""
    l_j(N)
Returns a tuple containing ``l_1, ..., l_N, ..., l_{20}`` as specified by the lift
matrix multiplication algorithm in

- Chan, Jesse and Tim Warburton (2017)
  GPU-accelerated Bernstein-Bezier discontinuous Galerkin methods for wave problems
  [DOI: 10.48550/arXiv.1512.06025](https://doi.org/10.48550/arXiv.1512.06025)
"""
function l_j(N)
    return ntuple(j -> (j <= N) ? ((isodd(j) ? -1.0 : 1.0) * binomial(N, j) / (1.0 + j)) : 0.0, 20) 
end

const l_j_table = tuple([SVector(l_j(N)) for N in 1:20]...)
const tri_offset_table = tuple([SVector(tri_offsets(N)) for N in 1:20]...)
const tet_offset_table = tuple([SVector(tet_offsets(N)) for N in 1:20]...)