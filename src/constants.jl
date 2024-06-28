# Lift coefficients
function l_j(N)
    return ntuple(j -> (j <= N) ? ((isodd(j) ? -1.0 : 1.0) * binomial(N, j) / (1.0 + j)) : 0.0, 20) 
end

const l_j_table = tuple([SVector(l_j(N)) for N in 1:20]...)
const tri_offset_table = tuple([SVector(tri_offsets(N)) for N in 1:20]...)
const tet_offset_table = tuple([SVector(tet_offsets(N)) for N in 1:20]...)