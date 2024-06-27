# Lift coefficients
function l_j(N)
    return ntuple(j -> (j <= N) ? ((isodd(j) ? -1.0 : 1.0) * binomial(N, j) / (1.0 + j)) : 0.0, 20) 
end

# TODO: Procompiling L0_table takes a very long time...
const L0_table = tuple([SparseMatrixCSR(transpose((N + 1)^2/2 * sparse(transpose(ElevationMatrix{N+1}()) * ElevationMatrix{N+1}()))) for N in 1:9]...)
# const L0_table = tuple([(N + 1)^2/2 * sparse(transpose(ElevationMatrix{N+1}()) * ElevationMatrix{N+1}()) for N in 1:9]...)
const l_j_table = tuple([SVector(l_j(N)) for N in 1:20]...)
const tri_offset_table = tuple([SVector(tri_offsets(N)) for N in 1:20]...)

tri_offset_table

l_j_table[7]
Matrix(L0_table[7])
using Plots
spy(L0_table[6], ms = 2)