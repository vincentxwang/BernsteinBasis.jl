using SparseArrays
using BenchmarkTools
using LinearAlgebra
using StaticArrays

const L0 = sprand(36, 36, 0.164)
const tri_offset_table = (
    (0, 2, 3, 3, 2, 0, -3, -7, -12, -18, -25, -33, -42, -52, -63, -75, -88, -102, -117, -133, -150), 
    (0, 3, 5, 6, 6, 5, 3, 0, -4, -9, -15, -22, -30, -39, -49, -60, -72, -85, -99, -114, -130), 
    (0, 4, 7, 9, 10, 10, 9, 7, 4, 0, -5, -11, -18, -26, -35, -45, -56, -68, -81, -95, -110), 
    (0, 5, 9, 12, 14, 15, 15, 14, 12, 9, 5, 0, -6, -13, -21, -30, -40, -51, -63, -76, -90), 
    (0, 6, 11, 15, 18, 20, 21, 21, 20, 18, 15, 11, 6, 0, -7, -15, -24, -34, -45, -57, -70), 
    (0, 7, 13, 18, 22, 25, 27, 28, 28, 27, 25, 22, 18, 13, 7, 0, -8, -17, -27, -38, -50), 
    (0, 8, 15, 21, 26, 30, 33, 35, 36, 36, 35, 33, 30, 26, 21, 15, 8, 0, -9, -19, -30), 
    (0, 9, 17, 24, 30, 35, 39, 42, 44, 45, 45, 44, 42, 39, 35, 30, 24, 17, 9, 0, -10), 
    (0, 10, 19, 27, 34, 40, 45, 49, 52, 54, 55, 55, 54, 52, 49, 45, 40, 34, 27, 19, 10), 
    (0, 11, 21, 30, 38, 45, 51, 56, 60, 63, 65, 66, 66, 65, 63, 60, 56, 51, 45, 38, 30)) 
const l_j_table = (-3.5, 7.0, -8.75, 7.0, -3.5, 1.0, -0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

function reduction_multiply!(out, N, x, offset)
    row = 1
    @inbounds @fastmath for j in 0:N-1
        for i in 0:N-1-j
            k = N-1-i-j

            # Below are derived from `ij_to_linear``
            linear_index1 = i+1 + offset[j+1] + 1
            linear_index2 = i + offset[j+2] + 1
            linear_index3 = i + offset[j+1] + 1

            val = muladd((i+1), x[linear_index1], 
            muladd((j+1), x[linear_index2],
            muladd((k+1), x[linear_index3], 0.0)))
            out[row] = val/N
            row += 1
        end
    end
    return out
end

function matrix_vector_multiply!(out, N, x, E)
    @inbounds @fastmath begin 
        LinearAlgebra.mul!(E, L0, x)
        index1 = div((N + 1) * (N + 2), 2)
        view(out,1:index1) .= E
        reduction_multiply!(E, N, E, tri_offset_table[N])
        for j in 1:N
            diff = div((N + 1 - j) * (N + 2 - j), 2)

            for i in 1:diff
                out[index1 + i] = l_j_table[j] * E[i]
            end

            if j < N
                index1 += diff
                reduction_multiply!(E, N-j, E, tri_offset_table[N-j])
            end
        end
    end
    return out
end

function matrix_matrix_multiply!(out, N, x, E)
    @simd for n in axes(x,2)
        @inbounds matrix_vector_multiply!(view(out,:,n), N, view(x,:,n), E)
    end
    return out
end

N = 7
Np2 = div((N + 1) * (N + 2), 2)
Np3 = div((N + 1) * (N + 2) * (N + 3), 6)
@benchmark matrix_vector_multiply!($(zeros(Np3)), $(N), $(rand(Float64, Np2)), $(zeros(Np2)))
@benchmark matrix_matrix_multiply!($(zeros(Np3, Np3)), $(N), $(rand(Float64, Np2, Np3)), $(zeros(Np2)))



