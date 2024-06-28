using BenchmarkTools
using BernsteinBasis
using LinearAlgebra

N = 7
Np2 = div((N + 1) * (N + 2), 2)
Np3 = div((N + 1) * (N + 2) * (N + 3), 6)

println("Lift matrix-vector multiplication for N = ", N)
@btime mul!($(zeros(Np3)), $(BernsteinLift(N)), $(rand(Float64, Np2)))
@btime mul!($(zeros(Np3)), $(rand(Float64, Np3, Np2)), $(rand(Float64, Np2)))

println("Lift matrix-matrix multiplication for N = ", N)
@btime mul!($(zeros(Np3, Np3)), $(BernsteinLift(N)), $(rand(Float64, Np2, Np3)))
@btime mul!($(zeros(Np3, Np3)), $(rand(Float64, Np3, Np2)), $(rand(Float64, Np2, Np3))) 

println("Derivative matrix-matrix multiplication for N = ", N)
@btime mul!($(zeros(Np3, Np3)), $(BernsteinDerivativeMatrix_3D_r(N)), $(rand(Float64, Np3, Np3))) 
@btime mul!($(zeros(Np3, Np3)), $(Matrix(BernsteinDerivativeMatrix_3D_r(N))), $(rand(Float64, Np3, Np3))) 

println("Derivative matrix-vector multiplication for N = ", N)
@btime mul!($(zeros(Np3)), $(BernsteinDerivativeMatrix_3D_r(N)), $(rand(Float64, Np3))) 
@btime mul!($(zeros(Np3)), $(Matrix(BernsteinDerivativeMatrix_3D_r(N))), $(rand(Float64, Np3))) 

# Naive matrix multiplication for benchmarking
function naive_mul!(C,A,B)
    n,m = size(A)
    @inbounds for i in 1:n
        for j in 1:m 
            for k in 1:m
                C[i,j] += A[i,k]*B[k,j]
            end
        end
    end
end

using StatsPlots
function make_lift_plot(K)

    bernstein_times = Float64[]
    opt_dense_times = Float64[]
    unopt_dense_times = Float64[]

    for N in 1:K
        Np2 = div((N + 1) * (N + 2), 2)
        Np3 = div((N + 1) * (N + 2) * (N + 3), 6)

        time1 = @benchmark mul!($(zeros(Float64, Np3, Np3)), $(BernsteinLift(N)), $(rand(Float64, Np2, Np3))) seconds=1
        push!(bernstein_times, median(time1).time) 

        time2 = @benchmark mul!($(zeros(Float64, Np3, Np3)), $(rand(Float64, Np3, Np2)), $(rand(Float64, Np2, Np3))) seconds=1
        push!(opt_dense_times, median(time2).time) 

        time3 = @benchmark naive_mul!($(zeros(Float64, Np3, Np3)), $(rand(Float64, Np3, Np2)), $(rand(Float64, Np2, Np3))) seconds=1
        push!(unopt_dense_times, median(time3).time) 
    end

    times = vcat(bernstein_times, opt_dense_times, unopt_dense_times)
    
    ctg = repeat(["Bernstein", "Dense opt", "Dense unopt"], inner = K)
    nam = repeat(string.(1:K), outer = 3)

    groupedbar(nam, times, group = ctg, xlabel = "N", ylabel = "Time",
            title = "Lift matrix-matrix multiplication, median times", bar_width = 0.67,
            lw = 0, framestyle = :box)
end

function make_der_plot(K)

    bernstein_times = Float64[]
    opt_dense_times = Float64[]
    # unopt_dense_times = Float64[]

    for N in 1:K
        Np2 = div((N + 1) * (N + 2), 2)
        Np3 = div((N + 1) * (N + 2) * (N + 3), 6)

        time1 = @benchmark mul!($(zeros(Float64, Np3, Np3)), $(BernsteinDerivativeMatrix_3D_r(N)), $(rand(Float64, Np3, Np3))) seconds=1
        push!(bernstein_times, median(time1).time) 

        time2 = @benchmark mul!($(zeros(Float64, Np3, Np3)), $(Matrix(BernsteinDerivativeMatrix_3D_r(N))), $(rand(Float64, Np3, Np3))) seconds=1
        push!(opt_dense_times, median(time2).time) 

        # time3 = @benchmark naive_mul!($(zeros(Float64, Np3, Np3)), $(Matrix(BernsteinDerivativeMatrix_3D_r(N))), $(rand(Float64, Np3, Np3))) seconds=1
        # push!(unopt_dense_times, median(time3).time) 
    end

    times = vcat(bernstein_times, opt_dense_times)
    
    ctg = repeat(["Bernstein", "Dense opt"], inner = K)
    nam = repeat(string.(1:K), outer = 2)

    groupedbar(nam, times, group = ctg, xlabel = "N", ylabel = "Time",
            title = "Derivative matrix-matrix multiplication, median times", bar_width = 0.67,
            lw = 0, framestyle = :box)
end

make_lift_plot(5)
make_lift_plot(9)
# make_der_plot(9)

function run_many_times()
    A = zeros(Np3, Np3)
    B = BernsteinLift(N)
    C = rand(Float64, Np2, Np3)
    for _ in 1:500000
        mul!(A, B, C)
    end
end

using Profile
@profile run_many_times()

using ProfileView
VSCodeServer.@profview run_many_times()


