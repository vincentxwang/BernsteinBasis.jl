# Generates a graph comparing multithreaded Bernstein to nodal. Takes a while to run to N = 10.
# Run using Julia v1.10.

using OrdinaryDiffEq
using StartUpDG
using StatsPlots
using LinearAlgebra
using SparseArrays
using StaticArrays
using BernsteinBasis
using TimerOutputs
using CategoricalArrays

BLAS.set_num_threads(1)


function fx(u)
    Hx, Hy, Hz, Ex, Ey, Ez = u
    return SVector(0, Ez, -Ey, 0, -Hz, Hy)
end

function fy(u)
    Hx, Hy, Hz, Ex, Ey, Ez = u
    return SVector(-Ez, 0, Ex, Hz, 0, -Hx)
end

function fz(u)
    Hx, Hy, Hz, Ex, Ey, Ez = u
    return SVector(Ey, -Ex, 0, -Hy, Hx, 0)
end

function rhs_matvec!(du, u, params, t)
    @timeit "total" begin 
        (; rd, md, Dr, Ds, Dt, LIFT) = params
        
        (; uM, interface_flux, dfxdr, dfxds, dfxdt, dfydr, dfyds, dfydt, dfzdr, dfzds, dfzdt, fxu, fyu, fzu) = params.cache
        
        uM .= view(u, rd.Fmask, :)

        @inbounds for e in axes(uM, 2)
            for i in axes(uM, 1)
                duM = uM[md.mapP[i,e]] - uM[i,e]
                ndotdH =    md.nxJ[i,e] * duM[1] + 
                            md.nyJ[i,e] * duM[2] +
                            md.nzJ[i,e] * duM[3]
                ndotdE =    md.nxJ[i,e] * duM[4] + 
                            md.nyJ[i,e] * duM[5] +
                            md.nzJ[i,e] * duM[6]
                interface_flux[i, e] = 0.5 * fx(duM) * md.nxJ[i,e] +
                                    0.5 * fy(duM) * md.nyJ[i,e] +
                                    0.5 * fz(duM) * md.nzJ[i,e] + 
                                    0.5 * (duM) -
                                    0.5 * SVector(
                                        ndotdH * md.nxJ[i,e],
                                        ndotdH * md.nyJ[i,e],
                                        ndotdH * md.nzJ[i,e],
                                        ndotdE * md.nxJ[i,e],
                                        ndotdE * md.nyJ[i,e],
                                        ndotdE * md.nzJ[i,e],
                                    )
            end
        end

        @timeit "volume kernel" begin  
            @inbounds for e in axes(du, 2)
                fxu .= fx.(view(u, :, e))
                fyu .= fy.(view(u, :, e))
                fzu .= fz.(view(u, :, e))

                mul!(view(dfxdr, :, e), Dr, fxu)
                mul!(view(dfxds, :, e), Ds, fxu)
                mul!(view(dfxdt, :, e), Dt, fxu)
                mul!(view(dfydr, :, e), Dr, fyu)
                mul!(view(dfyds, :, e), Ds, fyu)
                mul!(view(dfydt, :, e), Dt, fyu)
                mul!(view(dfzdr, :, e), Dr, fzu)
                mul!(view(dfzds, :, e), Ds, fzu)
                mul!(view(dfzdt, :, e), Dt, fzu)
            end
        end

        @timeit "surface kernel" begin
            @inbounds for e in axes(du, 2)
                mul!(view(du, :, e), LIFT, view(interface_flux, :, e))   
            end            
        end

        @timeit "adding" begin
            @inbounds for e in axes(du, 2)
                @inbounds for i in axes(du, 1)
                    du[i, e] += md.rxJ[1, e] * dfxdr[i, e] + md.sxJ[1, e] * dfxds[i, e] + md.txJ[1, e] * dfxdt[i, e] + 
                    md.ryJ[1, e] * dfydr[i, e] + md.syJ[1, e] * dfyds[i, e] + md.tyJ[1, e] * dfydt[i, e] + 
                    md.rzJ[1, e] * dfzdr[i, e] + md.szJ[1, e] * dfzds[i, e] + md.tzJ[1, e] * dfzdt[i, e]
                end
            end
        end

        # Note md.J is constant matrix.
        @. du = du / md.J[1,1]
    end
end

function rhs_matmat_mul!(du, u, params, t)
    @timeit "total" begin
        (; rd, md, Dr, Ds, Dt, LIFT) = params
    
        (; uM, interface_flux, dfxdr, dfxds, dfxdt, dfydr, dfyds, dfydt, dfzdr, dfzds, dfzdt, fxu, fyu, fzu) = params.cache
        
        uM .= view(u, rd.Fmask, :)

        @inbounds for e in axes(uM, 2)
            for i in axes(uM, 1)
                duM = uM[md.mapP[i,e]] - uM[i,e]
                ndotdH =    md.nxJ[i,e] * duM[1] + 
                            md.nyJ[i,e] * duM[2] +
                            md.nzJ[i,e] * duM[3]
                ndotdE =    md.nxJ[i,e] * duM[4] + 
                            md.nyJ[i,e] * duM[5] +
                            md.nzJ[i,e] * duM[6]
                interface_flux[i, e] = 0.5 * fx(duM) * md.nxJ[i,e] +
                                    0.5 * fy(duM) * md.nyJ[i,e] +
                                    0.5 * fz(duM) * md.nzJ[i,e] + 
                                    0.5 * (duM) -
                                    0.5 * SVector(
                                        ndotdH * md.nxJ[i,e],
                                        ndotdH * md.nyJ[i,e],
                                        ndotdH * md.nzJ[i,e],
                                        ndotdE * md.nxJ[i,e],
                                        ndotdE * md.nyJ[i,e],
                                        ndotdE * md.nzJ[i,e],
                                    )
            end
        end
        
        fxu .= fx.(u)
        fyu .= fy.(u)
        fzu .= fz.(u)

        @timeit "volume kernel" begin
            mul!(dfxdr, Dr, fxu)
            mul!(dfxds, Ds, fxu)
            mul!(dfxdt, Dt, fxu)
            mul!(dfydr, Dr, fyu)
            mul!(dfyds, Ds, fyu)
            mul!(dfydt, Dt, fyu)
            mul!(dfzdr, Dr, fzu)
            mul!(dfzds, Ds, fzu)
            mul!(dfzdt, Dt, fzu)
        end

        @timeit "adding" @. du = md.rxJ * dfxdr + md.sxJ * dfxds + md.txJ * dfxdt + 
                md.ryJ * dfydr + md.syJ * dfyds + md.tyJ * dfydt + 
                md.rzJ * dfzdr + md.szJ * dfzds + md.tzJ * dfzdt
        
        @timeit "surface kernel" mul!(du, LIFT, interface_flux, 1, -1)

        @. du = du ./ md.J[1, 1]
    end
end

function rhs_matmat_mymul!(du, u, params, t)
    @timeit "total" begin
        (; rd, md, Dr, Ds, Dt, LIFT) = params
    
        (; uM, interface_flux, dfxdr, dfxds, dfxdt, dfydr, dfyds, dfydt, dfzdr, dfzds, dfzdt, fxu, fyu, fzu) = params.cache
        
        uM .= view(u, rd.Fmask, :)

        @inbounds for e in axes(uM, 2)
            for i in axes(uM, 1)
                duM = uM[md.mapP[i,e]] - uM[i,e]
                ndotdH =    md.nxJ[i,e] * duM[1] + 
                            md.nyJ[i,e] * duM[2] +
                            md.nzJ[i,e] * duM[3]
                ndotdE =    md.nxJ[i,e] * duM[4] + 
                            md.nyJ[i,e] * duM[5] +
                            md.nzJ[i,e] * duM[6]
                interface_flux[i, e] = 0.5 * fx(duM) * md.nxJ[i,e] +
                                    0.5 * fy(duM) * md.nyJ[i,e] +
                                    0.5 * fz(duM) * md.nzJ[i,e] + 
                                    0.5 * (duM) -
                                    0.5 * SVector(
                                        ndotdH * md.nxJ[i,e],
                                        ndotdH * md.nyJ[i,e],
                                        ndotdH * md.nzJ[i,e],
                                        ndotdE * md.nxJ[i,e],
                                        ndotdE * md.nyJ[i,e],
                                        ndotdE * md.nzJ[i,e],
                                    )
            end
        end
        
        fxu .= fx.(u)
        fyu .= fy.(u)
        fzu .= fz.(u)

        @timeit "volume kernel" begin
            mymul!(dfxdr, Dr, fxu)
            mymul!(dfxds, Ds, fxu)
            mymul!(dfxdt, Dt, fxu)
            mymul!(dfydr, Dr, fyu)
            mymul!(dfyds, Ds, fyu)
            mymul!(dfydt, Dt, fyu)
            mymul!(dfzdr, Dr, fzu)
            mymul!(dfzds, Ds, fzu)
            mymul!(dfzdt, Dt, fzu)
        end

        @timeit "adding" @. du = md.rxJ * dfxdr + md.sxJ * dfxds + md.txJ * dfxdt + 
                md.ryJ * dfydr + md.syJ * dfyds + md.tyJ * dfydt + 
                md.rzJ * dfzdr + md.szJ * dfzds + md.tzJ * dfzdt
        
        @timeit "surface kernel" mymul!(du, LIFT, interface_flux, 1, -1)

        @. du = du ./ md.J[1, 1]
    end
end


function get_nodal_lift(N)
    rd = RefElemData(Tet(), N)
    rtri, stri = nodes(Tri(), N)
    rfq, sfq, wfq = quad_nodes(Tri(), rd.N)
    Vq_face = vandermonde(Tri(), rd.N, rfq, sfq) / vandermonde(Tri(), rd.N, rtri, stri)

    nodal_LIFT = rd.LIFT * kron(I(4), Vq_face)

    return nodal_LIFT
end

# Included custom matrix multiplication because mul! is slow in Julia 
# v1.10 for small matrices (dispatch issues...?)
function mymul!(C, A, B)
    @inbounds for j = axes(C, 2), i = axes(C, 1)
        C[i, j] = sum(A[i, k] * B[k, j] for k=axes(A, 2))
    end
end

function mymul!(C, A, B, a, b)
    @inbounds for j = axes(C, 2), i = axes(C, 1)
        C[i, j] = a * sum(A[i, k] * B[k, j] for k=axes(A, 2)) + b * C[i,j]
    end
end

function extract_times(to)
    return [
        TimerOutputs.time(to["total"]["volume kernel"]),
        TimerOutputs.time(to["total"]["surface kernel"]),
        TimerOutputs.time(to["total"]),
    ]
end

# See https://github.com/JuliaPlots/StatsPlots.jl/issues/437
function prepare_groupedbar_inputs!(names::Vector, data_matrix::Matrix,  group::Vector)
    
    # Redefine unique for `CategoricalArray` types to return a categorical array, rather than a regular vector/array. 
    @eval function Base.unique(ctg::CategoricalArray) # can be run in REPL instead
        l = levels(ctg)
        newctg = CategoricalArray(l)
        levels!(newctg, l)
    end

    data_matrix = data_matrix'
    
    @assert size(data_matrix)[1] % length(group) == 0 "The number of rows in the data matrix must be a multiple of the number of data categories."
    @assert size(data_matrix)[2] % length(names) == 0 "The number of column in the data matrix must be a multiple of the number of groups of bars."

    plot_names = repeat(names, outer = size(data_matrix)[1])
    plot_groups = repeat(group, inner = size(data_matrix)[2])

    plot_names = categorical(plot_names; levels = names)
    plot_groups = categorical(plot_groups; levels = group)

    return plot_names, data_matrix, plot_groups
end


function get_data(K, samples)

    ratio_times = Matrix{Float64}(undef, 0, 3)

    use_mul_only = false

    for N in 1:K
        time1 = TimerOutput()
        time2 = TimerOutput()
        time3 = TimerOutput()

        rd = RefElemData(Tet(), N)

        # Create interpolation matrix from Fmask node ordering to quadrature node ordering
        (; r, s, Fmask) = rd
        Fmask = reshape(Fmask, :, 4)

        # recreate RefElemData with nodal points instead of a quadrature rule
        rf, sf = rd.r[Fmask[:,1]], rd.t[Fmask[:,1]]
        rd = RefElemData(Tet(), N; quad_rule_face = (rf, sf, ones(length(rf))))
        md = MeshData(uniform_mesh(rd.element_type, 2), rd;               
                    is_periodic=true)

        (; x, y, z) = md
        u0 = @. SVector{6, Float64}(0, 0, 0, 0, 0, sin(pi * x) * sin(pi * y))

        Dr = BernsteinDerivativeMatrix_3D_r(N)
        Ds = BernsteinDerivativeMatrix_3D_s(N)
        Dt = BernsteinDerivativeMatrix_3D_t(N)
        LIFT = BernsteinLift{SVector{6, Float64}}(N)

        cache = (; uM = u0[rd.Fmask, :], interface_flux = u0[rd.Fmask, :], 
            duM = u0[rd.Fmask, :],
            dfxdr = similar(u0), dfxds = similar(u0), dfxdt = similar(u0),
            dfydr = similar(u0), dfyds = similar(u0), dfydt = similar(u0),
            dfzdr = similar(u0), dfzds = similar(u0), dfzdt = similar(u0),
            fxu = similar(u0[:, 1]), fyu = similar(u0[:, 1]), fzu = similar(u0[:, 1]))

        # Combine parameters
        params = (; rd, md, Dr, Ds, Dt, LIFT, cache)

        for i in 1:samples
            reset_timer!()
            rhs_matvec!(similar(u0), u0, params, 0)

            if i == 1 || extract_times(TimerOutputs.get_defaulttimer())[3] <= time1[3]
                time1 = extract_times(TimerOutputs.get_defaulttimer())
            end
        end

        cache = (; uM = u0[rd.Fmask, :], interface_flux = u0[rd.Fmask, :], 
            duM = u0[rd.Fmask, :],
            dfxdr = similar(u0), dfxds = similar(u0), dfxdt = similar(u0),
            dfydr = similar(u0), dfyds = similar(u0), dfydt = similar(u0),
            dfzdr = similar(u0), dfzds = similar(u0), dfzdt = similar(u0),
            fxu = similar(u0), fyu = similar(u0), fzu = similar(u0))
        
        (; Dr, Ds, Dt) = rd
        LIFT = get_nodal_lift(N)

        params = (; rd, md, Dr, Ds, Dt, LIFT, cache)

        for i in 1:samples
            reset_timer!()
            rhs_matmat_mul!(similar(u0), u0, params, 0)

            if i == 1 || extract_times(TimerOutputs.get_defaulttimer())[3] <= time2[3]
                time2 = extract_times(TimerOutputs.get_defaulttimer())
            end
        end

        if (!use_mul_only) 
            for i in 1:samples
                reset_timer!()
                rhs_matmat_mymul!(similar(u0), u0, params, 0)
    
                if i == 1 || extract_times(TimerOutputs.get_defaulttimer())[3] <= time3[3]
                    time3 = extract_times(TimerOutputs.get_defaulttimer())
                end
            end

            if time2[3] < time3[3]
                use_mul_only = true
            else 
                time2 = time3
            end
        end

        ratio_times = vcat(ratio_times, (time2./time1)')
    end
    return ratio_times
end 

function make_plot(ratio_times)

    K = size(ratio_times, 1)
    names = Vector(1:K)
    data_matrix = ratio_times
    group = ["Volume Kernel", "Surface Kernel", "Total"]
    
    names, data_matrix, group = prepare_groupedbar_inputs!(names, data_matrix, group)

    p = plot(groupedbar(ratio_times, bar_position = :dodge, group = group,color = [
        :cornflowerblue :coral :olivedrab
        ]),
        fontfamily = "Computer Modern",
        grid = false,
        title = "Speedup of Bernstein over nodal (Maxwell system timestep)",
        yaxis = "Time (Nodal) / Time (Bernstein)",
        xaxis = L"\textrm{Degree} \ N",
        xtickfontsize = 12,
        ytickfontsize = 12,
        titlefontsize = 12,
        legendfontsize = 16,
        legend = :topleft,
        xticks = 1:K)
    
    ## Plots the baseline if K < 8
    if (K < 8)
        plot!(x->1, c=:black, ls = :dash, lw = 1.5, linealpha = 0.5, label = "Reference (no speedup)", legendfontsize = 15,)
    end
    Plots.display(p)
end

maxwell = get_data(15, 200)

make_plot(maxwell)

