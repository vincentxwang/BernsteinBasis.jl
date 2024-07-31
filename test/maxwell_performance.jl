# Generates a graph comparing multithreaded Bernstein to nodal. Takes a while to run to N = 10.

using OrdinaryDiffEq
using StartUpDG
using Plots
using LinearAlgebra
using SparseArrays
using StaticArrays
using BernsteinBasis
using BenchmarkTools

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
    (; rd, md, Dr, Ds, Dt, LIFT) = params
    
    (; uM, interface_flux, duM, dfxdr, dfxds, dfxdt, dfydr, dfyds, dfydt, dfzdr, dfzds, dfzdt, fxu, fyu, fzu) = params.cache
    
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

        mul!(view(du, :, e), LIFT, view(interface_flux, :, e))

        for i in axes(du, 1)
            du[i, e] += md.rxJ[1, e] * dfxdr[i, e] + md.sxJ[1, e] * dfxds[i, e] + md.txJ[1, e] * dfxdt[i, e] + 
            md.ryJ[1, e] * dfydr[i, e] + md.syJ[1, e] * dfyds[i, e] + md.tyJ[1, e] * dfydt[i, e] + 
            md.rzJ[1, e] * dfzdr[i, e] + md.szJ[1, e] * dfzds[i, e] + md.tzJ[1, e] * dfzdt[i, e]
        end
    end

    # Note md.J is constant matrix.
    @. du = du / md.J[1,1]
end

function rhs_matmat!(du, u, params, t)
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

    mul!(dfxdr, Dr, fxu)
    mul!(dfxds, Ds, fxu)
    mul!(dfxdt, Dt, fxu)
    mul!(dfydr, Dr, fyu)
    mul!(dfyds, Ds, fyu)
    mul!(dfydt, Dt, fyu)
    mul!(dfzdr, Dr, fzu)
    mul!(dfzds, Ds, fzu)
    mul!(dfzdt, Dt, fzu)

    @. du = md.rxJ * dfxdr + md.sxJ * dfxds + md.txJ * dfxdt + 
            md.ryJ * dfydr + md.syJ * dfyds + md.tyJ * dfydt + 
            md.rzJ * dfzdr + md.szJ * dfzds + md.tzJ * dfzdt
    
    mul!(du, LIFT, interface_flux, 1, 1)

     # Note md.J is constant matrix.
    @. du = du / md.J[1, 1]
end

function get_bernstein_vandermonde(N)
    rd = RefElemData(Tet(), N)

    (; r, s, Fmask) = rd
    Fmask = reshape(Fmask, :, 4)
    rf, sf = rd.r[Fmask[:,1]], rd.t[Fmask[:,1]]

    rd = RefElemData(Tet(), N; quad_rule_face = (rf, sf, ones(length(rf))))

    (; r, s, t) = rd
    vande, _ = bernstein_basis(Tet(), N, r, s, t)
    return vande
end

function get_nodal_lift(N)
    rd = RefElemData(Tet(), N)
    rtri, stri = nodes(Tri(), N)
    rfq, sfq, wfq = quad_nodes(Tri(), rd.N)
    Vq_face = vandermonde(Tri(), rd.N, rfq, sfq) / vandermonde(Tri(), rd.N, rtri, stri)

    nodal_LIFT = rd.LIFT * kron(I(4), Vq_face)

    return nodal_LIFT
end

function naive_mul!(C, A, B)
    n, m = size(A)
    p = size(B, 2)

    C .= 0
    
    @inbounds for i in 1:n
        for j in 1:p
            for k in 1:m
                C[i,j] += A[i,k] * B[k,j]
            end
        end
    end
    return C
end

function naive_nodal_rhs_matmul!(du, u, params, t)
    (; rd, md, Dr, Ds, Dt, LIFT) = params
    (; uM, interface_flux, dudr, duds, dudt) = params.cache
    
    uM .= view(u, rd.Fmask, :)
    
    for e in axes(uM, 2)
        for i in axes(uM, 1)
            interface_flux[i, e] = 0.5 * (uM[md.mapP[i,e]] - uM[i,e]) * md.nxJ[i,e] - 
                                   0.5 * (uM[md.mapP[i,e]] - uM[i,e]) * md.Jf[i,e]
        end
    end

    # u(x,y,z) = u(x(r,s,t), y(r,s,t), z(r,s,t)) 
    # --> du/dx = du/dr * dr/dx + du/ds * ds/dx + du/dt * dt/dz
    naive_mul!(dudr, Dr, u) 
    naive_mul!(duds, Ds, u) 
    naive_mul!(dudt, Dt, u) 

    du .= 0
    naive_mul!(du, LIFT, interface_flux)

    @. du += md.rxJ * dudr + md.sxJ * duds + md.txJ * dudt
    @. du = -du ./ md.J
end


function make_rhs_plot(K)
    BenchmarkTools.DEFAULT_PARAMETERS.samples = 15

    ratio_times = Float64[]

    for N in 1:K
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

        time1 = @benchmark rhs_matvec!($(similar(u0)), $(u0), $(params), 0)

        cache = (; uM = u0[rd.Fmask, :], interface_flux = u0[rd.Fmask, :], 
            duM = u0[rd.Fmask, :],
            dfxdr = similar(u0), dfxds = similar(u0), dfxdt = similar(u0),
            dfydr = similar(u0), dfyds = similar(u0), dfydt = similar(u0),
            dfzdr = similar(u0), dfzds = similar(u0), dfzdt = similar(u0),
            fxu = similar(u0), fyu = similar(u0), fzu = similar(u0))

        (; Dr, Ds, Dt) = rd
        LIFT = get_nodal_lift(N)

        params = (; rd, md, Dr, Ds, Dt, LIFT, cache)

        time2 = @benchmark rhs_matmat!($(similar(u0)), $(u0), $(params), 0)

        push!(ratio_times, minimum(time2).time/minimum(time1).time)
    end

    plot(bar(1:K, ratio_times), 
        legend = false, 
        title = "Speedup of Bernstein over nodal (3D Maxwell DG), K = 2, min times over $(BenchmarkTools.DEFAULT_PARAMETERS.samples) samples",
        yaxis = ("Time (Nodal) / Time (Not-MT Bernstein)"),
        xaxis = ("Degree N"),
        titlefont = font(10),
        xticks = 1:K
        )
end 

make_rhs_plot(15)


