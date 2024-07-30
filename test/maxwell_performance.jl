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
    
    (; uM, interface_flux, dfxdr, dfxds, dfxdt, dfydr, dfyds, dfydt, dfzdr, dfzds, dfzdt) = params.cache
    
    uM .= view(u, rd.Fmask, :)

    @inbounds for e in axes(uM, 2)
        for i in axes(uM, 1)
            ndotdH =    md.nxJ[i,e] * (uM[md.mapP[i,e]][1] - uM[i,e][1]) + 
                        md.nyJ[i,e] * (uM[md.mapP[i,e]][2] - uM[i,e][2]) +
                        md.nzJ[i,e] * (uM[md.mapP[i,e]][3] - uM[i,e][3])
            ndotdE =    md.nxJ[i,e] * (uM[md.mapP[i,e]][4] - uM[i,e][4]) + 
                        md.nyJ[i,e] * (uM[md.mapP[i,e]][5] - uM[i,e][5]) +
                        md.nzJ[i,e] * (uM[md.mapP[i,e]][6] - uM[i,e][6])
            interface_flux[i, e] = 0.5 * fx(uM[md.mapP[i,e]] - uM[i,e]) * md.nxJ[i,e] +
                                0.5 * fy(uM[md.mapP[i,e]] - uM[i,e]) * md.nyJ[i,e] +
                                0.5 * fz(uM[md.mapP[i,e]] - uM[i,e]) * md.nzJ[i,e] + 
                                0.5 * (uM[md.mapP[i,e]] - uM[i,e]) -
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
        mul!(view(dfxdr, :, e), Dr, fx.(view(u, :, e)))
        mul!(view(dfxds, :, e), Ds, fx.(view(u, :, e)))
        mul!(view(dfxdt, :, e), Dt, fx.(view(u, :, e)))
        mul!(view(dfydr, :, e), Dr, fy.(view(u, :, e)))
        mul!(view(dfyds, :, e), Ds, fy.(view(u, :, e)))
        mul!(view(dfydt, :, e), Dt, fy.(view(u, :, e)))
        mul!(view(dfzdr, :, e), Dr, fz.(view(u, :, e)))
        mul!(view(dfzds, :, e), Ds, fz.(view(u, :, e)))
        mul!(view(dfzdt, :, e), Dt, fz.(view(u, :, e)))

        mul!(view(du, :, e), LIFT, view(interface_flux, :, e))

        du[:, e] += md.rxJ[1, e] * dfxdr[:, e] + md.sxJ[1, e] * dfxds[:, e] + md.txJ[1, e] * dfxdt[:, e] + 
                    md.ryJ[1, e] * dfydr[:, e] + md.syJ[1, e] * dfyds[:, e] + md.tyJ[1, e] * dfydt[:, e] + 
                    md.rzJ[1, e] * dfzdr[:, e] + md.szJ[1, e] * dfzds[:, e] + md.tzJ[1, e] * dfzdt[:, e]
        du[:, e] = du[:, e] / md.J[1, e]
    end
    return du
end

function rhs_matmat!(du, u, params, t)
    (; rd, md, Dr, Ds, Dt, LIFT) = params
    
    (; uM, interface_flux, dfxdr, dfxds, dfxdt, dfydr, dfyds, dfydt, dfzdr, dfzds, dfzdt) = params.cache
    
    uM .= view(u, rd.Fmask, :)

    @inbounds for e in axes(uM, 2)
        for i in axes(uM, 1)
            ndotdH =    md.nxJ[i,e] * (uM[md.mapP[i,e]][1] - uM[i,e][1]) + 
                        md.nyJ[i,e] * (uM[md.mapP[i,e]][2] - uM[i,e][2]) +
                        md.nzJ[i,e] * (uM[md.mapP[i,e]][3] - uM[i,e][3])
            ndotdE =    md.nxJ[i,e] * (uM[md.mapP[i,e]][4] - uM[i,e][4]) + 
                        md.nyJ[i,e] * (uM[md.mapP[i,e]][5] - uM[i,e][5]) +
                        md.nzJ[i,e] * (uM[md.mapP[i,e]][6] - uM[i,e][6])
            interface_flux[i, e] = 0.5 * fx(uM[md.mapP[i,e]] - uM[i,e]) * md.nxJ[i,e] +
                                0.5 * fy(uM[md.mapP[i,e]] - uM[i,e]) * md.nyJ[i,e] +
                                0.5 * fz(uM[md.mapP[i,e]] - uM[i,e]) * md.nzJ[i,e] + 
                                0.5 * (uM[md.mapP[i,e]] - uM[i,e]) -
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

    mul!(dfxdr, Dr, fx.(u))
    mul!(dfxds, Ds, fx.(u))
    mul!(dfxdt, Dt, fx.(u))
    mul!(dfydr, Dr, fy.(u))
    mul!(dfyds, Ds, fy.(u))
    mul!(dfydt, Dt, fy.(u))
    mul!(dfzdr, Dr, fz.(u))
    mul!(dfzds, Ds, fz.(u))
    mul!(dfzdt, Dt, fz.(u))

    @. du = md.rxJ * dfxdr + md.sxJ * dfxds + md.txJ * dfxdt + 
            md.ryJ * dfydr + md.syJ * dfyds + md.tyJ * dfydt + 
            md.rzJ * dfzdr + md.szJ * dfzds + md.tzJ * dfzdt
    
    mul!(du, LIFT, interface_flux, 1, 1)
    @. du = du ./ md.J
    return du
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
        modal_u0 = @. SVector{6, Float64}(0, 0, 0, 0, 0, sin(pi * x) * sin(pi * y))

        Dr = BernsteinDerivativeMatrix_3D_r(N)
        Ds = BernsteinDerivativeMatrix_3D_s(N)
        Dt = BernsteinDerivativeMatrix_3D_t(N)
        LIFT = BernsteinLift{SVector{6, Float64}}(N)

        cache = (; uM = modal_u0[rd.Fmask, :], interface_flux = modal_u0[rd.Fmask, :], 
           dfxdr = similar(modal_u0), dfxds = similar(modal_u0), dfxdt = similar(modal_u0),
           dfydr = similar(modal_u0), dfyds = similar(modal_u0), dfydt = similar(modal_u0),
           dfzdr = similar(modal_u0), dfzds = similar(modal_u0), dfzdt = similar(modal_u0),)

        # Combine parameters
        params = (; rd, md, Dr, Ds, Dt, LIFT, cache)

        time1 = @benchmark rhs_matvec!($(similar(modal_u0)), $(modal_u0), $(params), 0)

        (; Dr, Ds, Dt) = rd
        LIFT = get_nodal_lift(N)

        params = (; rd, md, Dr, Ds, Dt, LIFT, cache)

        time2 = @benchmark rhs_matmat!($(similar(modal_u0)), $(modal_u0), $(params), 0)

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


