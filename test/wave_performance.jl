# Generates a graph comparing multithreaded Bernstein to nodal. Takes a while to run to N = 10.

using OrdinaryDiffEq
using StartUpDG
using Plots
using LinearAlgebra
using SparseArrays
using StaticArrays
using BernsteinBasis
using BenchmarkTools

function fu(q)
    p, u, v, w = q
    return SVector(u, p, 0, 0)
end

function fv(q)
    p, u, v, w = q
    return SVector(v, 0, p, 0)
end

function fw(q)
    p, u, v, w = q
    return SVector(w, 0, 0, p)
end

function rhs_matvec!(du, u, params, t)
    (; rd, md, Dr, Ds, Dt, LIFT) = params
    
    (; uM, interface_flux, dfudr, dfuds, dfudt, dfvdr, dfvds, dfvdt, dfwdr, dfwds, dfwdt) = params.cache
    
    uM .= view(u, rd.Fmask, :)

    @inbounds for e in axes(uM, 2)
        for i in axes(uM, 1)
            ndotdU =    md.nxJ[i,e] * (uM[md.mapP[i,e]] - uM[i,e])[2] + 
                        md.nyJ[i,e] * (uM[md.mapP[i,e]] - uM[i,e])[3] +
                        md.nzJ[i,e] * (uM[md.mapP[i,e]] - uM[i,e])[4]
            interface_flux[i, e] = 0.5 * SVector(
                (uM[md.mapP[i,e]] - uM[i,e])[1] - ndotdU,
                (ndotdU - (uM[md.mapP[i,e]] - uM[i,e])[1]) * md.nxJ[i,e],
                (ndotdU - (uM[md.mapP[i,e]] - uM[i,e])[1]) * md.nyJ[i,e],
                (ndotdU - (uM[md.mapP[i,e]] - uM[i,e])[1]) * md.nzJ[i,e],
            )
        end
    end

    @inbounds for e in axes(du, 2)
        mul!(view(dfudr, :, e), Dr, fu.(view(u, :, e)))
        mul!(view(dfuds, :, e), Ds, fu.(view(u, :, e)))
        mul!(view(dfudt, :, e), Dt, fu.(view(u, :, e)))
        mul!(view(dfvdr, :, e), Dr, fv.(view(u, :, e)))
        mul!(view(dfvds, :, e), Ds, fv.(view(u, :, e)))
        mul!(view(dfvdt, :, e), Dt, fv.(view(u, :, e)))
        mul!(view(dfwdr, :, e), Dr, fw.(view(u, :, e)))
        mul!(view(dfwds, :, e), Ds, fw.(view(u, :, e)))
        mul!(view(dfwdt, :, e), Dt, fw.(view(u, :, e)))

        mul!(view(du, :, e), LIFT, view(interface_flux, :, e))

        du[:, e] -= (md.rxJ[1, e] * dfudr[:, e] + md.sxJ[1, e] * dfuds[:, e] + md.txJ[1, e] * dfudt[:, e] + 
                    md.ryJ[1, e] * dfvdr[:, e] + md.syJ[1, e] * dfvds[:, e] + md.tyJ[1, e] * dfvdt[:, e] + 
                    md.rzJ[1, e] * dfwdr[:, e] + md.szJ[1, e] * dfwds[:, e] + md.tzJ[1, e] * dfwdt[:, e])
                    
        du[:, e] = du[:, e] / md.J[1, e]
    end
end

# Computes d(p, u, v, w)/dt as a function of u = (p, u, v, w). (note abusive notation).
function rhs_matmat!(du, u, params, t)
    (; rd, md, Dr, Ds, Dt, LIFT) = params
    
    (; uM, interface_flux, dfudr, dfuds, dfudt, dfvdr, dfvds, dfvdt, dfwdr, dfwds, dfwdt) = params.cache
    
    uM .= view(u, rd.Fmask, :)

    @inbounds for e in axes(uM, 2)
        for i in axes(uM, 1)
            ndotdU =    md.nxJ[i,e] * (uM[md.mapP[i,e]] - uM[i,e])[2] + 
                        md.nyJ[i,e] * (uM[md.mapP[i,e]] - uM[i,e])[3] +
                        md.nzJ[i,e] * (uM[md.mapP[i,e]] - uM[i,e])[4]
            interface_flux[i, e] = 0.5 * SVector(
                (uM[md.mapP[i,e]] - uM[i,e])[1] - ndotdU,
                (ndotdU - (uM[md.mapP[i,e]] - uM[i,e])[1]) * md.nxJ[i,e],
                (ndotdU - (uM[md.mapP[i,e]] - uM[i,e])[1]) * md.nyJ[i,e],
                (ndotdU - (uM[md.mapP[i,e]] - uM[i,e])[1]) * md.nzJ[i,e],
            )
        end
    end

    mul!(dfudr, Dr, fu.(u))
    mul!(dfuds, Ds, fu.(u))
    mul!(dfudt, Dt, fu.(u))
    mul!(dfvdr, Dr, fv.(u))
    mul!(dfvds, Ds, fv.(u))
    mul!(dfvdt, Dt, fv.(u))
    mul!(dfwdr, Dr, fw.(u))
    mul!(dfwds, Ds, fw.(u))
    mul!(dfwdt, Dt, fw.(u))

    @. du = md.rxJ * dfudr + md.sxJ * dfuds + md.txJ * dfudt + 
            md.ryJ * dfvdr + md.syJ * dfvds + md.tyJ * dfvdt + 
            md.rzJ * dfwdr + md.szJ * dfwds + md.tzJ * dfwdt
    
    mul!(du, LIFT, interface_flux, 1, -1)

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
        u0 = @. SVector{4, Float64}(
            cos(pi * x) * cos(pi * y) * cos(pi * z), 
            0, 
            0, 
            0,
            )

        Dr = BernsteinDerivativeMatrix_3D_r(N)
        Ds = BernsteinDerivativeMatrix_3D_s(N)
        Dt = BernsteinDerivativeMatrix_3D_t(N)
        LIFT = BernsteinLift{SVector{4, Float64}}(N)

        cache = (; uM = u0[rd.Fmask, :], interface_flux = u0[rd.Fmask, :], 
        dfudr = similar(u0), dfuds = similar(u0), dfudt = similar(u0),
        dfvdr = similar(u0), dfvds = similar(u0), dfvdt = similar(u0),
        dfwdr = similar(u0), dfwds = similar(u0), dfwdt = similar(u0),)

        # Combine parameters
        params = (; rd, md, Dr, Ds, Dt, LIFT, cache)

        time1 = @benchmark rhs_matvec!($(similar(u0)), $(u0), $(params), 0)

        (; Dr, Ds, Dt) = rd
        LIFT = get_nodal_lift(N)

        params = (; rd, md, Dr, Ds, Dt, LIFT, cache)

        time2 = @benchmark rhs_matmat!($(similar(u0)), $(u0), $(params), 0)

        push!(ratio_times, minimum(time2).time/minimum(time1).time)
    end

    plot(bar(1:K, ratio_times), 
        legend = false, 
        title = "Speedup of Bernstein over nodal (3D Wave DG), K = 2, min times over $(BenchmarkTools.DEFAULT_PARAMETERS.samples) samples",
        yaxis = ("Time (Nodal) / Time (Not-MT Bernstein)"),
        xaxis = ("Degree N"),
        titlefont = font(10),
        xticks = 1:K
        )
end 

make_rhs_plot(15)


