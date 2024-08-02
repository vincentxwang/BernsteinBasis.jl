# A nodal basis DG solver for the 3D Maxwell's equation.
# Reference: Hesthaven and Warburton pg. 432-437

using OrdinaryDiffEq
using StartUpDG
using Plots
using LinearAlgebra
using SparseArrays
using BernsteinBasis
using StaticArrays

# Set polynomial order
N = 7

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

# Computes d(Hx, Hy, Hz, Ex, Ey, Ez)/dt as a function of u = (Hx, Hy, Hz, Ex, Ey, Ez).
function rhs_matvec!(du, u, params, t)
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

rd = RefElemData(Tet(), N)

(; r, s, Fmask) = rd
Fmask = reshape(Fmask, :, 4)
rf, sf = rd.r[Fmask[:,1]], rd.t[Fmask[:,1]]

rtri, stri = nodes(Tri(), N)
rfq, sfq, wfq = quad_nodes(Tri(), rd.N)
Vq_face = vandermonde(Tri(), rd.N, rfq, sfq) / vandermonde(Tri(), rd.N, rtri, stri)

nodal_LIFT = rd.LIFT * kron(I(4), Vq_face)

rd = RefElemData(Tet(), N; quad_rule_face = (rf, sf, ones(length(rf))))
md = MeshData(uniform_mesh(rd.element_type, 4), rd;               
              is_periodic=true)
              
# Problem setup
tspan = (0.0, 0.05)
(; x, y, z) = md

u0 = @. SVector{6, Float64}(0, 0, 0, 0, 0, sin(pi * x) * sin(pi * y))

# Initialize operators
(; Dr, Ds, Dt) = rd
LIFT = nodal_LIFT

# Cache temporary arrays (values are initialized to get the right dimensions)
cache = (; uM = u0[rd.Fmask, :], interface_flux = u0[rd.Fmask, :], 
           duM = u0[rd.Fmask, :],
           dfxdr = similar(u0), dfxds = similar(u0), dfxdt = similar(u0),
           dfydr = similar(u0), dfyds = similar(u0), dfydt = similar(u0),
           dfzdr = similar(u0), dfzds = similar(u0), dfzdt = similar(u0),
           fxu = similar(u0), fyu = similar(u0), fzu = similar(u0))

# Combine parameters
params = (; rd, md, Dr, Ds, Dt, LIFT, cache)

# Solve ODE system
ode = ODEProblem(rhs_matvec!, u0, tspan, params)
sol = solve(ode, Tsit5(), saveat=LinRange(tspan..., 25), dt = 0.01)

# Convert Bernstein coefficients back to point evaluations
u = sol.u[end]

# Test against analytical solution
u_exact = @. SVector{6, Float64}(
    -1 / sqrt(2) * sin(pi * x) * cos(pi * y) * sin(pi * sqrt(2) * tspan[2]), 
    1 / sqrt(2) * cos(pi * x) * sin(pi * y) * sin(pi * sqrt(2) * tspan[2]), 
    0, 
    0, 
    0, 
    sin(pi * x) * sin(pi * y) * cos(pi * sqrt(2) * tspan[2]))
@show norm(u - u_exact, Inf)