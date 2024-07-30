# A Bernstein basis DG solver for the 3D acoustic wave equation.
# Reference: https://arxiv.org/pdf/1512.06025

using OrdinaryDiffEq
using StartUpDG
using Plots
using LinearAlgebra
using SparseArrays
using BernsteinBasis
using StaticArrays

# Wave equation
# dpdt + du/dx + dv/dy + dw/dz = 0
# dudt + dp/dx                 = 0
# dvdt + dp/dy                 = 0
# dwdt + dp/dz                 = 0

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

# Computes d(p, u, v, w)/dt as a function of u = (p, u, v, w). (note two u's are not the same).
function rhs_matvec!(du, u, params, t)
    (; rd, md, Dr, Ds, Dt, LIFT) = params
    
    (; uM, interface_flux, dfudr, dfuds, dfudt, dfvdr, dfvds, dfvdt, dfwdr, dfwds, dfwdt) = params.cache
    
    uM .= view(u, rd.Fmask, :)

    @inbounds for e in axes(uM, 2)
        for i in axes(uM, 1)
            duM = uM[md.mapP[i,e]] - uM[i,e]
            ndotdU =    md.nxJ[i,e] * duM[2] + 
                        md.nyJ[i,e] * duM[3] +
                        md.nzJ[i,e] * duM[4]
            interface_flux[i, e] = 0.5 * SVector(
                duM[1] - ndotdU,
                (ndotdU - duM[1]) * md.nxJ[i,e],
                (ndotdU - duM[1]) * md.nyJ[i,e],
                (ndotdU - duM[1]) * md.nzJ[i,e],
            )
        end
    end

    @inbounds for e in axes(du, 2)
        # TODO: fu/fv/fw cache
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

        # TODO: broadcast or loop
        du[:, e] -= (md.rxJ[1, e] * dfudr[:, e] + md.sxJ[1, e] * dfuds[:, e] + md.txJ[1, e] * dfudt[:, e] + 
                    md.ryJ[1, e] * dfvdr[:, e] + md.syJ[1, e] * dfvds[:, e] + md.tyJ[1, e] * dfvdt[:, e] + 
                    md.rzJ[1, e] * dfwdr[:, e] + md.szJ[1, e] * dfwds[:, e] + md.tzJ[1, e] * dfwdt[:, e])
                    
        du[:, e] = du[:, e] / md.J[1, e]
    end
end

# Set polynomial order
N = 4

rd = RefElemData(Tet(), N)

(; r, s, Fmask) = rd
Fmask = reshape(Fmask, :, 4)
rf, sf = rd.r[Fmask[:,1]], rd.t[Fmask[:,1]]

rd = RefElemData(Tet(), N; quad_rule_face = (rf, sf, ones(length(rf))))
md = MeshData(uniform_mesh(rd.element_type, 4), rd;               
              is_periodic=true)
              
# Problem setup
tspan = (0.0, 0.05)
(; x, y, z) = md

u0 = @. SVector{4, Float64}(
    cos(pi * x) * cos(pi * y) * cos(pi * z), 
    0, 
    0, 
    0,
    )

# Convert initial conditions to Bernstein coefficients
(; r, s, t) = rd
vande, _ = bernstein_basis(Tet(), N, r, s, t)
modal_u0 = inv(vande) * u0

# Initialize operators
Dr = BernsteinDerivativeMatrix_3D_r(N)
Ds = BernsteinDerivativeMatrix_3D_s(N)
Dt = BernsteinDerivativeMatrix_3D_t(N)
LIFT = BernsteinLift{SVector{4, Float64}}(N)

# Cache temporary arrays (values are initialized to get the right dimensions)
cache = (; uM = modal_u0[rd.Fmask, :], interface_flux = modal_u0[rd.Fmask, :], 
           dfudr = similar(modal_u0), dfuds = similar(modal_u0), dfudt = similar(modal_u0),
           dfvdr = similar(modal_u0), dfvds = similar(modal_u0), dfvdt = similar(modal_u0),
           dfwdr = similar(modal_u0), dfwds = similar(modal_u0), dfwdt = similar(modal_u0),)

# Combine parameters
params = (; rd, md, Dr, Ds, Dt, LIFT, cache)

# Solve ODE system
ode = ODEProblem(rhs_matvec!, modal_u0, tspan, params)
sol = solve(ode, Tsit5(), saveat=LinRange(tspan..., 25), dt = 0.01)

# Convert Bernstein coefficients back to point evaluations
u = vande * sol.u[end]

# Test against analytical solution
u_exact = @. SVector{4, Float64}( 
    cos(pi * x) * cos(pi * y) * cos(pi * z) * cos(sqrt(3) * pi * tspan[2]), 
    1/sqrt(3) * sin(pi * x) * cos(pi * y) * cos(pi * z) * sin(sqrt(3) * pi * tspan[2]), 
    1/sqrt(3) * cos(pi * x) * sin(pi * y) * cos(pi * z) * sin(sqrt(3) * pi * tspan[2]), 
    1/sqrt(3) * cos(pi * x) * cos(pi * y) * sin(pi * z) * sin(sqrt(3) * pi * tspan[2]))

@show norm(u - u_exact, Inf)