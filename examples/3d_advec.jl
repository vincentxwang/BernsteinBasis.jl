# A Bernstein basis DG solver for the 3D advection equation, with a central
# numerical flux.

using OrdinaryDiffEq
using StartUpDG
using LinearAlgebra
using SparseArrays
using StaticArrays
using BernsteinBasis

###############################################################################
# Let q represent the state of the system where q = (u, v, w), where u, v, w
# are functions of x, y, z, t.
#
# We frame the problem as
#
# du/dt + dfx(q)/dx + dfy(q)/dy + dfz(q)/dz = 0.
# where fx(q) = (u, 0, 0), fy(q) = (0, 0, 0), fz(q) = (0, 0, 0).

function fx(q)
    u, v, w = q
    return SVector(u, 0, 0)
end

function fy(q)
    u, v, w = q
    return SVector(0, 0, 0)
end

function fz(q)
    u, v, w = q
    return SVector(0, 0, 0)
end

function rhs_matvec!(du, u, params, t)
    (; rd, md, Dr, Ds, Dt, LIFT) = params
    
    (; uM, interface_flux, dudr, duds, dudt, fxu) = params.cache

    uM .= view(u, rd.Fmask, :)

    @inbounds for e in axes(uM, 2)
        for i in axes(uM, 1)
            interface_flux[i, e] = 0.5 * (uM[md.mapP[i,e]] - uM[i,e]) * md.nxJ[i,e] - 
                                   0.5 * (uM[md.mapP[i,e]] - uM[i,e]) * md.Jf[i,e]
        end
    end

    @inbounds for e in axes(du, 2)
        fxu = fx.(view(u, :, e))

        mul!(view(dudr, :, e), Dr, fxu)
        mul!(view(duds, :, e), Ds, fxu)
        mul!(view(dudt, :, e), Dt, fxu)

        mul!(view(du, :, e), LIFT, view(interface_flux, :, e))

        for i in axes(du, 1)
            du[i, e] += md.rxJ[1, e] * dudr[i, e] + md.sxJ[1, e] * duds[i, e] + md.txJ[1, e] * dudt[i, e]
        end
    end
    @. du = -du / md.J[1,1]
end

# Set polynomial order
N = 7

rd = RefElemData(Tet(), N)

(; r, s, Fmask) = rd
Fmask = reshape(Fmask, :, 4)
rf, sf = rd.r[Fmask[:,1]], rd.t[Fmask[:,1]]

rd = RefElemData(Tet(), N; quad_rule_face = (rf, sf, ones(length(rf))))
md = MeshData(uniform_mesh(rd.element_type, 4), rd;               
              is_periodic=true)
        
# Generate vandermonde matrix
(; r, s, t) = rd
vande, _ = bernstein_basis(Tet(), N, r, s, t)

# Problem setup
tspan = (0.0, 0.1)
(; x, y, z) = md

# modal_u0 = ([hcat(x[:,e], y[:,e], z[:,e]) for e in axes(x,2)])

u0 = @. SVector{3, Float64}(sin(π * x) * sin(π * y) * sin(π * z), 0, 0)
modal_u0 = inv(vande) * u0

# Initialize operators
Dr = BernsteinDerivativeMatrix_3D_r(N)
Ds = BernsteinDerivativeMatrix_3D_s(N)
Dt = BernsteinDerivativeMatrix_3D_t(N)
LIFT = BernsteinLift{SVector{3,Float64}}(N)

# Cache temporary arrays (values are initialized to get the right dimensions)
cache = (; uM = modal_u0[rd.Fmask, :], interface_flux = modal_u0[rd.Fmask, :], 
           dudr = similar(modal_u0), duds = similar(modal_u0), dudt = similar(modal_u0),
           fxu = similar(modal_u0[:, 1]))

# Combine parameters
params = (; rd, md, Dr, Ds, Dt, LIFT, cache)


rhs_matvec!(similar(modal_u0), modal_u0, params, 0)

# Solve ODE system
ode = ODEProblem(rhs_matvec!, modal_u0, tspan, params)
sol = solve(ode, Tsit5(), saveat=LinRange(tspan..., 25), dt = 0.01)

# Convert Bernstein coefficients back to point evaluations
u = vande * sol.u[end]

# Test against analytical solution
u_exact = @. SVector{3, Float64}(sin(π * (x - tspan[2])) * sin(π * y) * sin(π * z), 0, 0)
@show norm(u - u_exact, Inf)


