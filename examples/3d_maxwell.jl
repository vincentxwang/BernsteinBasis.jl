# A Bernstein basis DG solver for the 3D Maxwell's equation.
# Reference: Hesthaven and Warburton pg. 432-437

using OrdinaryDiffEq
using StartUpDG
using Plots
using LinearAlgebra
using SparseArrays
using BernsteinBasis
using StaticArrays

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

function dot(u, md)
    Hx, Hy, Hz, Ex, Ey, Ez = u
    dotH = Hx * md.nxJ
end

# Computes d(Hx, Hy, Hz, Ex, Ey, Ez)/dt as a function of u = (Hx, Hy, Hz, Ex, Ey, Ez).
function rhs_matvec!(du, u, params, t)
    (; rd, md, Dr, Ds, Dt, LIFT) = params
    
    (; uM, interface_flux, dudr, duds, dudt) = params.cache
    
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
        # mul!(view(dudr, :, e), Dr, fx.(view(u, :, e)))
        # mul!(view(duds, :, e), Ds, fx.(view(u, :, e)))
        # mul!(view(dudt, :, e), Dt, fx.(view(u, :, e)))

        mul!(view(du, :, e), LIFT, view(interface_flux, :, e))

        du[:, e] += md.rxJ[1, e] * (Dr * fx.(view(u, :, e))) + md.sxJ[1, e] * (Ds * fx.(view(u, :, e))) + md.txJ[1, e] * (Dt * fx.(view(u, :, e))) + 
                    md.ryJ[1, e] * (Dr * fy.(view(u, :, e))) + md.syJ[1, e] * (Ds * fy.(view(u, :, e))) + md.tyJ[1, e] * (Dt * fy.(view(u, :, e))) + 
                    md.rzJ[1, e] * (Dr * fz.(view(u, :, e))) + md.szJ[1, e] * (Ds * fz.(view(u, :, e))) + md.tzJ[1, e] * (Dt * fz.(view(u, :, e)))
        du[:, e] = du[:, e] / md.J[1, e]
    end
    return du
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
              
# Problem setup
tspan = (0.0, 0.05)
(; x, y, z) = md


u0 = @. SVector{6, Float64}(0, 0, 0, 0, 0, sin(pi * x) * sin(pi * y))

# Convert initial conditions to Bernstein coefficients
(; r, s, t) = rd
vande, _ = bernstein_basis(Tet(), N, r, s, t)

modal_u0 = inv(vande) * u0

# Initialize operators
Dr = BernsteinDerivativeMatrix_3D_r(N)
Ds = BernsteinDerivativeMatrix_3D_s(N)
Dt = BernsteinDerivativeMatrix_3D_t(N)
LIFT = BernsteinLift{SVector{6, Float64}}(N)

# Cache temporary arrays (values are initialized to get the right dimensions)
cache = (; uM = modal_u0[rd.Fmask, :], interface_flux = modal_u0[rd.Fmask, :], 
           dudr = similar(modal_u0), duds = similar(modal_u0), dudt = similar(modal_u0))

# Combine parameters
params = (; rd, md, Dr, Ds, Dt, LIFT, cache)

# Solve ODE system
ode = ODEProblem(rhs_matvec!, modal_u0, tspan, params)
sol = solve(ode, Tsit5(), saveat=LinRange(tspan..., 25), dt = 0.01)

# Convert Bernstein coefficients back to point evaluations
u = vande * sol.u[end]

# Test against analytical solution
u_exact = @. SVector{6, Float64}(
    -1 / sqrt(2) * sin(pi * x) * cos(pi * y) * sin(pi * sqrt(2) * tspan[2]), 
    1 / sqrt(2) * cos(pi * x) * sin(pi * y) * sin(pi * sqrt(2) * tspan[2]), 
    0, 
    0, 
    0, 
    sin(pi * x) * sin(pi * y) * cos(pi * sqrt(2) * tspan[2]))
@show norm(u - u_exact, Inf)