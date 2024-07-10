using OrdinaryDiffEq
using StartUpDG
using Plots
using LinearAlgebra
using SparseArrays
using BenchmarkTools

# Set the polynomial order
N = 6

rd = RefElemData(Tet(), N)

# Create interpolation matrix from Fmask node ordering to quadrature node ordering
(; r, s, Fmask) = rd
Fmask = reshape(Fmask, :, 4)

rtri, stri = nodes(Tri(), N)
rfq, sfq, wfq = quad_nodes(Tri(), rd.N)
Vq_face = vandermonde(Tri(), rd.N, rfq, sfq) / vandermonde(Tri(), rd.N, rtri, stri)

nodal_LIFT = rd.LIFT * kron(I(4), Vq_face)

# Check correctness of LIFT matrix
u = randn(length(rd.r))
uf = rd.Vf * u
@assert norm(rd.LIFT * uf - nodal_LIFT * u[rd.Fmask]) < 100 * eps() * length(nodal_LIFT)

# recreate RefElemData with nodal points instead of a quadrature rule
rf, sf = rd.r[Fmask[:,1]], rd.t[Fmask[:,1]]
rd = RefElemData(Tet(), N; quad_rule_face = (rf, sf, ones(length(rf))))
md = MeshData(uniform_mesh(rd.element_type, 4), rd;               
              is_periodic=true)

function rhs_matmul!(du, u, params, t)
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
    mul!(dudr, Dr, u) 
    mul!(duds, Ds, u) 
    mul!(dudt, Dt, u) 

    @. du = md.rxJ * dudr + md.sxJ * duds + md.txJ * dudt

    mul!(du, LIFT, interface_flux, 1, 1)
    @. du = -du ./ md.J
end

function rhs_matvec!(du, u, params, t)
    (; rd, md, Dr, Ds, Dt, LIFT) = params
    (; uM, interface_flux, dudr, duds, dudt) = params.cache
    
    uM .= view(u, rd.Fmask, :)
    for e in axes(uM, 2)
        for i in axes(uM, 1)
            interface_flux[i, e] = 0.5 * (uM[md.mapP[i,e]] - uM[i,e]) * md.nxJ[i,e] - 
                                   0.5 * (uM[md.mapP[i,e]] - uM[i,e]) * md.Jf[i,e]
        end
    end

    for e in axes(du, 2)
        mul!(view(dudr, :, e), Dr, view(u, :, e))
        mul!(view(duds, :, e), Ds, view(u, :, e))
        mul!(view(dudt, :, e), Dt, view(u, :, e))

        mul!(view(du, :, e), LIFT, view(interface_flux, :, e))
        for i in axes(du, 1)
            du[i, e] += md.rxJ[1, e] * dudr[i, e] + 
                        md.sxJ[1, e] * duds[i, e] + 
                        md.txJ[1, e] * dudt[i, e]
            du[i, e] = -du[i, e] / md.J[1, e]
        end
    end
end

# Problem setup
tspan = (0.0, 2.0)
(; x, y, z) = md
u0 = @. sin(pi * x) * sin(pi * y) * sin(pi * z)

# Derivative operators
(; Dr, Ds, Dt) = rd

# Cache temporary arrays (values are initialized to get the right dimensions)
cache = (; uM = md.x[rd.Fmask, :], interface_flux = md.x[rd.Fmask, :], 
           dudr = similar(md.x), duds = similar(md.x), dudt = similar(md.x))

# Combine parameters
params = (; rd, md, Dr, Ds, Dt, LIFT=nodal_LIFT, cache)

# Solve ODE system
ode = ODEProblem(rhs_matmul!, u0, tspan, params)
sol = solve(ode, Tsit5(), saveat=LinRange(tspan..., 25))

@btime rhs_matvec!($(similar(u0)), $(u0), $params, 0)

u = sol.u[end]

# Test against analytical solution
u_exact = @. sin(pi * (x - tspan[2])) * sin(pi * y) * sin(pi * z)
@show norm(u - u_exact, Inf)