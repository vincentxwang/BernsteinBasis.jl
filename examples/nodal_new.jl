using OrdinaryDiffEq
using StartUpDG
using Plots
using LinearAlgebra
using SparseArrays

N = 2
rd = RefElemData(Tet(), N)

# create interp matrix from Fmask node ordering to quadrature node ordering
(; r, s, Fmask) = rd
Fmask = reshape(Fmask, :, 4)

rtri, stri = nodes(Tri(), N)
rfq, sfq, wfq = quad_nodes(Tri(), rd.N)
Vq_face = vandermonde(Tri(), rd.N, rfq, sfq) / vandermonde(Tri(), rd.N, rtri, stri)
VFmask = sparse(eachindex(vec(Fmask)), vec(Fmask), ones(length(Fmask)))
Vf_kron = kron(I(4), Vq_face) * VFmask


help_1, = bernstein_basis(Tri(), rd.N, rfq, sfq) 
help_2, = bernstein_basis(Tri(), rd.N, rtri, stri)
Vq_face = help_1 / help_2

nodal_LIFT = rd.LIFT * kron(I(4), Vq_face)

spy(droptol!(sparse(kron(I(4), Vq_face)), 1e-10))

# check for correctness of LIFT matrix
u = randn(length(rd.r))
uf = rd.Vf * u
@assert norm(rd.LIFT * uf - nodal_LIFT * u[rd.Fmask]) < 100 * eps() * length(LIFT)

# recreate RefElemData with nodal points instead of a quadrature rule
rf, sf = rd.r[Fmask[:,1]], rd.t[Fmask[:,1]]
rd = RefElemData(Tet(), N; quad_rule_face = (rf, sf, ones(length(rf))))
md = MeshData(uniform_mesh(rd.element_type, 4), rd;               
              is_periodic=true)

# PDE -> ODE system
#     -> du/dt = rhs(u, params, t)
# advection equation: du/dt + du/dx = 0
# wave equation: dp/dt + du/dx + dv/dy + dw/dz = 0
#                du/dt + dp/dx = 0
#                dv/dt + dp/dy = 0
#                dw/dt + dp/dz = 0
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

    du .= 0 
    # u(x,y,z) = u(x(r,s,t), y(r,s,t), z(r,s,t)) 
    # --> du/dx = du/dr * dr/dx + du/ds * ds/dx + du/dt * dt/dz
    mul!(dudr, Dr, u) 
    mul!(duds, Ds, u) 
    mul!(dudt, Dt, u) 

    @. du = md.rxJ * dudr + md.sxJ * duds + md.txJ * dudt

    mul!(du, LIFT, interface_flux, 1, 1)
    @. du = -du ./ md.J
    return du
end

# function rhs_matvec!(du, u, params, t)
#     (; rd, md, Dr, Ds, Dt, LIFT) = params
#     (; uM, interface_flux, dudr, duds, dudt) = params.cache
    
#     uM .= view(u, rd.Fmask, :)
#     for e in axes(uM, 2)
#         for i in axes(uM, 1)
#             interface_flux[i, e] = 0.5 * (uM[md.mapP[i,e]] - uM[i,e]) * md.nxJ[i,e] - 
#                                    0.5 * (uM[md.mapP[i,e]] - uM[i,e]) * md.Jf[i,e]
#         end
#     end

#     for e in axes(du, 2)
#         mul!(view(dudr, :, e), Dr, view(u, :, e))
#         mul!(view(duds, :, e), Ds, view(u, :, e))
#         mul!(view(dudt, :, e), Dt, view(u, :, e))

#         mul!(view(du, :, e), LIFT, view(interface_flux, :, e))
#         for i in axes(du, 1)
#             du[i, e] += md.rxJ[1, e] * dudr[i, e] + 
#                         md.sxJ[1, e] * duds[i, e] + 
#                         md.txJ[1, e] * dudt[i, e]
#             du[i, e] = -du[i, e] / md.J[1, e]
#         end
#     end
# end

tspan = (0.0, 0.01)

(; x, y, z) = md
u0 = @. sin(pi * x) * sin(pi * y) * sin(pi * z)

heatmap(u0)

(; Dr, Ds, Dt) = rd
cache = (; uM = md.x[rd.Fmask, :], interface_flux = md.x[rd.Fmask, :], 
           dudr = similar(md.x), duds = similar(md.x), dudt = similar(md.x))
params = (; rd, md, Dr, Ds, Dt, LIFT=nodal_LIFT, cache)
ode = ODEProblem(rhs_matmul!, u0, tspan, params)
sol = solve(ode, Tsit5(), saveat=LinRange(tspan..., 25), adaptive = false, dt = 0.01)



# A = similar(u0)
# spy(droptol!(sparse(rhs_matmul!(A, u0, params, 0)), 1e-10))
# spy(Matrix(droptol!(sparse(rhs_matmul!(similar(x), u0, params, 0)), 1e-10)))

heatmap(rhs_matmul!(similar(x), u0, params, 0))

A = rhs_matmul!(similar(x), u0, params, 0)

spy(A)

u = sol.u[end]

u_exact = @. sin(pi * (x - tspan[2])) * sin(pi * y) * sin(pi * z)

@show norm(u - u_exact, Inf)