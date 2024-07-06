

using OrdinaryDiffEq
using StartUpDG
using Plots
using LinearAlgebra
using SparseArrays
using BernsteinBasis


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
    mul!(du, LIFT, interface_flux)
    
    @. du += md.rxJ * dudr + md.sxJ * duds + md.txJ * dudt

    @. du = -du ./ md.J
    return du
end


# function rhs_matvec!(du, u, params, t)
#     (; rd, md, Dr, Ds, Dt, LIFT) = params
    
#     (; uM, interface_flux, dudr, duds, dudt) = params.cache
    
#     uM .= view(u, rd.Fmask, :)

#     for e in axes(uM, 2)
#         for i in axes(uM, 1)
#             interface_flux[i, e] = 0.5 * (uM[md.mapP[i,e]] + uM[i,e]) * md.nxJ[i,e] - 
#                                    0.5 * (uM[md.mapP[i,e]] + uM[i,e]) * md.Jf[i,e]
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


N = 10
rd = RefElemData(Tet(), N)

# create interp matrix from Fmask node ordering to quadrature node ordering
(; r, s, Fmask) = rd

Fmask = reshape(Fmask, :, 4)

rf, sf = rd.r[Fmask[:,1]], rd.t[Fmask[:,1]]

rd = RefElemData(Tet(), N; quad_rule_face = (rf, sf, ones(length(rf))))
md = MeshData(uniform_mesh(rd.element_type, 6), rd;               
              is_periodic=true)
              
Dr = BernsteinDerivativeMatrix_3D_r(N)
Ds = BernsteinDerivativeMatrix_3D_s(N)
Dt = BernsteinDerivativeMatrix_3D_t(N)
LIFT = BernsteinBasis.get_bernstein_lift(N)

tspan = (0.0, 2.0)

(; x, y, z) = md

u0 = @. cos(pi * x) * cos(pi * y) * cos(pi * z)

(; r, s, t) = rd

bary_coords = BernsteinBasis.cartesian_to_barycentric2(Tet(), r, s, t)

change = bernstein_basis(Tet(), N, 
    getfield.(bary_coords, 1), 
    getfield.(bary_coords, 2),
    getfield.(bary_coords, 3),
    getfield.(bary_coords, 4))[1]

modal_u0 = change \ u0

# (; Dr, Ds, Dt) = rd
# this is just filler to get the same size really.
cache = (; uM = md.x[rd.Fmask, :], interface_flux = md.x[rd.Fmask, :], 
           dudr = similar(md.x), duds = similar(md.x), dudt = similar(md.x))
params = (; rd, md, Dr, Ds, Dt, LIFT, cache)
ode = ODEProblem(rhs_matmul!, modal_u0, tspan, params)
sol = solve(ode, Tsit5(), saveat=LinRange(tspan..., 25))

u = change * sol.u[end]

u_exact = @. cos(pi * (x - tspan[2])) * cos(pi * y) * cos(pi * z)

@show norm(u - u_exact, Inf)

