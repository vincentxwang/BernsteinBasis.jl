using OrdinaryDiffEq
using StartUpDG
using Plots
using LinearAlgebra
using SparseArrays
using BernsteinBasis

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

    # u(x,y,z) = u(x(r,s,t), y(r,s,t), z(r,s,t)) 
    # --> du/dx = du/dr * dr/dx + du/ds * ds/dx + du/dt * dt/dz
    mul!(dudr, Dr, u) 
    mul!(duds, Ds, u) 
    mul!(dudt, Dt, u) 
    @. du = md.rxJ * dudr + md.sxJ * duds + md.txJ * dudt
    du = du + LIFT * interface_flux
    # mul!(du, LIFT, interface_flux, 1, 1)
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


N = 7
rd = RefElemData(Tet(), N)


# create interp matrix from Fmask node ordering to quadrature node ordering
(; r, s, Fmask) = rd

Fmask = reshape(Fmask, :, 4)

rf, sf = rd.r[Fmask[:,1]], rd.t[Fmask[:,1]]
rd = RefElemData(Tet(), N; quad_rule_face = (rf, sf, ones(length(rf))))
md = MeshData(uniform_mesh(rd.element_type, 2), rd;               
              is_periodic=true)
              
Dr = BernsteinDerivativeMatrix_3D_r(N)
Ds = BernsteinDerivativeMatrix_3D_s(N)
Dt = BernsteinDerivativeMatrix_3D_t(N)
LIFT = BernsteinLift(N)
# LIFT = Matrix(BernsteinBasis.get_bernstein_lift(N))

tspan = (0.0, 0.25)

(; x, y, z) = md

u0 = @. sin(pi * x) * sin(pi * y) * sin(pi * z)

spy(u0)

(; r, s, t) = rd

rd.r

bary_coords = BernsteinBasis.cartesian_to_barycentric(Tet(), r, s, t)

change = bernstein_basis(Tet(), N, 
    getfield.(bary_coords, 1), 
    getfield.(bary_coords, 2),
    getfield.(bary_coords, 3),
    getfield.(bary_coords, 4))[1]

for e in size(u0, 2)
    u0[:,e] = inv(change) * u0[:,e]
end


# (; Dr, Ds, Dt) = rd
# this is just filler to get the same size really.
cache = (; uM = md.x[rd.Fmask, :], interface_flux = md.x[rd.Fmask, :], 
           dudr = similar(md.x), duds = similar(md.x), dudt = similar(md.x))
params = (; rd, md, Dr, Ds, Dt, LIFT, cache)
ode = ODEProblem(rhs_matvec!, u0, tspan, params)
sol = solve(ode, Tsit5(), saveat=LinRange(tspan..., 25))


u = sol.u[end]

for e in size(u, 2)
    u[:,e] = change * u[:,e]
end

u_exact = @. sin(pi * (x - tspan[2])) * sin(pi * y) * sin(pi * z)

@show norm(u - u_exact, Inf)

# @gif for u in sol.u
#     scatter(vec(rd.Vp * md.x), vec(rd.Vp * md.y), vec(rd.Vp * md.z), zcolor=vec(rd.Vp * u), 
#         leg=false, msw=0, ms=4, ratio=1)
# end