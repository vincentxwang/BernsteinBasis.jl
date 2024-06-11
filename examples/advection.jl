using StartUpDG
using OrdinaryDiffEq
using Plots

N = 7
rd = RefElemData(Tet(), N)
md = MeshData(uniform_mesh(rd.element_type, 4), rd; is_periodic=true)

# PDE -> ODE system
#     -> du/dt = rhs(u, params, t)
# advection equation: du/dt + du/dx = 0
# wave equation: dp/dt + du/dx + dv/dy + dw/dz = 0
#                du/dt + dp/dx = 0
#                dv/dt + dp/dy = 0
#                dw/dt + dp/dz = 0
function rhs!(du, u, params, t)
    (; rd, md, Dr, Ds, Dt, LIFT) = params
    uM = rd.Vf * u
    uP = uM[md.mapP]                      
    interface_flux = @. 0.5 * (uP - uM) * md.nxJ - 0.5 * (uP - uM) * md.Jf
    # u(x,y,z) = u(x(r,s,t), y(r,s,t), z(r,s,t)) 
    # --> du/dx = du/dr * dr/dx + du/ds * ds/dx + du/dt * dt/dz
    dudxJ = md.rxJ .* (Dr * u) + md.sxJ .* (Ds * u) + md.txJ .* (Dt * u)
    du .= -(dudxJ + LIFT * interface_flux) ./ md.J
end

tspan = (0.0, 2.0)

(; x, y, z) = md
u0 = @. sin(pi * x) * sin(pi * y) * sin(pi * z)

(; Dr, Ds, Dt, LIFT) = rd
ode = ODEProblem(rhs!, u0, tspan, (; rd, md, Dr, Ds, Dt, LIFT))
sol = solve(ode, RK4(), saveat=LinRange(tspan..., 25))

u = sol.u[end]

@gif for u in sol.u
    scatter(vec(rd.Vp * md.x), vec(rd.Vp * md.y), vec(rd.Vp * md.z), zcolor=vec(rd.Vp * u), 
        leg=false, msw=0, ms=4, ratio=1)
end

# scatter(md.xyz..., leg=false, ms=2)
# scatter(md.xf[:,1], md.yf[:,1], md.zf[:,1], leg=false, ms=2)