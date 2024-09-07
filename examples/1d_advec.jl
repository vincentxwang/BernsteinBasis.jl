# A Bernstein basis DG solver for the 1D advection equation.

using StartUpDG
using OrdinaryDiffEq
using LinearAlgebra
using BernsteinBasis

###############################################################################
# Let u(x,t) be scalar.
#
# We solve:
#
# du/dt + du/dx = 0.

function rhs!(du, u, params, t)
    (; LIFT, Dr, rd, interface_flux) = params

    uM = view(u, rd.Fmask, :)
    
    for e in axes(uM, 2)
        for i in axes(uM, 1)
            interface_flux[i, e] = 0.5 * (uM[md.mapP[i,e]] - uM[i,e]) * md.nxJ[i,e] - 
                                0.5 * (uM[md.mapP[i,e]] - uM[i,e]) * md.Jf[i,e]
        end
    end

    dudxJ = md.rxJ .* (Dr * u)
    du .= -(dudxJ + LIFT * interface_flux) ./ md.J
end

# Set polynomial order
N = 7

# Problem setup
rd = RefElemData(Line(), N)
md = MeshData(uniform_mesh(rd.element_type, 100), rd; is_periodic=true)

tspan = (0.0, 0.2)
u0 = @. sin(pi * md.x)
vander, _ = bernstein_basis(Line(), N, rd.r)

# Convert nodal to Bernstein coefficients
u0_modal = vander \ u0

# rd.LIFT maps surface quad points -> volume points
# inv(vander) maps volume points -> bernstein basis coeffs in volume
# quad points = interp points but the polynomial is = 1 at the endpoints (i.e. surface),
# so LIFT maps interpolation surface points -> bernstein basis coeffs in volume
LIFT = inv(vander) * rd.LIFT 

# Setup and solve ODE system
ode = ODEProblem(rhs!, u0_modal, tspan, (; LIFT, Dr=(inv(vander) * rd.Dr * vander), rd, interface_flux = md.x[rd.Fmask, :]))
sol = solve(ode, RK4(), saveat=LinRange(tspan..., 25))

# Convert Bernstein back to nodal
u = vander * sol.u[end]

# Check error
u_exact = @. sin(pi * (md.x - tspan[2]))
@show norm(u - u_exact, Inf)



