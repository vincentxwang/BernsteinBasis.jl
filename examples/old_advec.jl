using StartUpDG 
using LinearAlgebra
using SparseArrays
using Plots
using BernsteinBasis

N = 5
rd = RefElemData(Tet(), N)
# LIFT: quad points on faces -> polynomial in volume
# ---> define Vq: polynomials on faces -> quad points on faces

# ∑ u(x_i) * w_i = ∫u(x)
# (A' * W * A)_ij = (a_i, a_j)_L2 = ∫a_i(x) a_j(x)
(; rf, sf, tf, wf) = rd
rf, sf, tf, wf = reshape.((rf, sf, tf, wf), :, 4) 

rd_tri = RefElemData(Tri(), rd.N, quad_rule_vol = (rf[:,4], sf[:,4], wf[:,4]))

# # ---> nodal_LIFT = LIFT * Vq: Lagrange polynomials on faces -> Lagrange polynomials in volume
nodal_LIFT = rd.LIFT * kron(I(4), rd_tri.Vq)

# nodal_to_bernstein_volume * nodal_LIFT * bernstein_to_nodal_face
VB, _ = bernstein_basis(Tri(), N, rd_tri.r, rd_tri.s)
bernstein_to_nodal_face = kron(I(4), VB)
bernstein_to_nodal_volume, _ = bernstein_basis(Tet(), N, rd.rst...)

# directly constructing the Bernstein lift matrix
Vq, _ = bernstein_basis(Tet(), N, rd.rstq...)
MB = Vq' * diagm(rd.wq) * Vq
Vf, _ = bernstein_basis(Tet(), N, rd.rstf...)
VBf2, _ = bernstein_basis(Tri(), N, rf[:,1], tf[:,1])
VBf3, _ = bernstein_basis(Tri(), N, rf[:,2], sf[:,2])
VBf4, _ = bernstein_basis(Tri(), N, sf[:,3], tf[:,3])
VBf1, _ = bernstein_basis(Tri(), N, rf[:,4], sf[:,4])
MBf = Vf' * diagm(rd.wf) * blockdiag(sparse.((VBf1, VBf2, VBf3, VBf4))...)

scatter(rf[:,2], sf[:,2], tf[:,2])


spy(MBf)

spy(Vf)
using SparseArrays
bernstein_LIFT = MB \ MBf # bernstein_to_nodal_volume \ (nodal_LIFT * bernstein_to_nodal_face)
bernstein_LIFT = Matrix(droptol!(sparse(bernstein_LIFT), 10 * length(bernstein_LIFT) * eps()))

spy(bernstein_LIFT)
using BernsteinBasis
spy(BernsteinBasis.get_bernstein_lift(5))

using OrdinaryDiffEq

N = 5
rd = RefElemData(Tet(), N)
md = MeshData(uniform_mesh(rd.element_type, 2), rd; is_periodic=true)


const BDr = Matrix(BernsteinBasis.BernsteinDerivativeMatrix_3D_r(N))
const BDs = Matrix(BernsteinDerivativeMatrix_3D_s(N))
const BDt = Matrix(BernsteinDerivativeMatrix_3D_t(N))
# PDE -> ODE system
#     -> du/dt = rhs(u, params, t)
# advection equation: du/dt + du/dx = 0
# wave equation: dp/dt + du/dx + dv/dy + dw/dz = 0
#                du/dt + dp/dx = 0
#                dv/dt + dp/dy = 0
#                dw/dt + dp/dz = 0
function rhs!(du, u, params, t)
    (; rd, md, Dr, Ds, Dt, LIFT) = params
    uM = Vf * u
    uP = uM[md.mapP]           
    interface_flux = @. 0.5 * (uP - uM) * md.nxJ - 0.5 * (uP - uM) * md.Jf
    # u(x,y,z) = u(x(r,s,t), y(r,s,t), z(r,s,t)) 
    # --> du/dx = du/dr * dr/dx + du/ds * ds/dx + du/dt * dt/dz
    dudxJ = md.rxJ .* (BDr * u) + md.sxJ .* (BDs * u) + md.txJ .* (BDt * u)
    du .= -(dudxJ + (bernstein_LIFT * (MBf \ Vf')) * interface_flux) ./ md.J
end

(; x, y, z) = md
u0 = @. sin(pi * x) * sin(pi * y) * sin(pi * z)

tspan = (0.0, 0.25)

(; Dr, Ds, Dt, LIFT) = rd
ode = ODEProblem(rhs!, u0, tspan, (; rd, md, Dr, Ds, Dt, LIFT))
sol = solve(ode, RK4(), saveat=LinRange(tspan..., 25))

u = sol.u[end]

u_exact = @. sin(pi * (x - tspan[2])) * sin(pi * y) * sin(pi * z)

@show norm(u - u_exact, Inf)


# @gif for u in sol.u
#     scatter(vec(rd.Vp * md.x), vec(rd.Vp * md.y), vec(rd.Vp * md.z), zcolor=vec(rd.Vp * u), 
#         leg=false, msw=0, ms=4, ratio=1)
# end