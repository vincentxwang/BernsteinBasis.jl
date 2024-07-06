using StartUpDG
using OrdinaryDiffEq
using Test
using LinearAlgebra


N = 7
rd = RefElemData(Line(), N)
md = MeshData(uniform_mesh(rd.element_type, 100), rd; is_periodic=true)

# does not work lol 

# struct BernsteinDerivativeMatrix_1D_r <: AbstractMatrix{Float64}
#     N::Int
# end

# function Base.size(Dr::BernsteinDerivativeMatrix_1D_r)
#     return (Dr.N + 1, Dr.N + 1)
# end

# function Base.getindex(Dr::BernsteinDerivativeMatrix_1D_r, m, n)
#     a1 = m
#     a2 = Dr.N - m
#     if m - n == 0
#         return 1/2 * (a1 - a2)
#     elseif m - n == 1
#         return 1/2 * (a2 + 1)
#     elseif m - n == -1
#         return -1/2 * (a1 + 1)
#     else
#         return 0
#     end
# end

function bernstein_basis_1d(::Line, N, r, s)
    V =  hcat(@. [(factorial(N)/(factorial(i) * factorial(N - i))) * r^i * s^(N - i) for i in 0:N]...)
    return V
end

a = @. (1+rd.r)/2
b = @. (1-rd.r)/2

vander = bernstein_basis_1d(Line(), N, a, b)

# @test inv(vander) * rd.Dr * vander ≈  BernsteinDerivativeMatrix_1D_r(N)

# quad points = interp points but the polynomial is = 1 at the endpoints
LIFT = inv(vander) * rd.LIFT 

u0 = @. sin(pi * md.x)

function rhs!(du, u, params, t)
    (; LIFT, Dr, rd, interface_flux) = params

    uM = view(u, rd.Fmask, :)
    
    for e in axes(uM, 2)
        for i in axes(uM, 1)
            interface_flux[i, e] = 0.5 * (uM[md.mapP[i,e]] - uM[i,e]) * md.nxJ[i,e] - 
                                   0.5 * (uM[md.mapP[i,e]] - uM[i,e]) * md.Jf[i,e]
        end
    end
    # interface_flux = @. 0.5 * (uP + uM)/2 + 1000/2 * (uP - uM) * md.nxJ
    dudxJ = md.rxJ .* (Dr * u)
    du .= -(dudxJ + LIFT * interface_flux) ./ md.J
end

for e in size(u0, 2)
    u0[:,e] = inv(vander) * u0[:,e]
end

tspan = (0.0, 1.0)

ode = ODEProblem(rhs!, u0, tspan, (; LIFT, Dr=(inv(vander) * rd.Dr * vander), rd, interface_flux = md.x[rd.Fmask, :]))
sol = solve(ode, RK4(), saveat=LinRange(tspan..., 25))

u = sol.u[end]

for e in size(u0, 2)
    u[:,e] = vander * u[:,e]
end

u_exact = @. sin(pi * (md.x - tspan[2]))

@show norm(u - u_exact, Inf)



