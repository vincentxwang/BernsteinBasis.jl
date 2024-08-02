# Generates a graph comparing multithreaded Bernstein to nodal. Takes a while to run to N = 10.

using OrdinaryDiffEq
using StartUpDG
using Plots
using LinearAlgebra
using SparseArrays
using BernsteinBasis
using BenchmarkTools

function bernstein_rhs_matvec!(du, u, params, t)
    (; rd, md, Dr, Ds, Dt, LIFT) = params
    
    (; uM, interface_flux, dudr, duds, dudt) = params.cache
    
    uM .= view(u, rd.Fmask, :)

    @inbounds for e in axes(uM, 2)
        for i in axes(uM, 1)
            interface_flux[i, e] = 0.5 * (uM[md.mapP[i,e]] - uM[i,e]) * md.nxJ[i,e] - 
                                   0.5 * (uM[md.mapP[i,e]] - uM[i,e]) * md.Jf[i,e]
        end
    end

    @inbounds for e in axes(du, 2)
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

function get_bernstein_vandermonde(N)
    rd = RefElemData(Tet(), N)

    (; r, s, Fmask) = rd
    Fmask = reshape(Fmask, :, 4)
    rf, sf = rd.r[Fmask[:,1]], rd.t[Fmask[:,1]]

    rd = RefElemData(Tet(), N; quad_rule_face = (rf, sf, ones(length(rf))))

    (; r, s, t) = rd
    vande, _ = bernstein_basis(Tet(), N, r, s, t)
    return vande
end

# non-multithreaded bernstein
function run_nmt_bernstein_dg(N, K, vande)
    rd = RefElemData(Tet(), N)

    (; r, s, Fmask) = rd
    Fmask = reshape(Fmask, :, 4)
    rf, sf = rd.r[Fmask[:,1]], rd.t[Fmask[:,1]]

    rd = RefElemData(Tet(), N; quad_rule_face = (rf, sf, ones(length(rf))))
    md = MeshData(uniform_mesh(rd.element_type, K), rd;               
                is_periodic=true)
                
    # Problem setup
    tspan = (0.0, 0.1)
    (; x, y, z) = md
    u0 = @. sin(pi * x) * sin(pi * y) * sin(pi * z)

    # Convert initial conditions to Bernstein coefficients
    modal_u0 = vande \ u0

    # Initialize operators
    Dr = BernsteinDerivativeMatrix_3D_r(N)
    Ds = BernsteinDerivativeMatrix_3D_s(N)
    Dt = BernsteinDerivativeMatrix_3D_t(N)
    LIFT = BernsteinLift(N)

    # Cache temporary arrays (values are initialized to get the right dimensions)
    cache = (; uM = md.x[rd.Fmask, :], interface_flux = md.x[rd.Fmask, :], 
            dudr = similar(md.x), duds = similar(md.x), dudt = similar(md.x))

    # Combine parameters
    params = (; rd, md, Dr, Ds, Dt, LIFT, cache)

    # Solve ODE system
    ode = ODEProblem(bernstein_rhs_matvec!, modal_u0, tspan, params)
    sol = solve(ode, RK4(), saveat=LinRange(tspan..., 25), dt = 0.01)

    # Convert Bernstein coefficients back to point evaluations
    u = vande * sol.u[end]
end

function get_nodal_lift(N)
    rd = RefElemData(Tet(), N)
    rtri, stri = nodes(Tri(), N)
    rfq, sfq, wfq = quad_nodes(Tri(), rd.N)
    Vq_face = vandermonde(Tri(), rd.N, rfq, sfq) / vandermonde(Tri(), rd.N, rtri, stri)

    nodal_LIFT = rd.LIFT * kron(I(4), Vq_face)

    return nodal_LIFT
end

function naive_mul!(C, A, B)
    n, m = size(A)
    p = size(B, 2)

    C .= 0
    
    @inbounds for i in 1:n
        for j in 1:p
            for k in 1:m
                C[i,j] += A[i,k] * B[k,j]
            end
        end
    end
    return C
end

function naive_nodal_rhs_matmul!(du, u, params, t)
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
    naive_mul!(dudr, Dr, u) 
    naive_mul!(duds, Ds, u) 
    naive_mul!(dudt, Dt, u) 

    du .= 0
    naive_mul!(du, LIFT, interface_flux)

    @. du += md.rxJ * dudr + md.sxJ * duds + md.txJ * dudt
    @. du = -du ./ md.J
end

function run_naive_nodal_dg(N, K, lift)

    rd = RefElemData(Tet(), N)

    # Create interpolation matrix from Fmask node ordering to quadrature node ordering
    (; r, s, Fmask) = rd
    Fmask = reshape(Fmask, :, 4)

    # recreate RefElemData with nodal points instead of a quadrature rule
    rf, sf = rd.r[Fmask[:,1]], rd.t[Fmask[:,1]]
    rd = RefElemData(Tet(), N; quad_rule_face = (rf, sf, ones(length(rf))))
    md = MeshData(uniform_mesh(rd.element_type, 2), rd;               
                is_periodic=true)

    # Problem setup
    tspan = (0.0, 0.1)
    (; x, y, z) = md
    u0 = @. sin(pi * x) * sin(pi * y) * sin(pi * z)

    # Derivative operators
    (; Dr, Ds, Dt) = rd

    # Cache temporary arrays (values are initialized to get the right dimensions)
    cache = (; uM = md.x[rd.Fmask, :], interface_flux = md.x[rd.Fmask, :], 
            dudr = similar(md.x), duds = similar(md.x), dudt = similar(md.x))

    # Combine parameters
    params = (; rd, md, Dr, Ds, Dt, LIFT=lift, cache)

    # Solve ODE system
    # Switch out naive_nodal_rhs_matmul / nodal_rhs_matmul
    ode = ODEProblem(naive_nodal_rhs_matmul!, u0, tspan, params)
    sol = solve(ode, RK4(), saveat=LinRange(tspan..., 25), dt = 0.01)

    u = sol.u[end]
    
    return u
end

# function make_dg_plot(K)
#     BenchmarkTools.DEFAULT_PARAMETERS.samples = 15

#     ratio_times = Float64[]

#     for N in 1:K
#         time1 = @benchmark run_nmt_bernstein_dg($N, 2, $(get_bernstein_vandermonde(N)))

#         time2 = @benchmark run_naive_nodal_dg($N, 2, $(get_nodal_lift(N)))

#         push!(ratio_times, minimum(time2).time/minimum(time1).time)
#     end

#     plot(bar(1:K, ratio_times), 
#         legend = false, 
#         title = "Speedup of Bernstein over nodal DG, K = 2, min times over $(BenchmarkTools.DEFAULT_PARAMETERS.samples) samples",
#         yaxis = ("Time (Naive Nodal) / Time (Not-MT Bernstein)"),
#         xaxis = ("Degree N"),
#         titlefont = font(10),
#         xticks = 1:K
#         )
# end 

function make_rhs_plot(A, B)
    BenchmarkTools.DEFAULT_PARAMETERS.samples = 15

    K = B - A + 1

    ratio_times = Float64[]

    for N in A:B

        rd = RefElemData(Tet(), N)

        # Create interpolation matrix from Fmask node ordering to quadrature node ordering
        (; r, s, Fmask) = rd
        Fmask = reshape(Fmask, :, 4)

        # recreate RefElemData with nodal points instead of a quadrature rule
        rf, sf = rd.r[Fmask[:,1]], rd.t[Fmask[:,1]]
        rd = RefElemData(Tet(), N; quad_rule_face = (rf, sf, ones(length(rf))))
        md = MeshData(uniform_mesh(rd.element_type, 2), rd;               
                    is_periodic=true)

        (; x, y, z) = md
        u0 = @. sin(pi * x) * sin(pi * y) * sin(pi * z)

        Dr = BernsteinDerivativeMatrix_3D_r(N)
        Ds = BernsteinDerivativeMatrix_3D_s(N)
        Dt = BernsteinDerivativeMatrix_3D_t(N)
        LIFT = BernsteinLift{Float64}(N)

        cache = (; uM = md.x[rd.Fmask, :], interface_flux = md.x[rd.Fmask, :], 
        dudr = similar(md.x), duds = similar(md.x), dudt = similar(md.x))

        # Combine parameters
        params = (; rd, md, Dr, Ds, Dt, LIFT, cache)

        time1 = @benchmark bernstein_rhs_matvec!($(similar(u0)), $(u0), $(params), 0)

        (; Dr, Ds, Dt) = rd
        LIFT = get_nodal_lift(N)

        params = (; rd, md, Dr, Ds, Dt, LIFT, cache)

        time2 = @benchmark naive_nodal_rhs_matmul!($(similar(u0)), $(u0), $(params), 0)

        push!(ratio_times, minimum(time2).time/minimum(time1).time)
    end

    plot(bar(1:K, ratio_times), 
        legend = false, 
        title = "Speedup of Bernstein over nodal DG, K = 2, min times over $(BenchmarkTools.DEFAULT_PARAMETERS.samples) samples",
        yaxis = ("Time (Naive Nodal) / Time (Not-MT Bernstein)"),
        xaxis = ("Degree N"),
        titlefont = font(10),
        xticks = 1:K
        )
end 

make_rhs_plot(1, 15)

@benchmark run_nmt_bernstein_dg(15, 2, $(get_bernstein_vandermonde(15)))
@benchmark run_naive_nodal_dg(15, 2, $(get_nodal_lift(15)))

using Profile
LIFT = get_bernstein_vandermonde(15)
@profile run_nmt_bernstein_dg(15, 2, LIFT)

using ProfileView
VSCodeServer.@profview run_nmt_bernstein_dg(15, 2, LIFT)

