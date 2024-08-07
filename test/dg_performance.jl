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

    @inbounds Threads.@threads for e in axes(du, 2)
        mul!(view(dudr, :, e), Dr, view(u, :, e))
        mul!(view(duds, :, e), Ds, view(u, :, e))
        mul!(view(dudt, :, e), Dt, view(u, :, e))

        threaded_mul!(view(du, :, e), LIFT, view(interface_flux, :, e), Threads.threadid())

        for i in axes(du, 1)
            du[i, e] += md.rxJ[1, e] * dudr[i, e] + 
                        md.sxJ[1, e] * duds[i, e] + 
                        md.txJ[1, e] * dudt[i, e]
            du[i, e] = -du[i, e] / md.J[1, e]
        end
    end
end

function run_bernstein_dg(N, K)

    rd = RefElemData(Tet(), N)

    (; r, s, Fmask) = rd
    Fmask = reshape(Fmask, :, 4)
    rf, sf = rd.r[Fmask[:,1]], rd.t[Fmask[:,1]]

    rd = RefElemData(Tet(), N; quad_rule_face = (rf, sf, ones(length(rf))))
    md = MeshData(uniform_mesh(rd.element_type, K), rd;               
                is_periodic=true)
                
    # Problem setup
    tspan = (0.0, 1.0)
    (; x, y, z) = md
    u0 = @. sin(pi * x) * sin(pi * y) * sin(pi * z)


    # Convert initial conditions to Bernstein coefficients
    (; r, s, t) = rd
    vande, _ = bernstein_basis(Tet(), N, r, s, t)
    modal_u0 = vande \ u0

    # Initialize operators
    Dr = BernsteinDerivativeMatrix_3D_r(N)
    Ds = BernsteinDerivativeMatrix_3D_s(N)
    Dt = BernsteinDerivativeMatrix_3D_t(N)
    LIFT = MultithreadedBernsteinLift(N, Threads.nthreads())

    # Cache temporary arrays (values are initialized to get the right dimensions)
    cache = (; uM = md.x[rd.Fmask, :], interface_flux = md.x[rd.Fmask, :], 
            dudr = similar(md.x), duds = similar(md.x), dudt = similar(md.x))

    # Combine parameters
    params = (; rd, md, Dr, Ds, Dt, LIFT, cache)

    # Solve ODE system
    ode = ODEProblem(bernstein_rhs_matvec!, modal_u0, tspan, params)
    sol = solve(ode, Tsit5(), saveat=LinRange(tspan..., 25))

    # Convert Bernstein coefficients back to point evaluations
    u = vande * sol.u[end]
end

function nodal_rhs_matmul!(du, u, params, t)
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

function get_nodal_lift(N)
    rd = RefElemData(Tet(), N)
    rtri, stri = nodes(Tri(), N)
    rfq, sfq, wfq = quad_nodes(Tri(), rd.N)
    Vq_face = vandermonde(Tri(), rd.N, rfq, sfq) / vandermonde(Tri(), rd.N, rtri, stri)

    nodal_LIFT = rd.LIFT * kron(I(4), Vq_face)

    return nodal_LIFT
end

function run_nodal_dg(N, K, lift)
    rd = RefElemData(Tet(), N)

    # Create interpolation matrix from Fmask node ordering to quadrature node ordering
    (; r, s, Fmask) = rd
    Fmask = reshape(Fmask, :, 4)

    # recreate RefElemData with nodal points instead of a quadrature rule
    rf, sf = rd.r[Fmask[:,1]], rd.t[Fmask[:,1]]
    rd = RefElemData(Tet(), N; quad_rule_face = (rf, sf, ones(length(rf))))
    md = MeshData(uniform_mesh(rd.element_type, K), rd;               
                is_periodic=true)

    # Problem setup
    tspan = (0.0, 1.0)
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
    ode = ODEProblem(nodal_rhs_matmul!, u0, tspan, params)
    sol = solve(ode, Tsit5(), saveat=LinRange(tspan..., 25))

    u = sol.u[end]
end

function make_plot(K)
    BenchmarkTools.DEFAULT_PARAMETERS.samples = 10

    ratio_times = Float64[]

    for N in 1:K
        time1 = @benchmark run_bernstein_dg($N, 2)

        time2 = @benchmark run_nodal_dg($N, 2, $(get_nodal_lift(N)))

        push!(ratio_times, minimum(time2).time/minimum(time1).time)
    end

    plot(bar(1:K, ratio_times), 
        legend = false, 
        title = "Speedup of Bernstein over nodal DG, K = 2, min times over 10 samples",
        yaxis = ("Time (Nodal) / Time (Bernstein)"),
        xaxis = ("Degree N"),
        titlefont = font(10),
        xticks = 1:K
        )
end 

make_plot(10)