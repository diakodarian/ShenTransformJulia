#! /usr/bin/julia
#=
Created on Fri 21 Oct 10:23:35 2016

1D Poisson solver with Dirichlet and Neumann BCs.
=#

using Base.Test
include("shentransformTypes.jl")
# using PyCall
# import IJulia
# using PyPlot

function solveDirichlet1D(f_hat, U_hat)
    N = length(f_hat)
    A = zeros(Float64, N-2, N-2)
    k = collect(0:N-3)
    for i = 1:N-2
        A[i,i] = -2.pi*(k[i]+1.).*(k[i]+2.)
    end
    for i = 1:N-2
        for j = (i+2):2:N-2
            A[i, j] = -4.pi*(k[i]+1.)
        end
    end
    U_hat[1:end-2] = A\f_hat[1:end-2]
    U_hat
end
function solveNeumann1D(f_hat, U_hat)
    N = length(f_hat)
    A = zeros(Float64, N-2, N-2)
    k = collect(Float64, 0:N-3)
    for i = 1:N-2
        A[i,i] = (-2.pi*(k[i]+1.).*k[i].^2)./(k[i]+2.)
    end
    for i = 1:N-2
        for j = (i+2):2:N-2
            A[i, j] = -4.pi*k[j].^2 .*(k[i]+1.)./(k[i]+2.).^2
        end
    end

    U_hat[1:end-2] = A[2:end, 1:end]\f_hat[2:end-2]
    U_hat
end
function poisson1d(N)

    ff = ["GC", "GL"]
    U = zeros(Float64, N)
    U_hat = zeros(Float64, N)
    V = similar(U)
    f, f_hat = similar(U), similar(U)
    x, w = similar(U), similar(U)
    for (j, F) in enumerate([Dirichlet{GC}(N), Dirichlet{GL}(N)])
        x, w = NodesWeights(F, x, w)
        U = ((1.0 - x.^2).^2).*cos(pi*x).*(x-0.25).^2
        f = -2pi*(2.(x-0.25).*(1-x.^2).^2 - 4.((x-0.25).^2).*x.*(1-x.^2)).*sin(pi*x)-
            pi^2*((x-0.25).^2).*((1-x.^2).^2).*cos(pi*x)+
            ((8.x.^2-4.(1-x.^2)).*(x-0.25).^2 - 16.x.*(1-x.^2).*(x-0.25)+2.((1-x.^2).^2)).*cos(pi*x)

        f_hat = fastShenScalar(F, f, f_hat)
        U_hat = solveDirichlet1D(f_hat, U_hat)
        V = ifst(F, U_hat, V)

        @test isapprox(U, V)
        println("Solving Poisson equation with Dirichlet basis for ", ff[j], " nodes succeeded." )
        # fig = figure(figsize=(8,6))
        # subplot(211)
        # plot(x, U)
        # ylabel("U")
        # subplot(212)
        # plot(x, V)
        # ylabel("V")
    end
    for (j, F) in enumerate([Neumann{GC}(N), Neumann{GL}(N)])
        x, w = NodesWeights(F, x, w)
        U = x.- (1./3.)*x.^3
        f = -2.*x

        f_hat = fastShenScalar(F, f, f_hat)
        U_hat = solveNeumann1D(f_hat, U_hat)
        V = ifst(F, U_hat, V)
        @test isapprox(U, V)
        println("Solving Poisson equation with Neumann basis for ", ff[j], " nodes succeeded." )
    end
end

N = 60
poisson1d(N)
