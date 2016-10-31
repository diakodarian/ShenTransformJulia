#! /usr/bin/julia
#=
Created on Fri 28 Oct 21:48:02 2016

1D Helmholtz solver with Dirichlet and Neumann BCs.
=#
include("LinearAlgebraSolvers.jl")
using Base.Test
include("shentransformTypes.jl")

function DirichletMatrices(A, B, ck)
    N = last(size(A))
    k = collect(0:N-1)
    for i = 1:N
        A[i,i] = -2.pi*(k[i]+1.).*(k[i]+2.)
        B[i,i] = 0.5pi*(ck[i]+1.)
        if i < N-2
            B[i,i+2] = -0.5pi
        end
        if i > 2
            B[i, i-2] = -0.5pi
        end
    end
    for i = 1:N
        for j = (i+2):2:N
            A[i, j] = -4.pi*(k[i]+1.)
        end
    end
    A, B
end
function solveDirichlet1D(f_hat, U_hat, H)
    U_hat[1:end-2] = H\f_hat[1:end-2]
    U_hat
end
function NeumannMatrices(A, B, ck)
    N = last(size(A))
    k = collect(0:N-1)
    for i = 1:N
        A[i,i] = (-2.pi*(k[i]+1.).*k[i].^2)./(k[i]+2.)
        B[i,i] = 0.5pi*(ck[i]+  ( (k[i].^4) ./ (k[i]+2.).^4 ) )
        if i < N-2
            B[i,i+2] = -0.5pi*(k[i].^2)./(k[i]+2.).^2
        end
        if i > 2
            B[i, i-2] = -0.5pi*((k[i]-2.).^2)./(k[i].^2)
        end
    end
    for i = 1:N
        for j = (i+2):2:N
            A[i, j] = -4.pi*k[j].^2 .*(k[i]+1.)./(k[i]+2.).^2
        end
    end
    A, B
end
function solveNeumann1D(f_hat, U_hat, H)
    U_hat[1:end-2] = H[2:end, 1:end]\f_hat[2:end-2]
    U_hat
end
function Helmholtz1d(N)
    ff = ["GL", "GC"]
    # solver = "sparse"
    solver = "lu"
    U = zeros(Float64, N)
    U_hat = zeros(Float64, N)
    V = similar(U)
    f, f_hat = similar(U), similar(U)
    x, w = similar(U), similar(U)
    H = zeros(Float64, N-2, N-2)
    A = zeros(Float64, N-2, N-2)
    B = zeros(Float64, N-2, N-2)
    alpha = 2.0;

    # Dirichlet BC
    for (j, F) in enumerate([Dirichlet{GL}(N), Dirichlet{GL}(N)])
        x, w = NodesWeights(F, x, w)
        U = ((1.0 - x.^2).^2).*cos(pi*x).*(x-0.25).^2
        f = alpha*U - (-2pi*(2.(x-0.25).*(1-x.^2).^2 - 4.((x-0.25).^2).*x.*(1-x.^2)).*sin(pi*x)-
            pi^2*((x-0.25).^2).*((1-x.^2).^2).*cos(pi*x)+
            ((8.x.^2-4.(1-x.^2)).*(x-0.25).^2 - 16.x.*(1-x.^2).*(x-0.25)+2.((1-x.^2).^2)).*cos(pi*x))

        if j == 2
            ck = ones(eltype(Float64), N-2); ck[1] = 2.; ck[end] = 2.
        elseif j == 1
            ck = ones(eltype(Float64), N-2); ck[1] = 2.
        end

        A, B = DirichletMatrices(A, B, ck)

        f_hat = fastShenScalar(F, f, f_hat)
        if solver == "sparse"
            H = -A+alpha*B
            @time U_hat = solveDirichlet1D(f_hat, U_hat, H)
        elseif solver == "lu"
            M = div(N-3,2)
            Mo = div(N-4,2)
            d0 = zeros(2, M+1)
            d1 = zeros(2, M)
            d2 = zeros(2, M-1)
            L  = zeros(2, M)

            d0, d1, d2, L = LU_Helmholtz_1D(N, 0, ff[j]=="GL", sqrt(alpha), d0, d1, d2, L)
            U_hat[1:end-2] = Solve_Helmholtz_1D(N, 0, f_hat[1:end-2], U_hat[1:end-2], d0, d1, d2, L)
        end
        V = ifst(F, U_hat, V)
        @test isapprox(U, V)
        b = zeros(Float64, N-2)
        b = Mult_Helmholtz_1D(N, ff[j]=="GL", 1, alpha, U_hat[1:end-2], b)
        @test isapprox(b, f_hat[1:end-2])
        println("Solving Helmholtz equation with Dirichlet basis for ", ff[j], " nodes succeeded." )
    end
    # Neumann BC
    for (j, F) in enumerate([Neumann{GC}(N), Neumann{GL}(N)])
        x, w = NodesWeights(F, x, w)
        U = x.- (1./3.)*x.^3
        f = alpha*U+2.*x
        if j == 2
            ck = ones(eltype(Float64), N-2); ck[1] = 2.; ck[end] = 2.
        elseif j == 1
            ck = ones(eltype(Float64), N-2); ck[1] = 2.
        end

        A, B = NeumannMatrices(A, B, ck)

        f_hat = fastShenScalar(F, f, f_hat)
        if solver == "sparse"
            H = -A+alpha*B
            @time U_hat = solveNeumann1D(f_hat, U_hat, H)
        elseif solver == "lu"
            M = div(N-4,2)
            u0 = zeros(2, M+1)
            u1 = zeros(2, M)
            u2 = zeros(2, M-1)
            L  = zeros(2, M)

            u0, u1, u2, L = LU_Helmholtz_1D(N, 1, ff[j]=="GL", sqrt(alpha), u0, u1, u2, L)
            U_hat[2:end-2] = Solve_Helmholtz_1D(N, 1, f_hat[2:end-2], U_hat[2:end-2], u0, u1, u2, L)
        end
        V = ifst(F, U_hat, V)
        @test isapprox(U, V)
        # b = zeros(Float64, N-3)
        # b = Mult_Helmholtz_1D(N, ff[j]=="GL", 1, alpha, U_hat[2:end-2], b)
        # @test isapprox(b, f_hat[2:end-2])
        println("Solving Helmholtz equation with Neumann basis for ", ff[j], " nodes succeeded." )
    end
end

N = 64
Helmholtz1d(N)
