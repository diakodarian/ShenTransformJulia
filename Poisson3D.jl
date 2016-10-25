#! /usr/bin/julia
#=
Created on Sun 23 Oct 13:12:36 2016

3D Poisson solver with Dirichlet and Neumann BCs.
=#
include("shentransform_v4_Parallel.jl")
# using shentransform_v4_Parallel
using Base.Test
using Compat
# using PyCall
# import IJulia
# using PyPlot

view(k::Int, N::Int=4) = [fill(Colon(), N-1)..., k]
@compat function call{T, N}(A::Array{T, N}, k::Int)
    @assert 1 <= k <= size(A, N)
    indices = [fill(Colon(), N-1)..., k]
    slice(A, indices...)
end
function fftfreq(n::Int, d::Real=1.0)
    val = 1.0/(n*d)
    results = Array{Int}(n)
    N = (n-1)÷2 + 1
    p1 = 0:(N-1)
    results[1:N] = p1
    p2 = -n÷2:-1
    results[N+1:end] = p2
    results * val
end
function rfftfreq(n::Int, d::Real=1.0)
    val = 1.0/(n*d)
    N = (n)÷2 + 1
    results = Array{Int}(N)
    results[1:N] = 0:(N-1)
    results * val
end
function ndgrid_fill(a, v, s, snext)
    for j = 1:length(a)
        a[j] = v[div(rem(j-1, snext), s)+1]
    end
end
function ndgrid{T}(vs::AbstractVector{T}...)
    n = length(vs)
    sz = map(length, vs)
    out = ntuple(i->Array{T}(sz), n)
    s = 1
    for i=1:n
        a = out[i]::Array
        v = vs[i]
        snext = s*size(a,i)
        ndgrid_fill(a, v, s, snext)
        s = snext
    end
    out
end

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
function poisson3d(n)
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    num_processes = MPI.Comm_size(comm)
    const N = [n, n, n]
    const L = [2pi, 2pi, 2.] # Real size of mesh
    FFT = r2c{Float64}(N, L, comm)
    # DNS shapes
    rshape = real_shape(FFT)
    rvector_shape = tuple(push!([rshape...], 3)...)

    cshape = complex_shape(FFT)
    cvector_shape = tuple(push!([cshape...], 3)...)

    # Real vectors
    U = Array{Float64}(rvector_shape)
    V, f = similar(U), similar(U)
    # Complex vectors
    U_hat = Array{Complex{Float64}}(cvector_shape)
    f_hat = similar(U_hat)

    H = zeros(Float64, N[3]-2, N[3]-2)
    A = zeros(Float64, N[3]-2, N[3]-2)
    B = zeros(Float64, N[3]-2, N[3]-2)
    ff = ["GC", "GL"]
    # Dirichlet
    for (l, F) in enumerate([Dirichlet{GC}(N, comm), Dirichlet{GL}(N, comm)])

        X = get_local_mesh(FFT, F)

        k = collect(Float64, 0:N[3]-3)
        kx = rfftfreq(N[1], 1.0/N[1])
        ky = fftfreq(N[2], 1.0/N[2])[rank*div(N[2], num_processes)+1:(rank+1)*div(N[2], num_processes)]
        K = Array{Float64}(tuple(push!([N[1]÷2+1, N[2]÷num_processes, N[3]-2], 3)...))
        for (i, Ki) in enumerate(ndgrid(kx, ky, k)) K[view(i)...] = Ki end

        if l == 2
            ck = ones(eltype(Float64), N[3]-2); ck[1] = 2.; ck[end] = 2.
        elseif l == 1
            ck = ones(eltype(Float64), N[3]-2); ck[1] = 2.
        end

        A, B = DirichletMatrices(A, B, ck)

        U[view(1)...] = sin(X(3)).*cos(X(1)).*cos(X(2))
        U[view(2)...] = -cos(X(3)).*sin(X(1)).*cos(X(2))
        U[view(3)...] = (1. - X(3).^2).*sin(X(1)).*cos(X(2))
        f[view(3)...] = -2.0*(2.-X(3).^2).*sin(X(1)).*cos(X(2))

        f_hat[view(3)...] = FSS(FFT, F, f(3), f_hat(3));

        for i in 1:size(f_hat,1)
            for j in 1:size(f_hat, 2)
                alpha = K[i,1,1,1].^2 + K[1,j,1,2].^2
                H = A - alpha*B
                U_hat[i,j,:,3] = solveDirichlet1D(f_hat[i,j,:,3], U_hat[i,j,:,3], H)
            end
        end
        V[view(3)...]  = IFST(FFT, F, U_hat(3), V(3));
        #@test isapprox(U(3), V(3))
        eu = MPI.Reduce(sumabs2(U(3)), MPI.SUM, 0, comm)
        ev = MPI.Reduce(sumabs2(V(3)), MPI.SUM, 0, comm)
        if rank == 0
            println(" U(3): ", eu, " V(3): ", ev)
            println("Solving Poisson equation with Dirichlet basis for ", ff[l], " nodes succeeded." )
        end
    end
    for (l, F) in enumerate([Neumann{GC}(N, comm), Neumann{GL}(N, comm)])
        X = get_local_mesh(FFT, F)

        k = collect(Float64, 0:N[3]-3)
        kx = rfftfreq(N[1], 1.0/N[1])
        ky = fftfreq(N[2], 1.0/N[2])[rank*div(N[2], num_processes)+1:(rank+1)*div(N[2], num_processes)]
        K = Array{Float64}(tuple(push!([N[1]÷2+1, N[2]÷num_processes, N[3]-2], 3)...))
        for (i, Ki) in enumerate(ndgrid(kx, ky, k)) K[view(i)...] = Ki end

        if l == 2
            ck = ones(eltype(Float64), N[3]-2); ck[1] = 2.; ck[end] = 2.
        elseif l == 1
            ck = ones(eltype(Float64), N[3]-2); ck[1] = 2.
        end

        A, B = NeumannMatrices(A, B, ck)

        U[view(1)...] = sin(X(3)).*cos(X(1)).*cos(X(2))
        U[view(2)...] = -cos(X(3)).*sin(X(1)).*cos(X(2))
        U[view(3)...] = (X(3) - (1./3.)X(3).^3).*sin(X(1)).*cos(X(2))
        f[view(3)...] = -2.0*(2.X(3)-(1./3.)X(3).^3).*sin(X(1)).*cos(X(2))

        f_hat[view(3)...] = FSS(FFT, F, f(3), f_hat(3))

        for i in 1:size(f_hat,1)
            for j in 1:size(f_hat, 2)
                alpha = K[i,1,1,1].^2 + K[1,j,1,2].^2
                H = A - alpha*B
                U_hat[i,j,:,3] = solveNeumann1D(f_hat[i,j,:,3], U_hat[i,j,:,3], H)
            end
        end
        V[view(3)...]  = IFST(FFT, F, U_hat(3), V(3));
        # @test isapprox(U(3), V(3))
        eu = MPI.Reduce(sumabs2(U(3)), MPI.SUM, 0, comm)
        ev = MPI.Reduce(sumabs2(V(3)), MPI.SUM, 0, comm)
        if rank == 0
            println(" U(3): ", eu, " V(3): ", ev)
            println("Solving Poisson equation with Neumann basis for ", ff[l], " nodes succeeded." )
        end
    end
    MPI.Finalize()
end

n = 60
poisson3d(n)
