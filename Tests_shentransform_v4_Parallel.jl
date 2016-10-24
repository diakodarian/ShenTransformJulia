include("shentransform_v4_Parallel.jl")
using Base.Test
using Compat

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
#-------------------------------------------------------------------------------
#   Tests: Shen transforms 3D
#-------------------------------------------------------------------------------
function tests(N)
    axis = 3
    z = zeros(Float64, N)
    w = similar(z)
    ff = ["GC", "GL"]
    ff_R = ["ND + GC", "ND + GL", "DN + GC", "DN + GL"]
    ff_B = ["DB + GC", "DB + GL", "ND + GC", "ND + GL"]
    # Chebyshev
    for F in [Chebyshev{GC}(), Chebyshev{GL}()]
        z, w = NodesWeights(F, z, w)
        x = collect(0:N-1)*2*pi/N
        X = Array{Float64}(N, N, N, 3)
        for (i, Xi) in enumerate(ndgrid(x, x, z)) X[view(i)...] = Xi end
        U = similar(X)
        U[view(1)...] = sin(X(3)).*cos(X(1)).*cos(X(2))
        U[view(2)...] = -cos(X(3)).*sin(X(1)).*cos(X(2))
        U[view(3)...] = 1. - X(3).^2
        V, U_hat = similar(U),  similar(U)

        U_hat[view(3)...] = fct(F, U(3), U_hat(3));
        V[view(3)...]  = ifct(F, U_hat(3), V(3));
        @test isapprox(U(3), V(3))
        U_hat[view(3)...] = -2.*X(3)
        V[view(3)...] = fastChebDerivative(F, U(3), V(3))
        @test isapprox(U_hat(3), V(3))
        println("Test: Chebyshev transform for ", F, " succeeded.")
    end
    # Dirichlet
    for (j, F) in enumerate([Dirichlet{GC}(N), Dirichlet{GL}(N)])
        z, w = NodesWeights(F, z, w)
        x = collect(0:N-1)*2*pi/N
        X = Array{Float64}(N, N, N, 3)
        for (i, Xi) in enumerate(ndgrid(x, x, z)) X[view(i)...] = Xi end
        U = similar(X)
        U[view(1)...] = sin(X(3)).*cos(X(1)).*cos(X(2))
        U[view(2)...] = -cos(X(3)).*sin(X(1)).*cos(X(2))
        U[view(3)...] = 1. - X(3).^2
        V, U_hat = similar(U),  similar(U)
        U_hat[view(3)...] = fst(F, U(3), U_hat(3));
        V[view(3)...]  = ifst(F, U_hat(3), V(3));
        @test isapprox(U(3), V(3))

        U_hat[view(3)...] = -2.*X(3)
        V[view(3)...] = 0.0
        V[view(3)...] = fastShenDerivative(F, U(3), V(3))
        @test isapprox(U_hat(3), V(3))
        println("Test: Dirichlet transform ", ff[j]," succeeded.")
    end
    # General Dirichlet
    a = -2.0; b = 2.0;
    for (j,F) in enumerate([GeneralDirichlet{GC}(a, b, N), GeneralDirichlet{GL}(a, b, N)])
        z, w = NodesWeights(F, z, w)
        x = collect(0:N-1)*2*pi/N
        X = Array{Float64}(N, N, N, 3)
        for (i, Xi) in enumerate(ndgrid(x, x, z)) X[view(i)...] = Xi end
        U = similar(X)
        U[view(1)...] = sin(X(3)).*cos(X(1)).*cos(X(2))
        U[view(2)...] = -cos(X(3)).*sin(X(1)).*cos(X(2))
        U[view(3)...] = -2.+10.*X(3).^2-8.*X(3).^4 +2.0*(-3.*X(3)+4.*X(3).^3)
        V, U_hat = similar(U),  similar(U)
        U_hat[view(3)...] = fst(F, U(3), U_hat(3));
        V[view(3)...]  = ifst(F, U_hat(3), V(3));
        @test isapprox(U(3), V(3))

        U_hat[view(3)...] = 20.*X(3)-32.*X(3).^3-6.0+24.*X(3).^2
        V[view(3)...] = 0.0
        V[view(3)...] = fastShenDerivative(F, U(3), V(3))
        @test isapprox(U_hat(3), V(3))
        println("Test: General Dirichlet transform ", ff[j]," succeeded.")
    end
    # Neumann
    i = 1
    for (j,F) in enumerate([Neumann{GC}(N), Neumann{GL}(N)])
        z, w = NodesWeights(F, z, w)
        x = collect(0:N-1)*2*pi/N
        X = Array{Float64}(N, N, N,3)
        for (i, Xi) in enumerate(ndgrid(x, x, z)) X[view(i)...] = Xi end
        U, V, U_hat = similar(X),  similar(X),  similar(X)
        U[view(1)...] = sin(X(3)).*cos(X(1)).*cos(X(2))
        U[view(2)...] = -cos(X(3)).*sin(X(1)).*cos(X(2))
        U[view(3)...] = X(3) -(1./3.)*X(3).^3

        U_hat[view(3)...] = fst(F, U(3), U_hat(3));
        V[view(3)...]  = ifst(F, U_hat(3), V(3));
        @test isapprox(U(3), V(3))

        U_hat[view(3)...] = 1.0-X(3).^2
        V[view(3)...] = 0.0
        V[view(3)...] = fastShenDerivative(F, U(3), V(3))
        @test isapprox(U_hat(3), V(3))
        println("Test: Neumann transform for ", ff[j], " succeeded.")
    end
    # Robin
    jj = 0
    for BC in RobinBC
        for (j, F) in enumerate([Robin{GC}(BC, N), Robin{GL}(BC, N)])
            z, w = NodesWeights(F, z, w)
            x = collect(0:N-1)*2*pi/N
            X = Array{Float64}(N, N, N,3)
            for (i, Xi) in enumerate(ndgrid(x, x, z)) X[view(i)...] = Xi end
            U, V, U_hat = similar(X),  similar(X),  similar(X)
            U[view(1)...] = sin(X(3)).*cos(X(1)).*cos(X(2))
            U[view(2)...] = -cos(X(3)).*sin(X(1)).*cos(X(2))
            if eval(BC) == "ND"
                U[view(3)...] = X(3)-(8./13.)*(-1. + 2*X(3).^2) - (5./13.)*(-3*X(3) + 4*X(3).^3);
            elseif eval(BC) == "DN"
                U[view(3)...] = X(3) + (8./13.)*(-1. + 2.*X(3).^2) - (5./13.)*(-3*X(3) + 4*X(3).^3);
            end
            U_hat[view(3)...] = fst(F, U(3), U_hat(3));
            V[view(3)...]  = ifst(F, U_hat(3), V(3));
            @test isapprox(U(3), V(3))

            if eval(BC) == "ND"
                U_hat[view(3)...] = (28./13.) - (32./13.)*X(3) -(60./13.)*X(3).^2;
            elseif eval(BC) == "DN"
                U_hat[view(3)...] = (28./13.) + (32./13.)*X(3) -(60./13.)*X(3).^2;
            end
            V[view(3)...] = 0.0
            V[view(3)...] = fastShenDerivative(F, U(3), V(3))
            @test isapprox(U_hat(3), V(3))
            println("Test: Robin transform for ", ff_R[j+jj], " succeeded.")
        end
        jj+=2
    end
    # Biharmonic
    jj = 0
    for BiharmonicBC in BiharmonicBCsymbols
        for (j,F) in enumerate([Biharmonic{GC}(BiharmonicBC, N), Biharmonic{GL}(BiharmonicBC, N)])
            z, w = NodesWeights(F, z, w)
            x = collect(0:N-1)*2*pi/N
            X = Array{Float64}(N, N, N,3)
            for (i, Xi) in enumerate(ndgrid(x, x, z)) X[view(i)...] = Xi end
            U, V, U_hat = similar(X),  similar(X),  similar(X)
            U[view(1)...] = sin(X(3)).*cos(X(1)).*cos(X(2))
            U[view(2)...] = -cos(X(3)).*sin(X(1)).*cos(X(2))
            if eval(BiharmonicBC) == "DB"
                U[view(3)...] = X(3) -(3./2.)*(4.*X(3).^3 - 3.*X(3))+(1./2.)*(16.*X(3).^5 -20.*X(3).^3 +5.*X(3))
            elseif eval(BiharmonicBC) == "NB"
                U[view(3)...] = -1. + 2.*X(3).^2 - (2./5.)*(1. - 8.*X(3).^2 + 8.*X(3).^4) +
                (1./15.)*(-1. + 18.*X(3).^2 - 48.*X(3).^4 + 32.*X(3).^6)
            end
            U_hat[view(3)...] = fst(F, U(3), U_hat(3));
            V[view(3)...]  = ifst(F, U_hat(3), V(3));
            @test isapprox(U(3), V(3))

            if eval(BiharmonicBC) == "DB"
                U_hat[view(3)...] = 8. - 48.*X(3).^2 +40.*X(3).^4;
            elseif eval(BiharmonicBC) == "NB"
                U_hat[view(3)...] = (64./5.)*X(3) - (128./5.)*X(3).^3 +(64./5.)*X(3).^5;
            end
            V[view(3)...] = 0.0
            V[view(3)...] = fastShenDerivative(F, U(3), V(3))
            @test isapprox(U_hat(3), V(3))

            println("Test: Biharmonic transform for  ", ff_B[j+jj], " succeeded.")
        end
        jj+=2
    end
end
#-------------------------------------------------------------------------------
#   Tests: Shen transforms 3D with FFT in x and y directions
#-------------------------------------------------------------------------------
function Spectraltests(n)
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
    V = similar(U)
    # Complex vectors
    U_hat = Array{Complex{Float64}}(cvector_shape)

    z = zeros(Float64, N[3])
    w = similar(z)

    ff = ["GC", "GL"]
    ff_R = ["ND + GC", "ND + GL", "DN + GC", "DN + GL"]
    ff_B = ["DB + GC", "DB + GL", "ND + GC", "ND + GL"]
    # Chebyshev
    for (j, F) in enumerate([Chebyshev{GC}(), Chebyshev{GL}()])
        # z, w = NodesWeights(F, z, w)
        # x = collect(0:N[1]-1)*L[1]/N[1]
        # y = collect(0:N[2]-1)*L[2]/N[2]
        #
        # X = Array{Float64}(tuple(push!([rshape...], 3)...))
        # for (i, Xi) in enumerate(ndgrid(x, y, z[rank*N[3]÷num_processes+1:(rank+1)*N[3]÷num_processes])) X[view(i)...] = Xi end
        X = get_local_mesh(FFT, F)
        U[view(1)...] = sin(X(3)).*cos(X(1)).*cos(X(2))
        U[view(2)...] = -cos(X(3)).*sin(X(1)).*cos(X(2))
        U[view(3)...] = (1. - X(3).^2).*sin(X(1)).*cos(X(2))

        U_hat[view(3)...] = FCT(FFT, F, U(3), U_hat(3));
        V[view(3)...]  = IFCT(FFT, F, U_hat(3), V(3));

        eu = MPI.Reduce(sumabs2(U(3)), MPI.SUM, 0, comm)
        ev = MPI.Reduce(sumabs2(V(3)), MPI.SUM, 0, comm)
        # U[:,:,:] = MPI.Gather(U[:,:,:], 0, comm)
        # V[:,:,:] = MPI.Gather(V[:,:,:], 0, comm)
        if rank == 0
            println(" U(3): ", eu, " V(3): ", ev)
            println("Test: Chebyshev transform for ", ff[j], " succeeded.")
        end
    end
    # Dirichlet
    for (j, F) in enumerate([Dirichlet{GC}(N, comm), Dirichlet{GL}(N, comm)])
        X = get_local_mesh(FFT, F)
        U[view(1)...] = sin(X(3)).*cos(X(1)).*cos(X(2))
        U[view(2)...] = -cos(X(3)).*sin(X(1)).*cos(X(2))
        U[view(3)...] = (1. - X(3).^2).*sin(X(1)).*cos(X(2))

        U_hat[view(3)...] = FST(FFT, F, U(3), U_hat(3));
        V[view(3)...]  = IFST(FFT, F, U_hat(3), V(3));
        #@test isapprox(U(3), V(3))
        eu = MPI.Reduce(sumabs2(U(3)), MPI.SUM, 0, comm)
        ev = MPI.Reduce(sumabs2(V(3)), MPI.SUM, 0, comm)
        if rank == 0
            println(" U(3): ", eu, " V(3): ", ev)
            println("Test: Dirichlet transform for ", ff[j]," succeeded.")
        end
    end
    # GeneralDirichlet
    a = -2.0; b = 2.0;
    for (j, F) in enumerate([GeneralDirichlet{GC}(a, b, N, comm), GeneralDirichlet{GL}(a, b, N, comm)])
        X = get_local_mesh(FFT, F)
        U[view(1)...] = sin(X(3)).*cos(X(1)).*cos(X(2))
        U[view(2)...] = -cos(X(3)).*sin(X(1)).*cos(X(2))
        U[view(3)...] = (-2.+10.*X(3).^2-8.*X(3).^4 +2.0*(-3.*X(3)+4.*X(3).^3)).*sin(X(1)).*cos(X(2))

        U_hat[view(3)...] = FST(FFT, F, U(3), U_hat(3));
        V[view(3)...]  = IFST(FFT, F, U_hat(3), V(3));
        # @test isapprox(U(3), V(3))
        eu = MPI.Reduce(sumabs2(U(3)), MPI.SUM, 0, comm)
        ev = MPI.Reduce(sumabs2(V(3)), MPI.SUM, 0, comm)
        if rank == 0
            println(" U(3): ", eu, " V(3): ", ev)
            println("Test: GeneralDirichlet transform for ", ff[j]," succeeded.")
        end
    end
    # Neumann
    for (j, F) in enumerate([Neumann{GC}(N, comm), Neumann{GL}(N, comm)])
        X = get_local_mesh(FFT, F)
        U[view(1)...] = sin(X(3)).*cos(X(1)).*cos(X(2))
        U[view(2)...] = -cos(X(3)).*sin(X(1)).*cos(X(2))
        U[view(3)...] = (X(3) - (1./3.)X(3).^3).*sin(X(1)).*cos(X(2))

        U_hat[view(3)...] = FST(FFT, F, U(3), U_hat(3));
        V[view(3)...]  = IFST(FFT, F, U_hat(3), V(3));
        # @test isapprox(U(3), V(3))
        eu = MPI.Reduce(sumabs2(U(3)), MPI.SUM, 0, comm)
        ev = MPI.Reduce(sumabs2(V(3)), MPI.SUM, 0, comm)
        if rank == 0
            println(" U(3): ", eu, " V(3): ", ev)
            println("Test: Neumann transform for ", ff[j]," succeeded.")
        end
    end
    # Robin
    jj = 0
    for BC in RobinBC
        for (j, F) in enumerate([Robin{GC}(BC, N, comm), Robin{GL}(BC, N, comm)])
            X = get_local_mesh(FFT, F)
            U[view(1)...] = sin(X(3)).*cos(X(1)).*cos(X(2))
            U[view(2)...] = -cos(X(3)).*sin(X(1)).*cos(X(2))
            if eval(BC) == "ND"
                U[view(3)...] = (X(3)-(8./13.)*(-1. + 2*X(3).^2) - (5./13.)*(-3*X(3) + 4*X(3).^3)).*sin(X(1)).*cos(X(2))
            elseif eval(BC) == "DN"
                U[view(3)...] = (X(3) + (8./13.)*(-1. + 2.*X(3).^2) - (5./13.)*(-3*X(3) + 4*X(3).^3)).*sin(X(1)).*cos(X(2))
            end
            U_hat[view(3)...] = FST(FFT, F, U(3), U_hat(3));
            V[view(3)...]  = IFST(FFT, F, U_hat(3), V(3));
            # @test isapprox(U(3), V(3))
            eu = MPI.Reduce(sumabs2(U(3)), MPI.SUM, 0, comm)
            ev = MPI.Reduce(sumabs2(V(3)), MPI.SUM, 0, comm)
            if rank == 0
                println(" U(3): ", eu, " V(3): ", ev)
                println("Test: Robin transform for ", ff_R[j+jj], " succeeded.")
            end
        end
        jj+=2
    end
    # Biharmonic
    jj = 0
    for BiharmonicBC in BiharmonicBCsymbols
        for (j, F) in enumerate([Biharmonic{GC}(BiharmonicBC, N, comm), Biharmonic{GL}(BiharmonicBC, N, comm)])
            X = get_local_mesh(FFT, F)
            U[view(1)...] = sin(X(3)).*cos(X(1)).*cos(X(2))
            U[view(2)...] = -cos(X(3)).*sin(X(1)).*cos(X(2))
            if eval(BiharmonicBC) == "DB"
                U[view(3)...] = (X(3) -(3./2.)*(4.*X(3).^3 - 3.*X(3))+(1./2.)*(16.*X(3).^5 -20.*X(3).^3 +5.*X(3))).*sin(X(1)).*cos(X(2))
            elseif eval(BiharmonicBC) == "NB"
                U[view(3)...] = (-1. + 2.*X(3).^2 - (2./5.)*(1. - 8.*X(3).^2 + 8.*X(3).^4) +
                (1./15.)*(-1. + 18.*X(3).^2 - 48.*X(3).^4 + 32.*X(3).^6)).*sin(X(1)).*cos(X(2))
            end
            U_hat[view(3)...] = FST(FFT, F, U(3), U_hat(3));
            V[view(3)...]  = IFST(FFT, F, U_hat(3), V(3));
            # @test isapprox(U(3), V(3))
            eu = MPI.Reduce(sumabs2(U(3)), MPI.SUM, 0, comm)
            ev = MPI.Reduce(sumabs2(V(3)), MPI.SUM, 0, comm)
            if rank == 0
                println(" U(3): ", eu, " V(3): ", ev)
                println("Test: Biharmonic transform for  ", ff_B[j+jj], " succeeded.")
            end
        end
        jj+=2
    end
    MPI.Finalize()
end

n = 2^6;
BC1 = "ND"; BC2 = "DN";
sym1 = :BC1
sym2 = :BC2
RobinBC = [sym1, sym2]

BiharmBC1 = "DB"; BiharmBC2 = "NB";
symbol1 = :BiharmBC1
symbol2 = :BiharmBC2
BiharmonicBCsymbols = [symbol1, symbol2]

#tests(n)
Spectraltests(n)
