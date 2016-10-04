include("TDMA.jl")
include("PDMA.jl")
using TDMA
using PDMA
#-----------------------------------------------------
#      Discrete cosine transform from LAPACK
#-----------------------------------------------------
function dct_type(x, i::Int, axis::Int)
    if i == 1
        FFTW.r2r(x, FFTW.REDFT00, (axis, ))
    elseif i == 2
        FFTW.r2r(x, FFTW.REDFT10, (axis, ))
    elseif i == 3
        FFTW.r2r(x, FFTW.REDFT01, (axis, ))
    elseif i == 4
        FFTW.r2r(x, FFTW.REDFT11, (axis, ))
    end
end
function dct(x, i::Int, axis::Int=3)
    if typeof(x)==Complex{Float64}
        xreal = dct_type(real(x), i, axis)
        ximag = dct_type(imag(x), i, axis)
        return xreal + ximag*1.0im
    else
        return dct_type(x, i, axis)
    end
end
#----------------------ooo----------------------------
#-----------------------------------------------------
#       Abstract type for all transforms
#-----------------------------------------------------
abstract SpecTransf
#----------------------ooo----------------------------
#-----------------------------------------------------
#      Functions needed for all transforms
#-----------------------------------------------------
function wavenumbers3D{T<:Int}(N::T)
    ky = fftfreq(N, 1./N)
    #kz = kx[1:(N÷2+1)]; kz[end] *= -1
    kx = collect(0:N-3)
    K = Array{Float64}(N, N, N-2, 3)
    for (i, Ki) in enumerate(ndgrid(ky, ky, kx)) K[view(i)...] = Ki end
    return K
end
function wavenumbers{T<:Int}(N::T)
    return collect(0:N-3)
end
function Biharmonicwavenumbers{T<:Int}(N::T)
    return collect(0:N-5)
end
function nodesWeights{T<:SpecTransf, S}(F::T, x::Array{S, 1}, w::Array{S, 1})
    if F.quad == "GC"
        points, weights = chebyshevGaussNodesWeights(x, w)
    elseif F.quad == "GL"
        points, weights = chebyshevGaussLobattoNodesWeights(x, w)
    end
    return points, weights
end
function chebyshevGaussNodesWeights{T, S}(x::Array{T, 1}, w::Array{S, 1})
    N = length(x)
    @fastmath for i = 1:N
        @inbounds x[i] = cos((2.0*i-1.0)*pi/(2.0*N))
    end
    w[1:end] = pi/N
    return x, w
end
function chebyshevGaussLobattoNodesWeights{T, S}(x::Array{T, 1}, w::Array{S, 1})
    N = length(x)
    @fastmath for i = 1:N
        @inbounds x[i] = cos((i-1.0)*pi/(N-1))
    end
    w[1] = w[end] = pi/(2*(N-1))
    w[2:end-1] = pi/(N-1)
    return x, w
end
#----------------------ooo----------------------------
#-----------------------------------------------------
#       Chebyshev transforms
#-----------------------------------------------------
type Chebyshev <: SpecTransf
    axis::Int64
    quad::ASCIIString
end
function fastChebScalar{T<:SpecTransf}(F::T, fj)
    # Fast Chebyshev scalar product.
    N = last(size(fj))
    if F.quad == "GC"
        return dct(fj, 2, F.axis)*pi/(2.*N)
    elseif F.quad == "GL"
        return dct(fj, 1, F.axis)*pi/(2.*(N-1))
    end
end
function fct{T<:SpecTransf}(F::T, fj, fk)
    # Fast Chebyshev transform.
    N = last(size(fj))
    if F.quad == "GC"
        fk = dct(fj, 2, F.axis)
        fk /= N
        fk[:, :, 1] /= 2
    elseif F.quad == "GL"
        fk = dct(fj, 1, F.axis)/(N-1)
        fk[:,:,1] /= 2.
        fk[:,:,end] /= 2.
    end
    return fk
end
function ifct{T<:SpecTransf}(F::T, fk, fj)
    # Inverse fast Chebyshev transform.
    if F.quad == "GC"
        fj = 0.5*dct(fk, 3, F.axis)
        fj += 0.5*fk[1]
    elseif F.quad == "GL"
        N = last(size(fk))
        fj = 0.5*dct(fk, 1, F.axis)
        fj += 0.5*fk[1]
        for k in 1:div((N+1),2)
            fj[:,:,(2*k-1)] += 0.5*fk[:, :, end]
        end
        for k in 1:div(N,2)
            fj[:,:,2*k] -= 0.5*fk[:, :, end]
        end
    end
    return fj
end
function chebDerivativeCoefficients{T<:SpecTransf}(F::T, fk, fl)
    N = length(fk)
    fl[end] = 0.0
    fl[end-1] = 2*(N-1)*fk[end]
    for k in N-2:-1:2
        fl[k] = 2*k*fk[k+1]+fl[k+2]
    end
    fl[1] = fk[2] + 0.5*fl[3]
    return fl
end
function chebDerivativeCoefficients_3D{T<:SpecTransf}(F::T, fk, fl)
    for i in 1:size(fk,1)
        for j in 1:size(fk,2)
            fl[i,j,1:end] = chebDerivativeCoefficients(F, fk[i,j,1:end], fl[i,j,1:end])
        end
    end
    return fl
end
function fastChebDerivative(F::Chebyshev, fj, fd)
    # Compute derivative of fj at the same points.
    fk = similar(fj)
    fkd = similar(fj)
    fk = fct(F, fj, fk)
    fkd = chebDerivativeCoefficients_3D(F, fk, fkd)
    fd  = ifct(F, fkd, fd)
    return fd
end
#----------------------ooo----------------------------
#-----------------------------------------------------
#           Shen Dirichlet transform
#-----------------------------------------------------
type Dirichlet <: SpecTransf
    axis::Int64
    quad::ASCIIString
end

function fastShenScalar(F::Dirichlet, fj, fk)
    fk = fastChebScalar(F, fj)
    fk[:,:,1:end-2] -= fk[:,:,3:end]
    return fk
end
function ifst(F::Dirichlet, fk, fj)
    """Fast inverse Shen transform
    Transform needs to take into account that phi_k = T_k - T_{k+2}
    fk contains Shen coefficients in the first fk.shape[0]-2 positions
    """
    N = last(size(fj))
    if length(size(fk)) == 3
        w_hat = zeros(eltype(fk), size(fk))
        w_hat[:,:,1:end-2] = fk[:,:,1:end-2]
        w_hat[:,:,3:end] -= fk[:,:,1:end-2]
    end
    if length(size(fk)) == 1
        w_hat = zeros(eltype(fk), N)
        w_hat[1:end-2] = fk[1:end-2]
        w_hat[3:end] -= fk[1:end-2]
    end
    fj = ifct(F, w_hat, fj)
    return fj
end
function fst(F::Dirichlet, fj, fk)
    """Fast Shen transform
    """
    fk = fastShenScalar(F, fj, fk)
    N = last(size(fj))
    if F.quad == "GC"
        ck = ones(eltype(fk), N-2); ck[1] = 2
    elseif F.quad == "GL"
        ck = ones(eltype(fk), N-2); ck[1] = 2; ck[end] = 2
    end
    a = ones(eltype(fk), N-4)*(-pi/2)
    b = pi/2*(ck+1)
    c = zeros(eltype(a), N-4)
    c[:] = a
    if length(size(fk)) == 3
        fk[:,:,1:end-2] = TDMA_SymSolve3D(b,a,c,fk[:,:,1:end-2])
    elseif length(size(fk)) == 1
        fk[1:end-2] = TDMA_1D(a, b, c, fk[1:end-2])
    end
    return fk
end
function fastShenDerivative(F::Dirichlet, fj, df)
    """
    Fast derivative of fj using Shen Dirichlet basis
    """
    N = last(size(fj))
    fk = similar(fj); fk_1 = Array(eltype(fj), N, N, N-2)
    fk_0 = similar(fk_1);fk_2 = similar(fk_1)
    fk = fst(F, fj, fk)
    k = wavenumbers(N)
    for i in 1:size(fj,1)
        for j in 1:size(fj,2)
            for l in 1:size(fk_1,3)
                fk_0[i,j,l] = fk[i,j,l]*(1.-((k[l]+2)/k[l]))
            end
        end
    end
    fk_1 = chebDerivativeCoefficients_3D(F, fk_0, fk_1)

    for i in 1:size(fj,1)
        for j in 1:size(fj,2)
            for l in 1:size(fk_2,3)
                fk_2[i,j,l] = 2*fk[i,j,l]*(k[l]+2)
            end
        end
    end

    df_hat = similar(fj)
    df_hat[:] = 0.0
    for i in 1:size(fj,1)
        for j in 1:size(fj,2)
            df_hat[i,j,1:end-2] = fk_1[i,j,1:end] - cat(3, 0.0, fk_2[i,j,1:end-1])
        end
    end

    df_hat[:,:,end-1] = -fk_2[:,:,end]
    df = ifct(F, df_hat, df)
    return df
end
#----------------------ooo----------------------------
#-----------------------------------------------------
#           Shen Neumann transform
#-----------------------------------------------------
type Neumann <: SpecTransf
    axis::Int64
    quad::ASCIIString
end

function fastShenScalar(F::Neumann, fj, fk)
    """Fast Shen scalar product.
    Chebyshev transform taking into account that phi_k = T_k - (k/(k+2))**2*T_{k+2}
    Note, this is the non-normalized scalar product
    """
    # k  = wavenumbers(last(size(fj)))
    k = wavenumbers3D(last(size(fj)))
    fk = fastChebScalar(F, fj)
    println(size(k))
    #println("bølgetall: ", ((k./(k+2)).^2)[:])
    #println("fk:  ", (fk[1,1,3:end][:]).*((k./(k+2)).^2))
    fk[:,:,1:end-2] -= (fk[:,:,3:end]).*((k./(k+2)).^2)
    # for i in 1:size(fk,1)
    #     for j in 1:size(fk,2)
    #         fk[i,j,1:end-2] -= (fk[i,j,3:end][:]).*((k./(k+2)).^2)
    #     end
    # end
    return fk
end
function ifst(F::Neumann, fk, fj)
    """Fast inverse Shen scalar transform
    """
    if length(size(fk))==3
        k = wavenumbers(last(size(fk)))
        w_hat = zeros(eltype(fk), size(fk))
        w_hat[:,:,2:end-2] = fk[:,:,2:end-2]
        w_hat[:,:,4:end] -= (k[2:end]./(k[2:end]+2)).^(2.).*fk[:,:,2:end-2]
    elseif length(size(fk))==1
        k = wavenumbers(length(fk))
        w_hat = zeros(eltype(fk), length(fk))
        w_hat[2:end-2] = fk[2:end-2]
        w_hat[4:end] -= (k[2:end]./(k[2:end]+2)).^2.*fk[2:end-2]
    end
    fj = ifct(F, w_hat, fj)
    return fj
end
function fst(F::Neumann, fj, fk)
    """Fast Shen transform.
    """
    fk = fastShenScalar(F, fj, fk)
    N = last(size(fj))
    k = wavenumbers(N)
    ck = ones(eltype(k), N-3)
    if F.quad == "GL" ck[end] = 2 end # Note not the first since basis phi_0 is not included
    a = (-pi/2)*ones(eltype(fk), N-5).*(k[2:end-2]./(k[2:end-2]+2.)).^2
    b = pi/2*(1.+ck.*(k[2:end]./(k[2:end]+2.)).^4)
    c = a
    if length(size(fk)) == 3
        fk[:,:,2:end-2] = TDMA_SymSolve3D(b,a,c,fk[:,:,2:end-2])
    elseif length(size(fk)) == 1
        fk[2:end-2] = TDMA_1D(a, b, c, fk[2:end-2])
    end
    return fk
end
function fastShenDerivative(F::Neumann, fj, df)
    """
    Fast derivative of fj using Shen Neumann basis
    """
    N = last(size(fj))
    fk = similar(fj); fk_1 = Array(eltype(fj), N, N, N-2)
    fk_0 = similar(fk_1);fk_2 = similar(fk_1)
    fk = fst(F, fj, fk)
    k = wavenumbers(N)
    fk_0[:] = fk[:,:,2:end-2].*(1.0 - ( k[2:end]./(k[2:end]+2) ) )
    fk_tmp = cat(3, 0.0, fk_0)

    fk_1 = chebDerivativeCoefficients_3D(F, fk_tmp, fk_1)

    fk_2[:] = 2*fk[:,:,2:end-2].*(k[2:end].^2)./(k[2:end]+2)

    df_hat = zeros(eltype(fk), N,N,N)
    df_hat[:,:,1] = fk_1[:,:,1]
    df_hat[:,:,2:end-2] = fk_1[:,:,2:end] - cat(3, 0.0, fk_2[:,:,1:end-1])
    df_hat[:,:,end-1] = -fk_2[:,:,end]
    df = ifct(F, df_hat, df)
    return df
end
#----------------------ooo----------------------------
#-----------------------------------------------------
#           Shen Robin transform
#-----------------------------------------------------
type Robin <: SpecTransf
    axis::Int64
    quad::ASCIIString
    BC::ASCIIString
    N::Int64
    k::Array{Float64, 1}
    ak::Array{Float64, 1}
    bk::Array{Float64, 1}
    k1::Array{Float64, 1}
    ak1::Array{Float64, 1}
    bk1::Array{Float64, 1}

    function Robin(axis, quad, BC, N)
        """
        Shen basis functions given by
        phi_k = T_k + a_k*T_{k+1} + b_k*T_{k+2},
        satisfy the imposed Robin (mixed) boundary conditions for a unique set of {a_k, b_k}.
        """
        k = collect(0:N-3)
        k1 = collect(0:N-2)
        if BC == "ND"
            ak = -4*(k+1)./((k+1).^2 .+ (k+2).^2)
            ak1 = -4*(k1+1)./((k1+1).^2 .+ (k1+2).^2)
        elseif BC == "DN"
            ak = 4*(k+1)./((k+1).^2 .+ (k+2).^2)
            ak1 = 4*(k1+1)./((k1+1).^2 .+ (k1+2).^2)
        end
        bk = -((k.^2 + (k+1).^2)./((k+1).^2 .+ (k+2).^2))
        bk1 = -((k1.^2 + (k1+1).^2)./((k1+1).^2 .+ (k1+2).^2))
        new(axis, quad, BC, N, k, ak, bk, k1, ak1, bk1)
    end
end
function fastShenScalar{T<:Real}(F::Robin, fj::Array{T, 1}, fk::Array{T, 1})
    """Fast Shen scalar product
    B u_hat = sum_{j=0}{N} u_j phi_k(x_j) w_j,
    for Shen basis functions given by
    phi_k = T_k + a_k*T_{k+1} + b_k*T_{k+2}
    """
    fk = fastChebScalar(F, fj)
    fk_tmp = fk
    fk[1:end-2] = fk_tmp[1:end-2] + (F.ak).*fk_tmp[2:end-1] + (F.bk).*fk_tmp[3:end]
    return fk
end
function ifst{T<:Real}(F::Robin, fk::Array{T, 1}, fj::Array{T, 1})
    """Fast inverse Shen scalar transform for Robin BC.
    """
    w_hat = zeros(eltype(fk), length(fk))
    w_hat[1:end-2] = fk[1:end-2]
    w_hat[2:end-1] += (F.ak).*fk[1:end-2]
    w_hat[3:end]   += (F.bk).*fk[1:end-2]
    fj = ifct(F, w_hat, fj)
    return fj
end
function fst{T<:Real}(F::Robin, fj::Array{T, 1}, fk::Array{T, 1})
    """Fast Shen transform for Robin BC.
    """
    fk = fastShenScalar(F, fj, fk)
    N = length(fj)
    if F.quad == "GL"
        ck = ones(eltype(fj), N-2); ck[1] = 2
    elseif F.quad == "GC"
        ck = ones(eltype(fj), N-2); ck[1] = 2; ck[end] = 2
    end
    a = (pi/2)*(ck .+ (F.ak).^2 .+ (F.bk).^2)
    b = (pi/2)*ones(eltype(fj), N-3).*((F.ak)[1:end-1] .+ (F.ak1[2:end-1]).*(F.bk[1:end-1]))
    c = (pi/2)*ones(eltype(fj), N-4).*(F.bk[1:end-2])

    fk[1:end-2] = SymmetricalPDMA_1D(a, b, c, fk[1:end-2])
    return fk
end
#----------------------ooo----------------------------
# ----------------------------------------------------
#          Shen Biharmonic transform
# ----------------------------------------------------
type Biharmonic <: SpecTransf
    axis::Int64
    quad::ASCIIString
    BiharmonicBC::ASCIIString
    N::Int64
    k::Array{Float64, 1}
    factor1::Array{Float64, 1}
    factor2::Array{Float64, 1}

    function Biharmonic(axis, quad, BiharmonicBC, N)
        k = collect(0:N-5)
        if BiharmonicBC == "NB"
            factor1 = -2*(k.^2)./((k+2).*(k+3))
            factor2 = (k.^2).*(k+1)./((k+3).*(k+4).^2)
        elseif BiharmonicBC == "DB"
            factor1 = -2*(k+2)./(k+3)
            factor2 = (k+1)./(k+3)
        end
        new(axis, quad, BiharmonicBC, N, k, factor1, factor2)
    end
end
function fastShenScalar{T<:Real}(F::Biharmonic, fj::Array{T, 1}, fk::Array{T, 1})
    """Fast Shen scalar product.
    """
    Tk = fk
    Tk = fastChebScalar(F, fj)
    fk[:] = Tk
    fk[1:end-4] += F.factor1.*Tk[3:end-2]
    fk[1:end-4] += F.factor2.*Tk[5:end]
    fk[end-3:end] = 0.0
    return fk
end
function ifst{T<:Real}(F::Biharmonic, fk::Array{T, 1}, fj::Array{T, 1})
    """Fast inverse Shen scalar transform
    """
    w_hat = zeros(eltype(fk), length(fk))
    w_hat[1:end-4] = fk[1:end-4]
    w_hat[3:end-2] += F.factor1.*fk[1:end-4]
    w_hat[5:end]   += F.factor2.*fk[1:end-4]
    fj = ifct(F, w_hat, fj)
    return fj
end
function fst{T<:Real}(F::Biharmonic, fj::Array{T, 1}, fk::Array{T, 1})
    """Fast Shen transform .
    """
    fk = fastShenScalar(F, fj, fk)
    N = length(fj)
    N -= 4
    if F.quad == "GL"
        ck = ones(N); ck[1] = 2
    elseif F.quad == "GC"
        ck = ones(N); ck[1] = 2; ck[end] = 2
    end
    c = (ck + F.factor1.^2 + F.factor2.^2)*pi/2.
    d = (F.factor1[1:end-2] + F.factor1[3:end].*F.factor2[1:end-2])*pi/2.
    e = F.factor2[1:end-4]*pi/2.

    fk[1:end-4] = PDMA_Symsolve(c, d, e,fk[1:end-4])
    return fk
end
#----------------------ooo----------------------------

# ----------------------------------------------------
#                       Tests
# ----------------------------------------------------
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

function tests(N)

    axis = 3
    z = zeros(Float64, N)
    w = similar(z)
    # Chebyshev
    for quad in ["GC", "GL"]
        F = Chebyshev(axis, quad)
        z, w = nodesWeights(F, z, w)
        x = collect(0:N-1)*2*pi/N
        X = Array{Float64}(N, N, N,3)
        for (i, Xi) in enumerate(ndgrid(x, x, z)) X[view(i)...] = Xi end
        U, V, U_hat = similar(X),  similar(X),  similar(X)
        U[view(1)...] = sin(X(3)).*cos(X(1)).*cos(X(2))
        U[view(2)...] = -cos(X(3)).*sin(X(1)).*cos(X(2))
        U[view(3)...] = 1. - X(3).^2

        U_hat[view(3)...] = fct(F, U(3), U_hat(3));
        V[view(3)...]  = ifct(F, U_hat(3), V(3));
        @test isapprox(U(3), V(3))
        U_hat[view(3)...] = -2.*X(3)
        V[view(3)...] = fastChebDerivative(F, U(3), V(3))
        @test isapprox(U_hat(3), V(3))
        println("Test: Chebyshev transform for ", quad, " succeeded.")
    end
    # Dirichlet
    for quad in ["GL", "GC"]
        F = Dirichlet(axis, quad)
        z, w = nodesWeights(F, z, w)
        x = collect(0:N-1)*2*pi/N
        X = Array{Float64}(N, N, N,3)
        for (i, Xi) in enumerate(ndgrid(x, x, z)) X[view(i)...] = Xi end
        U, V, U_hat = similar(X),  similar(X),  similar(X)
        U[view(1)...] = sin(X(3)).*cos(X(1)).*cos(X(2))
        U[view(2)...] = -cos(X(3)).*sin(X(1)).*cos(X(2))
        U[view(3)...] = 1. - X(3).^2

        U_hat[view(3)...] = fst(F, U(3), U_hat(3));
        V[view(3)...]  = ifst(F, U_hat(3), V(3));
        @test isapprox(U(3), V(3))

        U_hat[view(3)...] = -2.*X(3)
        V[view(3)...] = 0.0
        V[view(3)...] = fastShenDerivative(F, U(3), V(3))
        @test isapprox(U_hat(3), V(3))
        println("Test: Dirichlet transform for ", quad, " succeeded.")
    end
    # Neumann
    for quad in ["GL", "GC"]
        F = Neumann(axis, quad)
        z, w = nodesWeights(F, z, w)
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
        println("Test: Neumann transform for ", quad, " succeeded.")
    end
    # # Robin
    # for BC in ["ND", "DN"]
    #     for quad in ["GL", "GC"]
    #         F = Robin(axis, quad, BC, N)
    #         x, w = nodesWeights(F, x, w)
    #         if BC == "ND"
    #             U = x-(8./13.)*(-1. + 2*x.^2) - (5./13.)*(-3*x + 4*x.^3);
    #         elseif BC == "DN"
    #             U = x + (8./13.)*(-1. + 2.*x.^2) - (5./13.)*(-3*x + 4*x.^3);
    #         end
    #         U_hat = fst(F, U, U_hat)
    #         V = ifst(F, U_hat, V)
    #         @test isapprox(U, V)
    #         println("Test: Robin transform for ", BC,"  ",  quad, " succeeded.")
    #     end
    # end
    # # Biharmonic
    # for BiharmonicBC in ["DB", "NB"]
    #     for quad in ["GL", "GC"]
    #         F = Biharmonic(axis, quad, BiharmonicBC, N)
    #         x, w = nodesWeights(F, x, w)
    #         if BiharmonicBC == "DB"
    #             U = x -(3./2.)*(4.*x.^3 - 3.*x)+(1./2.)*(16.*x.^5 -20.*x.^3 +5.*x)
    #         elseif BiharmonicBC == "NB"
    #             U = -1. + 2.*x.^2 - (2./5.)*(1. - 8.*x.^2 + 8.*x.^4) +
    #             (1./15.)*(-1. + 18.*x.^2 - 48.*x.^4 + 32.*x.^6)
    #         end
    #         U_hat = fst(F, U, U_hat)
    #         V = ifst(F, U_hat, V)
    #         @test isapprox(U, V)
    #         println("Test: Biharmonic transform for  ", BiharmonicBC, "  ", quad, " succeeded.")
    #     end
    # end
end

n = 2^3;
tests(n)
