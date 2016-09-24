include("TDMA.jl")
include("PDMA.jl")
using TDMA
using PDMA
#-----------------------------------------------------
#      Discrete cosine transform from LAPACK
#-----------------------------------------------------
function dct_type{T}(x::Array{T,1}, i::Int64, axis::Int64)
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
function dct{T}(x::Array{T,1}, i::Int64, axis::Int64=1)
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
function fastChebScalar{T<:SpecTransf, S}(F::T, fj::Array{S, 1})
    # Fast Chebyshev scalar product.
    N = length(fj)
    if F.quad == "GC"
        return dct(fj, 2, F.axis)*pi/(2.*N)
    elseif F.quad == "GL"
        return dct(fj, 1, F.axis)*pi/(2.*(N-1))
    end
end
function fct{T<:SpecTransf, S}(F::T, fj::Array{S, 1}, fk::Array{S, 1})
    # Fast Chebyshev transform.
    N = length(fj)
    if F.quad == "GC"
        fk = dct(fj, 2, F.axis)/N
        fk[1] /= 2.
    elseif F.quad == "GL"
        fk = dct(fj, 1, F.axis)/(N-1)
        fk[1] /= 2.
        fk[end] /= 2.
    end
    return fk
end
function ifct{T<:SpecTransf, S}(F::T, fk::Array{S, 1}, fj::Array{S, 1})
    # Inverse fast Chebyshev transform.
    if F.quad == "GC"
        fj = 0.5*dct(fk, 3, F.axis)
        fj += 0.5*fk[1]
    elseif F.quad == "GL"
        fj = 0.5*dct(fk, 1, F.axis)
        fj += 0.5*fk[1]
        fj[1:2:end] += 0.5*fk[end]
        fj[2:2:end] -= 0.5*fk[end]
    end
    return fj
end
function chebDerivativeCoefficients{T<:SpecTransf, S}(F::T, fk::Array{S, 1}, fl::Array{S, 1})
    N = length(fk)
    fl[end] = 0.0
    fl[end-1] = 2*(N-1)*fk[end]
    for k in N-2:-1:2
        fl[k] = 2*k*fk[k+1]+fl[k+2]
    end
    fl[1] = fk[2] + 0.5*fl[3]
    return fl
end
function fastChebDerivative{T<:Real}(F::Chebyshev, fj::Array{T, 1}, fd::Array{T, 1})
    # Compute derivative of fj at the same points.
    fk = Array{T}(length(fj))
    fkd = Array{T}(length(fj))
    fk = fct(F, fj, fk)
    fkd = chebDerivativeCoefficients(F, fk, fkd)
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

function fastShenScalar{T<:Real}(F::Dirichlet, fj::Array{T, 1}, fk::Array{T, 1})
    fk = fastChebScalar(F, fj)
    fk[1:end-2] -= fk[3:end]
    return fk
end
function ifst{T<:Real}(F::Dirichlet, fk::Array{T, 1}, fj::Array{T, 1})
    """Fast inverse Shen transform
    Transform needs to take into account that phi_k = T_k - T_{k+2}
    fk contains Shen coefficients in the first fk.shape[0]-2 positions
    """
    N = length(fj)
    if length(size(fk)) == 3
        w_hat = zeros(eltype(fk), size(fj))
        w_hat[1:end-2] = fk[1:end-2]
        w_hat[2:end] -= fk[1:end-2]
    end
    if length(size(fk)) == 1
        w_hat = zeros(eltype(fk), N)
        w_hat[1:end-2] = fk[1:end-2]
        w_hat[3:end] -= fk[1:end-2]
    end
    fj = ifct(F, w_hat, fj)
    return fj
end
function fst{T<:Real}(F::Dirichlet, fj::Array{T, 1}, fk::Array{T, 1})
    """Fast Shen transform
    """
    fk = fastShenScalar(F, fj, fk)
    N = length(fj)
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
        bc = zeros(eltype(b), N-2)
        bc[:] = b
        fk[1:end-2] = TDMA_3D_complex(a, b, bc, c, fk[1:end-2])
    elseif length(size(fk)) == 1
        fk[1:end-2] = TDMA_1D(a, b, c, fk[1:end-2])
    end
    return fk
end
function fastShenDerivative{T<:Real}(F::Dirichlet, fj::Array{T, 1}, df::Array{T, 1})
    """
    Fast derivative of fj using Shen Dirichlet basis
    """
    N = length(fj)
    fk = Array{T}(N); fk_1 = Array{T}(N-2)
    fk_0 = similar(fk_1);
    fk = fst(F, fj, fk)
    k = wavenumbers(N)
    fk_0 = fk[1:end-2].*(1.-((k+2)./k))
    fk_1 = chebDerivativeCoefficients(F, fk_0, fk_1)
    fk_2 = 2*fk[1:end-2].*(k+2)

    df_hat = zeros(eltype(fj), N)
    df_hat[1:end-2] = fk_1 - vcat(0, fk_2[1:end-1])

    df_hat[end-1] = -fk_2[end]
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

function fastShenScalar{T<:Real}(F::Neumann, fj::Array{T, 1}, fk::Array{T, 1})
    """Fast Shen scalar product.
    Chebyshev transform taking into account that phi_k = T_k - (k/(k+2))**2*T_{k+2}
    Note, this is the non-normalized scalar product
    """
    k  = wavenumbers(length(fj))
    fk = fastChebScalar(F, fj)
    fk[1:end-2] -= ((k./(k+2)).^2).*fk[3:end]
    return fk
end
function ifst{T<:Real}(F::Neumann, fk::Array{T, 1}, fj::Array{T, 1})
    """Fast inverse Shen scalar transform
    """
    if length(size(fk))==3
        k = wavenumbers(size(fk))
        w_hat = zeros(eltype(fk), size(fk))
        w_hat[2:end-2] = fk[2:end-2]
        w_hat[4:end] -= (k[2:end]./(k[2:end]+2)).^2.*fk[2:end-2]
    elseif length(size(fk))==1
        k = wavenumbers(length(fk))
        w_hat = zeros(eltype(fk), length(fk))
        w_hat[2:end-2] = fk[2:end-2]
        w_hat[4:end] -= (k[2:end]./(k[2:end]+2)).^2.*fk[2:end-2]
    end
    fj = ifct(F, w_hat, fj)
    return fj
end
function fst{T<:Real}(F::Neumann, fj::Array{T, 1}, fk::Array{T, 1})
    """Fast Shen transform.
    """
    fk = fastShenScalar(F, fj, fk)
    N = length(fj)
    k = wavenumbers(N)
    ck = ones(eltype(k), N-3)
    if F.quad == "GL" ck[end] = 2 end # Note not the first since basis phi_0 is not included
    a = (-pi/2)*ones(eltype(fk), N-5).*(k[2:end-2]./(k[2:end-2]+2.)).^2
    b = pi/2*(1.+ck.*(k[2:end]./(k[2:end]+2.)).^4)
    c = a
    fk[2:end-2] = TDMA_1D(a, b, c, fk[2:end-2])
    return fk
end
function fastShenDerivative{T<:Real}(F::Neumann, fj::Array{T, 1}, df::Array{T, 1})
    """
    Fast derivative of fj using Shen Neumann basis
    """
    N = length(fj)
    fk = Array{T}(N); fk_1 = Array{T}(N-2)
    fk = fst(F, fj, fk)
    k = wavenumbers(N)
    fk_0 = fk[2:end-2].*(1.0 - ( k[2:end]./(k[2:end]+2) ) )
    fk_tmp = vcat(0, fk_0)
    fk_1 = chebDerivativeCoefficients(F, fk_tmp, fk_1)
    fk_2 = 2*fk[2:end-2].*(k[2:end].^2)./(k[2:end]+2)

    df_hat = zeros(eltype(fk), N)
    df_hat[1] = fk_1[1]
    df_hat[2:end-2] = fk_1[2:end] - vcat(0, fk_2[1:end-1])
    df_hat[end-1] = -fk_2[end]
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
end
function shenCoefficients{T<:Real}(F::Robin, k::Array{T, 1})
    """
    Shen basis functions given by
    phi_k = T_k + a_k*T_{k+1} + b_k*T_{k+2},
    satisfy the imposed Robin (mixed) boundary conditions for a unique set of {a_k, b_k}.
    """
    if F.BC == "ND"
        ak = -4*(k+1)./((k+1).^2 .+ (k+2).^2)
    elseif F.BC == "DN"
        ak = 4*(k+1)./((k+1).^2 .+ (k+2).^2)
    end
    bk = -((k.^2 + (k+1).^2)./((k+1).^2 .+ (k+2).^2))
    return ak, bk
end
function fastShenScalar{T<:Real}(F::Robin, fj::Array{T, 1}, fk::Array{T, 1})
    """Fast Shen scalar product
    B u_hat = sum_{j=0}{N} u_j phi_k(x_j) w_j,
    for Shen basis functions given by
    phi_k = T_k + a_k*T_{k+1} + b_k*T_{k+2}
    """
    fk = fastChebScalar(F, fj)
    k  = wavenumbers(length(fj))
    ak, bk = shenCoefficients(F, k)
    fk_tmp = fk
    fk[1:end-2] = fk_tmp[1:end-2] + ak.*fk_tmp[2:end-1] + bk.*fk_tmp[3:end]
    return fk
end
function ifst{T<:Real}(F::Robin, fk::Array{T, 1}, fj::Array{T, 1})
    """Fast inverse Shen scalar transform for Robin BC.
    """
    k = wavenumbers(length(fk))
    w_hat = zeros(eltype(fk), length(fk))
    ak, bk = shenCoefficients(F, k)
    w_hat[1:end-2] = fk[1:end-2]
    w_hat[2:end-1] += ak.*fk[1:end-2]
    w_hat[3:end]   += bk.*fk[1:end-2]
    fj = ifct(F, w_hat, fj)
    return fj
end
function fst{T<:Real}(F::Robin, fj::Array{T, 1}, fk::Array{T, 1})
    """Fast Shen transform for Robin BC.
    """
    fk = fastShenScalar(F, fj, fk)
    N = length(fj)
    k = wavenumbers(N)
    k1 = wavenumbers(N+1)
    ak, bk = shenCoefficients(F, k)
    ak1, bk1 = shenCoefficients(F, k1)
    if F.quad == "GL"
        ck = ones(eltype(k), N-2); ck[1] = 2
    elseif F.quad == "GC"
        ck = ones(eltype(k), N-2); ck[1] = 2; ck[end] = 2
    end
    a = (pi/2)*(ck .+ ak.^2 .+ bk.^2)
    b = (pi/2)*ones(eltype(k), N-3).*(ak[1:end-1] .+ ak1[2:end-1].*bk[1:end-1])
    c = (pi/2)*ones(eltype(k), N-4).*bk[1:end-2]

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
end
function BiharmonicFactor{T<:Real}(F::Biharmonic, k::Array{T, 1})
    if F.BiharmonicBC == "NB"
        factor1 = -2*(k.^2)./((k+2).*(k+3))
        factor2 = (k.^2).*(k+1)./((k+3).*(k+4).^2)
    elseif F.BiharmonicBC == "DB"
        factor1 = -2*(k+2)./(k+3)
        factor2 = (k+1)./(k+3)
    end
    return factor1, factor2
end
function fastShenScalar{T<:Real}(F::Biharmonic, fj::Array{T, 1}, fk::Array{T, 1})
    """Fast Shen scalar product.
    """
    k  = Biharmonicwavenumbers(length(fj))
    factor1, factor2 = BiharmonicFactor(F, k)
    Tk = fk
    Tk = fastChebScalar(F, fj)
    fk[:] = Tk
    fk[1:end-4] += factor1.*Tk[3:end-2]
    fk[1:end-4] += factor2.*Tk[5:end]
    fk[end-3:end] = 0.0
    return fk
end
function ifst{T<:Real}(F::Biharmonic, fk::Array{T, 1}, fj::Array{T, 1})
    """Fast inverse Shen scalar transform
    """
    k = Biharmonicwavenumbers(length(fk))
    factor1, factor2 = BiharmonicFactor(F, k)
    w_hat = zeros(eltype(fk), length(fk))
    w_hat[1:end-4] = fk[1:end-4]
    w_hat[3:end-2] += factor1.*fk[1:end-4]
    w_hat[5:end]   += factor2.*fk[1:end-4]
    fj = ifct(F, w_hat, fj)
    return fj
end
function fst{T<:Real}(F::Biharmonic, fj::Array{T, 1}, fk::Array{T, 1})
    """Fast Shen transform .
    """
    fk = fastShenScalar(F, fj, fk)
    N = length(fj)
    k = Biharmonicwavenumbers(N)
    factor1, factor2 = BiharmonicFactor(F, k)
    N -= 4
    if F.quad == "GL"
        ck = ones(N); ck[1] = 2
    elseif F.quad == "GC"
        ck = ones(N); ck[1] = 2; ck[end] = 2
    end
    c = (ck + factor1.^2 + factor2.^2)*pi/2.
    d = (factor1[1:end-2] + factor1[3:end].*factor2[1:end-2])*pi/2.
    e = factor2[1:end-4]*pi/2.

    fk[1:end-4] = PDMA_Symsolve(c, d, e,fk[1:end-4])
    return fk
end
#----------------------ooo----------------------------

# ----------------------------------------------------
#                       Tests
# ----------------------------------------------------
using Base.Test

function tests(N)

    axis = 1
    U = zeros(Float64, N)
    V, U_hat = similar(U), similar(U)
    x, w = similar(U), similar(U)
    # Chebyshev
    for quad in ["GL", "GC"]
        F = Chebyshev(axis, quad)
        x, w = nodesWeights(F, x, w)
        U = 1.0 - x.^2;
        U_hat = fct(F, U, U_hat)
        V = ifct(F, U_hat, V)
        @test isapprox(U, V)
        U_hat = -2.*x
        V = fastChebDerivative(F, U, V)
        @test isapprox(U_hat, V)
        println("Test: Chebyshev transform for ", quad, " succeeded.")
    end
    # Dirichlet
    for quad in ["GL", "GC"]
        F = Dirichlet(axis, quad)
        x, w = nodesWeights(F, x, w)
        U = 1.0 - x.^2;
        U_hat = fst(F, U, U_hat)
        V = ifst(F, U_hat, V)
        @test isapprox(U, V)
        U_hat = -2.*x
        V = fastShenDerivative(F, U, V)
        @test isapprox(U_hat, V)
        println("Test: Dirichlet transform for ", quad, " succeeded.")
    end
    # Neumann
    for quad in ["GL", "GC"]
        F = Neumann(axis, quad)
        x, w = nodesWeights(F, x, w)
        U = x.- (1./3.)*x.^3;
        U_hat = fst(F, U, U_hat)
        V = ifst(F, U_hat, V)
        @test isapprox(U, V)
        println("Test: Neumann transform for ", quad, " succeeded.")
    end
    # Robin
    for BC in ["ND", "DN"]
        for quad in ["GL", "GC"]
            F = Robin(axis, quad, BC)
            x, w = nodesWeights(F, x, w)
            if BC == "ND"
                U = x-(8./13.)*(-1. + 2*x.^2) - (5./13.)*(-3*x + 4*x.^3);
            elseif BC == "DN"
                U = x + (8./13.)*(-1. + 2.*x.^2) - (5./13.)*(-3*x + 4*x.^3);
            end
            U_hat = fst(F, U, U_hat)
            V = ifst(F, U_hat, V)
            @test isapprox(U, V)
            println("Test: Robin transform for ", BC,"  ",  quad, " succeeded.")
        end
    end
    # Biharmonic
    for BiharmonicBC in ["DB", "NB"]
        for quad in ["GL", "GC"]
            F = Biharmonic(axis, quad, BiharmonicBC)
            x, w = nodesWeights(F, x, w)
            if BiharmonicBC == "DB"
                U = x -(3./2.)*(4.*x.^3 - 3.*x)+(1./2.)*(16.*x.^5 -20.*x.^3 +5.*x)
            elseif BiharmonicBC == "NB"
                U = -1. + 2.*x.^2 - (2./5.)*(1. - 8.*x.^2 + 8.*x.^4) +
                (1./15.)*(-1. + 18.*x.^2 - 48.*x.^4 + 32.*x.^6)
            end
            U_hat = fst(F, U, U_hat)
            V = ifst(F, U_hat, V)
            @test isapprox(U, V)
            println("Test: Biharmonic transform for  ", BiharmonicBC, "  ", quad, " succeeded.")
        end
    end
end

n = 2^6;
tests(n)
