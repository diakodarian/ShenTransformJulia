include("TDMA.jl")
include("PDMA.jl")
using TDMA
using PDMA

#-------------------------------------------------------------------------------
#     Wavenumber vector
#-------------------------------------------------------------------------------
function wavenumbers{T}(N::T)
    return collect(0:N-3)
end
function Biharmonicwavenumbers{T}(N::T)
    return collect(0:N-5)
end
#-------------------------------------------------------------------------------
#   dct - discrete cosine transform
#-------------------------------------------------------------------------------
type DctType{N}
    axis::Int64
end
# Generate various DCTs of real data
@generated function dct{T<:Real, N}(t::DctType{N}, x::Vector{T})
    dct_lookup = Dict(1 => FFTW.REDFT00, 2 => FFTW.REDFT10, 3 => FFTW.REDFT01, 4 => FFTW.REDFT11)
    fftw = dct_lookup[N]
    quote
        FFTW.r2r(x, $(fftw), (t.axis, ))
    end
end
# Generate various DCTs of complex data
@generated function dct{T<:Real, N}(t::DctType{N}, x::Vector{Complex{T}})
    dct_lookup = Dict(1 => FFTW.REDFT00, 2 => FFTW.REDFT10, 3 => FFTW.REDFT01, 4 => FFTW.REDFT11)
    fftw = dct_lookup[N]
    quote
        FFTW.r2r(real(x), $(fftw), (t.axis, )) + FFTW.r2r(imag(x), $(fftw), (t.axis, ))*one(T)
    end
end
#-------------------------------------------------------------------------------
#   Spectral nodes
#-------------------------------------------------------------------------------
abstract NodeType
type GL <: NodeType end
type GC <: NodeType end

#-------------------------------------------------------------------------------
#   Shen spectral transforms
#-------------------------------------------------------------------------------
abstract SpecTransf{T<:NodeType}
type Chebyshev{T<:NodeType} <: SpecTransf{T} end
type Dirichlet{T<:NodeType} <: SpecTransf{T} end
type Neumann{T<:NodeType} <: SpecTransf{T} end
type GeneralDirichlet{T<:NodeType} <: SpecTransf{T}
    a::Float64
    b::Float64
    N::Int64
    k::Vector{Float64}
    ak::Vector{Float64}
    bk::Vector{Float64}
    k1::Vector{Float64}
    ak1::Vector{Float64}
    bk1::Vector{Float64}

    function GeneralDirichlet(a, b, N)
        k = collect(0:N-3)
        k1 = collect(0:N-2)
        ak = 0.5*(b-((-1.).^(-k))*a)
        ak1 = 0.5*(b-((-1.).^(-k1))*a)
        bk = -1. +0.5*(b+((-1.).^(-k))*a)
        bk1 = -1. +0.5*(b+((-1.).^(-k))*a)
        new(a, b, N, k, ak, bk, k1, ak1, bk1)
    end
end
type Robin{T<:NodeType} <: SpecTransf{T}
    BC::Symbol#ASCIIString
    N::Int64
    k::Vector{Float64}
    ak::Vector{Float64}
    bk::Vector{Float64}
    k1::Vector{Float64}
    ak1::Vector{Float64}
    bk1::Vector{Float64}

    function Robin(BC, N)
        k = collect(0:N-3)
        k1 = collect(0:N-2)
        if eval(BC) == "ND"
            ak = -4*(k+1)./((k+1).^2 .+ (k+2).^2)
            ak1 = -4*(k1+1)./((k1+1).^2 .+ (k1+2).^2)
        elseif eval(BC) == "DN"
            ak = 4*(k+1)./((k+1).^2 .+ (k+2).^2)
            ak1 = 4*(k1+1)./((k1+1).^2 .+ (k1+2).^2)
        end
        bk = -((k.^2 + (k+1).^2)./((k+1).^2 .+ (k+2).^2))
        bk1 = -((k1.^2 + (k1+1).^2)./((k1+1).^2 .+ (k1+2).^2))
        new(BC, N, k, ak, bk, k1, ak1, bk1)
    end
end
type Biharmonic{T<:NodeType} <: SpecTransf{T}
    BiharmonicBC::ASCIIString
    N::Int64
    k::Vector{Float64}
    factor1::Vector{Float64}
    factor2::Vector{Float64}

    function Biharmonic(BiharmonicBC, N)
        k = collect(0:N-5)
        if BiharmonicBC == "NB"
            factor1 = -2*(k.^2)./((k+2).*(k+3))
            factor2 = (k.^2).*(k+1)./((k+3).*(k+4).^2)
        elseif BiharmonicBC == "DB"
            factor1 = -2*(k+2)./(k+3)
            factor2 = (k+1)./(k+3)
        end
        new(BiharmonicBC, N, k, factor1, factor2)
    end

end

# Below we will dispatch alot on types which have GL, GC nodes. Typealias to
# save some typing
typealias SpecTGL SpecTransf{GL}
typealias SpecTGC SpecTransf{GC}
typealias DirGL Dirichlet{GL}
typealias DirGC Dirichlet{GC}
typealias NeuGL Neumann{GL}
typealias NeuGC Neumann{GC}
typealias GenDirGL GeneralDirichlet{GL}
typealias GenDirGC GeneralDirichlet{GC}
typealias RobinGL Robin{GL}
typealias RobinGC Robin{GC}
typealias BiharmGL Biharmonic{GL}
typealias BiharmGC Biharmonic{GC}

# Define nodes and weights in terms of transforms
function NodesWeights{T<:Real}(::SpecTGL, x::Vector{T}, w::Vector{T})
    N = length(x)
    @fastmath for i = 1:N
        @inbounds x[i] = cos((i-1.0)*pi/(N-1))
    end
    w[1] = w[end] = pi/(2*(N-1))
    w[2:end-1] = pi/(N-1)
    (x, w)
end
function NodesWeights{T<:Real}(::SpecTGC, x::Vector{T}, w::Vector{T})
    N = length(x)
    @fastmath for i = 1:N
        @inbounds x[i] = cos((2.0*i-1.0)*pi/(2.0*N))
    end
    w[1:end] = pi/N
    (x, w)
end
#-------------------------------------------------------------------------------
#   Chebyshev scalar product
#-------------------------------------------------------------------------------
function fastChebScalar{T}(::SpecTGC, fj::Vector{T})
    F = DctType{2}(1)
    N = length(fj)
    dct(F, fj)*pi/(2.*N)
end
function fastChebScalar{T}(::SpecTGL, fj::Vector{T})
    F = DctType{1}(1)
    N = length(fj)
    dct(F, fj)*pi/(2.*(N-1))
end
#-------------------------------------------------------------------------------
#    Forward Chebyshev transform
#-------------------------------------------------------------------------------
function fct{T}(::SpecTGC, fj::Vector{T}, fk::Vector{T})
    F = DctType{2}(1)
    N = length(fj)
    fk = dct(F, fj)/N
    fk[1] /= 2.
    fk
end
function fct{T}(::SpecTGL, fj::Vector{T}, fk::Vector{T})
    F = DctType{1}(1)
    N = length(fj)
    fk = dct(F, fj)/(N-1)
    fk[1] /= 2.
    fk[end] /= 2.
    fk
end
#-------------------------------------------------------------------------------
#   Backward Chebyshev transform
#-------------------------------------------------------------------------------
function ifct{T}(::SpecTGC, fk::Vector{T}, fj::Vector{T})
    F = DctType{3}(1)
    fj = 0.5*dct(F, fk)
    fj += 0.5*fk[1]
    fj
end
function ifct{T}(::SpecTGL, fk::Array{T}, fj::Array{T})
    F = DctType{1}(1)
    fj = 0.5*dct(F, fk)
    fj += 0.5*fk[1]
    fj[1:2:end] += 0.5*fk[end]
    fj[2:2:end] -= 0.5*fk[end]
    fj
end
#-------------------------------------------------------------------------------
# Spectral Chebyshev coefficients of the first derivative
#-------------------------------------------------------------------------------
function chebDerivativeCoefficients{T}(fk::Vector{T}, fl::Vector{T})
    N = length(fk)
    fl[end] = 0.0
    fl[end-1] = 2*(N-1)*fk[end]
    for k in N-2:-1:2
        fl[k] = 2*k*fk[k+1]+fl[k+2]
    end
    fl[1] = fk[2] + 0.5*fl[3]
    fl
end
#-------------------------------------------------------------------------------
# The first derivative - (Chebyshev basis)
#-------------------------------------------------------------------------------
function fastChebDerivative{T}(F::SpecTransf, fj::Vector{T}, fd::Vector{T})
    fk = similar(fj)
    fkd = similar(fj)
    fk = fct(F, fj, fk)
    fkd = chebDerivativeCoefficients(fk, fkd)
    ifct(F, fkd, fd)
end
#-------------------------------------------------------------------------------
#     Shen scalar product
#-------------------------------------------------------------------------------
@generated function fastShenScalar{T<:Real}(t::SpecTransf, fj::Vector{T}, fk::Vector{T})
    if t == DirGL || t == DirGC
        quote
            fk = fastChebScalar(t, fj)
            fk[1:end-2] -= fk[3:end]
            fk
        end
    elseif t == NeuGL || t == NeuGC
        quote
            N = length(fj)
            k = wavenumbers(N)
            fk = fastChebScalar(t, fj)
            fk[1:end-2] -= ((k./(k+2)).^2).*fk[3:end]
            fk
        end
    elseif t == RobinGL || t == RobinGC || t == GenDirGL || t == GenDirGC
        quote
            fk = fastChebScalar(t, fj)
            fk_tmp = fk
            fk[1:end-2] = fk_tmp[1:end-2] + (t.ak).*fk_tmp[2:end-1] + (t.bk).*fk_tmp[3:end]
            return fk
        end
    elseif t == BiharmGL || t == BiharmGC
        quote
            Tk = fk
            Tk = fastChebScalar(t, fj)
            fk[:] = Tk
            fk[1:end-4] += t.factor1.*Tk[3:end-2]
            fk[1:end-4] += t.factor2.*Tk[5:end]
            fk[end-3:end] = 0.0
            fk
        end
    end
end
#-------------------------------------------------------------------------------
#   ifst - Backward Shen transform
#-------------------------------------------------------------------------------
@generated function ifst{T<:Real}(t::SpecTransf, fk::Vector{T}, fj::Vector{T})
    if t == DirGL || t == DirGC
        quote
            w_hat = zeros(eltype(fk), N)
            w_hat[1:end-2] = fk[1:end-2]
            w_hat[3:end] -= fk[1:end-2]
            ifct(t, w_hat, fj)
        end
    elseif t == NeuGL || t == NeuGC
        quote
            k = wavenumbers(length(fk))
            w_hat = zeros(eltype(fk), length(fk))
            w_hat[2:end-2] = fk[2:end-2]
            w_hat[4:end] -= (k[2:end]./(k[2:end]+2)).^2.*fk[2:end-2]
            ifct(t, w_hat, fj)
        end
    elseif t == RobinGL || t == RobinGC || t == GenDirGL || t == GenDirGC
        quote
            w_hat = zeros(eltype(fk), length(fk))
            w_hat[1:end-2] = fk[1:end-2]
            w_hat[2:end-1] += (t.ak).*fk[1:end-2]
            w_hat[3:end]   += (t.bk).*fk[1:end-2]
            ifct(t, w_hat, fj)
        end
    elseif t == BiharmGL || t == BiharmGC
        quote
            w_hat = zeros(eltype(fk), length(fk))
            w_hat[1:end-4] = fk[1:end-4]
            w_hat[3:end-2] += t.factor1.*fk[1:end-4]
            w_hat[5:end]   += t.factor2.*fk[1:end-4]
            ifct(t, w_hat, fj)
        end
    end
end
#-------------------------------------------------------------------------------
#   fst - Forward Shen transform
#-------------------------------------------------------------------------------
@generated function fst{T<:Real}(t::SpecTransf, fj::Vector{T}, fk::Vector{T})
    if t == DirGC
        quote
            fk = fastShenScalar(t, fj, fk)
            N = length(fj)
            ck = ones(eltype(fk), N-2); ck[1] = 2
            a = ones(eltype(fk), N-4)*(-pi/2)
            b = pi/2*(ck+1)
            c = zeros(eltype(a), N-4)
            c[:] = a
            fk[1:end-2] = TDMA_1D(a, b, c, fk[1:end-2])
            fk
        end
    elseif t == DirGL
        quote
            fk = fastShenScalar(t, fj, fk)
            N = length(fj)
            ck = ones(eltype(fk), N-2); ck[1] = 2; ck[end] = 2
            a = ones(eltype(fk), N-4)*(-pi/2)
            b = pi/2*(ck+1)
            c = zeros(eltype(a), N-4)
            c[:] = a
            fk[1:end-2] = TDMA_1D(a, b, c, fk[1:end-2])
            fk
        end
    elseif t == NeuGC
        quote
            fk = fastShenScalar(t, fj, fk)
            N = length(fj)
            k = wavenumbers(N)
            ck = ones(eltype(k), N-3)
            a = (-pi/2)*ones(eltype(fk), N-5).*(k[2:end-2]./(k[2:end-2]+2.)).^2
            b = pi/2*(1.+ck.*(k[2:end]./(k[2:end]+2.)).^4)
            c = a
            fk[2:end-2] = TDMA_1D(a, b, c, fk[2:end-2])
            fk
        end
    elseif t == NeuGL
        quote
            fk = fastShenScalar(t, fj, fk)
            N = length(fj)
            k = wavenumbers(N)
            ck = ones(eltype(k), N-3)
            ck[end] = 2
            a = (-pi/2)*ones(eltype(fk), N-5).*(k[2:end-2]./(k[2:end-2]+2.)).^2
            b = pi/2*(1.+ck.*(k[2:end]./(k[2:end]+2.)).^4)
            c = a
            fk[2:end-2] = TDMA_1D(a, b, c, fk[2:end-2])
            fk
        end
    elseif t == RobinGC || t == GenDirGC
        quote
            fk = fastShenScalar(t, fj, fk)
            N = length(fj)
            ck = ones(eltype(fj), N-2); ck[1] = 2; ck[end] = 2
            a = (pi/2)*(ck .+ (t.ak).^2 .+ (t.bk).^2)
            b = (pi/2)*ones(eltype(fj), N-3).*((t.ak)[1:end-1] .+ (t.ak1[2:end-1]).*(t.bk[1:end-1]))
            c = (pi/2)*ones(eltype(fj), N-4).*(t.bk[1:end-2])

            fk[1:end-2] = SymmetricalPDMA_1D(a, b, c, fk[1:end-2])
            fk
        end
    elseif t == RobinGL || t == GenDirGL
        quote
            fk = fastShenScalar(t, fj, fk)
            N = length(fj)
            ck = ones(eltype(fj), N-2); ck[1] = 2
            a = (pi/2)*(ck .+ (t.ak).^2 .+ (t.bk).^2)
            b = (pi/2)*ones(eltype(fj), N-3).*((t.ak)[1:end-1] .+ (t.ak1[2:end-1]).*(t.bk[1:end-1]))
            c = (pi/2)*ones(eltype(fj), N-4).*(t.bk[1:end-2])

            fk[1:end-2] = SymmetricalPDMA_1D(a, b, c, fk[1:end-2])
            fk
        end
    elseif t == BiharmGC
        quote
            fk = fastShenScalar(t, fj, fk)
            ck = ones(t.N-4); ck[1] = 2; ck[end] = 2
            c = (ck + t.factor1.^2 + t.factor2.^2)*pi/2.
            d = (t.factor1[1:end-2] + t.factor1[3:end].*t.factor2[1:end-2])*pi/2.
            e = t.factor2[1:end-4]*pi/2.

            fk[1:end-4] = PDMA_Symsolve(c, d, e,fk[1:end-4])
            fk
        end
    elseif t == BiharmGL
        quote
            fk = fastShenScalar(t, fj, fk)
            ck = ones(t.N-4); ck[1] = 2
            c = (ck + t.factor1.^2 + t.factor2.^2)*pi/2.
            d = (t.factor1[1:end-2] + t.factor1[3:end].*t.factor2[1:end-2])*pi/2.
            e = t.factor2[1:end-4]*pi/2.

            fk[1:end-4] = PDMA_Symsolve(c, d, e,fk[1:end-4])
            fk
        end
    end
end
#-------------------------------------------------------------------------------
#   The first derivative - Shen basis
#-------------------------------------------------------------------------------
@generated function fastShenDerivative{T<:Real}(t::SpecTransf, fj::Vector{T}, df::Vector{T})
    if t == DirGL || t == DirGC
        quote
            N = length(fj)
            fk = Array{T}(N); fk_1 = Array{T}(N-2)
            fk_0 = similar(fk_1);
            fk = fst(t, fj, fk)
            k = wavenumbers(N)
            fk_0 = fk[1:end-2].*(1.-((k+2)./k))
            fk_1 = chebDerivativeCoefficients(fk_0, fk_1)
            fk_2 = 2*fk[1:end-2].*(k+2)
            df_hat = zeros(eltype(fj), N)
            df_hat[1:end-2] = fk_1 - vcat(0, fk_2[1:end-1])
            df_hat[end-1] = -fk_2[end]
            ifct(t, df_hat, df)
        end
    elseif t == NeuGL || t == NeuGC
        quote
            N = length(fj)
            fk = Array{T}(N); fk_1 = Array{T}(N-2)
            fk = fst(t, fj, fk)
            k = wavenumbers(N)
            fk_0 = fk[2:end-2].*(1.0 - ( k[2:end]./(k[2:end]+2) ) )
            fk_tmp = vcat(0, fk_0)
            fk_1 = chebDerivativeCoefficients(fk_tmp, fk_1)
            fk_2 = 2*fk[2:end-2].*(k[2:end].^2)./(k[2:end]+2)
            df_hat = zeros(eltype(fk), N)
            df_hat[1] = fk_1[1]
            df_hat[2:end-2] = fk_1[2:end] - vcat(0, fk_2[1:end-1])
            df_hat[end-1] = -fk_2[end]
            ifct(t, df_hat, df)
        end
    elseif t == RobinGL || t == RobinGC || t == GenDirGL || t == GenDirGC
        quote
            fk = Vector{T}(t.N); fk_1 = Vector{T}(t.N-2)
            fk_2 = Vector{T}(t.N-1)
            fk = fst(t, fj, fk)
            fk_tmp = fk[2:end-2].*t.ak[2:end].*((t.k[2:end]+1.)./(t.k[2:end]-1.))
            push!(fk_tmp, 0)
            fk_0 = fk[1:end-2].*(1.0 + t.bk.*(t.k+2.)./t.k)+ fk_tmp
            fk_1 = chebDerivativeCoefficients(fk_0, fk_1)

            fk_tmp2 = 2.*fk[1:end-3].*t.bk[1:end-1].*(t.k[1:end-1]+2.)
            fk_2[1:end-1] = 2*fk[1:end-2].*t.ak.*(t.k+1.) + vcat(0.0 ,fk_tmp2)
            fk_2[end] = 2*fk[end-2]*t.bk[end]*(t.k[end]+2)

            df_hat = zeros(eltype(fk), t.N)
            df_hat[1:end-2] = fk_1 + fk_2[1:end-1]
            df_hat[end-1] = fk_2[end]
            ifct(t, df_hat, df)
        end
    elseif t == BiharmGL || t == BiharmGC
        quote
            fk = Vector{T}(t.N); df_hat = zeros(eltype(fk), t.N)
            fk_1 = Vector{T}(t.N-2); fk_0 = Vector{T}(t.N-2)
            fk_2 = Vector{T}(t.N-1)

            fk = fst(t, fj, fk)

            fk_tmp = fk[1:end-6].*t.factor2[1:end-2].*((t.k[1:end-2]+4.)./(t.k[1:end-2]+2.))
            fk_0[1:end-2] = fk[1:end-4].*(1.0 + t.factor1.*(t.k+2.)./t.k) + vcat(0.,0.,fk_tmp)
            fk_0[end-1] = fk[end-5]*t.factor2[end-1]*(t.k[end-1]+4.)/(t.k[end-1]+2)
            fk_0[end] = fk[end-4]*t.factor2[end]*(t.k[end]+4.)/(t.k[end]+2)
            fk_1 = chebDerivativeCoefficients(fk_0, fk_1)

            fk_tmp2 = 2.*fk[1:end-4].*t.factor1.*(t.k+2.)
            fk_tmp3 = 2.*fk[1:end-6].*t.factor2[1:end-2].*(t.k[1:end-2]+4.)
            fk_2[1:end-2] = vcat(0.,fk_tmp2) + vcat(0.,0.,0.,fk_tmp3)
            fk_2[end-1] = 2.*fk[end-5]*t.factor2[end-1]*(t.k[end-1]+4.)
            fk_2[end] = 2.*fk[end-4]*t.factor2[end]*(t.k[end]+4.)

            df_hat[1:end-2] = fk_1 + fk_2[1:end-1]
            df_hat[end-1] = fk_2[end]
            ifct(t, df_hat, df)
        end
    end
end
# ----------------------------------------------------------------------------
using Base.Test
function tests(N)
    axis = 1
    U = zeros(Float64, N)
    V, U_hat = similar(U), similar(U)
    x, w = similar(U), similar(U)
    # Chebyshev
    for F in [Chebyshev{GC}(), Chebyshev{GL}()]
        x, w = NodesWeights(F, x, w)
        U = 1.0 - x.^2;
        U_hat = fct(F, U, U_hat)
        V = ifct(F, U_hat, V)
        @test isapprox(U, V)
        U_hat = -2.*x
        V = fastChebDerivative(F, U, V)
        @test isapprox(U_hat, V)
        println("Test: Chebyshev transform for ", F, " succeeded.")
    end
    # Dirichlet
    for F in [Dirichlet{GC}(), Dirichlet{GL}()]
        x, w = NodesWeights(F, x, w)
        U = 1.0 - x.^2;
        U_hat = fst(F, U, U_hat)
        V = ifst(F, U_hat, V)
        @test isapprox(U, V)
        U_hat = -2.*x
        V = fastShenDerivative(F, U, V)
        @test isapprox(U_hat, V)
        println("Test: Dirichlet transform for ", F, " succeeded.")
    end
    # General Dirichlet
    a = -2.0; b = 2.0;
    for F in [GeneralDirichlet{GL}(a, b, N), GeneralDirichlet{GL}(a, b, N)]
        x, w = NodesWeights(F, x, w)
        U = -2.+10.*x.^2-8.*x.^4 +2.0*(-3.*x+4.*x.^3)#-2.*x+4.*x.^3;
        U_hat = fst(F, U, U_hat)
        V = ifst(F, U_hat, V)
        @test isapprox(U, V)
        U_hat = 20.*x-32.*x.^3-6.0+24.*x.^2#-2.0+12.*x.^2;
        V = fastShenDerivative(F, U, V)
        @test isapprox(U_hat, V)
        println("Test: General Dirichlet transform for ", F, " succeeded.")
    end
    # Neumann
    for F in [Neumann{GC}(), Neumann{GL}()]
        x, w = NodesWeights(F, x, w)
        U = x.- (1./3.)*x.^3;
        U_hat = fst(F, U, U_hat)
        V = ifst(F, U_hat, V)
        @test isapprox(U, V)
        U_hat = 1.-x.^2
        V = fastShenDerivative(F, U, V)
        @test isapprox(U_hat, V)
        println("Test: Neumann transform for ", F, " succeeded.")
    end
    # Robin
    for BC in syms
        for F in [Robin{GC}(BC, N), Robin{GL}(BC, N)]
            x, w = NodesWeights(F, x, w)
            if eval(BC) == "ND"
                U = x-(8./13.)*(-1. + 2*x.^2) - (5./13.)*(-3*x + 4*x.^3);
            elseif eval(BC) == "DN"
                U = x + (8./13.)*(-1. + 2.*x.^2) - (5./13.)*(-3*x + 4*x.^3);
            end
            U_hat = fst(F, U, U_hat)
            V = ifst(F, U_hat, V)
            @test isapprox(U, V)
            if eval(BC) == "ND"
                U_hat = (28./13.) - (32./13.)*x -(60./13.)*x.^2;
            elseif eval(BC) == "DN"
                U_hat = (28./13.) + (32./13.)*x -(60./13.)*x.^2;
            end
            V = fastShenDerivative(F, U, V)
            @test isapprox(U_hat, V)
            println("Test: Robin transform for ", eval(BC), " succeeded.")
        end
    end
    # Biharmonic
    for BiharmonicBC in ["DB", "NB"]
        for F in [Biharmonic{GC}(BiharmonicBC, N), Biharmonic{GL}(BiharmonicBC, N)]
            x, w = NodesWeights(F, x, w)
            if BiharmonicBC == "DB"
                U = x -(3./2.)*(4.*x.^3 - 3.*x)+(1./2.)*(16.*x.^5 -20.*x.^3 +5.*x)
            elseif BiharmonicBC == "NB"
                U = -1. + 2.*x.^2 - (2./5.)*(1. - 8.*x.^2 + 8.*x.^4) +
                (1./15.)*(-1. + 18.*x.^2 - 48.*x.^4 + 32.*x.^6)
            end
            U_hat = fst(F, U, U_hat)
            V = ifst(F, U_hat, V)
            @test isapprox(U, V)
            if BiharmonicBC == "DB"
                U_hat = 8. - 48.*x.^2 +40.*x.^4;
            elseif BiharmonicBC == "NB"
                U_hat = (64./5.)*x - (128./5.)*x.^3 +(64./5.)*x.^5;
            end
            V = fastShenDerivative(F, U, V)
            @test isapprox(U_hat, V)
            println("Test: Biharmonic transform for  ", BiharmonicBC, " succeeded.")
        end
    end
end
N = 2^7;
BC1 = "ND"; BC2 = "DN";
sym1 = :BC1
sym2 = :BC2
syms = [sym1, sym2]
tests(N)
