include("TDMA.jl")
include("PDMA.jl")
using TDMA
using PDMA
import MPI
#-----------------------------------------------------
#      Functions needed for all transforms
#-----------------------------------------------------
function wavenumbers3D{T<:Int}(N::T)
    Nh = N÷2+1
    ky = fftfreq(N, 1./N)
    kz = ky[1:(N÷2+1)]; kz[end] *= -1
    kx = collect(Float64, 0:N-3)
    K = Array{Float64}(Nh, N, N-2, 3)
    for (i, Ki) in enumerate(ndgrid(kz, ky, kx)) K[view(i)...] = Ki end
    return K
end
function Biharmonicwavenumbers3D{T<:Int}(N::T)
    Nh = N÷2+1
    ky = fftfreq(N, 1./N)
    kz = ky[1:(N÷2+1)]; kz[end] *= -1
    kx = collect(Float64, 0:N-5)
    K = Array{Float64}(Nh, N, N-4, 3)
    for (i, Ki) in enumerate(ndgrid(kz, ky, kx)) K[view(i)...] = Ki end
    return K
end
#-------------------------------------------------------------------------------
#   dct - discrete cosine transform
#-------------------------------------------------------------------------------
type DctType{N}
    axis::Int64
end
# Generate various DCTs of real data
@generated function dct{T<:Real, N}(t::DctType{N}, x::AbstractArray{T, 3})
    dct_lookup = Dict(1 => FFTW.REDFT00, 2 => FFTW.REDFT10, 3 => FFTW.REDFT01, 4 => FFTW.REDFT11)
    fftw = dct_lookup[N]
    quote
        FFTW.r2r(x, $(fftw), (t.axis, ))
    end
end
# Generate various DCTs of complex data
@generated function dct{T<:Real, N}(t::DctType{N}, x::AbstractArray{Complex{T},3})
    dct_lookup = Dict(1 => FFTW.REDFT00, 2 => FFTW.REDFT10, 3 => FFTW.REDFT01, 4 => FFTW.REDFT11)
    fftw = dct_lookup[N]
    quote
        FFTW.r2r(real(x), $(fftw), (t.axis, )) + FFTW.r2r(imag(x), $(fftw), (t.axis, ))*Complex{T}(im)
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
type Dirichlet{T<:NodeType} <: SpecTransf{T}
    N::Int64
    k::Array{Float64, 4}
    ck::Vector{Float64}
    function Dirichlet(N)
        k = wavenumbers3D(N)
        if T == GL
            ck = ones(eltype(Float64), N-2); ck[1] = 2; ck[end] = 2
        elseif T == GC
            ck = ones(eltype(Float64), N-2); ck[1] = 2
        end
        new(N, k, ck)
    end
end
type Neumann{T<:NodeType} <: SpecTransf{T}
    N::Int64
    k::Vector{Float64}
    K::Array{Float64, 4}
    ck::Vector{Float64}
    function Neumann(N)
        k = collect(0:N-3)
        K = wavenumbers3D(N)
        if T == GL
            ck = ones(eltype(Float64), N-3); ck[end] = 2
        elseif T == GC
            ck = ones(eltype(Float64), N-3)
        end
        new(N, k, K, ck)
    end
end
type GeneralDirichlet{T<:NodeType} <: SpecTransf{T}
    a::Float64
    b::Float64
    N::Int64
    k::Vector{Float64}
    ck::Vector{Float64}
    ak::Vector{Float64}
    bk::Vector{Float64}
    K::Array{Float64, 4}
    aK::Array{Float64, 3}
    bK::Array{Float64, 3}
    k1::Vector{Float64}
    ak1::Vector{Float64}
    bk1::Vector{Float64}

    function GeneralDirichlet(a, b, N)
        k = collect(0:N-3)
        K = wavenumbers3D(N)
        k1 = collect(0:N-2)
        ak = 0.5*(b-((-1.).^(-k))*a)
        aK = 0.5*(b-((-1.).^(-K(3)))*a)
        ak1 = 0.5*(b-((-1.).^(-k1))*a)
        bk = -1. +0.5*(b+((-1.).^(-k))*a)
        bK = -1. +0.5*(b+((-1.).^(-K(3)))*a)
        bk1 = -1. +0.5*(b+((-1.).^(-k))*a)
        if T == GL
            ck = ones(eltype(Float64), N-2); ck[1] = 2; ck[end] = 2
        elseif T == GC
            ck = ones(eltype(Float64), N-2); ck[1] = 2
        end
        new(a, b, N, k, ck, ak, bk, K, aK, bK, k1, ak1, bk1)
    end
end
type Robin{T<:NodeType} <: SpecTransf{T}
    BC::Symbol
    N::Int64
    k::Vector{Float64}
    ck::Vector{Float64}
    ak::Vector{Float64}
    bk::Vector{Float64}
    K::Array{Float64, 4}
    aK::Array{Float64, 3}
    bK::Array{Float64, 3}
    k1::Vector{Float64}
    ak1::Vector{Float64}
    bk1::Vector{Float64}

    function Robin(BC, N)
        K = wavenumbers3D(N)
        k = collect(0:N-3)
        k1 = collect(0:N-2)
        if eval(BC) == "ND"
            aK  = -4*(K(3)+1.)./((K(3)+1.).^2 + (K(3)+2.).^2)
            ak = -4*(k+1)./((k+1).^2 .+ (k+2).^2)
            ak1 = -4*(k1+1)./((k1+1).^2 .+ (k1+2).^2)
        elseif eval(BC) == "DN"
            aK  = 4*(K(3)+1.)./((K(3)+1.).^2 + (K(3)+2.).^2)
            ak = 4*(k+1)./((k+1).^2 .+ (k+2).^2)
            ak1 = 4*(k1+1)./((k1+1).^2 .+ (k1+2).^2)
        end
        bK = -((K(3).^2 + (K(3)+1.).^2)./((K(3)+1.).^2 + (K(3)+2.).^2))
        bk = -((k.^2 + (k+1).^2)./((k+1).^2 .+ (k+2).^2))
        bk1 = -((k1.^2 + (k1+1).^2)./((k1+1).^2 .+ (k1+2).^2))
        if T == GL
            ck = ones(eltype(Float64), N-2); ck[1] = 2; ck[end] = 2
        elseif T == GC
            ck = ones(eltype(Float64), N-2); ck[1] = 2
        end
        new(BC, N, k, ck, ak, bk, K, aK, bK, k1, ak1, bk1)
    end
end
type Biharmonic{T<:NodeType} <: SpecTransf{T}
    BiharmonicBC::Symbol
    N::Int64
    k::Vector{Float64}
    K::Array{Float64, 4}
    ck::Vector{Float64}
    ak::Vector{Float64}
    bk::Vector{Float64}
    aK::Array{Float64, 3}
    bK::Array{Float64, 3}

    function Biharmonic(BiharmonicBC, N)
        k = collect(0:N-5)
        K = Biharmonicwavenumbers3D(N)
        if eval(BiharmonicBC) == "NB"
            ak = -2*(k.^2)./((k+2).*(k+3))
            bk = (k.^2).*(k+1)./((k+3).*(k+4).^2)
            aK = -2*(K(3).^2)./((K(3)+2.).*(K(3)+3.))
            bK = (K(3).^2).*(K(3)+1.)./((K(3)+3.).*(K(3)+4.).^2)
        elseif eval(BiharmonicBC) == "DB"
            ak = -2*(k+2)./(k+3)
            bk = (k+1)./(k+3)
            aK = -2*(K(3)+2.)./(K(3)+3.)
            bK = (K(3)+1.)./(K(3)+3.)
        end
        if T == GL
            ck = ones(eltype(Float64), N-4); ck[1] = 2; ck[end] = 2
        elseif T == GC
            ck = ones(eltype(Float64), N-4); ck[1] = 2
        end
        new(BiharmonicBC, N, k, K, ck, ak, bk, aK, bK)
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

#-------------------------------------------------------------------------------
#   Nodes and weights
#-------------------------------------------------------------------------------
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
function fastChebScalar{S<:Real}(::SpecTGC, fj::AbstractArray{Complex{S}, 3})
    F = DctType{2}(3)
    N = last(size(fj))
    dct(F, fj)*pi/(2.*N)
end
function fastChebScalar{S<:Real}(::SpecTGL, fj::AbstractArray{Complex{S}, 3})
    F = DctType{1}(3)
    N = last(size(fj))
    dct(F, fj)*pi/(2.*(N-1))
end
#-------------------------------------------------------------------------------
#    Forward Chebyshev transform
#-------------------------------------------------------------------------------
function fct{S<:Real}(::SpecTGC, fj::AbstractArray{Complex{S}}, fk::AbstractArray{Complex{S}})
    F = DctType{2}(3)
    N = last(size(fj))
    fk = dct(F, fj)/N
    fk[:,:,1] /= 2.
    fk
end
function fct{S<:Real}(::SpecTGL, fj::AbstractArray{Complex{S}, 3}, fk::AbstractArray{Complex{S}, 3})
    F = DctType{1}(3)
    N = last(size(fj))
    fk = dct(F, fj)/(N-1)
    fk[:,:,1] /= 2.
    fk[:,:,end] /= 2.
    fk
end
#-------------------------------------------------------------------------------
#   Backward Chebyshev transform
#-------------------------------------------------------------------------------
function ifct{S<:Real}(::SpecTGC, fk::AbstractArray{Complex{S}, 3}, fj::AbstractArray{Complex{S}, 3})
    F = DctType{3}(3)
    fj = 0.5*dct(F, fk)
    for i in 1:last(size(fj))
        fj[:,:,i] += 0.5*fk[:,:,1]
    end
    fj
end
function ifct{S<:Real}(::SpecTGL, fk::AbstractArray{Complex{S}, 3}, fj::AbstractArray{Complex{S}, 3})
    F = DctType{1}(3)
    fj = 0.5*dct(F, fk)
    for i in 1:last(size(fj))
        fj[:,:,i] += 0.5*fk[:,:,1]
    end
    for k in 1:div((N+1),2)
        fj[:,:,(2*k-1)] += 0.5*fk[:, :, end]
    end
    for k in 1:div(N,2)
        fj[:,:,2*k] -= 0.5*fk[:, :, end]
    end
    fj
end
#-------------------------------------------------------------------------------
# Spectral Chebyshev coefficients of the first derivative
#-------------------------------------------------------------------------------
function chebDerivativeCoefficients{S<:Real}(fk::AbstractArray{S, 3}, fl::AbstractArray{S, 3})
    N = length(fk)
    fl[end] = 0.0
    fl[end-1] = 2*(N-1)*fk[end]
    for k in N-2:-1:2
        fl[k] = 2*k*fk[k+1]+fl[k+2]
    end
    fl[1] = fk[2] + 0.5*fl[3]
    fl
end
function chebDerivativeCoefficients_3D{S<:Real}(fk::AbstractArray{S, 3}, fl::AbstractArray{S, 3})
    for i in 1:size(fk,1)
        for j in 1:size(fk,2)
            fl[i,j,1:end] = chebDerivativeCoefficients(fk[i,j,1:end], fl[i,j,1:end])
        end
    end
    return fl
end
#-------------------------------------------------------------------------------
# The first derivative - (Chebyshev basis)
#-------------------------------------------------------------------------------
function fastChebDerivative{T<:Real}(F::SpecTransf, fj::AbstractArray{T, 3}, fd::AbstractArray{T, 3})
    fk = similar(fj)
    fkd = similar(fj)
    fk = fct(F, fj, fk)
    fkd = chebDerivativeCoefficients_3D(fk, fkd)
    ifct(F, fkd, fd)
end
#-------------------------------------------------------------------------------
#     Shen scalar product
#-------------------------------------------------------------------------------
@generated function fastShenScalar{T<:Real}(t::SpecTransf, fj::AbstractArray{Complex{T}, 3}, fk::AbstractArray{Complex{T}, 3})
    if t == DirGL || t == DirGC
        quote
            fk = fastChebScalar(t, fj)
            fk[:,:,1:end-2] -= fk[:,:,3:end]
            fk
        end
    elseif t == NeuGL || t == NeuGC
        quote
            fk = fastChebScalar(t, fj)
            fk[:,:,1:end-2] -= ((t.K(3)./(t.K(3)+2.)).^2).*fk[:,:,3:end]
            fk
        end
    elseif t == RobinGL || t == RobinGC || t == GenDirGL || t == GenDirGC
        quote
            fk = fastChebScalar(t, fj)
            fk_tmp = fk
            fk[:,:,1:end-2] = fk_tmp[:,:,1:end-2] + (t.aK).*fk_tmp[:,:,2:end-1] + (t.bK).*fk_tmp[:,:,3:end]
            return fk
        end
    elseif t == BiharmGL || t == BiharmGC
        quote
            Tk = fk
            Tk = fastChebScalar(t, fj)
            fk[:] = Tk
            fk[:,:,1:end-4] += t.aK.*Tk[:,:,3:end-2]
            fk[:,:,1:end-4] += t.bK.*Tk[:,:,5:end]
            fk[:,:,end-3:end] = 0.0
            fk
        end
    end
end
#-------------------------------------------------------------------------------
#   ifst - Backward Shen transform
#-------------------------------------------------------------------------------
@generated function ifst{T<:Real}(t::SpecTransf, fk::AbstractArray{Complex{T}, 3}, fj::AbstractArray{Complex{T}, 3})
    if t == DirGL || t == DirGC
        quote
            w_hat = zeros(eltype(fk), size(fk))
            w_hat[:,:,1:end-2] = fk[:,:,1:end-2]
            w_hat[:,:,3:end] -= fk[:,:,1:end-2]
            ifct(t, w_hat, fj)
        end
    elseif t == NeuGL || t == NeuGC
        quote
            w_hat = zeros(eltype(fk), size(fk))
            w_hat[:,:,2:end-2] = fk[:,:,2:end-2]
            w_hat[:,:,4:end] -= (t.K(3)[:,:,2:end]./(t.K(3)[:,:,2:end]+2.)).^2.*fk[:,:,2:end-2]
            ifct(t, w_hat, fj)
        end
    elseif t == RobinGL || t == RobinGC || t == GenDirGL || t == GenDirGC
        quote
            w_hat = zeros(eltype(fk), size(fk))
            w_hat[:,:,1:end-2] = fk[:,:,1:end-2]
            w_hat[:,:,2:end-1] += (t.aK).*fk[:,:,1:end-2]
            w_hat[:,:,3:end]   += (t.bK).*fk[:,:,1:end-2]
            ifct(t, w_hat, fj)
        end
    elseif t == BiharmGL || t == BiharmGC
        quote
            w_hat = zeros(eltype(fk), size(fk))
            w_hat[:,:,1:end-4] = fk[:,:,1:end-4]
            w_hat[:,:,3:end-2] += t.aK.*fk[:,:,1:end-4]
            w_hat[:,:,5:end]   += t.bK.*fk[:,:,1:end-4]
            ifct(t, w_hat, fj)
        end
    end
end
#-------------------------------------------------------------------------------
#   fst - Forward Shen transform
#-------------------------------------------------------------------------------
@generated function fst{T<:Real}(t::SpecTransf, fj::AbstractArray{Complex{T}, 3}, fk::AbstractArray{Complex{T}, 3})
    if t == DirGC || t == DirGL
        quote
            fk = fastShenScalar(t, fj, fk)
            N = last(size(fj))
            a = ones(eltype(fk), N-4)*(-pi/2)
            b = (pi/2.)*(t.ck+1.)
            c = zeros(eltype(a), N-4)
            c[:] = a
            fk[:,:,1:end-2] = TDMA_SymSolve3D(b, a, c, fk[:,:,1:end-2])
            fk
        end
    elseif t == NeuGC || t == NeuGL
        quote
            fk = fastShenScalar(t, fj, fk)
            N = last(size(fj))
            a = (-pi/2.)*ones(eltype(fk), N-5).*(t.k[2:end-2]./(t.k[2:end-2]+2.)).^2
            b = (pi/2.)*(1.+t.ck.*(t.k[2:end]./(t.k[2:end]+2.)).^4)
            c = a
            fk[:,:,2:end-2] = TDMA_SymSolve3D(b,a,c,fk[:,:,2:end-2])
            fk
        end
    elseif t == RobinGC || t == GenDirGC || t == RobinGL || t == GenDirGL
        quote
            fk = fastShenScalar(t, fj, fk)
            N = last(size(fj))
            a = (pi/2)*(t.ck + (t.ak).^2 + (t.bk).^2)
            b = (pi/2)*ones(eltype(fj), N-3).*((t.ak)[1:end-1] .+ (t.ak1[2:end-1]).*(t.bk[1:end-1]))
            c = (pi/2)*ones(eltype(fj), N-4).*(t.bk[1:end-2])

            fk[:,:,1:end-2] = SymmetricalPDMA_3D(a, b, c, fk[:,:,1:end-2])
            fk
        end
    elseif t == BiharmGC || t == BiharmGL
        quote
            fk = fastShenScalar(t, fj, fk)
            c = (t.ck + t.ak.^2 + t.bk.^2)*pi/2.
            d = (t.ak[1:end-2] + t.ak[3:end].*t.bk[1:end-2])*pi/2.
            e = t.bk[1:end-4]*pi/2.

            fk[:,:,1:end-4] = PDMA_Symsolve3D(c, d, e,fk[:,:,1:end-4])
            fk
        end
    end
end
#-------------------------------------------------------------------------------
#   The first derivative - Shen basis
#-------------------------------------------------------------------------------
@generated function fastShenDerivative{T<:Real}(t::SpecTransf, fj::AbstractArray{T, 3}, df::AbstractArray{T, 3})
    if t == DirGL || t == DirGC
        quote
            N = last(size(fj))
            fk = similar(fj); fk_1 = Array(eltype(fj), N, N, N-2)
            fk_0 = similar(fk_1);fk_2 = similar(fk_1)
            fk = fst(t, fj, fk)

            fk_0 = fk[:,:,1:end-2].*(1.-((t.k(3)+2.)./t.k(3)))
            fk_1 = chebDerivativeCoefficients_3D(fk_0, fk_1)
            fk_2 = 2.*fk[:,:,1:end-2].*(t.k(3)+2.)
            df_hat = zeros(eltype(fj), N, N, N)
            df_hat[:,:,1:end-2] = fk_1[:,:,1:end] - cat(3, zeros(N,N), fk_2[:,:,1:end-1])
            df_hat[:,:,end-1] = -fk_2[:,:,end]
            ifct(t, df_hat, df)
        end
    elseif t == NeuGL || t == NeuGC
        quote
            N = last(size(fj))
            fk = similar(fj); fk_1 = Array(eltype(fj), N, N, N-2)
            fk = fst(t, fj, fk)
            fk_0 = fk[:,:,2:end-2].*(1.0 - ( t.K(3)[:,:,2:end]./(t.K(3)[:,:,2:end]+2.) ) )
            fk_tmp = cat(3, zeros(N,N), fk_0)
            fk_1 = chebDerivativeCoefficients_3D(fk_tmp, fk_1)
            fk_2 = 2*fk[:,:,2:end-2].*(t.K(3)[:,:,2:end].^2)./(t.K(3)[:,:,2:end]+2.)
            df_hat = zeros(eltype(fk), N,N,N)
            df_hat[:,:,1] = fk_1[:,:,1]
            df_hat[:,:,2:end-2] = fk_1[:,:,2:end] - cat(3, zeros(N,N), fk_2[:,:,1:end-1])
            df_hat[:,:,end-1] = -fk_2[:,:,end]
            ifct(t, df_hat, df)
        end
    elseif t == RobinGL || t == RobinGC || t == GenDirGL || t == GenDirGC
        quote
            N = last(size(fj))
            fk = similar(fj); fk_1 = Array(eltype(fj), N, N, N-2)
            fk_2 = Array(eltype(fj), N, N, N-1)
            fk_tmp = zeros(eltype(fj), N, N, N-2)
            fk = fst(t, fj, fk)
            fk_tmp[:,:,1:end-1] = fk[:,:,2:end-2].*t.aK[:,:,2:end].*((t.K(3)[:,:,2:end]+1.)./(t.K(3)[:,:,2:end]-1.))
            fk_0 = fk[:,:,1:end-2].*(1.0 + t.bK.*(t.K(3)+2.)./t.K(3)) .+ fk_tmp
            fk_1 = chebDerivativeCoefficients_3D(fk_0, fk_1)

            fk_tmp2 = 2.*fk[:,:,1:end-3].*t.bK[:,:,1:end-1].*(t.K(3)[:,:,1:end-1]+2.)
            fk_2[:,:,1:end-1] = 2.*fk[:,:,1:end-2].*t.aK.*(t.K(3)+1.)
            fk_2[:,:,2:end-1] += fk_tmp2
            fk_2[:,:,end] = 2.*fk[:,:,end-2].*t.bK[:,:,end].*(t.K(3)[:,:,end]+2.)

            df_hat = zeros(eltype(fk), size(fk))
            df_hat[:,:,1:end-2] = fk_1[:,:,1:end] + fk_2[:,:,1:end-1]
            df_hat[:,:,end-1] = fk_2[:,:,end]

            ifct(t, df_hat, df)
        end
    elseif t == BiharmGL || t == BiharmGC
        quote
            N = last(size(fj))
            fk = similar(fj); fk_1 = Array(eltype(fj), N, N, N-2)
            fk_0 = Array(eltype(fj), N, N, N-2)
            fk_2 = zeros(eltype(fj), N, N, N-1)
            df_hat = zeros(eltype(fk),N,N,N)

            fk = fst(t, fj, fk)

            fk_tmp = fk[:,:,1:end-6].*t.bK[:,:,1:end-2].*((t.K(3)[:,:,1:end-2]+4.)./(t.K(3)[:,:,1:end-2]+2.))
            fk_0[:,:,1:end-2] = fk[:,:,1:end-4].*(1.0 + t.aK.*(t.K(3)+2.)./t.K(3))
            fk_0[:,:,3:end-2] += fk_tmp
            fk_0[:,:,end-1] = fk[:,:,end-5].*t.bK[:,:,end-1].*(t.K(3)[:,:,end-1]+4.)./(t.K(3)[:,:,end-1]+2.)
            fk_0[:,:,end] = fk[:,:,end-4].*t.bK[:,:,end].*(t.K(3)[:,:,end]+4.)./(t.K(3)[:,:,end]+2.)
            fk_1 = chebDerivativeCoefficients_3D(fk_0, fk_1)

            fk_tmp2 = 2.*fk[:,:,1:end-4].*t.aK.*(t.K(3)+2.)
            fk_tmp3 = 2.*fk[:,:,1:end-6].*t.bK[:,:,1:end-2].*(t.K(3)[:,:,1:end-2]+4.)
            fk_2[:,:,2:end-2] = fk_tmp2
            fk_2[:,:,4:end-2] += fk_tmp3
            fk_2[:,:,end-1] = 2.*fk[:,:,end-5].*t.bK[:,:,end-1].*(t.K(3)[:,:,end-1]+4.)
            fk_2[:,:,end] = 2.*fk[:,:,end-4].*t.bK[:,:,end].*(t.K(3)[:,:,end]+4.)

            df_hat[:,:,1:end-2] = fk_1 + fk_2[:,:,1:end-1]
            df_hat[:,:,end-1] = fk_2[:,:,end]
            ifct(t, df_hat, df)
        end
    end
end

#-------------------------------------------------------------------------------
#   Spectral transforms
#-------------------------------------------------------------------------------
type r2c{T<:SpecTransf, S<:Real}
    N::Int64
    plan12::FFTW.rFFTWPlan{S}
    vT::Array{Complex{S}, 3}
    v::Array{Complex{S}, 3}

    function r2c(N)
        Nh = N÷2+1
        A = zeros(S, (N, N, N))
        plan12 = plan_rfft(A, (1, 2))
        vT, v = Array{Complex{S}}(Nh, N, N), Array{Complex{S}}(Nh, N, N)
        inv(plan12)
        new(N, plan12, vT, v)
    end
end

@generated function FCS{S<:SpecTransf, T<:Real}(F::r2c{S, T}, u::AbstractArray{T}, fu::AbstractArray{Complex{T}, 3})
    node = [Chebyshev{GL}(), Chebyshev{GC}()]
    if r2c{S, T} == r2c{SpecTransf{GL}, Float64}
        t = node[1]
    elseif r2c{S, T} == r2c{SpecTransf{GC}, Float64}
        t = node[2]
    end
    quote
        A_mul_B!(F.vT, F.plan12, u)
        fu = fastChebScalar($t, F.vT, fu)
        fu
    end
end
@generated function IFCT{S<:SpecTransf, T<:Real}(F::r2c{S, T}, fu::AbstractArray{Complex{T}}, u::AbstractArray{T})
    node = [Chebyshev{GL}(), Chebyshev{GC}()]
    if r2c{S, T} == r2c{SpecTransf{GL}, Float64}
        t = node[1]
    elseif r2c{S, T} == r2c{SpecTransf{GC}, Float64}
        t = node[2]
    end
    quote
        F.v = ifct($t, fu, F.v)
        A_mul_B!(u, F.plan12.pinv, F.v)
        u
    end
end
@generated function FCT{S<:SpecTransf, T<:Real}(F::r2c{S, T}, u::AbstractArray{T}, fu::AbstractArray{Complex{T}})
    node = [Chebyshev{GL}(), Chebyshev{GC}()]
    if r2c{S, T} == r2c{SpecTransf{GL}, Float64}
        t = node[1]
    elseif r2c{S, T} == r2c{SpecTransf{GC}, Float64}
        t = node[2]
    end
    quote
        A_mul_B!(F.vT, F.plan12, u)
        fu = fct($t, F.vT, fu)
        fu
    end
end

@generated function FSS{S<:SpecTransf, T<:Real}(F::r2c{S, T}, u::AbstractArray{T}, fu::AbstractArray{Complex{T}, 3})
    node = [Dirichlet{GL}, Dirichlet{GC},
            GeneralDirichlet{Gl}, GeneralDirichlet{GC},
            Neumann{GL}, Neumann{GC}, Robin{GL}, Robin{GC},
            Biharmonic{GL}, Biharmonic{GC}]
    if r2c{S, T} == r2c{Dirichlet{GL}, Float64}
        t = node[1](F.N)
    elseif r2c{S, T} == r2c{Dirichlet{GC}, Float64}
        t = node[2](F.N)
    elseif r2c{S, T} == r2c{GeneralDirichlet{GL}, Float64}
        t = node[3]
    elseif r2c{S, T} == r2c{GeneralDirichlet{GC}, Float64}
        t = node[4]
    elseif r2c{S, T} == r2c{Neumann{GL}, Float64}
        t = node[5]
    elseif r2c{S, T} == r2c{Neumann{GC}, Float64}
        t = node[6]
    elseif r2c{S, T} == r2c{Robin{GL}, Float64}
        t = node[7]
    elseif r2c{S, T} == r2c{Robin{GC}, Float64}
        t = node[8]
    elseif r2c{S, T} == r2c{Biharmonic{GL}, Float64}
        t = node[9]
    elseif r2c{S, T} == r2c{Biharmonic{GC}, Float64}
        t = node[10]
    end
    quote
        A_mul_B!(F.vT, F.plan12, u)
        fu = fastShenScalar($t, F.vT, fu)
        fu
    end
end
@generated function IFST{S<:SpecTransf, T<:Real}(F::r2c{S, T}, fu::AbstractArray{Complex{T}}, u::AbstractArray{T})
    node = [Dirichlet{GL}, Dirichlet{GC}, Neumann{GL}, Neumann{GC}]
    if r2c{S, T} == r2c{Dirichlet{GL}, Float64}
        t = node[1]
    elseif r2c{S, T} == r2c{Dirichlet{GC}, Float64}
        t = node[2]
    elseif r2c{S, T} == r2c{Neumann{GL}, Float64}
        t = node[3]
    elseif r2c{S, T} == r2c{Neumann{GC}, Float64}
        t = node[4]
    end
    quote
        F.v = ifst($t(F.N), fu, F.v)
        A_mul_B!(u, F.plan12.pinv, F.v)
        u
    end
end
@generated function FST{S<:SpecTransf, T<:Real}(F::r2c{S, T}, u::AbstractArray{T}, fu::AbstractArray{Complex{T}})
    node = [Dirichlet{GL}, Dirichlet{GC}, Neumann{GL}, Neumann{GC}]
    if r2c{S, T} == r2c{Dirichlet{GL}, Float64}
        t = node[1]
    elseif r2c{S, T} == r2c{Dirichlet{GC}, Float64}
        t = node[2]
    elseif r2c{S, T} == r2c{Neumann{GL}, Float64}
        t = node[3]
    elseif r2c{S, T} == r2c{Neumann{GC}, Float64}
        t = node[4]
    end
    quote
        A_mul_B!(F.vT, F.plan12, u)
        fu = fst($t(F.N), F.vT, fu)
        fu
    end
end
# ----------------------------------------------------------------------------
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
#   Tests: Shen transforms
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
    for (j,F) in enumerate([GeneralDirichlet{GL}(a, b, N), GeneralDirichlet{GL}(a, b, N)])
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
        for (j,F) in enumerate([Robin{GC}(BC, N), Robin{GL}(BC, N)])
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
N = 2^3;
BC1 = "ND"; BC2 = "DN";
sym1 = :BC1
sym2 = :BC2
RobinBC = [sym1, sym2]

BiharmBC1 = "DB"; BiharmBC2 = "NB";
symbol1 = :BiharmBC1
symbol2 = :BiharmBC2
BiharmonicBCsymbols = [symbol1, symbol2]

#tests(N)

function Spectraltests(N)
    axis = 3
    Nh = N÷2+1
    z = zeros(Float64, N)
    w = similar(z)
    ff = ["GC", "GL"]
    # Chebyshev
    for (j, F) in enumerate([r2c{SpecTransf{GL}, Float64}, r2c{SpecTransf{GC}, Float64}])
        if F == r2c{SpecTransf{GL}, Float64}
            C = Chebyshev{GL}()
            z, w = NodesWeights(C, z, w)
        elseif  F == r2c{SpecTransf{GC}, Float64}
            C = Chebyshev{GC}()
            z, w = NodesWeights(C, z, w)
        end
        x = collect(0:N-1)*2*pi/N
        X = Array{Float64}(N, N, N, 3)
        for (i, Xi) in enumerate(ndgrid(x, x, z)) X[view(i)...] = Xi end
        U = similar(X)
        U[view(1)...] = sin(X(3)).*cos(X(1)).*cos(X(2))
        U[view(2)...] = -cos(X(3)).*sin(X(1)).*cos(X(2))
        U[view(3)...] = (1. - X(3).^2).*sin(X(1))#.*cos(X(1))
        V = similar(U)
        U_hat = Array{Complex{Float64}}(Nh, N, N, 3)

        U_hat[view(3)...] = FCT(F(N), U(3), U_hat(3));
        V[view(3)...]  = IFCT(F(N), U_hat(3), V(3));
        @test isapprox(U(3), V(3))
        println("Test: Chebyshev transform for ", ff[j], " succeeded.")
    end
    # Dirichlet
    for (j, F) in enumerate([r2c{Dirichlet{GL}, Float64}, r2c{Dirichlet{GC}, Float64}])
        if F == r2c{Dirichlet{GL}, Float64}
            C = Dirichlet{GL}(N)
            z, w = NodesWeights(C, z, w)
        elseif  F == r2c{Dirichlet{GC}, Float64}
            C = Dirichlet{GC}(N)
            z, w = NodesWeights(C, z, w)
        end
        x = collect(0:N-1)*2*pi/N
        X = Array{Float64}(N, N, N, 3)
        for (i, Xi) in enumerate(ndgrid(x, x, z)) X[view(i)...] = Xi end
        U = similar(X)
        U[view(1)...] = sin(X(3)).*cos(X(1)).*cos(X(2))
        U[view(2)...] = -cos(X(3)).*sin(X(1)).*cos(X(2))
        U[view(3)...] = (1. - X(3).^2).*sin(X(1))
        V = similar(U)
        U_hat = Array{Complex{Float64}}(Nh, N, N, 3)

        U_hat[view(3)...] = FST(F(N), U(3), U_hat(3));
        V[view(3)...]  = IFST(F(N), U_hat(3), V(3));
        @test isapprox(U(3), V(3))
        println("Test: Dirichlet transform ", ff[j]," succeeded.")
    end
    # Neumann
    for (j, F) in enumerate([r2c{Neumann{GL}, Float64}, r2c{Neumann{GC}, Float64}])
        if F == r2c{Neumann{GL}, Float64}
            C = Dirichlet{GL}(N)
            z, w = NodesWeights(C, z, w)
        elseif  F == r2c{Neumann{GC}, Float64}
            C = Dirichlet{GC}(N)
            z, w = NodesWeights(C, z, w)
        end
        x = collect(0:N-1)*2*pi/N
        X = Array{Float64}(N, N, N, 3)
        for (i, Xi) in enumerate(ndgrid(x, x, z)) X[view(i)...] = Xi end
        U = similar(X)
        U[view(1)...] = sin(X(3)).*cos(X(1)).*cos(X(2))
        U[view(2)...] = -cos(X(3)).*sin(X(1)).*cos(X(2))
        U[view(3)...] = (X(3) - (1./3.)X(3).^3).*sin(X(1))
        V = similar(U)
        U_hat = Array{Complex{Float64}}(Nh, N, N, 3)

        U_hat[view(3)...] = FST(F(N), U(3), U_hat(3));
        V[view(3)...]  = IFST(F(N), U_hat(3), V(3));
        @test isapprox(U(3), V(3))
        println("Test: Neumann transform ", ff[j]," succeeded.")
    end
end

Spectraltests(N)
