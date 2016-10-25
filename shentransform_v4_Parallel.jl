# module shentransform_v4_Parallel

include("TDMA.jl")
include("PDMA.jl")

import MPI

# export *
#-----------------------------------------------------
#      Functions needed for all transforms
#-----------------------------------------------------
function wavenumbers3D{T<:Int}(N::T)
    Nh = N÷2+1
    ky = fftfreq(N, 1./N)
    kx = ky[1:(N÷2+1)]; kx[end] *= -1
    kz = collect(Float64, 0:N-3)
    K = Array{Float64}(Nh, N, N-2, 3)
    for (i, Ki) in enumerate(ndgrid(kx, ky, kz)) K[view(i)...] = Ki end
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
    # Global shape
    N::Array{Int, 1}
    # Communicator
    comm::MPI.Comm
    num_processes::Int
    rank::Int
    k::Array{Int, 4}
    ck::Vector{Float64}
    function Dirichlet(N, comm)
        num_processes = MPI.Comm_size(comm)
        rank = MPI.Comm_rank(comm)
        kz = collect(Float64, 0:N[3]-3)
        kx = rfftfreq(N[1], 1.0/N[1])
        ky = fftfreq(N[2], 1.0/N[2])[rank*div(N[2], num_processes)+1:(rank+1)*div(N[2], num_processes)]
        k = Array{Int}(tuple(push!([N[1]÷2+1, N[2]÷num_processes, N[3]-2], 3)...))
        for (i, Ki) in enumerate(ndgrid(kx, ky, kz)) k[view(i)...] = Ki end

        if T == GL
            ck = ones(eltype(Float64), N[3]-2); ck[1] = 2; ck[end] = 2
        elseif T == GC
            ck = ones(eltype(Float64), N[3]-2); ck[1] = 2
        end
        new(N, comm, num_processes, rank, k, ck)
    end
end
type Neumann{T<:NodeType} <: SpecTransf{T}
    # Global shape
    N::Array{Int, 1}
    # Communicator
    comm::MPI.Comm
    num_processes::Int
    rank::Int
    k::Vector{Float64}
    K::Array{Int, 4}
    ck::Vector{Float64}
    function Neumann(N, comm)
        num_processes = MPI.Comm_size(comm)
        rank = MPI.Comm_rank(comm)
        k = collect(Float64, 0:N[3]-3)
        kx = rfftfreq(N[1], 1.0/N[1])
        ky = fftfreq(N[2], 1.0/N[2])[rank*div(N[2], num_processes)+1:(rank+1)*div(N[2], num_processes)]
        K = Array{Int}(tuple(push!([N[1]÷2+1, N[2]÷num_processes, N[3]-2], 3)...))
        for (i, Ki) in enumerate(ndgrid(kx, ky, k)) K[view(i)...] = Ki end

        if T == GL
            ck = ones(eltype(Float64), N[3]-3); ck[end] = 2
        elseif T == GC
            ck = ones(eltype(Float64), N[3]-3)
        end
        new(N, comm, num_processes, rank, k, K, ck)
    end
end
type GeneralDirichlet{T<:NodeType} <: SpecTransf{T}
    a::Float64
    b::Float64
    # Global shape
    N::Array{Int, 1}
    # Communicator
    comm::MPI.Comm
    num_processes::Int
    rank::Int
    k::Vector{Float64}
    ck::Vector{Float64}
    ak::Vector{Float64}
    bk::Vector{Float64}
    K::Array{Int, 4}
    aK::Array{Float64, 3}
    bK::Array{Float64, 3}
    k1::Vector{Float64}
    ak1::Vector{Float64}
    bk1::Vector{Float64}

    function GeneralDirichlet(a, b, N, comm)
        num_processes = MPI.Comm_size(comm)
        rank = MPI.Comm_rank(comm)

        k = collect(Float64, 0:N[3]-3)
        k1 = collect(Float64, 0:N[3]-2)
        kx = rfftfreq(N[1], 1.0/N[1])
        ky = fftfreq(N[2], 1.0/N[2])[rank*div(N[2], num_processes)+1:(rank+1)*div(N[2], num_processes)]
        K = Array{Int}(tuple(push!([N[1]÷2+1, N[2]÷num_processes, N[3]-2], 3)...))
        for (i, Ki) in enumerate(ndgrid(kx, ky, k)) K[view(i)...] = Ki end

        ak = 0.5*(b-((-1.).^(-k))*a)
        aK = 0.5*(b-((-1.).^(-K(3)))*a)
        ak1 = 0.5*(b-((-1.).^(-k1))*a)
        bk = -1. +0.5*(b+((-1.).^(-k))*a)
        bK = -1. +0.5*(b+((-1.).^(-K(3)))*a)
        bk1 = -1. +0.5*(b+((-1.).^(-k))*a)
        if T == GL
            ck = ones(eltype(Float64), N[3]-2); ck[1] = 2; ck[end] = 2
        elseif T == GC
            ck = ones(eltype(Float64), N[3]-2); ck[1] = 2
        end
        new(a, b, N, comm, num_processes, rank, k, ck, ak, bk, K, aK, bK, k1, ak1, bk1)
    end
end
type Robin{T<:NodeType} <: SpecTransf{T}
    BC::Symbol
    # Global shape
    N::Array{Int, 1}
    # Communicator
    comm::MPI.Comm
    num_processes::Int
    rank::Int
    k::Vector{Float64}
    ck::Vector{Float64}
    ak::Vector{Float64}
    bk::Vector{Float64}
    K::Array{Int, 4}
    aK::Array{Float64, 3}
    bK::Array{Float64, 3}
    k1::Vector{Float64}
    ak1::Vector{Float64}
    bk1::Vector{Float64}

    function Robin(BC, N, comm)
        num_processes = MPI.Comm_size(comm)
        rank = MPI.Comm_rank(comm)

        k = collect(Float64, 0:N[3]-3)
        k1 = collect(Float64, 0:N[3]-2)
        kx = rfftfreq(N[1], 1.0/N[1])
        ky = fftfreq(N[2], 1.0/N[2])[rank*div(N[2], num_processes)+1:(rank+1)*div(N[2], num_processes)]
        K = Array{Int}(tuple(push!([N[1]÷2+1, N[2]÷num_processes, N[3]-2], 3)...))
        for (i, Ki) in enumerate(ndgrid(kx, ky, k)) K[view(i)...] = Ki end

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
            ck = ones(eltype(Float64), N[3]-2); ck[1] = 2; ck[end] = 2
        elseif T == GC
            ck = ones(eltype(Float64), N[3]-2); ck[1] = 2
        end
        new(BC, N, comm, num_processes, rank, k, ck, ak, bk, K, aK, bK, k1, ak1, bk1)
    end
end
type Biharmonic{T<:NodeType} <: SpecTransf{T}
    BiharmonicBC::Symbol
    # Global shape
    N::Array{Int, 1}
    # Communicator
    comm::MPI.Comm
    num_processes::Int
    rank::Int
    k::Vector{Float64}
    K::Array{Int, 4}
    ck::Vector{Float64}
    ak::Vector{Float64}
    bk::Vector{Float64}
    aK::Array{Float64, 3}
    bK::Array{Float64, 3}

    function Biharmonic(BiharmonicBC, N, comm)
        num_processes = MPI.Comm_size(comm)
        rank = MPI.Comm_rank(comm)

        k = collect(Float64, 0:N[3]-5)
        kx = rfftfreq(N[1], 1.0/N[1])
        ky = fftfreq(N[2], 1.0/N[2])[rank*div(N[2], num_processes)+1:(rank+1)*div(N[2], num_processes)]
        K = Array{Int}(tuple(push!([N[1]÷2+1, N[2]÷num_processes, N[3]-4], 3)...))
        for (i, Ki) in enumerate(ndgrid(kx, ky, k)) K[view(i)...] = Ki end

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
            ck = ones(eltype(Float64), N[3]-4); ck[1] = 2; ck[end] = 2
        elseif T == GC
            ck = ones(eltype(Float64), N[3]-4); ck[1] = 2
        end
        new(BiharmonicBC, N, comm, num_processes, rank, k, K, ck, ak, bk, aK, bK)
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
    N = last(size(fj))
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
#   FFT in x and y directions
#-------------------------------------------------------------------------------
type r2c{T<:Real}
    # Global shape
    N::Array{Int, 1}
    # Global size of domain
    L::Array{T, 1}
    # Communicator
    comm::MPI.Comm
    num_processes::Int
    rank::Int
    chunk::Int    # Amount of data to be send by MPI
    # Plans
    plan12::FFTW.rFFTWPlan{T}
    # Work arrays for transformations
    vT::Array{Complex{T}, 3}
    vT_view::Array{Complex{T}, 4}
    v::Array{Complex{T}, 3}
    v_view::Array{Complex{T}, 4}
    v_recv::Array{Complex{T}, 3}
    v_recv_view::Array{Complex{T}, 4}
    dealias::Array{Int, 1}

    # Constructor
    function r2c(N, L, comm)
        # Verify input
        Nh = N[1]÷2+1
        p = MPI.Comm_size(comm)
        Np = N÷p

        # Allocate work arrays
        vT, v = Array{Complex{T}}(Nh, N[2], Np[3]), Array{Complex{T}}(Nh, Np[2], N[3])
        vT_view, v_view = reshape(vT, (Nh, Np[2], p, Np[3])), reshape(v, (Nh, Np[2], Np[3], p))
        # For MPI.Alltoall! preallocate the receiving buffer
        v_recv = similar(v); v_recv_view = reshape(v_recv, (Nh, Np[2], Np[3], p))

        # Plan Fourier transformations
        A = zeros(T, (N[1], N[2], Np[3]))
        plan12 = plan_rfft(A, (1, 2), flags=FFTW.MEASURE)

        # Compute the inverse plans
        inv(plan12)

        chunk = Nh*Np[2]*Np[3]
        # Now we are ready
        new(N, L, comm, p, MPI.Comm_rank(comm), chunk,
            plan12, vT, vT_view, v, v_view, v_recv, v_recv_view)
    end
end
# Constructor
r2c{T<:Real}(N::Array{Int, 1}, L::Array{T, 1}, comm::Any) = r2c{T}(N, L, comm)

function real_shape{T<:Real}(F::r2c{T})
    (F.N[1], F.N[2], F.N[3]÷F.num_processes)
end

function complex_shape{T<:Real}(F::r2c{T})
    (F.N[1]÷2+1, F.N[2]÷F.num_processes, F.N[3])
end

function complex_shape_T{T<:Real}(F::r2c{T})
    (F.N[1]÷2+1, F.N[2], F.N[3]÷F.num_processes)
end

function complex_local_slice{T<:Real}(F::r2c{T})
    ((1, F.N[1]÷2+1),
     (F.rank*F.N[2]÷F.num_processes+1, (F.rank+1)*F.N[2]÷F.num_processes),
     (1, F.N[3]))
end

function complex_local_wavenumbers{T<:Real}(F::r2c{T}, Biharm::Int)
    if Biharm == 1
        z = collect(T, 0:F.N[3]-5)
    else
        z = collect(T, 0:F.N[3]-3)
    end
    (rfftfreq(F.N[1], 1.0/F.N[1]),
     fftfreq(F.N[2], 1.0/F.N[2])[F.rank*div(F.N[2], F.num_processes)+1:(F.rank+1)*div(F.N[2], F.num_processes)],
     z)
end

function get_local_wavenumbermesh{T<:Real}(F::r2c{T}, Biharm::Int)
    K = Array{T}(tuple(push!([complex_shape(F)...], 3)...))
    k = complex_local_wavenumbers(F, Biharm)
    for (i, Ki) in enumerate(ndgrid(k[1], k[2], k[3])) K[view(i)...] = Ki end
    K
end
function get_local_mesh{T<:Real}(F::r2c{T}, t::SpecTransf)
    # Real grid
    z = zeros(T, F.N[3])
    w = similar(z)
    z, w = NodesWeights(t, z, w)
    x = collect(0:F.N[1]-1)*F.L[1]/F.N[1]
    y = collect(0:F.N[2]-1)*F.L[2]/F.N[2]

    X = Array{T}(tuple(push!([real_shape(F)...], 3)...))
    for (i, Xi) in enumerate(ndgrid(x, y, z[F.rank*F.N[3]÷F.num_processes+1:(F.rank+1)*F.N[3]÷F.num_processes])) X[view(i)...] = Xi end
    X
end

function dealias{T<:Real}(F::r2c{T}, fu::AbstractArray{Complex{T}, 3})
    kk = complex_local_wavenumbers(F, 0)
    for (k, kz) in enumerate(kk[3])
        x = false
        # if abs(kz) > div(F.N[3], 3)
        # @inbounds fu[:, :, k] = 0.0
        #     continue
        # end
        for (j, ky) in enumerate(kk[2])
            if abs(ky) > div(F.N[2], 3)
               @inbounds fu[:, j, k] = 0
                continue
            end
            for (i, kx) in enumerate(kk[1])
                if (abs(kx) > div(F.N[1], 3))
                    @inbounds fu[i, j, k] = 0.0
                end
            end
        end
    end
end
@generated function FCS{T<:Real}(F::r2c{T}, t::SpecTransf, u::AbstractArray{T}, fu::AbstractArray{Complex{T}, 3})
    if t == Chebyshev{GL} || t == Chebyshev{GC}
        quote
            if F.num_processes > 1
                A_mul_B!(F.vT, F.plan12, u)
                permutedims!(F.v_view, F.vT_view, [1, 2, 4, 3])
                MPI.Alltoall!(F.v_recv_view, F.v_view, F.chunk, F.comm)
                fu = fastChebScalar(t, F.v_recv, fu)
                fu
            else
                A_mul_B!(F.vT, F.plan12, u)
                fu = fastChebScalar(t, F.vT, fu)
                fu
            end
        end
    end
end
@generated function IFCT{T<:Real}(F::r2c{T}, t::SpecTransf, fu::AbstractArray{Complex{T}}, u::AbstractArray{T}, dealias_fu::Int=0)
    if t == Chebyshev{GL} || t == Chebyshev{GC}
        quote
            if F.num_processes > 1
                F.v = ifct(t, fu, F.v)
                # F.v[:] = fu
                # if dealias_fu == 1
                #     dealias(F, F.v)
                MPI.Alltoall!(F.v_recv_view, F.v_view, F.chunk, F.comm)
                permutedims!(F.vT_view, F.v_recv_view, [1, 2, 4, 3])
                A_mul_B!(u, F.plan12.pinv, F.vT)
                u
            else
                F.v = ifct(t, fu, F.v)
                A_mul_B!(u, F.plan12.pinv, F.v)
                u
            end
        end
    end
end
@generated function FCT{T<:Real}(F::r2c{T}, t::SpecTransf, u::AbstractArray{T}, fu::AbstractArray{Complex{T}})
    if t == Chebyshev{GL} || t == Chebyshev{GC}
        quote
            if F.num_processes > 1
                A_mul_B!(F.vT, F.plan12, u)
                permutedims!(F.v_view, F.vT_view, [1, 2, 4, 3])
                MPI.Alltoall!(F.v_recv_view, F.v_view, F.chunk, F.comm)
                fu = fct(t, F.v_recv, fu)
                fu
            else
                A_mul_B!(F.vT, F.plan12, u)
                fu = fct(t, F.vT, fu)
                fu
            end
        end
    end
end

@generated function FSS{T<:Real}(F::r2c{T}, t::SpecTransf, u::AbstractArray{T}, fu::AbstractArray{Complex{T}, 3})
    quote
        if F.num_processes > 1
            A_mul_B!(F.vT, F.plan12, u)
            permutedims!(F.v_view, F.vT_view, [1, 2, 4, 3])
            MPI.Alltoall!(F.v_recv_view, F.v_view, F.chunk, F.comm)
            fu = fastShenScalar(t, F.v_recv, fu)
            fu
        else
            A_mul_B!(F.vT, F.plan12, u)
            fu = fastShenScalar(t, F.vT, fu)
            fu
        end
    end
end
@generated function IFST{T<:Real}(F::r2c{T}, t::SpecTransf, fu::AbstractArray{Complex{T}}, u::AbstractArray{T}, dealias_fu::Int=1)
    quote
        if F.num_processes > 1
            F.v[:] = ifst(t, fu, F.v)
            if dealias_fu == 1
                dealias(F, F.v)
            end
            MPI.Alltoall!(F.v_recv_view, F.v_view, F.chunk, F.comm)
            permutedims!(F.vT_view, F.v_recv_view, [1, 2, 4, 3])
            A_mul_B!(u, F.plan12.pinv, F.vT)
            u
        else
            F.v = ifst(t, fu, F.v)
            A_mul_B!(u, F.plan12.pinv, F.v)
            u
        end
    end
end
@generated function FST{T<:Real}(F::r2c{T}, t::SpecTransf, u::AbstractArray{T}, fu::AbstractArray{Complex{T}})
    quote
        if F.num_processes > 1
            A_mul_B!(F.vT, F.plan12, u)
            permutedims!(F.v_view, F.vT_view, [1, 2, 4, 3])
            MPI.Alltoall!(F.v_recv_view, F.v_view, F.chunk, F.comm)
            fu = fst(t, F.v_recv, fu)
            fu
        else
            A_mul_B!(F.vT, F.plan12, u)
            fu = fst(t, F.vT, fu)
            fu
        end
    end
end

# end
