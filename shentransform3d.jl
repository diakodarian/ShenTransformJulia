#! /usr/bin/julia
#=
Created on Sun 11 Sep 21:03:09 2016

@author: Diako Darian

Fast transforms for pure Chebyshev basis or
Shen's Chebyshev basis:

  phi_k = T_k + a_k*T_{k+1} + b_k*T_{k+2},

where for homogeneous Dirichlet boundary conditions:

    a_k = 0  and  b_k = -1

For homogeneous Neumann boundary conditions:

     a_k = 0  and  b_k = -(k/k+2)**2

For Robin/mixed boundary conditions:

     a_k = \pm 4*(k+1)/((k+1)**2 + (k+2)**2)  and
     b_k = -(k**2 + (k+1)**2)/((k+1)**2 + (k+2)**2)

a_k is positive for Dirichlet BC at x = -1 and Neumann BC at x = +1 (DN),
and it is negative for Neumann BC at x = -1 and Dirichlet BC at x = +1 (ND)

It is therefore possible to choose DN boundary conditions (BC = "DN")
or ND boundary conditions (BC = "ND").

Use either Chebyshev-Gauss (GC) or Gauss-Lobatto (GL)
points in real space.

The ChebyshevTransform may be used to compute derivatives
through fast Chebyshev transforms.
=#
function chebyshevPolynomials(N, x)
    l = length(x)
    T = zeros(l, N)
    for i in 1:N
        for j in 1:l
            T[j,i] = cos((i-1)*acos(x[j]))
        end
    end
    return T
end
function wavenumbers{T<:Int}(N::T)
    return collect(1:N-2)
end
function chebyshevGaussNodesWeights{T<:Int}(N::T)
    x = Array{Float64}(N)
    w = similar(x)
    for i = 1:N
        x[i] = cos((2.0*i-1.0)*pi/(2.0*N))
    end
    w[:] = pi/N
    return x, w
end
function chebyshevGaussLobattoNodesWeights(N)
    x =  zeros(Float64, N)
    w = similar(x)
    for i = 1:N
        x[i] = cos((i-1.0)*pi/(N-1))
    end
    w[1] = w[end] = pi/(2*(N-1))
    w[2:end-1] = pi/(N-1)
    return x, w
end
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
function chebDerivativeCoefficients(fk, fl)
    N = length(fk)
    fl[end] = 0.0
    fl[end-1] = 2*(N-1)*fk[end]
    for k in N-2:-1:2
        fl[k] = 2*k*fk[k+1]+fl[k+2]
    end
    fl[1] = fk[2] + 0.5*fl[3]
end

function chebDerivativeCoefficients_3D(fk, fl)
    for i in 1:size(fk,1)
        for j in 1:size(fk,2)
            chebDerivativeCoefficients(fk[i,j,1:end], fl[i,j,1:end])
        end
    end
end

function chebDerivative_3D(fj, fd, fk, fkd)
    fk = fct(fj, fk)
    fkd = chebDerivativeCoefficients_3D(fk, fkd)
    fd = ifct(fl, fd)
    return fd
end

# This function is not complete
function fastChebDerivative(fj, fd, fk, fkd)
    # Compute derivative of fj at the same points.
    fk = fct(fj, fk, quad)
    fkd = chebDerivativeCoefficients(fk, fkd)
    fd  = ifct(fkd, fd, quad)
    return fd
end

function fct(fj, cj, quad, axis)
    # Fast Chebyshev transform.
    N = last(size(fj))
    if quad == "GC"
        cj = dct(fj, 2, axis)
        cj /= N
        cj[:, :, 1] /= 2
    elseif quad == "GL"
        cj = dct(fj, 1, axis)/(N-1)
        cj[:,:, 1] /= 2
        cj[:,:, end] /= 2
    end
    return cj
end

function ifct(fk, cj, quad, axis)
    # Inverse fast Chebyshev transform.
    if quad == "GC"
        cj = 0.5*dct(fk, 3, axis)
        cj += 0.5*fk[1]
    elseif quad == "GL"
        cj = 0.5*dct(fk, 1, axis)
        cj += 0.5*fk[1]
        for k in 1:div((N+1),2)
            cj[:,:,(2*k-1)] += 0.5*fk[:, :, end]
        end
        for k in 1:div(N,2)
            cj[:,:,2*k] -= 0.5*fk[:, : , end]
        end
    end
    return cj
end

function fastChebScalar(fj, fk, quad, axis)
    # Fast Chebyshev scalar product.
    N = length(fj)
    if quad == "GC"
        fk = dct(fj, 2, axis)*pi/(2*N)
    elseif quad == "GL"
        fk = dct(fj, 1, axis)*pi/(2*(N-1))
    end
    return fk
end

# --------------------------------------------------
using Base.Test
using Compat
view(k::Int, N::Int=4) = [fill(Colon(), N-1)..., k]

"View of A with last coordinate fixed at k"
@compat function call{T, N}(A::Array{T, N}, k::Int)
    @assert 1 <= k <= size(A, N)
    indices = [fill(Colon(), N-1)..., k]
    slice(A, indices...)
end

"Helper"
function ndgrid_fill(a, v, s, snext)
    for j = 1:length(a)
        a[j] = v[div(rem(j-1, snext), s)+1]
    end
end

"numpy.mgrid[v1, v2, v3, ...]"
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
# --------------------------------------------------
# Tests
# --------------------------------------------------
N = 2^2;
axis = 3
quad = "GL"
if quad == "GC"
    points, weights = chebyshevGaussNodesWeights(N);
elseif quad == "GL"
    points, weights = chebyshevGaussLobattoNodesWeights(N);
end
x = collect(0:N-1)*2*pi/N
z = points
X = Array{Float64}(N, N, N,3)
for (i, Xi) in enumerate(ndgrid(x, x, z)) X[view(i)...] = Xi end
U, V, U_hat = similar(X),  similar(X),  similar(X)
U[view(1)...] = sin(X(3)).*cos(X(1)).*cos(X(2))
U[view(2)...] = -cos(X(3)).*sin(X(1)).*cos(X(2))
U[view(3)...] = X(3).^2

U_hat[view(3)...] = fct(U(3), U_hat(3), quad, 3);
V[view(3)...]  = ifct(U_hat(3), V(3), quad, 3);
@test isapprox(U(3), V(3))
println("3D test succeeded!")
