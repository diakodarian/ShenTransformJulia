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
function chebyshevDirichletPolynomials(N, x)
    l = length(x)
    phi = zeros(l, N)
    for k in 1:N-2
        for j in 1:l
            phi[j,k] = cos((k-1)*acos(x[j])) - cos((k+1)*acos(x[j]))
        end
    end
    return phi
end
function chebyshevNeumannPolynomials(N, x)
    l = length(x)
    phi = zeros(l, N)
    k = wavenumbers(N)
    Tk = chebyshevPolynomials(N, x)
    for i in 2:N-2
        for j in 1:l
            phi[j,i] = Tk[j,i] - (k[i]./(k[i]+2.)).^2.*Tk[j,i+2]
        end
    end
    return phi
end
function chebyshevRobinPolynomials(BC, N, x)
    l = length(x)
    phi = zeros(l, N)
    k = wavenumbers(N)
    ak, bk = shenCoefficients(k, BC)
    Tk = chebyshevPolynomials(N, x)
    for i in 1:N-2
        for j in 1:l
            phi[j,i] = Tk[j,i] + ak[i]*Tk[j,i+1]+ bk[i]*Tk[j,i+2]
        end
    end
    return phi
end
function wavenumbers{T<:Int}(N::T)
    return collect(0:N-3)
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
function dct(x, i::Int, axis::Int=1)
    if typeof(x)==Complex{Float64}
        xreal = dct_type(real(x), i, axis)
        ximag = dct_type(imag(x), i, axis)
        return xreal + ximag*1.0im
    else
        return dct_type(x, i, axis)
    end
end
#-----------------------------------------------------
#       Chebyshev transforms
#-----------------------------------------------------
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
function fct(fj, cj, quad, axis)
    # Fast Chebyshev transform.
    N = length(fj)
    if quad == "GC"
        cj = dct(fj, 2, axis)
        cj /= N
        cj[1] /= 2
    elseif quad == "GL"
        cj = dct(fj, 1, axis)/(N-1)
        cj[1] /= 2
        cj[end] /= 2
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
        cj[1:2:end] += 0.5*fk[end]
        cj[2:2:end] -= 0.5*fk[end]
    end
    return cj
end
function chebDerivativeCoefficients(fk, fl)
    N = length(fk)
    fl[end] = 0.0
    fl[end-1] = 2*(N-1)*fk[end]
    for k in N-2:-1:2
        fl[k] = 2*k*fk[k+1]+fl[k+2]
    end
    fl[1] = fk[2] + 0.5*fl[3]
    return fl
end
function fastChebDerivative(fj, fd, quad, axis)
    # Compute derivative of fj at the same points.
    fk = Array{Float64}(length(fj))
    fkd = Array{Float64}(length(fj))
    fk = fct(fj, fk, quad, axis)
    fkd = chebDerivativeCoefficients(fk, fkd)
    fd  = ifct(fkd, fd, quad, axis)
    return fd
end
#----------------------ooo----------------------------
#-----------------------------------------------------
#                TDMA_1D
#-----------------------------------------------------
function TDMA_1D(a, b, c, d)
    n = length(b)
    m = length(a)
    k = n - m
    for i in 1:m
        d[i + k] -= d[i] * a[i] / b[i]
        b[i + k] -= c[i] * a[i] / b[i]
    end
    for i in m:-1:1
        d[i] -= d[i + k] * c[i] / b[i + k]
    end
    for i in 1:n
        d[i] /= b[i]
    end
    return d
end
#----------------------ooo----------------------------
#-----------------------------------------------------
#                PDMA_1D
#-----------------------------------------------------
function PDMA_1D(d, e, f, b)
    n = length(d)
    for k in 1:(n-2)
        lam = e[k]/d[k]
        d[k+1] = d[k+1] - lam*e[k]
        e[k+1] = e[k+1] - lam*f[k]
        e[k] = lam
        lam = f[k]/d[k]
        d[k+2] = d[k+2] - lam*f[k]
        f[k] = lam
    end
    lam = e[n-1]/d[n-1]
    d[n] = d[n] - lam*e[n-1]
    e[n-1] = lam

    b[2] = b[2] - e[1].*b[1]
    for k in 3:n
        b[k] = b[k] - e[k-1]*b[k-1] - f[k-2]*b[k-2]
    end
    b[end] = b[end]/d[end]
    b[end-1] = b[end-1]/d[end-1] - e[n-1]*b[end]
    for k in n-2:-1:1
        b[k] = b[k]/d[k] - e[k]*b[k+1] - f[k]*b[k+2]
    end
    return b
end
#----------------------ooo----------------------------
#-----------------------------------------------------
#           Shen Dirichlet transform
#-----------------------------------------------------
function fastShenScalarD(fj, fk, quad, axis)
        fk = fastChebScalar(fj, fk, quad, axis)
        fk[1:end-2] -= fk[3:end]
    return fk
end
function ifstD(fk, fj, quad, axis)
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
    fj = ifct(w_hat, fj, quad, axis)
    return fj
end
function fstD(fj, fk, quad, aixs)
    """Fast Shen transform
    """
    fk = fastShenScalarD(fj, fk, quad, axis)
    N = length(fj)
    if quad == "GC"
        ck = ones(eltype(fk), N-2); ck[1] = 2
    elseif quad == "GL"
        ck = ones(eltype(fk), N-2); ck[1] = 2; ck[end] = 2  # Note!! Shen paper has only ck[0] = 2, not ck[-1] = 2. For Gauss points ck[-1] = 1, but not here!
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
function fastShenDerivativeD(fj, df, quad, axis)
    """
    Fast derivative of fj using Shen Dirichlet basis
    """
    N = length(fj)
    fk = Array{Float64}(N); fk_1 = Array{Float64}(N-2)
    fk_0 = similar(fk_1);
    fk = fstD(fj, fk, quad, axis)
    k = wavenumbers(N)
    fk_0 = fk[1:end-2].*(1.-((k+2)./k))
    fk_1 = chebDerivativeCoefficients(fk_0, fk_1)
    fk_2 = 2*fk[1:end-2].*(k+2)

    df_hat = zeros(eltype(fj), N)
    df_hat[1:end-2] = fk_1 - vcat(0, fk_2[1:end-1])

    df_hat[end-1] = -fk_2[end]
    df = ifct(df_hat, df, quad, axis)
    return df
end
#----------------------ooo----------------------------
#-----------------------------------------------------
#           Shen Neumann transform
#-----------------------------------------------------
function fastShenScalarN(fj, fk, quad, axis)
    """Fast Shen scalar product.
    Chebyshev transform taking into account that phi_k = T_k - (k/(k+2))**2*T_{k+2}
    Note, this is the non-normalized scalar product
    """
    k  = wavenumbers(length(fj))
    fk = fastChebScalar(fj, fk, quad, axis)
    fk[1:end-2] -= ((k./(k+2)).^2).*fk[3:end]
    return fk
end
function ifstN(fk, fj, quad, axis)
    """Fast inverse Shen scalar transform
    """
    if length(size(fk))==3
        k = wavenumbers(fk.shape)
        w_hat = zeros(eltype(fk), size(fk))
        w_hat[2:end-2] = fk[2:end-2]
        w_hat[4:end] -= (k[2:end]./(k[2:end]+2)).^2.*fk[2:end-2]
    elseif length(size(fk))==1
        k = wavenumbers(length(fk))
        w_hat = zeros(eltype(fk), length(fk))
        w_hat[2:end-2] = fk[2:end-2]
        w_hat[4:end] -= (k[2:end]./(k[2:end]+2)).^2.*fk[2:end-2]
    end
    fj = ifct(w_hat, fj, quad, axis)
    return fj
end
function fstN(fj, fk, quad, axis)
    """Fast Shen transform.
    """
    fk = fastShenScalarN(fj, fk, quad, axis)
    N = length(fj)
    k = wavenumbers(N)
    ck = ones(eltype(k), N-3)
    if quad == "GL" ck[end] = 2 end # Note not the first since basis phi_0 is not included
    a = (-pi/2)*ones(eltype(fk), N-5).*(k[2:end-2]./(k[2:end-2]+2.)).^2
    b = pi/2*(1.+ck.*(k[2:end]./(k[2:end]+2.)).^4)
    c = a
    fk[2:end-2] = TDMA_1D(a, b, c, fk[2:end-2])
    return fk
end
function fastShenDerivativeN(fj, df, quad, axis)
    """
    Fast derivative of fj using Shen Neumann basis
    """
    N = length(fj)
    fk = Array{Float64}(N); fk_1 = Array{Float64}(N-2)
    fk = fstN(fj, fk, quad, axis)
    k = wavenumbers(N)
    fk_0 = fk[2:end-2].*(1.0 - ( k[2:end]./(k[2:end]+2) ) )
    fk_tmp = vcat(0, fk_0)
    fk_1 = chebDerivativeCoefficients(fk_tmp, fk_1)
    fk_2 = 2*fk[2:end-2].*(k[2:end].^2)./(k[2:end]+2)

    df_hat = zeros(eltype(fk), N)
    df_hat[1] = fk_1[1]
    df_hat[2:end-2] = fk_1[2:end] - vcat(0, fk_2[1:end-1])
    df_hat[end-1] = -fk_2[end]
    df = ifct(df_hat, df, quad, axis)
    return df
end
#----------------------ooo----------------------------
#-----------------------------------------------------
#           Shen Robin transform
#-----------------------------------------------------
function shenCoefficients(k, BC)
    """
    Shen basis functions given by
    phi_k = T_k + a_k*T_{k+1} + b_k*T_{k+2},
    satisfy the imposed Robin (mixed) boundary conditions for a unique set of {a_k, b_k}.
    """
    if BC == "ND"
        ak = -4*(k+1)./((k+1).^2 .+ (k+2).^2)
    elseif BC == "DN"
        ak = 4*(k+1)./((k+1).^2 .+ (k+2).^2)
    end
    bk = -((k.^2 + (k+1).^2)./((k+1).^2 .+ (k+2).^2))
    return ak, bk
end
function fastShenScalarR(fj, fk, quad, axis, BC)
    """Fast Shen scalar product
    B u_hat = sum_{j=0}{N} u_j phi_k(x_j) w_j,
    for Shen basis functions given by
    phi_k = T_k + a_k*T_{k+1} + b_k*T_{k+2}
    """
    fk = fastChebScalar(fj, fk, quad, axis)
    k  = wavenumbers(length(fj))
    ak, bk = shenCoefficients(k, BC)
    fk_tmp = fk
    fk[1:end-2] = fk_tmp[1:end-2] + ak.*fk_tmp[2:end-1] + bk.*fk_tmp[3:end]
    return fk
end
function ifstR(fk, fj, quad, axis, BC)
    """Fast inverse Shen scalar transform for Robin BC.
    """
    k = wavenumbers(length(fk))
    w_hat = zeros(eltype(fk), length(fk))
    ak, bk = shenCoefficients(k, BC)
    w_hat[1:end-2] = fk[1:end-2]
    w_hat[2:end-1] += ak.*fk[1:end-2]
    w_hat[3:end]   += bk.*fk[1:end-2]
    fj = ifct(w_hat, fj, quad, axis)
    return fj
end
function fstR(fj, fk, quad, axis, BC)
    """Fast Shen transform for Robin BC.
    """
    fk = fastShenScalarR(fj, fk, quad, axis, BC)
    N = length(fj)
    k = wavenumbers(N)
    k1 = wavenumbers(N+1)
    ak, bk = shenCoefficients(k, BC)
    ak1, bk1 = shenCoefficients(k1, BC)
    if quad == "GL"
        ck = ones(eltype(k), N-2); ck[1] = 2
    elseif quad == "GC"
        ck = ones(eltype(k), N-2); ck[1] = 2; ck[end] = 2
    end
    a = (pi/2)*(ck .+ ak.^2 .+ bk.^2)
    b = (pi/2)*ones(eltype(k), N-3).*(ak[1:end-1] .+ ak1[2:end-1].*bk[1:end-1])
    c = (pi/2)*ones(eltype(k), N-4).*bk[1:end-2]

    fk[1:end-2] = PDMA_1D(a, b, c, fk[1:end-2])
    return fk
end
#----------------------ooo----------------------------
# ----------------------------------------------------
#               Shen Biharmonic transform
# ----------------------------------------------------
function Biharmonicwavenumbers{T<:Int}(N::T)
    return collect(0:N-5)
end
function fastShenScalarB(fj, fk, quad, axis)
    """Fast Shen scalar product.
    """
    k  = Biharmonicwavenumbers(length(fj))
    Tk = fk
    Tk = fastChebScalar(fj, Tk, quad, axis)
    fk[:] = Tk[:]
    fk[1:end-4] -= 2*((k+2)./(k+3)).*Tk[3:end-2]
    fk[1:end-4] += ((k+1)./(k+3)).* Tk[5:end]
    fk[end-3:end] = 0.0
    return fk
end
function ifstB(fk, fj, quad, axis)
    """Fast inverse Shen scalar transform
    """
    w_hat = fk
    k = Biharmonicwavenumbers(length(fk))
    factor1 = -2*(k+2)./(k+3)
    factor2 = (k+1)./(k+3)

    w_hat[:] = 0.0
    w_hat[1:end-4] = fk[1:end-4]
    w_hat[3:end-2] += factor1.*fk[1:end-4]
    w_hat[5:end]   += factor2.*fk[1:end-4]
    fj = ifct(w_hat, fj, quad, axis)
    return fj
end

function fstB(fj, fk, quad, axis)
    """Fast Shen transform .
    """
    fk = fastShenScalar(fj, fk, quad, axis)
    N = length(fj)
    k = Biharmonicwavenumbers(N)
    N -= 4
    ckp = ones(N)
    if quad == "GL"
        ck = ones(N); ck[1] = 2
    elseif quad == "GC"
        ck = ones(N); ck[1] = 2; ck[end] = 2
    end
    c = (ck + 4*((k+2)./(k+3)).^2 + ckp.*((k+1)./(k+3)).^2)*pi/2.
    d = -((k[1:end-2]+2)./(k[1:end-2]+3) + (k[1:end-2]+4).*(k[1:end-2]+1)./((k[1:end-2]+5).*(k[1:end-2]+3)))*pi
    e = (k[1:end-4]+1)./(k[1:end-4]+3)*pi/2
    b = d
    a = e
    fk[1:end-4] = PDMA_1D_2Version(a, b, c, d, e,fk[1:end-4],fk[1:end-4])
    return fk
end
#----------------------ooo----------------------------
# ----------------------------------------------------
#                       Tests
# ----------------------------------------------------
using Base.Test
N = 2^3;
axis = 1
BC = "ND"
quad = "GL"
if quad == "GC"
    points, weights = chebyshevGaussNodesWeights(N);
elseif quad == "GL"
    points, weights = chebyshevGaussLobattoNodesWeights(N);
end
x = collect(0:N-1)*2*pi/N
z = points
U, V, U_hat = similar(z), similar(z), similar(z)
U = z .- (1./3.)*z.^3;
#U = z-(8./13.)*(-1. + 2*z.^2) - (5./13.)*(-3*z + 4*z.^3)
println(U)
U_hat = fastShenScalarB(U, U_hat, quad, axis)
println(U_hat)
# U_hat = fstR(U, U_hat, quad, axis, BC)
# V = ifstR(U_hat, V, quad, axis, BC);
# @test isapprox(U, V)

# V = fastShenDerivativeN(U, V, quad, axis)
# U_hat = 1.- z.^2;
# @test isapprox(U_hat, V)

#U_hat = fstR(U, U_hat, quad, axis, BC);
# V = ifstN(U_hat, V, quad, axis);
# @test isapprox(U, V)
# println("1D test succeeded!")

# Test the derivative
# V = 2*z;
# U_hat = fastChebDerivative(U, U_hat, quad, axis)
# @test isapprox(U_hat, V)
# println("Derivative test succeeded!")

# The plots of polynomials
# testPolys = 1
# using PyCall
# using PyPlot
# if testPolys == 1
#
#     n = 8
#     #phi = chebyshevDirichletPolynomials(n, z)
#     #phi = chebyshevNeumannPolynomials(n, z)
#     # phi = chebyshevPolynomials(n, z)
#     phi = chebyshevRobinPolynomials(BC, n, z)
#     for k in 1:n
#         plot(z[1:end], phi[1:end, k])
#     end
# end
#----------------------ooo----------------------------
