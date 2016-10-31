# Linear algebra solvers from:
# https://github.com/spectralDNS/spectralDNS/blob/master/spectralDNS/shen/LUsolve.pyx

function BackSubstitution_1D(u, f)
    """
    Solve Ux = f, where U is an upper diagonal Shen Poisson matrix aij
    """
    n = length(f)
    for i in n:-1:1
        for l in i+2:2:n
            f[i] += 4.*pi*i*u[l]
        end
        u[i] = -f[i] / (2.*pi*i*(i+1.))
    end
    u
end
function BackSubstitution_1D_complex(u, f)
    """
    Solve Ux = f, where U is an upper diagonal Shen Poisson matrix aij
    """
    n = length(f)
    for i in n:-1:1
        for l in i+2:2:n
            f[i] += 4.*pi*i*u[l]
            # real(f[i]) = real(f[i]) + 4.*pi*i*real(u[l])
            # imag(f[i]) = imag(f[i]) + 4.*pi*i*imag(u[l])
        end
        u[i] = -f[i] / (2.pi*i*(i+1.))
    end
    u
end
function BackSubstitution_3D(u, f)
    n = size(u, 3)
    fc = zeros(eltype(f), n)
    uc = zeros(eltype(u), n)
    for ii in 1:size(u,1)
        for jj in 1:size(u,2)
            for i in 1:n
                fc[i] = f[ii, jj, i]
                uc[i] = u[ii, jj, i]
            end
            u[ii, jj, :] = BackSubstitution_1D(uc, fc)
        end
    end
    u
end
function BackSubstitution_3D_complex(u, f)
    n = size(u, 3)
    fc = Array(eltype(f), n)
    uc = similar(fc)
    for ii in 1:size(u, 1)
        for jj in 1:size(u, 2)
            for i in 1:n
                fc[i] = f[ii, jj, i]
                uc[i] = u[ii, jj, i]
            end
            u[ii, jj, :] = BackSubstitution_1D_complex(uc, fc)
        end
    end
    u
end
function LU_Helmholtz_3D(N, neumann, GC, alfa, d0, d1, d2, L)
    for jj in 1:size(d0, 2)
        for ii in 1:size(d0, 1)
            d0[ii, jj, :, :], d1[ii, jj, :, :], d2[ii, jj, :, :], L[ii, jj, :, :] = LU_Helmholtz_1D(N, neumann, GC, alfa[ii, jj], d0[ii, jj, :, :], d1[ii, jj, :, :], d2[ii, jj, :, :], L[ii, jj, :, :])
        end
    end
    d0, d1, d2, L
end
function LU_Helmholtz_1D(N, neumann, GC, alfa, d0,  d1, d2, L)
    if neumann == 0
        if ndims(d0)>2
            d0[1,1,1,:], d1[1,1,1,:], d2[1,1,1,:], L[1,1,1,:] = LU_oe_Helmholtz_1D(N, 0, GC, alfa, d0[1,1,1,:], d1[1,1,1,:], d2[1,1,1,:], L[1,1,1,:])
            d0[1,1,2,:], d1[1,1,2,:], d2[1,1,2,:], L[1,1,2,:] = LU_oe_Helmholtz_1D(N, 1, GC, alfa, d0[1,1,2,:], d1[1,1,2,:], d2[1,1,2,:], L[1,1,2,:])
        else
            d0[1,:], d1[1,:], d2[1,:], L[1,:] = LU_oe_Helmholtz_1D(N, 0, GC, alfa, d0[1,:], d1[1,:], d2[1,:], L[1,:])
            d0[2,:], d1[2,:], d2[2,:], L[2,:] = LU_oe_Helmholtz_1D(N, 1, GC, alfa, d0[2,:], d1[2,:], d2[2,:], L[2,:])
        end
    else
        if ndims(d0)>2
            d0[1,1,1,:], d1[1,1,1,:], d2[1,1,1,:], L[1,1,1,:] = LU_oe_HelmholtzN_1D(N, 0, GC, alfa, d0[1,1,1,:], d1[1,1,1,:], d2[1,1,1,:], L[1,1,1,:])
            d0[1,1,2,:], d1[1,1,2,:], d2[1,1,2,:], L[1,1,2,:] = LU_oe_HelmholtzN_1D(N, 1, GC, alfa, d0[1,1,2,:], d1[1,1,2,:], d2[1,1,2,:], L[1,1,2,:])
        else
            d0[1,:], d1[1,:], d2[1,:], L[1,:] = LU_oe_HelmholtzN_1D(N, 0, GC, alfa, d0[1,:], d1[1,:], d2[1,:], L[1,:])
            d0[2,:], d1[2,:], d2[2,:], L[2,:] = LU_oe_HelmholtzN_1D(N, 1, GC, alfa, d0[2,:], d1[2,:], d2[2,:], L[2,:])
        end
    end
    d0, d1, d2, L
end
function LU_oe_Helmholtz_1D(N, odd, GC, alfa, d0, d1, d2, L)
    c0 = 0
    kx = alfa*alfa
    if odd == 0
        M = div(N-3,2)
    else
        M = div(N-4,2)
    end
    bij_e = Array{Float64}(M+1)
    # Direct LU decomposition using the fact that there are only three unique diagonals in U and one in L
    bij_e[:] = pi
    if odd == 0
        bij_e[1] *= 1.5
        if (N % 2 == 1) & GC
            bij_e[M+1] *= 1.5
        end
    else
        if (N % 2 == 0) & GC
            bij_e[M+1] *= 1.5
        end
    end
    d = Array{Float64}(M+1)
    s = Array{Float64}(M)
    g = Array{Float64}(M-1)

    if odd == 1
        c0 = 1
    end
    for i in 1:M+1
        d[i] = 2*pi*(2*(i-1)+1+c0)*(2*(i-1)+2+c0) + kx*bij_e[i]
        if i <= M
            s[i] = 4*pi*(2*(i-1)+1+c0) - kx*pi/2
        end
        if i <= M-1
            g[i] = 4*pi*(2*(i-1)+1+c0)
        end
    end
    d0[1] = d[1]
    d1[1] = s[1]
    d2[1] = g[1]
    L[1] = (-kx*pi/2) / d0[1]
    d0[2] = d[2] - L[1]*s[1]
    d1[2] = s[2] - L[1]*g[1]
    d2[2] = g[2] - L[1]*g[1]
    for i in 2:M
        L[i] = (-kx*pi/2) / d0[i]
        d0[i+1] = d[i+1] - L[i]*d1[i]
        if i <= M-1
            d1[i+1] = s[i+1] - L[i]*d2[i]
        end
        if i <= M-2
            d2[i+1] = g[i+1] - L[i]*d2[i]
        end
    end
    d0, d1, d2, L
end
function LU_oe_HelmholtzN_1D(N, odd, GC, alfa, d0, d1, d2, L)
    c0 = 0
    halfpi = pi/2.0
    kx = alfa*alfa

    if odd == 0
        M = div(N-4,2)
        bii = Array{Float64}(div(N-2,2))
        bip = Array{Float64}(div(N-4,2))
        bim = Array{Float64}(div(N-4,2))
    else
        M = div(N-5,2)
    end
    # Direct LU decomposition using the fact that there are only three unique diagonals in U and one in L
    if odd == 1
        c0 = 1
        bii = Array{Float64}(div(N-3,2))
        bip = Array{Float64}(div(N-5,2))
        bim = Array{Float64}(div(N-5,2))
    end
    for i in 1+c0:2:N-3
        ks = i*i
        kp2 = i+2
        kkp2 = (i*1.0)/(i+2)
        kp24 = kkp2*kkp2/(kp2*kp2)
        push!(bii, halfpi*(1.0/ks+kp24))
        shift!(bii)
        if i < N-4
            push!(bip, -halfpi*kp24)
            shift!(bip)
        end
        if i > 2
            push!(bim, -halfpi/ks)
            shift!(bim)
        end
    end
    if GC
        if odd == 0
            if N % 2 == 0
                bii[M] = pi/2.0*(1.0/((N-3)*(N-3))+2.0*(((N-3)*1.0)/(N-1))^2/((N-1)*(N-1)))
            end
        else
            if N % 2 == 1
                bii[M] = pi/2.0*(1.0/((N-3)*(N-3))+2.0*(((N-3)*1.0)/(N-1))^2/((N-1)*(N-1)))
            end
        end
    end
    d = Array{Float64}(M+1)
    s = Array{Float64}(M)
    g = Array{Float64}(M-1)

    for i in 1:M+1
        kk = 2.*(i-1)+1+c0
        kp2 = kk+2.
        d[i] = 2*(pi*(kk+1))/kp2 + kx*bii[i]
        if i < M+1
            kd = 4*(pi*(kk+1))/(kp2*kp2)
            s[i] = kd + kx*bip[i]
        end
        if i < M
            g[i] = kd
        end
    end

    d0[1] = d[1]
    d1[1] = s[1]
    d2[1] = g[1]
    L[1] = (kx*bim[1]) / d0[1]
    d0[2] = d[2] - L[1]*s[1]
    d1[2] = s[2] - L[1]*g[1]
    d2[2] = g[2] - L[1]*g[1]
    for i in 2:M
        L[i] = (kx*bim[i]) / d0[i]
        d0[i+1] = d[i+1] - L[i]*d1[i]
        if i < M
            d1[i+1] = s[i+1] - L[i]*d2[i]
        end
        if i < M-1
            d2[i+1] = g[i+1] - L[i]*d2[i]
        end
    end
    d0, d1, d2, L
end
function Solve_Helmholtz_1D(N, neumann, fk, uk, d0, d1, d2, L)
    if neumann == 0
        uk = Solve_oe_Helmholtz_1D(N, 0, fk, uk, d0[1,:], d1[1,:], d2[1,:], L[1,:])
        uk = Solve_oe_Helmholtz_1D(N, 1, fk, uk, d0[2,:], d1[2,:], d2[2,:], L[2,:])
    else
        uk = Solve_oe_Helmholtz_1D(N-1, 0, fk, uk, d0[1,:], d1[1,:], d2[1,:], L[1,:])
        uk = Solve_oe_Helmholtz_1D(N-1, 1, fk, uk, d0[2,:], d1[2,:], d2[2,:], L[2,:])
        for i in 1:N-3
            uk[i] = uk[i] / (i^2)
        end
    end
    uk
end
function Solve_oe_Helmholtz_1D(N, odd, fk, u_hat, d0, d1, d2, L)
    """
    Solve (A+k**2*B)x = f, where A and B are stiffness and mass matrices of Shen with Dirichlet BC
    """
    if odd == 0
        M = div(N-3,2)
    else
        M = div(N-4,2)
    end
    y = Array{Float64}(M+1)
    y = ForwardSolve_L(y, L, odd, fk)

    # Solve Backward U u = y
    u0 = Array{Float64}(M+1)
    u_hat = BackSolve_U(M, odd, y, u0, d0, d1, d2, u_hat)
    u_hat
end
function BackSolve_U(M, odd, y, u0, d0, d1, d2, u_hat)
    sum_u0 = 0.0
    u0[M+1] = y[M+1] / d0[M+1]
    for i in M:-1:1
        u0[i] = y[i] - d1[i]*u0[i+1]
        if i <= M-1
            sum_u0 += u0[i+2]
            u0[i] -= sum_u0*d2[i]
        end
        u0[i] /= d0[i]
        u_hat[2*(i-1)+odd+1] = u0[i]
    end
    u_hat[2*M+odd+1] = u0[M+1]
    u_hat
end

function ForwardSolve_L(y, L, odd, fk)
    # Solve Forward Ly = f
    y[1] = fk[odd+1]
    for i in 2:length(y)
        y[i] = fk[2*(i-1)+odd+1] - L[i-1]*y[i-1]
    end
    y
end
function Solve_Helmholtz_3D_n(N, neumann, fk, uk, d0, d1, d2, L)
    y = zeros(eltype(uk), size(uk, 1), size(uk, 2), size(uk, 3))
    s1 = zeros(eltype(uk), size(uk, 1), size(uk, 2))
    s2 = zeros(eltype(uk), size(uk, 1), size(uk, 2))

    M = last(size(d0))
    for k in 1:size(uk, 2)
        for j in 1:size(uk, 1)
            y[j, k, 1] = fk[j, k, 1]
            y[j, k, 2] = fk[j, k, 2]
        end
    end

    if neumann == 1
        for i in 2:M
            ke = 2*i-1
            ko = ke+1
            for k in 1:size(uk, 2)
                for j in 1:size(uk, 1)
                    y[j, k, ke] = fk[j, k, ke] - L[j, k, 1, i-1]*y[j, k, ke-2]
                    if i <= M-1
                        y[j, k, ko] = fk[j, k, ko] - L[j, k, 2, i-1]*y[j, k, ko-2]
                    end
                end
            end
        end
        for k in 1:size(uk, 2)
            for j in 1:size(uk, 1)
                ke = 2*M-1
                uk[j, k, ke] = y[j, k, ke] / d0[j, k, 1, M]
            end
        end

        for i in M-1:-1:1
            ke = 2*i-1
            ko = ke+1
            for k in 1:size(uk, 2)
                for j in 1:size(uk, 1)
                    uk[j, k, ke] = y[j, k, ke] - d1[j, k, 1, i]*uk[j, k, ke+2]
                    if i == M-1
                        uk[j, k, ko] = y[j, k, ko]
                    else
                        uk[j, k, ko] = y[j, k, ko] - d1[j, k, 2, i]*uk[j, k, ko+2]
                    end
                    if i < M-1
                        s1[j, k] += uk[j, k, ke+4]
                        uk[j, k, ke] -= s1[j, k]*d2[j, k, 1, i]
                    end
                    if i < M-2
                        s2[j, k] += uk[j, k, ko+4]
                        uk[j, k, ko] -= s2[j, k]*d2[j, k, 2, i]
                    end
                    uk[j, k, ke] /= d0[j, k, 1, i]
                    uk[j, k, ko] /= d0[j, k, 2, i]
                end
            end
        end
        for i in 1:N-3
            ii = i*i
            for k in 1:size(uk, 2)
                for j in 1:size(uk, 1)
                    uk[j, k, i] = uk[j, k, i] / ii
                end
            end
        end
    else
        for i in 2:M
            ke = 2*i-1
            ko = ke+1
            kem2 = ke-2
            kom2 = ko-2
            im1 = i-1
            for k in 1:size(uk, 2)
                for j in 1:size(uk, 1)
                    y[j, k, ke] = fk[j, k, ke] - L[j, k, 1, im1]*y[j, k, kem2]
                    y[j, k, ko] = fk[j, k, ko] - L[j, k, 2, im1]*y[j, k, kom2]
                end
            end
        end
        ke = 2*M-1
        ko = ke+1
        ii = M
        for k in 1:size(uk, 2)
            for j in 1:size(uk, 1)
                uk[j, k, ke] = y[j, k, ke] / d0[j, k, 1, ii]
                uk[j, k, ko] = y[j, k, ko] / d0[j, k, 2, ii]
            end
        end
        for i in M-1:-1:1
            ke = 2*i-1
            ko = ke+1
            kep2 = ke+2
            kop2 = ko+2
            kep4 = ke+4
            kop4 = ko+4
            for k in 1:size(uk, 2)
                for j in 1:size(uk, 1)
                    uk[j, k, ke] = y[j, k, ke] - d1[j, k, 1, i]*uk[j, k, kep2]
                    uk[j, k, ko] = y[j, k, ko] - d1[j, k, 2, i]*uk[j, k, kop2]
                    if i < M-2
                        s1[j, k] += uk[j, k, kep4]
                        s2[j, k] += uk[j, k, kop4]
                        uk[j, k, ke] -= s1[j, k]*d2[j, k, 1, i]
                        uk[j, k, ko] -= s2[j, k]*d2[j, k, 2, i]
                    end
                    uk[j, k, ke] /= d0[j, k, 1, i]
                    uk[j, k, ko] /= d0[j, k, 2, i]
                end
            end
        end
    end
    uk
end
function Mult_Helmholtz_3D(N, GC, factor, alfa, u_hat, b)
    for jj in 1:size(u_hat, 2)
        for ii in 1:size(u_hat,1)
            b[ii,jj,:] = Mult_Helmholtz_1D(N, GC, factor,alfa[ii, jj],u_hat[ii, jj, :],b[ii, jj,:])
        end
    end
    b
end
function Mult_Helmholtz_1D(N, GC, factor, kx, u_hat, b)
    if ndims(b)>1
        b[1,1,:] = Mult_oe_Helmholtz_1D(N, 0, GC, factor, kx, u_hat[1,1,:], b[1,1,:])
        b[1,1,:] = Mult_oe_Helmholtz_1D(N, 1, GC, factor, kx, u_hat[1,1,:], b[1,1,:])
    else
        b = Mult_oe_Helmholtz_1D(N, 0, GC, factor, kx, u_hat, b)
        b = Mult_oe_Helmholtz_1D(N, 1, GC, factor, kx, u_hat, b)
    end
    b
end

function Mult_oe_Helmholtz_1D(N, odd, GC, factor, kx, u_hat, b)
    c0 = 0
    sum_u0 = 0.0

    if odd == 0
        M = div(N-3,2)
    else
        M = div(N-4,2)
    end
    # Direct matvec using the fact that there are only three unique diagonals in matrix
    bij = Array{Float64}(M+1)
    bij[:] = pi

    if odd == 0
        bij[1] *= 1.5
        if (N % 2 == 1) & GC
            bij[M+1] *= 1.5
        end
    else
        if (N % 2 == 0) & GC
            bij[M+1] *= 1.5
        end
    end
    d = Array{Float64}(M+1)
    s = Array{Float64}(M)
    g = Array{Float64}(M-1)
    if odd == 1
        c0 = 1
    end
    for i in 1:M+1
        d[i] = 2*pi*(2*(i-1)+1+c0)*(2*(i-1)+2+c0) + kx*bij[i]
        if i <= M
            s[i] = 4*pi*(2*(i-1)+1+c0) - kx*pi/2
        end
        if i <= M-1
            g[i] = 4*pi*(2*(i-1)+1+c0)
        end
    end
    b[2*M+odd+1] += factor*(-pi/2*kx*u_hat[2*M+odd-1] + d[M+1]*u_hat[2*M+odd+1])
    for i in M-1:-1:2
        b[2*(i-1)+odd+1] += factor*(-pi/2*kx*u_hat[2*(i-1)+odd-1] + d[i]*u_hat[2*(i-1)+odd+1] + s[i]*u_hat[2*(i-1)+odd+2+1])
        if i < M-1
            sum_u0 += u_hat[2*(i-1)+odd+1+4]
            b[2*(i-1)+odd+1] += factor*sum_u0*g[i]
        end
    end
    b[odd+1] += factor*(d[1]*u_hat[odd+1] + s[1]*u_hat[odd+2+1] + (sum_u0+u_hat[odd+4+1])*g[1])
    b
end
