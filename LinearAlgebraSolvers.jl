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

# Not implemented!!!!!
function LU_Helmholtz_3D(N, neumann, GC, alfa, d0, d1, d2, L)
    for ii in 1:size(d0, 3)
        for jj in 1:size(d0, 4)
            LU_Helmholtz_1D(N, neumann, GC, alfa[ii, jj], d0[1:end, 1:end, ii, jj],d1[1:end, 1:end, ii, jj],d2[1:end, 1:end, ii, jj],L[1:end, 1:end, ii, jj])
        end
    end
end
function LU_Helmholtz_1D(N, neumann, GC, alfa, d0,  d1, d2, L)
    if neumann == 0
        d0[1,:], d1[1,:], d2[1,:], L[1,:] = LU_oe_Helmholtz_1D(N, 0, GC, alfa, d0[1,:], d1[1,:], d2[1,:], L[1,:])
        d0[2,:], d1[2,:], d2[2,:], L[2,:] = LU_oe_Helmholtz_1D(N, 1, GC, alfa, d0[2,:], d1[2,:], d2[2,:], L[2,:])
    else
        d0[1,:], d1[1,:], d2[1,:], L[1,:] = LU_oe_HelmholtzN_1D(N, 0, GC, alfa, d0[1,:], d1[1,:], d2[1,:], L[1,:])
        d0[2,:], d1[2,:], d2[2,:], L[2,:] = LU_oe_HelmholtzN_1D(N, 1, GC, alfa, d0[2,:], d1[2,:], d2[2,:], L[2,:])
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
        if N % 2 == 1 & GC
            bij_e[M+1] *= 1.5
        end
    else
        if N % 2 == 0 & GC
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
