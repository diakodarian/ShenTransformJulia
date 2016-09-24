module PDMA
#-----------------------------------------------------
#                Symmetrical PDMA_1D
#-----------------------------------------------------
function SymmetricalPDMA_1D(d, e, f, b)
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
#                Non-symmertrical PDMA
#-----------------------------------------------------
function PDMA_1D(e,a,d,c,f,b)
    N = length(d)
    for i in 2:N-1
        xmult = a[i-1]/d[i-1]
        d[i] = d[i] - xmult*c[i-1]
        c[i] = c[i] - xmult*f[i-1]
        b[i] = b[i] - xmult*b[i-1]
        xmult = e[i-1]/d[i-1]
        a[i]   = a[i] - xmult*c[i-1]
        d[i+1] = d[i+1] - xmult*f[i-1]
        b[i+1] = b[i+1] - xmult*b[i-1]
    end
    xmult = a[N-1]/d[N-1]
    d[N] = d[N] - xmult*c[N-1]
    b[N] = (b[N] - xmult*b[N-1])/d[N]
    b[N-1] = (b[N-1] - c[N-1]*b[N])/d[N-1]
    for i in N-2:-1:1
        b[i] = (b[i] - f[i]*b[i+2] - c[i]*b[i+1])/d[i]
    end
    return b
end
#----------------------ooo----------------------------
#-----------------------------------------------------
#        Non-symmertrical Biharmonic PDMA
#-----------------------------------------------------
function BiharmonicPDMA_1D(a, b, c, d, e, r)

    N = length(r)
    rho = zeros(eltype(r), N)
    alpha = zeros(eltype(r), N-2)
    beta = zeros(eltype(r), N-4)
    x = zeros(eltype(r), N)

    alpha[1] = d[1]/c[1]
    beta[1] = e[1]/c[1]

    alpha[2] = d[2]/c[2]
    beta[2] = e[2]/c[2]

    alpha[3] = (d[3]-b[1]*beta[1])/(c[3]-b[1]*alpha[1])
    beta[3] = e[3]/(c[3]-b[1]*alpha[1])

    alpha[4] = (d[4]-b[2]*beta[2])/(c[4]-b[2]*alpha[2])
    beta[4] = e[4]/(c[4]-b[2]*alpha[2])

    rho[1] = r[1]/c[1]
    rho[2] = r[2]/c[2]
    rho[3] = (r[3] - b[1]*rho[1])/(c[3]-b[1]*alpha[1])
    rho[4] = (r[4] - b[2]*rho[2])/(c[4]-b[2]*alpha[2])

    for i in 5:N
        rho[i] = (r[i] - a[i-4]*rho[i-4] - rho[i-2]*(b[i-2] - a[i-4]*alpha[i-4])) / (c[i] - a[i-4]*beta[i-4] - alpha[i-2]*(b[i-2] - a[i-4]*alpha[i-4]))
        if i <= (N-2)
            alpha[i] = (d[i] - beta[i-2]*(b[i-2] - a[i-4]*alpha[i-4])) / (c[i] - a[i-4]*beta[i-4] - alpha[i-2]*(b[i-2] - a[i-4]*alpha[i-4]))
        end
        if i <= (N-4)
            beta[i] = e[i]/(c[i] - a[i-4]*beta[i-4] - alpha[i-2]*(b[i-2] - a[i-4]*alpha[i-4]))
        end
    end
    for i in N:-1:1
        x[i] = rho[i]
        if i<=(N-2)
            x[i] -= alpha[i]*x[i+2]
        end
        if i<=(N-4)
            x[i] -= beta[i]*x[i+4]
        end
    end
    return x
end
#----------------------ooo----------------------------
#-----------------------------------------------------
#        Symmertrical Biharmonic PDMA
#-----------------------------------------------------
function PDMA_SymLU(d,e,f)
    n = length(d)
    m = length(e)
    k = n - m
    for i in 1:(n-2*k)
        lam = e[i]/d[i]
        d[i+k] -= lam*e[i]
        e[i+k] -= lam*f[i]
        e[i] = lam
        lam = f[i]/d[i]
        d[i+2*k] -= lam*f[i]
        f[i] = lam
    end
    lam = e[n-3]/d[n-3]
    d[n-1] -= lam*e[n-3]
    e[n-3] = lam
    lam = e[n-2]/d[n-2]
    d[n] -= lam*e[n-2]
    e[n-2] = lam
    return d,e,f
end
function PDMA_Symsolve(d,e,f,b)
    d,e,f = PDMA_SymLU(d,e,f)
    n = length(d)

    b[3] -= e[1]*b[1]
    b[4] -= e[2]*b[2]
    for k in 5:n
        b[k] -= (e[k-2]*b[k-2] + f[k-4]*b[k-4])
    end
    b[n] /= d[n]
    b[n-1] /= d[n-1]
    b[n-2] /= d[n-2]
    b[n-2] -= e[n-2]*b[n]
    b[n-3] /= d[n-3]
    b[n-3] -= e[n-3]*b[n-1]
    for k in (n-4):-1:1
        b[k] /= d[k]
        b[k] -= (e[k]*b[k+2] + f[k]*b[k+4])
    end
    return b
end
#----------------------ooo----------------------------

export PDMA_Symsolve, BiharmonicPDMA_1D, PDMA_1D, SymmetricalPDMA_1D
end
