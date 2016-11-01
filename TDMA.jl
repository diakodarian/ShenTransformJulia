module TDMA
#-----------------------------------------------------
#                TDMA_1D
#-----------------------------------------------------
"""
    TDMA_1D(a, b, c, d)

Solves a symmetrical TDMA system. a is the main diagonal, b is lower diagonal
and c is the upper diagonal. d is the right hand side.
"""
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
"""
    TDMA_SymLU(d,a,l)

LU ecomposition of a TDMA matrix. d is the main diagonal, l is lower diagonal
and a is the upper diagonal.
"""
function TDMA_SymLU(d, a, l)
    n = length(d)
    for i in 3:n
        l[i-2] = a[i-2]/d[i-2]
        d[i] = d[i] - l[i-2]*a[i-2]
    end
    return d, a, l
end
"""
    TDMA_SymSolve(d,a,l, x)

Solves a symmetrical TDMA matrix. d is the main diagonal, l is lower diagonal
and a is the upper diagonal. x is the right hand side of the system.
"""
function TDMA_SymSolve(d, a, l, x)
    d,a,l = TDMA_SymLU(d,a,l)
    n = length(d)
    y = zeros(eltype(x), n)

    y[1] = x[1]
    y[2] = x[2]
    for i in 3:n
        y[i] = x[i] - l[i-2]*y[i-2]
    end
    x[n] = y[n]/d[n]
    x[n-1] = y[n-1]/d[n-1]
    for i in n-2:-1:1
        x[i] = (y[i] - a[i]*x[i+2])/d[i]
    end
    return x
end
"""
    TDMA_SymSolve3D(d,a,l, x)

Solves a symmetrical TDMA matrix. d is the main diagonal, l is lower diagonal
and a is the upper diagonal. x is the right hand side of the system.
"""
function TDMA_SymSolve3D(d,a,l,x)
    d,a,l = TDMA_SymLU(d,a,l)
    n = length(d)
    y = similar(x)
    for k in 1:size(x,2)
        for j in 1:size(x,1)
            y[j, k, 1] = x[j, k, 1]
            y[j, k, 2] = x[j, k, 2]
        end
    end
    for i in 3:n
        for  k in 1:size(x,2)
            for j in 1:size(x,1)
                y[j, k, i] = x[j, k, i] - l[i-2]*y[j, k, i-2]
            end
        end
    end
    for k in 1:size(x,2)
        for j in 1:size(x,1)
            x[j, k, n] = y[j, k, n]/d[n]
            x[j, k, n-1] = y[j, k, n-1]/d[n-1]
        end
    end
    for i in n-2:-1:1
        for k in 1:size(x,2)
            for j in 1:size(x,1)
                x[j, k, i] = (y[j, k, i] - a[i]*x[j, k, i+2])/d[i]
            end
        end
    end
    return x
end
#----------------------ooo----------------------------

export TDMA_1D, TDMA_SymSolve, TDMA_SymSolve3D
end
