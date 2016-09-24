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
#----------------------ooo----------------------------

export TDMA_1D
end
