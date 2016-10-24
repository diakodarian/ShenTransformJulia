include("shentransformTypes.jl")
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
    for F in [Dirichlet{GC}(N), Dirichlet{GL}(N)]
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
    for F in [Neumann{GC}(N), Neumann{GL}(N)]
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
    for BC in RobinBC
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
    for BiharmonicBC in BiharmonicBCsymbols
        for F in [Biharmonic{GC}(BiharmonicBC, N), Biharmonic{GL}(BiharmonicBC, N)]
            x, w = NodesWeights(F, x, w)
            if eval(BiharmonicBC) == "DB"
                U = x -(3./2.)*(4.*x.^3 - 3.*x)+(1./2.)*(16.*x.^5 -20.*x.^3 +5.*x)
            elseif eval(BiharmonicBC) == "NB"
                U = -1. + 2.*x.^2 - (2./5.)*(1. - 8.*x.^2 + 8.*x.^4) +
                (1./15.)*(-1. + 18.*x.^2 - 48.*x.^4 + 32.*x.^6)
            end
            U_hat = fst(F, U, U_hat)
            V = ifst(F, U_hat, V)
            @test isapprox(U, V)
            if eval(BiharmonicBC) == "DB"
                U_hat = 8. - 48.*x.^2 +40.*x.^4;
            elseif eval(BiharmonicBC) == "NB"
                U_hat = (64./5.)*x - (128./5.)*x.^3 +(64./5.)*x.^5;
            end
            V = fastShenDerivative(F, U, V)
            @test isapprox(U_hat, V)
            println("Test: Biharmonic transform for  ", BiharmonicBC, " succeeded.")
        end
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

tests(N)
