@testset "Field                         " begin
    # construct grid
    Nx = 16; Ny = 11
    L = 10*rand()
    g = FakeGrid(rand(Float64, Nx), Ny, L)
    x, y   = points(g, dealias=false)
    xd, yd = points(g, dealias=true )

    # test field constructors
    f(x, y)=x^2*cos(2π*y/L)
    u1 = Field(g, f, dealias=false)
    u2 = Field(g, f, dealias=true )
    u3 = Field(g,    dealias=false)
    u4 = Field(g,    dealias=true )
    @test size(u1) == size(u3) == (16, 11)
    @test size(u2) == size(u4) == (16, 17)
    @test begin
        res = true
        for nx in 1:16, ny in 1:11
            u1[nx, ny] != f(x[nx], y[ny]) && (res = false; break)
        end; res
    end
    @test begin
        res = true
        for nx in 1:16, ny in 1:17
            u2[nx, ny] != f(xd[nx], yd[ny]) && (res = false; break)
        end; res
    end

    # test broadcasting
    foo(u, v, w) = (@allocated u .= v.*2 .+ w./3)
    bar(u)       = (@allocated u .= 0.0)
    u = Field(g,    dealias=false)
    v = Field(g, f, dealias=false)
    w = Field(g, f, dealias=false)
    @test foo(u, v, w) == 0
    @test u == 2*v + w/3
    @test bar(u) == 0
    u = Field(g,    dealias=true)
    v = Field(g, f, dealias=true)
    w = Field(g, f, dealias=true)
    @test foo(u, v, w) == 0
    @test u == 2*v + w/3
    @test bar(u) == 0
end

@testset "FTField                       " begin
    # construct grid
    Nx = 16; Ny = 11
    L = 10*rand()
    g = FakeGrid(rand(Float64, Nx), Ny, L)

    # test ftfield constructors
    data = randn(ComplexF64, Nx, (Ny >> 1) + 1)
    u1 = FTField(g)
    u2 = FTField(g, data)
    @test size(u1) == size(u2) == (16, 6)
    @test norm(imag.(u1[:, 1])) == norm(imag.(u2[:, 1])) == 0

    # test symmetry enforcement
    data = randn(ComplexF64, 16, 6, 11)
    NSEBase.apply_symmetry!(data, Val((2, 3)))
    @test begin
        res = true
        for i in axes(data, 1), k in 1:(size(data, 3) >> 1)
            data[i, 1, k+1] != conj(data[i, 1, end-k+1]) && (res = false; break)
        end; res
    end
    data = randn(ComplexF64, 7, 6, 11, 7)
    NSEBase.apply_symmetry!(data, Val((3, 1, 4)))
    @test begin
        res = true
        for j in axes(data, 2)
            for i in -(size(data, 1) >> 1):(size(data, 1) >> 1), l in -(size(data, 4) >> 1):(size(data, 4) >> 1)
                (i == 0 && l == 0) && continue
                _i     = i< 0 ? size(data, 1)+i+1 :               i+1
                _i_sym = i< 0 ?              -i+1 : size(data, 1)-i+1
                _i_sym = i==0 ?               i+1 : _i_sym
                _l     = l< 0 ? size(data, 4)+l+1 :               l+1
                _l_sym = l< 0 ?              -l+1 : size(data, 4)-l+1
                _l_sym = l==0 ?               l+1 : _l_sym
                data[_i, j, 1, _l] != conj(data[_i_sym, j, 1, _l_sym]) && (res = false; break)
            end
        end; res
    end

    # test broadcasting
    foo(u, v, w) = (@allocated u .= v.*2 .+ w./3)
    bar(u)       = (@allocated u .= 0.0)
    u = FTField(g)
    v = FTField(g, randn(ComplexF64, Nx, (Ny >> 1) + 1))
    w = FTField(g, randn(ComplexF64, Nx, (Ny >> 1) + 1))
    @test foo(u, v, w) == 0
    @test u == 2*v + w/3
    @test bar(u) == 0
end

@testset "VectorField                   " begin
    # construct grid
    Nx = 16; Ny = 11
    L = 10*rand()
    g = FakeGrid(rand(Float64, Nx), Ny, L)

    # test vectorfield constructors
    f1(x, y)=x^2*cos(2π*y/L)
    f2(x, y)=x^3*sin(2π*y/L)
    f3(x, y)=exp(x)*exp(cos(2π*y/L))
    u1 = VectorField(g)
    u2 = VectorField(g, Field, N=2)
    u3 = VectorField(g, Field, N=5, dealias=true)
    u4 = VectorField(g, f1, f2, f3)
    u5 = VectorField(g, f1, f2, f3, dealias=true)
    @test u1 isa VectorField{3, <:FTField}
    @test u2 isa VectorField{2, <:Field}
    @test size(u1[1]) == (16, 6)
    @test size(u2[1]) == size(u4[1]) == (16, 11)
    @test size(u3[1]) == size(u5[1]) == (16, 17)

    # test interface
    @test size(u1) == (3,)
    @test size(u2) == (2,)
    @test length(u2) == 2
    @test eltype(u1) <: FTField
    @test eltype(u2) <: Field
    @test similar(u2) isa VectorField{2, <:Field}
    @test all(parent(copy(u1)) .== parent(u1))
    @test all(parent(copy(u2)) .== parent(u2))
    @test all(parent(zero(u1)[1]) .== 0.0)
    @test all(parent(zero(u1)[2]) .== 0.0)
    @test all(parent(zero(u2)[1]) .== 0.0)
    @test all(parent(zero(u2)[2]) .== 0.0)

    # test inner-product
    u = VectorField(FTField(g, 0.5*ones(ComplexF64, Nx, (Ny >> 1) + 1)),
                    FTField(g, 0.5*ones(ComplexF64, Nx, (Ny >> 1) + 1)),
                    FTField(g, 0.5*ones(ComplexF64, Nx, (Ny >> 1) + 1)))
    @test dot(u, u) == 66

    # test broadcasting
    foo(u, v, w) = (@allocated u .= v.*2 .+ w./3)
    bar(u)       = (@allocated u .= 0.0)
    u = VectorField(g)
    v = VectorField(FTField(g, randn(ComplexF64, Nx, (Ny >> 1) + 1)),
                    FTField(g, randn(ComplexF64, Nx, (Ny >> 1) + 1)),
                    FTField(g, randn(ComplexF64, Nx, (Ny >> 1) + 1)))
    w = VectorField(FTField(g, randn(ComplexF64, Nx, (Ny >> 1) + 1)),
                    FTField(g, randn(ComplexF64, Nx, (Ny >> 1) + 1)),
                    FTField(g, randn(ComplexF64, Nx, (Ny >> 1) + 1)))
    @test foo(u, v, w) == 0
    for n in 1:3; @test parent(u[n]) == 2*parent(v[n]) + parent(w[n])/3; end
    @test bar(u) == 0
end

@testset "ProjectedField                " begin
    # construct grid
    Nx = 16; Ny = 11
    L = 10*rand()
    g = FakeGrid(rand(Float64, Nx), Ny, L)

    # generate modes
    M = 10
    Ψ = [zeros(ComplexF64, Nx, M, (Ny >> 1) + 1),
         zeros(ComplexF64, Nx, M, (Ny >> 1) + 1),
         zeros(ComplexF64, Nx, M, (Ny >> 1) + 1)]
    for ny in 1:(Ny >> 1) + 1
        tmp = qr(randn(ComplexF64, 3*Nx, M)).Q[:, 1:M]
        Ψ[1][:, :, ny] .= tmp[     1:1*Nx, :]
        Ψ[2][:, :, ny] .= tmp[  Nx+1:2*Nx, :]
        Ψ[3][:, :, ny] .= tmp[2*Nx+1:3*Nx, :]
    end

    # construct projected field from grid
    a1 = ProjectedField(        g,  Ψ)
    a2 = ProjectedField(FTField(g), Ψ)
    a3 = ProjectedField(  Field(g), Ψ)
    @test eltype(a1) == ComplexF64
    @test size(a1) == size(a2) == size(a3) == (10, 6)
    @test similar(a1) isa typeof(a1)
    @test zero(a1) isa typeof(a1)
    @test norm(zero(a1)) == 0

    # test expand and project
    a1 .= randn(ComplexF64, 10, 6)
    @test project(expand(a1), Ψ) ≈ a1

    # test braodcasting
    foo(a, b, c) = (@allocated a .= b.*2 .+ c./3)
    bar(a)       = (@allocated a .= 0.0)
    a = ProjectedField(g, Ψ)
    b = ProjectedField(g, randn(ComplexF64, M, (Ny >> 1) + 1), Ψ)
    c = ProjectedField(g, randn(ComplexF64, M, (Ny >> 1) + 1), Ψ)
    @test foo(a, b, c) == 0
    @test a == 2*b + c/3
    @test bar(a) == 0
end
