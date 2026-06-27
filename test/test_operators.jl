@testset "Projected Navier-Stokes equations                                   " begin
    # construct grid
    Nx = 16; Ny = 11
    L = 10*rand()
    g = FakeGrid(rand(Float64, Nx), Ny, L)

    # generate modes — shape (Nm, inh_sz..., kH_sz...) = (M, Nx, Nk)
    using Random; Random.seed!(0)
    M = 10
    Ψ = [zeros(ComplexF64, M, Nx, (Ny >> 1) + 1),
         zeros(ComplexF64, M, Nx, (Ny >> 1) + 1),
         zeros(ComplexF64, M, Nx, (Ny >> 1) + 1)]
    for ny in 1:(Ny >> 1) + 1
        tmp = qr(randn(ComplexF64, 3*Nx, M)).Q[:, 1:M]
        Ψ[1][:, :, ny] .= transpose(tmp[     1:1*Nx, :])
        Ψ[2][:, :, ny] .= transpose(tmp[  Nx+1:2*Nx, :])
        Ψ[3][:, :, ny] .= transpose(tmp[2*Nx+1:3*Nx, :])
    end
    Ψ[1][:, :, 1] .= real.(Ψ[1][:, :, 1])
    Ψ[2][:, :, 1] .= real.(Ψ[2][:, :, 1])
    Ψ[3][:, :, 1] .= real.(Ψ[3][:, :, 1])

    # operator construction
    nl = CartesianPrimitive3DNSE(g, 100; flags=FFTW.ESTIMATE)
    ln = CartesianPrimitive3DLNSE(g, 100; mode=AdjointDiscrete(), flags=FFTW.ESTIMATE)
    op = construct_equations(g, 100, nothing, CartesianPrimitive3D(); mode=AdjointDiscrete(), flags=FFTW.ESTIMATE)
    @test op.cache1 isa VectorField{3, <:FTField{FakeGrid}}
    @test op.cache2 isa VectorField{3, <:FTField{FakeGrid}}

    # test operation
    a = ProjectedField(g, randn(ComplexF64, M, (Ny >> 1) + 1), Ψ)
    b = ProjectedField(g, randn(ComplexF64, M, (Ny >> 1) + 1), Ψ)
    u = expand(a)
    v = expand(b)
    @test op(similar(a), a)    ≈ project(nl(0.0, u,    similar(u)), Ψ)
    @test op(similar(a), a, b) ≈ project(ln(0.0, u, v, similar(u)), Ψ)
end
