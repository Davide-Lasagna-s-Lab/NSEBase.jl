@testset "Projected Navier-Stokes equations " begin
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

    # test construction
    op = ProjectedNSE(g)
    @test op.cache1 isa VectorField{3, <:FTField{FakeGrid}}
    @test op.cache2 isa VectorField{3, <:FTField{FakeGrid}}

    # test operation
    a = ProjectedField(g, randn(ComplexF64, M, (Ny >> 1) + 1), Ψ)
    @test op(zero(a), a) ≈ 2*a
    @test op(similar(a), a, a) ≈ 4*a
end
