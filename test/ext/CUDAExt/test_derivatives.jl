@testset "CUDA derivatives uses correct methods                               " begin
    gd = CUDA.cu(g)
    u = FTField(gd, CUDA.randn(ComplexF32, Ny, (Nx >> 1) + 1, Nz, Nt))

    # make sure no errors are thrown during execution (the analytical tests are located in downstream packages)
    @test_nowarn ddx!(FTField(gd), u)
    @test_nowarn ddy!(FTField(gd), u)
    @test_nowarn ddz!(FTField(gd), u)
    @test_nowarn ddt!(FTField(gd), u)
end
