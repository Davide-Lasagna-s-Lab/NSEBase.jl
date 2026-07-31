@testset verbose=true "CUDA derivatives use the correct methods                    " begin
    g = CUDA_CHANNEL_GRID
    gd = CUDA.cu(g)
    u = FTField(gd, CUDA.randn(ComplexF32, transform_size(g)...))

    # Analytical accuracy belongs to the rectangular-grid package. Here we
    # verify that every Cartesian derivative reaches its CUDA specialization.
    @test_nowarn ddx!(FTField(gd), u)
    @test_nowarn ddy!(FTField(gd), u)
    @test_nowarn ddz!(FTField(gd), u)
    @test_nowarn ddt!(FTField(gd), u)
end
