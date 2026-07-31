@testset verbose=true "CUDA dot product                                            " begin
    g = CUDA_CHANNEL_GRID
    a = cuda_test_projected_field(g; nmodes=5, seed=1)
    b = cuda_test_projected_field(g; nmodes=5, seed=3)
    ad, bd = CUDA.cu(a), CUDA.cu(b)

    @testset verbose=true "Explicit methods agree with the host dot product            " begin
        method_twostage = CUDAExt.DotTwoStage(ad)
        method_atomic   = CUDAExt.DotAtomic(ad)
        method_shared   = CUDAExt.DotShared(ad)

        res_host = dot(a, b)
        @test abs(res_host - dot(ad, bd, method_twostage)) < 4e-4
        @test abs(res_host - dot(ad, bd, method_atomic))   < 4e-4
        @test abs(res_host - dot(ad, bd, method_shared))   < 4e-4
    end

    # A distinct production-grid type provides a second autotuning cache key.
    g2 = channel_grid(Nx=3, Ny=16, Nz=1, Nt=1)
    ad2 = CUDA.cu(cuda_test_projected_field(g2; nmodes=2, seed=5))

    @testset verbose=true "Autotuning caches methods by concrete field type            " begin
        reset_dot_cache!()
        @test isempty(CUDAExt.DOT_METHODS)
        CUDAExt.dot_method(ad)
        @test length(CUDAExt.DOT_METHODS) == 1
        reset_dot_cache!()
        @test isempty(CUDAExt.DOT_METHODS)
        CUDAExt.dot_method(ad)
        CUDAExt.dot_method(ad2)
        @test length(CUDAExt.DOT_METHODS) == 2
        reset_dot_cache!(ad2)
        @test length(CUDAExt.DOT_METHODS) == 1
        CUDAExt.dot_method(ad)
        @test length(CUDAExt.DOT_METHODS) == 1
        CUDAExt.dot_method(ad2)
        reset_dot_cache!()
        @test isempty(CUDAExt.DOT_METHODS)
    end
end
