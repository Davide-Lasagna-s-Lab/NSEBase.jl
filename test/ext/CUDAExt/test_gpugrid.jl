@testset verbose=true "CUDA grid                                                   " begin
    g = CUDA_CHANNEL_GRID
    gd = CUDA.cu(g)

    @test gd isa CUDAExt.GPUGrid{Float32}
    @test size(gd) == size(g)
    @test weights(gd) isa CuArray{Float32}
    @test Array(weights(gd)) == Float32.(weights(g))
    @test map(d -> wavenumber_scale(gd, d), fft_storage_dims(gd)) ==
          Float32.(map(d -> wavenumber_scale(g, d), fft_storage_dims(g)))
end
