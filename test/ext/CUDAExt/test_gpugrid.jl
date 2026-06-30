@testset "GPU grid                                                            " begin
    gd = CUDA.cu(g)

    @test gd isa CUDAExt.GPUGrid{Float32}
    @test size(gd) == (16, 15, 15, 15)
    @test weights(gd) isa CuArray{Float32}
    @test Array(weights(gd)) == Float32.(weights(g))
    @test map(d -> wavenumber_scale(gd, d), 2:4) == Float32.([2π, 2π, 1.0])
end
