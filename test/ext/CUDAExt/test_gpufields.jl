@testset verbose=true "CUDA fields                                                 " begin
    g = CUDA_CHANNEL_GRID
    gd = CUDA.cu(g)

    @testset verbose=true "FTField                                                     " begin
        u  = FTField(g)
        ud = FTField(gd)

        @test ud isa FTField{<:CUDAExt.GPUGrid, <:CuArray{ComplexF32}}
        @test size(ud) == transform_size(g)
        @test typeof(CUDA.cu(u)) == typeof(ud)
    end

    @testset verbose=true "Field                                                       " begin
        u   = Field(g)
        ud  = Field(gd; dealias=false)
        udd = Field(gd; dealias=true)

        @test ud  isa Field{<:CUDAExt.GPUGrid, <:CuArray{Float32}}
        @test udd isa Field{<:CUDAExt.GPUGrid, <:CuArray{Float32}}
        @test size(ud) == size(g)
        @test size(udd) == NSEBase.get_padded_size(size(g), fft_storage_dims(g))
        @test typeof(CUDA.cu(u)) == typeof(ud) == typeof(udd)
    end

    @testset verbose=true "VectorField                                                 " begin
        N = 3
        û = VectorField(g, FTField; N)
        u = VectorField(g, Field; N)
        ûd = VectorField(gd, FTField; N)
        ud = VectorField(gd, Field; N, dealias=false)
        udd = VectorField(gd, Field; N, dealias=true)

        @test ûd isa VectorField{N, <:FTField{<:CUDAExt.GPUGrid, <:CuArray{ComplexF32}}}
        @test ud isa VectorField{N, <:Field{<:CUDAExt.GPUGrid, <:CuArray{Float32}}}
        @test udd isa VectorField{N,   <:Field{<:CUDAExt.GPUGrid, <:CuArray{Float32}}}
        @test typeof(CUDA.cu(û)) == typeof(ûd)
        @test typeof(CUDA.cu(u)) == typeof(ud) == typeof(udd)
    end

    @testset verbose=true "ProjectedField                                              " begin
        M, N = 3, 2
        modes = test_modes(g; ncomponents=N, nmodes=M)
        a = ProjectedField(g, modes)
        ad = ProjectedField(gd, map(CUDA.cu, modes))
        expected_size = (M, map(dim -> transform_size(g)[dim], fft_storage_dims(g))...)

        @test ad isa ProjectedField{<:CUDAExt.GPUGrid, <:NTuple{N, <:CuArray{ComplexF32}}, <:CuArray{ComplexF32}}
        @test size(ad) == expected_size
        @test typeof(CUDA.cu(a)) == typeof(ad)
    end
end
