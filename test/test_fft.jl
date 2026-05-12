@testset "_get_padded_shape                 " begin
    # single transformed dimension
    @test NSEBase._get_padded_shape((4,), (1,)) == (7,)
    @test NSEBase._get_padded_shape((6,), (1,)) == (9,)
    @test NSEBase._get_padded_shape((8,), (1,)) == (13,)

    # untransformed dimensions are unchanged
    @test NSEBase._get_padded_shape((4, 5), (1,)) == (7, 5)
    @test NSEBase._get_padded_shape((4, 5), (2,)) == (4, 9)

    # multiple transformed dimensions
    @test NSEBase._get_padded_shape((4, 6), (1, 2))       == (7, 9)
    @test NSEBase._get_padded_shape((4, 6, 8), (1, 2, 3)) == (7, 9, 13)
    @test NSEBase._get_padded_shape((4, 5, 6), (1, 3))    == (7, 5, 9)

    # padded size must satisfy the 3/2 dealiasing rule for all transformed dimensions
    for s in 1:32, ndim in 1:3
        shape  = ntuple(_ -> s, ndim)
        order  = ntuple(identity, ndim)
        padded = NSEBase._get_padded_shape(shape, order)
        for d in order
            @test padded[d] >= 3s/2
        end
    end
end

@testset "_get_transform_shape              " begin
    # 1D: first (and only) dimension is halved + 1
    @test NSEBase._get_transform_shape((4,), 1) == (3,)
    @test NSEBase._get_transform_shape((6,), 1) == (4,)
    @test NSEBase._get_transform_shape((8,), 1) == (5,)

    # 2D: only the selected dimension changes
    @test NSEBase._get_transform_shape((4, 6), 1) == (3, 6)
    @test NSEBase._get_transform_shape((4, 6), 2) == (4, 4)

    # 3D
    @test NSEBase._get_transform_shape((4, 6, 8), 1) == (3, 6, 8)
    @test NSEBase._get_transform_shape((4, 6, 8), 2) == (4, 4, 8)
    @test NSEBase._get_transform_shape((4, 6, 8), 3) == (4, 6, 5)
end

@testset "Fourier transforms                " begin
    # construct grid
    Nx = 16; Ny = 11
    L = 10*rand()
    g = FakeGrid(rand(Float64, Nx), Ny, L)

    # create plans
    plans  = FFTPlans(g, dealias=false, flags=FFTW.ESTIMATE)
    plansd = FFTPlans(g, dealias=true,  flags=FFTW.ESTIMATE)

    # random signal
    U  = FTField(g, randn(ComplexF64, Nx, (Ny >> 1) + 1))
    u  = Field(g, dealias=false)
    ud = Field(g, dealias=true)
    Uv = VectorField(FTField(g, randn(ComplexF64, Nx, (Ny >> 1) + 1)),
                     FTField(g, randn(ComplexF64, Nx, (Ny >> 1) + 1)),
                     FTField(g, randn(ComplexF64, Nx, (Ny >> 1) + 1)))
    uv = VectorField(g, Field)

    # test transforms
    @test  plans(    FTField(g),  plans(u,  U))           ≈   U
    @test plansd(    FTField(g), plansd(ud, U))           ≈   U
    @test plansd(       copy(U), plansd(ud, U), add=true) ≈ 2*U
    @test  plans(VectorField(g),  plans(uv, Uv))           ≈ Uv

    # test allocation
    funf(plan, A, B) = @allocated plan(A, B)
    funb(plan, B, A) = @allocated plan(B, A, add=false)
    # funf(plansd, ud, U) # run first to avoid initial allocation
    # funb(plansd, U, ud)
    @test funf(plans,  u,  U) == 0
    # @test funf(plansd, ud, U) == 0 # ! where do allocations come from here? The views that davide mentioned?
    @test funb(plans,  U,  u) == 0
    # @test funb(plansd, U, ud) == 0

    # test constructing transforms
    @test FFT(plans(similar(u), U)) ≈ U
    @test IFFT(U) == plans(similar(u), U)
    @test FFT(plans(similar(uv), Uv)) ≈ Uv
    @test IFFT(Uv) == plans(similar(uv), Uv)
end
