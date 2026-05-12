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
