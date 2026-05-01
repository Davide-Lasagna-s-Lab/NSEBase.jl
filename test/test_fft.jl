@testset "Transform Plans               " begin
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
    @test parent( plans(    FTField(g),  plans(u,  U)))           ≈   parent(U)
    @test parent(plansd(    FTField(g), plansd(ud, U)))           ≈   parent(U)
    @test parent(plansd(       copy(U), plansd(ud, U), add=true)) ≈ 2*parent(U)
    out = plans(VectorField(g),  plans(uv, Uv))
    for n in 1:3; @test parent(out[n]) ≈ parent(Uv[n]); end

    # test constructing transforms
    @test parent(FFT(plans(similar(u), U))) ≈ parent(U)
    @test IFFT(U) == plans(similar(u), U)
    out = FFT(plans(similar(uv), Uv))
    for n in 1:3; @test parent(out[n]) ≈ parent(Uv[n]); end
    @test IFFT(Uv) == plans(similar(uv), Uv)
end
