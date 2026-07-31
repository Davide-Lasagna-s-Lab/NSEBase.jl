@testset verbose=true "Projected Navier-Stokes equations                           " begin
    g = steady_channel_grid(; Nx=7, Ny=9, Nz=5)
    modes = test_modes(g; ncomponents=3, nmodes=4, seed=11)

    rng = MersenneTwister(12)
    coefficient_size = (size(modes[1], 1), map(dim -> transform_size(g)[dim], fft_storage_dims(g))...)
    a = ProjectedField(g, randn(rng, ComplexF64, coefficient_size), modes)
    b = ProjectedField(g, randn(rng, ComplexF64, coefficient_size), modes)

    nl = CartesianPrimitive3DNSE(g, 100; flags=FFTW.ESTIMATE)
    ln = CartesianPrimitive3DLNSE(g, 100; mode=AdjointDiscrete(), flags=FFTW.ESTIMATE)
    base = (nothing, nothing, nothing)
    equations = construct_equations(g, 100, base, CartesianPrimitive3D();
                                    mode=AdjointDiscrete(), flags=FFTW.ESTIMATE)

    @test equations.cache1 isa VectorField{3, <:FTField{typeof(g)}}
    @test equations.cache2 isa VectorField{3, <:FTField{typeof(g)}}

    u = expand(a)
    v = expand(b)
    @test equations(similar(a), a) ≈ project(nl(0.0, u, similar(u)), modes)
    @test equations(similar(a), a, b) ≈ project(ln(0.0, u, v, similar(u)), modes)
end
