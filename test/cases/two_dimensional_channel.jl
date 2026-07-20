@testset verbose=true "Two-dimensional channel case                                " begin
    @testset verbose=true "Grid layout and coordinates                                 " begin
        Nx, Ny, Nt, α = 9, 17, 5, 0.5
        g = TwoDimensionalChannelGrid(Nx, Ny; Nt, α, width=3)
        positional = TwoDimensionalChannelGrid(Nx, Ny, Nt, 3, α)
        y = g.xs[1]
        grid_points = points(g)

        @test g isa TwoDimensionalChannelGrid
        @test g isa AbstractTwoDimensionalChannelGrid
        @test g isa AbstractChannelGrid
        @test TWO_DIMENSIONAL_CHANNEL_AXES == (2, 1, nothing, 3)
        @test size(g) == (Ny, Nx, Nt)
        @test grid_points == points(positional)
        @test fft_storage_dims(g) == TWO_DIMENSIONAL_CHANNEL_FFT_ORDER
        @test fft_physical_dims(g) == (:x, :t)
        @test inhomogeneous_storage_dims(g) == TWO_DIMENSIONAL_CHANNEL_INHOMOGENEOUS_DIMS
        @test inhomogeneous_physical_dims(g) == (:y,)
        @test storage_dim(g, :x) == 2
        @test storage_dim(g, :y) == 1
        @test map(size, grid_points) == ((Ny, 1, 1), (1, Nx, 1), (1, 1, Nt))
        @test vec(grid_points[1]) ≈ y
        @test vec(grid_points[2]) ≈ range(0, 2π / α - 2π / (α * Nx); length=Nx)
        @test vec(grid_points[3]) ≈ range(0, 1 - 1 / Nt; length=Nt)
        @test wavenumber_scale(g, 1) == 1.0
        @test wavenumber_scale(g, 2) == α
        @test wavenumber_scale(g, 3) == 2π

        mapped = Field(g, (X, Y, _, T) -> @. X + 2Y + 0T)
        @test parent(mapped) ≈ grid_points[2] .+ 2 .* grid_points[1] .+ 0 .* grid_points[3]

        @test_throws ArgumentError TwoDimensionalChannelGrid(2, Ny; Nt, α, width=3)
        @test_throws ArgumentError TwoDimensionalChannelGrid(Nx, Ny; Nt=2, α, width=3)
    end

    @testset verbose=true "Analytic streamwise and wall-normal derivatives             " begin
        Nx, Ny, Nt, α = 9, 17, 3, 0.75
        g = TwoDimensionalChannelGrid(Nx, Ny; Nt, α, width=5)
        plans = FFTPlans(g; flags=FFTW.ESTIMATE, dealias=false)
        u, _ = case_physical_to_spectral(
            g, plans, (X, Y, _, T) -> @. (1 - Y^2) * case_periodic_profile(α * X) + 0T)
        exact_dx, _ = case_physical_to_spectral(
            g, plans, (X, Y, _, T) -> @. α * (1 - Y^2) * case_periodic_profile_d1(α * X) + 0T)
        exact_dy, _ = case_physical_to_spectral(
            g, plans, (X, Y, _, T) -> @. -2Y * case_periodic_profile(α * X) + 0T)
        computed_dx, computed_dy = similar(u), similar(u)

        ddx!(computed_dx, u)
        ddy!(computed_dy, u)
        @test computed_dx ≈ exact_dx rtol=1e-11 atol=1e-11
        @test computed_dy ≈ exact_dy rtol=1e-10 atol=1e-10
    end

    @testset verbose=true "Analytical two-dimensional norms                            " begin
        α = 0.75
        g = TwoDimensionalChannelGrid(9, 17; Nt=9, α, width=5)
        f(X, Y, _, T) = (1 - Y^2) * case_periodic_profile(α*X) * case_periodic_profile(2π*T)
        twice_f(X, Y, Z, T) = 2f(X, Y, Z, T)
        physical = Field(g, f)
        spectral = FFT(physical)
        exact_norm2 = CASE_WALL_POLY_NORM2 * CASE_PERIODIC_PROFILE_NORM2^2

        @test case_physical_dot(g, parent(physical)) ≈ exact_norm2 rtol=1e-12
        @test dot(spectral, spectral) ≈ exact_norm2 rtol=1e-12
        @test norm(spectral) ≈ sqrt(exact_norm2) rtol=1e-12
        @test norm(FFT(VectorField(g, f, twice_f)))^2 ≈ 5exact_norm2 rtol=1e-12
    end

    @testset verbose=true "Plane Couette and Poiseuille constructors                   " begin
        Re = 40.0
        g = TwoDimensionalChannelGrid(9, 17; Nt=3, α=0.5, width=5)
        Uc, Up = plane_couette_base(g), plane_poiseuille_base(g)
        couette = PlaneCouetteFlow(g, Re; fftw_flags=FFTW.ESTIMATE, dealias=false)
        poiseuille = PlanePoiseuilleFlow(g, Re; f=2 / Re,
                                         fftw_flags=FFTW.ESTIMATE, dealias=false)
        rotating = PlanePoiseuilleFlow(g, Re; Ro=0.2, f=2 / Re,
                                       fftw_flags=FFTW.ESTIMATE, dealias=false)

        @test couette isa ProjectedNSE
        @test couette.nl isa CartesianPrimitive2DNSE
        @test couette.ln isa CartesianPrimitive2DLNSE{AdjointDiscrete}
        @test couette.nl.force isa NoForce
        @test couette.base == (Uc, nothing)

        @test poiseuille isa ProjectedNSE
        @test poiseuille.nl isa CartesianPrimitive2DNSE
        @test poiseuille.ln isa CartesianPrimitive2DLNSE{AdjointDiscrete}
        @test poiseuille.nl.force isa ConstantBodyForce
        @test poiseuille.nl.force.value == 2 / Re
        @test poiseuille.nl.force.component == 1
        @test poiseuille.base == (Up, nothing)
        @test rotating.nl.force isa CompoundForcing
        @test rotating.nl.force.forces[1] isa ConstantBodyForce
        @test rotating.nl.force.forces[2] isa CoriolisForce
        @test rotating.nl.force.forces[2].components == (1, 2)

        streamwise_velocity = (X, Y, _, T) -> @. (1 - Y^2) + 0X + 0T
        wall_normal_velocity = (X, Y, _, T) -> @. 0X + 0Y + 0T
        laminar = FFT(VectorField(g, streamwise_velocity, wall_normal_velocity))
        residual = similar(laminar)
        poiseuille.nl(0.0, laminar, residual)
        @test norm(residual) < 1e-10
    end
end
