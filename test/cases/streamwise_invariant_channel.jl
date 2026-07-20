@testset verbose=true "Streamwise-invariant channel case                           " begin
    @testset verbose=true "Grid layout and coordinates                                 " begin
        g = StreamwiseInvariantChannelGrid(17, 9; Nt=5, β=0.5, width=3)
        positional = StreamwiseInvariantChannelGrid(17, 9, 5, 3, 0.5)
        y = g.xs[1]
        grid_points = points(g)

        @test g isa StreamwiseInvariantChannelGrid
        @test g isa AbstractStreamwiseInvariantChannelGrid
        @test g isa AbstractChannelGrid
        @test STREAMWISE_INVARIANT_CHANNEL_AXES == (nothing, 1, 2, 3)
        @test size(g) == (17, 9, 5)
        @test grid_points == points(positional)
        @test fft_storage_dims(g) == STREAMWISE_INVARIANT_CHANNEL_FFT_ORDER
        @test inhomogeneous_storage_dims(g) == STREAMWISE_INVARIANT_CHANNEL_INHOMOGENEOUS_DIMS
        @test fft_physical_dims(g) == (:z, :t)
        @test inhomogeneous_physical_dims(g) == (:y,)
        @test storage_dim(g, :x) === nothing
        @test storage_dim(g, :y) == 1
        @test storage_dim(g, :z) == 2
        @test physical_dim.(Ref(g), 1:3) == [:y, :z, :t]
        @test length(grid_points) == 3
        @test map(size, grid_points) == ((17, 1, 1), (1, 9, 1), (1, 1, 5))
        @test vec(grid_points[1]) ≈ y
        @test vec(grid_points[2]) ≈ range(0, 2π / 0.5 - 2π / (0.5 * 9); length=9)
        @test vec(grid_points[3]) ≈ range(0, 1 - 1 / 5; length=5)
        @test wavenumber_scale(g, 1) == 1.0
        @test wavenumber_scale(g, 2) == 0.5
        @test wavenumber_scale(g, 3) == 2π
        @test plane_couette_base(g) ≈ y

        @test_throws ArgumentError StreamwiseInvariantChannelGrid(17, 2; Nt=5, β=0.5, width=3)
        @test_throws ArgumentError StreamwiseInvariantChannelGrid(17, 9; Nt=2, β=0.5, width=3)
    end

    @testset verbose=true "Wall-normal finite-difference contract                      " begin
        g = StreamwiseInvariantChannelGrid(32, 1; Nt=1, β=2π,
                                           dist=FDGrids.GaussLobattoGrid(), width=9)
        @test g isa AbstractStreamwiseInvariantChannelGrid
        @test size(g) == (32, 1, 1)
        @test fft_storage_dims(g) == STREAMWISE_INVARIANT_CHANNEL_FFT_ORDER
        @test inhomogeneous_storage_dims(g) == STREAMWISE_INVARIANT_CHANNEL_INHOMOGENEOUS_DIMS
        test_one_inhomogeneous_fd_contract(g, (63, 3))
    end

    @testset verbose=true "Physical-coordinate derivative mapping                      " begin
        α = 0.75
        g = StreamwiseInvariantChannelGrid(17, 9; Nt=3, β=α, width=5)
        plans = FFTPlans(g; flags=FFTW.ESTIMATE, dealias=false)
        u, _ = case_physical_to_spectral(
            g, plans, (_, Y, Z, T) -> @. (1 - Y^2) * case_periodic_profile(α * Z) + 0T)
        exact_dy, _ = case_physical_to_spectral(
            g, plans, (_, Y, Z, T) -> @. -2Y * case_periodic_profile(α * Z) + 0T)
        exact_dz, _ = case_physical_to_spectral(
            g, plans, (_, Y, Z, T) -> @. α * (1 - Y^2) * case_periodic_profile_d1(α * Z) + 0T)
        computed_dx, computed_dy, computed_dz = zero(u), similar(u), similar(u)

        ddx!(computed_dx, u); ddy!(computed_dy, u); ddz!(computed_dz, u)
        @test iszero(parent(computed_dx))
        @test computed_dy ≈ exact_dy rtol=1e-10 atol=1e-10
        @test computed_dz ≈ exact_dz rtol=1e-11 atol=1e-11
        @test @inferred(NSEBase._check_cartesian_2d3c_grid(g)) === nothing

        two_dimensional = TwoDimensionalChannelGrid(9, 17; Nt=1, α=0.75, width=5)
        @test_throws ArgumentError CartesianPrimitive2D3CNSE(two_dimensional, 100; flags=FFTW.ESTIMATE)
        @test_throws ArgumentError CartesianPrimitive2D3CLNSE(two_dimensional, 100; flags=FFTW.ESTIMATE)
    end

    @testset verbose=true "Nonlinear and linearised 2D3C equations                     " begin
        Re, β = 37.0, 0.75
        g = StreamwiseInvariantChannelGrid(17, 9; Nt=1, β, width=5)
        equations = PlaneCouetteFlow(g, Re; fftw_flags=FFTW.ESTIMATE, dealias=false)
        v(_, Y, Z, T) = @. 1 + 0Y + 0Z + 0T
        w(_, Y, Z, T) = @. 1 + 0Y + 0Z + 0T
        u(_, Y, Z, T) = @. (1 - Y^2) * sin(β * Z) + 0T
        zero_rhs(_, Y, Z, T) = @. 0Y + 0Z + 0T
        u_rhs(_, Y, Z, T) = @. (-2 - β^2 * (1 - Y^2)) * sin(β * Z) / Re +
                                  2Y * sin(β * Z) - β * (1 - Y^2) * cos(β * Z) + 0T
        state = FFT(VectorField(g, v, w, u))
        exact = FFT(VectorField(g, zero_rhs, zero_rhs, u_rhs))
        @test equations.nl(0.0, state, similar(state)) ≈ exact rtol=1e-10 atol=1e-10

        p₁(_, Y, Z, T) = @. (1 - Y^2) * cos(β * Z) + 0T
        p₂(_, Y, Z, T) = @. (1 - Y^2) * sin(β * Z) + 0T
        p₃(_, Y, Z, T) = @. Y * (1 - Y^2) * cos(β * Z) + 0T
        q₁(_, Y, Z, T) = @. Y * (1 - Y^2) * sin(β * Z) + 0T
        q₂(_, Y, Z, T) = @. (1 - Y^2)^2 * cos(β * Z) + 0T
        q₃(_, Y, Z, T) = @. (1 - Y^2) * sin(β * Z) + 0T
        perturbation = FFT(VectorField(g, p₁, p₂, p₃))
        test_field = FFT(VectorField(g, q₁, q₂, q₃))
        forward = CartesianPrimitive2D3CLNSE(g, Re; mode=Forward(), flags=FFTW.ESTIMATE)
        adjoint = CartesianPrimitive2D3CLNSE(g, Re; mode=AdjointDiscrete(), flags=FFTW.ESTIMATE)
        lhs = dot(forward(0.0, state, perturbation, similar(state)), test_field)
        rhs = dot(perturbation, adjoint(0.0, state, test_field, similar(state)))
        @test lhs ≈ rhs rtol=1e-11 atol=1e-11
    end

    @testset verbose=true "Flow constructors                                           " begin
        g = StreamwiseInvariantChannelGrid(17, 9; Nt=5, β=0.5, width=3)
        no_rotation = PlaneCouetteFlow(g, 500; fftw_flags=FFTW.ESTIMATE, dealias=false)
        with_rotation = PlaneCouetteFlow(g, 500; Ro=0.2, fftw_flags=FFTW.ESTIMATE, dealias=false)
        poiseuille = PlanePoiseuilleFlow(g, 500; Ro=0.2, f=1.5,
                                         fftw_flags=FFTW.ESTIMATE, dealias=false)
        @test no_rotation isa ProjectedNSE
        @test no_rotation.nl.force isa NoForce
        @test with_rotation isa ProjectedNSE
        @test with_rotation.nl.force isa CoriolisForce
        @test with_rotation.nl.force.components == (3, 1)
        @test with_rotation.base == (nothing, nothing, plane_couette_base(g))
        @test poiseuille.base == (nothing, nothing, plane_poiseuille_base(g))
        @test poiseuille.nl.force isa CompoundForcing
        @test poiseuille.nl.force.forces[1].component == 3
        @test poiseuille.nl.force.forces[2].components == (3, 1)
    end

    @testset verbose=true "Analytical scalar and vector norms                          " begin
        g = StreamwiseInvariantChannelGrid(17, 9; Nt=9, β=π,
                                           dist=FDGrids.GaussLobattoGrid(), width=3)
        plans = FFTPlans(g; flags=FFTW.ESTIMATE, dealias=false)
        wall_normal_norm = CASE_WALL_POLY_NORM2

        y_only, _ = case_physical_to_spectral(g, plans, (_, Y, Z, T) -> @. (1 - Y^2) + 0 * Z + 0 * T)
        @test dot(y_only, y_only) ≈ wall_normal_norm rtol=1e-12
        @test norm(y_only) ≈ sqrt(wall_normal_norm) rtol=1e-12

        yz, _ = case_physical_to_spectral(g, plans,
            (_, Y, Z, T) -> @. (1 - Y^2) * case_periodic_profile(π * Z) + 0 * T)
        @test dot(yz, yz) ≈ wall_normal_norm * CASE_PERIODIC_PROFILE_NORM2 rtol=1e-10
        @test norm(yz) ≈ sqrt(wall_normal_norm * CASE_PERIODIC_PROFILE_NORM2) rtol=1e-10

        f(_, Y, Z, T) = (1 - Y^2) * case_periodic_profile(π * Z) * case_periodic_profile(2π * T)
        twice_f(X, Y, Z, T) = 2f(X, Y, Z, T)
        thrice_f(X, Y, Z, T) = 3f(X, Y, Z, T)
        yzt, physical = case_physical_to_spectral(g, plans, f)
        Ny, Nz, Nt = size(g)
        physical_norm = sum(g.ws[1][j] * physical[j, k, n]^2
                            for j in 1:Ny, k in 1:Nz, n in 1:Nt) / (Nz * Nt)
        exact_norm = wall_normal_norm * CASE_PERIODIC_PROFILE_NORM2^2
        @test physical_norm ≈ exact_norm rtol=1e-10
        @test dot(yzt, yzt) ≈ physical_norm rtol=1e-10
        @test norm(yzt) ≈ sqrt(exact_norm) rtol=1e-10
        @test norm(FFT(VectorField(g, f, twice_f, thrice_f)))^2 ≈ 14exact_norm rtol=1e-10
    end
end
