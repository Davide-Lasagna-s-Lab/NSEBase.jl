@testset verbose=true "Rayleigh–Bénard case                                        " begin
    @testset verbose=true "Grid layout and coordinates                                 " begin
        g = ChannelGrid(9, 17, 7; Nt=1, α=0.5, β=0.25, width=3)
        positional = ChannelGrid(9, 17, 7, 1, 3, 0.5, 0.25)
        Y, X, Z, T = points(g)

        @test g isa AbstractChannel3DGrid
        @test g isa AbstractChannelGrid
        @test size(g) == (17, 9, 7, 1)
        @test points(g) == points(positional)
        @test fft_storage_dims(g) == CHANNEL_3D_FFT_ORDER
        @test inhomogeneous_storage_dims(g) == CHANNEL_3D_INHOMOGENEOUS_DIMS
        @test length(points(g)) == 4
        @test vec(Y) ≈ g.xs[1]
        @test vec(X) ≈ range(0, 2π / 0.5 - 2π / (0.5 * 9); length=9)
        @test vec(Z) ≈ range(0, 2π / 0.25 - 2π / (0.25 * 7); length=7)
        @test vec(T) ≈ range(0, 0; length=1)
        @test wavenumber_scale(g, 1) == 1.0
        @test wavenumber_scale(g, 2) == 0.5
        @test wavenumber_scale(g, 3) == 0.25
        @test wavenumber_scale(g, 4) == 2π

        @test_throws ArgumentError ChannelGrid(2, 17, 7; Nt=1, α=0.5, β=0.25, width=3)
        @test_throws ArgumentError ChannelGrid(9, 17, 2; Nt=1, α=0.5, β=0.25, width=3)
        @test_throws ArgumentError ChannelGrid(9, 17, 7; Nt=2, α=0.5, β=0.25, width=3)
    end

    @testset verbose=true "Wall-normal finite-difference contract                      " begin
        g = ChannelGrid(1, 32, 1; Nt=1, α=π, β=π, dist=FDGrids.GaussLobattoGrid(), width=9)
        @test g isa AbstractChannel3DGrid
        @test size(g) == (32, 1, 1, 1)
        @test fft_storage_dims(g) == CHANNEL_3D_FFT_ORDER
        @test inhomogeneous_storage_dims(g) == CHANNEL_3D_INHOMOGENEOUS_DIMS
        test_one_inhomogeneous_fd_contract(g, (63, 63, 3))
    end

    @testset verbose=true "Base temperature and flow constructors                      " begin
        g = ChannelGrid(9, 17, 7; Nt=1, α=0.5, β=0.25, width=3)
        Θ = rbc_base_temperature(g)
        @test length(Θ) == 17
        @test Θ[firstindex(Θ)] ≈ 1 rtol=1e-12
        @test Θ[lastindex(Θ)] ≈ 0 rtol=1e-12
        @test Θ ≈ (1 .- g.xs[1]) ./ 2

        equations = RayleighBenardFlow(g, 1.0, 0.71, 1000.0; fftw_flags=FFTW.ESTIMATE, dealias=false)
        passive_scalar = RayleighBenardFlow(g, 1.0, 0.71, 0.0; fftw_flags=FFTW.ESTIMATE, dealias=false)
        @test equations isa ProjectedNSE
        @test passive_scalar isa ProjectedNSE
        @test equations.nl.Pr ≈ 0.71
        @test equations.nl.Ri ≈ 1000 / 0.71
        @test equations.nl.grav == 2
        @test equations.nl.plans isa FFTPlans{false}
        @test equations.base == (nothing, nothing, nothing, Θ)
    end

    @testset verbose=true "Two-dimensional construction and conduction profile         " begin
        g = TwoDimensionalChannelGrid(9, 17; Nt=1, α=0.5, lim=(2, 6), width=3)
        Θ = rbc_base_temperature(g)
        equations = RayleighBenardFlow(g, 2.0, 0.71, 1000.0;
                                        fftw_flags=FFTW.ESTIMATE, dealias=false)
        passive_scalar = RayleighBenardFlow(g, 2.0, 0.71, 0.0;
                                            fftw_flags=FFTW.ESTIMATE, dealias=false)

        @test Θ[firstindex(Θ)] ≈ 1 rtol=1e-12
        @test Θ[lastindex(Θ)] ≈ 0 rtol=1e-12
        @test Θ ≈ (1 .- plane_couette_base(g)) ./ 2
        @test equations isa ProjectedNSE
        @test equations.nl isa CartesianPrimitive2DBoussinesqNSE
        @test equations.ln isa CartesianPrimitive2DBoussinesqLNSE{AdjointDiscrete}
        @test equations.base == (nothing, nothing, Θ)
        @test equations.nl.Pr ≈ 0.71
        @test equations.nl.Ri ≈ 1000 / (2^2 * 0.71)
        @test equations.nl.grav == 2
        @test equations.nl.plans isa FFTPlans{false}
        @test length(equations.nl.scache) == 3
        @test length(equations.nl.pcache) == 6
        @test passive_scalar.nl.Ri == 0

        formulation = CartesianPrimitive2DBoussinesq(0.71, 3.0)
        direct_nonlinear = CartesianPrimitive2DBoussinesqNSE(
            g, 2.0, formulation; flags=FFTW.ESTIMATE, dealias=false)
        direct_forward = CartesianPrimitive2DBoussinesqLNSE(
            g, 2.0, formulation; mode=Forward(), flags=FFTW.ESTIMATE, dealias=false)
        @test direct_nonlinear isa CartesianPrimitive2DBoussinesqNSE
        @test direct_forward isa CartesianPrimitive2DBoussinesqLNSE{Forward}
        @test_throws ArgumentError CartesianPrimitive2DBoussinesq(0.71, 3.0; grav=3)
        @test_throws ArgumentError RayleighBenardFlow(
            g, 2.0, 0.71, 1000.0; base=(nothing, nothing), fftw_flags=FFTW.ESTIMATE)

        streamwise_invariant = StreamwiseInvariantChannelGrid(17, 9; Nt=1, β=0.5, width=3)
        @test !applicable(RayleighBenardFlow, streamwise_invariant, 2.0, 0.71, 1000.0)
        @test_throws ArgumentError CartesianPrimitive2DBoussinesqNSE(
            streamwise_invariant, 2.0, formulation; flags=FFTW.ESTIMATE, dealias=false)
    end

    @testset verbose=true "Two-dimensional nonlinear Boussinesq equations              " begin
        Re, Pr, Ri, α = 37.0, 0.71, 2.3, 0.75
        g = TwoDimensionalChannelGrid(17, 25; Nt=3, α, width=7)
        formulation = CartesianPrimitive2DBoussinesq(Pr, Ri)
        equations = CartesianPrimitive2DBoussinesqNSE(g, Re, formulation; flags=FFTW.ESTIMATE)
        horizontal_buoyancy = CartesianPrimitive2DBoussinesqNSE(
            g, Re, CartesianPrimitive2DBoussinesq(Pr, Ri; grav=1); flags=FFTW.ESTIMATE)

        u(X, Y, _, T) = @. (1 - Y^2) * sin(α * X) + 0T
        v(X, Y, _, T) = @. 1 + 0X + 0Y + 0T
        θ(X, Y, _, T) = @. (1 - Y^2) * cos(α * X) + 0T
        zero_rhs(X, Y, _, T) = @. 0X + 0Y + 0T
        u_rhs(X, Y, _, T) = @. (-2 - α^2 * (1 - Y^2)) * sin(α * X) / Re -
                                α * (1 - Y^2)^2 * sin(α * X) * cos(α * X) +
                                2Y * sin(α * X) + 0T
        v_rhs(X, Y, _, T) = @. Ri * (1 - Y^2) * cos(α * X) + 0T
        θ_rhs(X, Y, _, T) = @. (-2 - α^2 * (1 - Y^2)) * cos(α * X) / (Re * Pr) +
                                α * (1 - Y^2)^2 * sin(α * X)^2 + 2Y * cos(α * X) + 0T
        horizontal_u_rhs(X, Y, _, T) = u_rhs(X, Y, nothing, T) .+
                                        Ri .* (1 .- Y .^ 2) .* cos.(α .* X)
        state = FFT(VectorField(g, u, v, θ))
        exact = FFT(VectorField(g, u_rhs, v_rhs, θ_rhs))
        exact_horizontal = FFT(VectorField(g, horizontal_u_rhs, zero_rhs, θ_rhs))

        @test equations(0.0, state, similar(state)) ≈ exact rtol=1e-10 atol=1e-10
        @test horizontal_buoyancy(0.0, state, similar(state)) ≈ exact_horizontal rtol=1e-10 atol=1e-10
    end

    @testset verbose=true "Conduction-state linearised equations                       " begin
        Re, Pr, Ri, α = 31.0, 0.71, 1.7, 0.75
        g = TwoDimensionalChannelGrid(17, 25; Nt=1, α, width=7)
        formulation = CartesianPrimitive2DBoussinesq(Pr, Ri)
        forward = CartesianPrimitive2DBoussinesqLNSE(
            g, Re, formulation; mode=Forward(), flags=FFTW.ESTIMATE, dealias=false)
        continuous = CartesianPrimitive2DBoussinesqLNSE(
            g, Re, formulation; mode=AdjointContinuous(), flags=FFTW.ESTIMATE, dealias=false)

        zero_state(X, Y, _, T) = @. 0X + 0Y + 0T
        conduction(X, Y, _, T) = @. (1 - Y) / 2 + 0X + 0T
        p₁(X, Y, _, T) = @. (1 - Y^2) * sin(α * X) + 0T
        p₂(X, Y, _, T) = @. (1 - Y^2) * cos(α * X) + 0T
        p₃(X, Y, _, T) = @. (1 - Y^2) * sin(α * X) + 0T
        Lp₁(X, Y, _, T) = @. (-2 - α^2 * (1 - Y^2)) * sin(α * X) / Re + 0T
        Lp₂(X, Y, _, T) = @. (-2 - α^2 * (1 - Y^2)) * cos(α * X) / Re +
                                Ri * (1 - Y^2) * sin(α * X) + 0T
        Lp₃(X, Y, _, T) = @. (-2 - α^2 * (1 - Y^2)) * sin(α * X) / (Re * Pr) +
                                (1 - Y^2) * cos(α * X) / 2 + 0T
        Ap₁(X, Y, _, T) = @. (-2 - α^2 * (1 - Y^2)) * sin(α * X) / Re + 0T
        Ap₂(X, Y, _, T) = @. (-2 - α^2 * (1 - Y^2)) * cos(α * X) / Re +
                                (1 - Y^2) * sin(α * X) / 2 + 0T
        Ap₃(X, Y, _, T) = @. (-2 - α^2 * (1 - Y^2)) * sin(α * X) / (Re * Pr) +
                                Ri * (1 - Y^2) * cos(α * X) + 0T
        base = FFT(VectorField(g, zero_state, zero_state, conduction))
        perturbation = FFT(VectorField(g, p₁, p₂, p₃))
        exact_forward = FFT(VectorField(g, Lp₁, Lp₂, Lp₃))
        exact_continuous = FFT(VectorField(g, Ap₁, Ap₂, Ap₃))

        @test forward(0.0, base, perturbation, similar(base)) ≈ exact_forward rtol=1e-10 atol=1e-10
        @test continuous(0.0, base, perturbation, similar(base)) ≈ exact_continuous rtol=1e-10 atol=1e-10
    end

    @testset verbose=true "Linearisation and discrete-adjoint identity                 " begin
        Re, Pr, Ri, α = 29.0, 0.71, 1.3, 0.75
        g = TwoDimensionalChannelGrid(17, 25; Nt=1, α, width=7)
        formulation = CartesianPrimitive2DBoussinesq(Pr, Ri)
        nonlinear = CartesianPrimitive2DBoussinesqNSE(
            g, Re, formulation; flags=FFTW.ESTIMATE, dealias=false)
        forward = CartesianPrimitive2DBoussinesqLNSE(
            g, Re, formulation; mode=Forward(), flags=FFTW.ESTIMATE, dealias=false)

        U(X, Y, _, T) = @. (1 - Y^2) * sin(α * X) + 0T
        V(X, Y, _, T) = @. 0.2 * (1 - Y^2) * cos(α * X) + 0T
        Θ(X, Y, _, T) = @. (1 - Y) / 2 + 0.1 * (1 - Y^2) * cos(α * X) + 0T
        p₁(X, Y, _, T) = @. Y * (1 - Y^2) * cos(α * X) + 0T
        p₂(X, Y, _, T) = @. (1 - Y^2)^2 * sin(α * X) + 0T
        p₃(X, Y, _, T) = @. (1 - Y^2) * cos(α * X) + 0T
        q₁(X, Y, _, T) = @. (1 - Y^2)^2 * cos(α * X) + 0T
        q₂(X, Y, _, T) = @. Y * (1 - Y^2) * sin(α * X) + 0T
        q₃(X, Y, _, T) = @. (1 - Y^2) * sin(α * X) + 0T
        base = FFT(VectorField(g, U, V, Θ))
        perturbation = FFT(VectorField(g, p₁, p₂, p₃))
        test_field = FFT(VectorField(g, q₁, q₂, q₃))

        ε = 1e-6
        forward_difference = nonlinear(0.0, base .+ ε .* perturbation, similar(base))
        backward_difference = nonlinear(0.0, base .- ε .* perturbation, similar(base))
        finite_difference = (forward_difference - backward_difference) ./ (2ε)
        linearised = forward(0.0, base, perturbation, similar(base))
        @test finite_difference ≈ linearised rtol=1e-8 atol=1e-9

        for dealias in (false, true)
            forward_adjoint_check = CartesianPrimitive2DBoussinesqLNSE(
                g, Re, formulation; mode=Forward(), flags=FFTW.ESTIMATE, dealias)
            adjoint = CartesianPrimitive2DBoussinesqLNSE(
                g, Re, formulation; mode=AdjointDiscrete(), flags=FFTW.ESTIMATE, dealias)
            lhs = dot(forward_adjoint_check(0.0, base, perturbation, similar(base)), test_field)
            rhs = dot(perturbation, adjoint(0.0, base, test_field, similar(base)))
            @test lhs ≈ rhs rtol=1e-11 atol=1e-11
        end
    end

    @testset verbose=true "Analytical Rayleigh–Bénard norms                            " begin
        @testset verbose=true "Two-dimensional Boussinesq norms                            " begin
            g = TwoDimensionalChannelGrid(9, 17; Nt=9, α=π,
                                          dist=FDGrids.GaussLobattoGrid(), width=3)
            f(X, Y, _, T) = (1 - Y^2) * case_periodic_profile(π*X) * case_periodic_profile(2π*T)
            twice_f(X, Y, Z, T) = 2f(X, Y, Z, T)
            thrice_f(X, Y, Z, T) = 3f(X, Y, Z, T)
            physical = Field(g, f)
            spectral = FFT(physical)
            exact_norm2 = CASE_WALL_POLY_NORM2 * CASE_PERIODIC_PROFILE_NORM2^2

            @test case_physical_dot(g, parent(physical)) ≈ exact_norm2 rtol=1e-12
            @test dot(spectral, spectral) ≈ exact_norm2 rtol=1e-12
            @test norm(spectral) ≈ sqrt(exact_norm2) rtol=1e-12
            @test norm(FFT(VectorField(g, f, twice_f, thrice_f)))^2 ≈ 14exact_norm2 rtol=1e-12
        end

        @testset verbose=true "Three-dimensional Boussinesq norms                          " begin
            g = ChannelGrid(9, 17, 9; Nt=9, α=π, β=π,
                            dist=FDGrids.GaussLobattoGrid(), width=3)
            plans = FFTPlans(g; flags=FFTW.ESTIMATE, dealias=false)
            wall_normal_norm2 = CASE_WALL_POLY_NORM2

            y_only, _ = case_physical_to_spectral(g, plans,
                (X, Y, Z, T) -> @. (1 - Y^2) + 0X + 0Z + 0T)
            @test dot(y_only, y_only) ≈ wall_normal_norm2 rtol=1e-12
            @test norm(y_only) ≈ sqrt(wall_normal_norm2) rtol=1e-12

            yx, _ = case_physical_to_spectral(g, plans,
                (X, Y, Z, T) -> @. (1 - Y^2) * case_periodic_profile(π * X) + 0Z + 0T)
            @test dot(yx, yx) ≈ wall_normal_norm2 * CASE_PERIODIC_PROFILE_NORM2 rtol=1e-12
            @test norm(yx) ≈ sqrt(wall_normal_norm2 * CASE_PERIODIC_PROFILE_NORM2) rtol=1e-12

            f(X, Y, Z, T) = (1 - Y^2) * case_periodic_profile(π*X) * case_periodic_profile(π*Z) * case_periodic_profile(2π*T)
            twice_f(X, Y, Z, T) = 2f(X, Y, Z, T)
            thrice_f(X, Y, Z, T) = 3f(X, Y, Z, T)
            four_times_f(X, Y, Z, T) = 4f(X, Y, Z, T)
            full, physical = case_physical_to_spectral(g, plans, f)
            exact_norm2 = wall_normal_norm2 * CASE_PERIODIC_PROFILE_NORM2^3
            @test case_physical_dot(g, physical) ≈ exact_norm2 rtol=1e-12
            @test dot(full, full) ≈ exact_norm2 rtol=1e-12
            @test norm(full) ≈ sqrt(exact_norm2) rtol=1e-12
            @test norm(FFT(VectorField(g, f, twice_f, thrice_f, four_times_f)))^2 ≈ 30exact_norm2 rtol=1e-12
        end
    end
end
