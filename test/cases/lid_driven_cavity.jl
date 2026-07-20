@testset verbose=true "Lid-driven cavity case                                      " begin
    @testset verbose=true "Grid layout, coordinates, and growth                        " begin
        g = LidDrivenCavityGrid(13, 11; Nt=5, dist=FDGrids.GaussLobattoGrid(), width=5)
        x, y = g.xs
        X, Y, T = points(g)

        @test g isa NSEBase.AbstractLidDrivenCavity2DGrid
        @test g isa AbstractLidDrivenCavityGrid
        @test size(g) == (13, 11, 5) # Rectangular cavities replace the old unequal-size error.
        @test fft_storage_dims(g) == LID_DRIVEN_CAVITY_2D_FFT_ORDER
        @test inhomogeneous_storage_dims(g) == LID_DRIVEN_CAVITY_2D_INHOMOGENEOUS_DIMS
        @test map(size, (X, Y, T)) == ((13, 1, 1), (1, 11, 1), (1, 1, 5))
        @test vec(X) == x
        @test vec(Y) == y
        @test vec(T) ≈ (0:4) ./ 5
        @test wavenumber_scale(g, 3) ≈ 2π
        @test weights(g) === g.w
        @test weights(g) ≈ g.ws[1] * transpose(g.ws[2])
        @test points(g, (7,)) == points(growto(g, (7,)))

        bounded = LidDrivenCavityGrid(17, 17, 13; spanwise=:bounded, dist=FDGrids.GaussLobattoGrid(), width=5)
        periodic = LidDrivenCavityGrid(17, 17, 9; Nt=5, spanwise=:periodic, Lz=2,
                                       dist=FDGrids.GaussLobattoGrid(), width=5)
        periodic_points = points(periodic)
        @test bounded isa NSEBase.AbstractLidDrivenCavity3DBoundedGrid && bounded isa AbstractLidDrivenCavity3DGrid
        @test periodic isa NSEBase.AbstractLidDrivenCavity3DPeriodicGrid && periodic isa AbstractLidDrivenCavity3DGrid
        @test LID_DRIVEN_CAVITY_3D_AXES == (1, 2, 3, 4)
        @test size(bounded) == (17, 17, 13, 1)
        @test size(periodic) == (17, 17, 9, 5)
        @test fft_storage_dims(bounded) == LID_DRIVEN_CAVITY_3D_BOUNDED_FFT_ORDER
        @test fft_storage_dims(periodic) == LID_DRIVEN_CAVITY_3D_PERIODIC_FFT_ORDER
        @test inhomogeneous_storage_dims(bounded) == (1, 2, 3)
        @test inhomogeneous_storage_dims(periodic) == (1, 2)
        @test map(size, periodic_points) == ((17, 1, 1, 1), (1, 17, 1, 1), (1, 1, 9, 1), (1, 1, 1, 5))
        @test wavenumber_scale(periodic, 3) ≈ π

        grown = growto(periodic, (11, 7))
        @test size(grown) == (17, 17, 11, 7)
        @test grown.xs == periodic.xs
        @test grown.w == periodic.w

        compact = LidDrivenCavityGrid(13, 11; Nt=1, dist=FDGrids.GaussLobattoGrid(), width=5)
        @test size(compact) == (13, 11, 1)
    end

    @testset verbose=true "Finite-difference derivatives                               " begin
        g = LidDrivenCavityGrid(21, 21; dist=FDGrids.GaussLobattoGrid(), width=9)
        f(x, y, _, _) = x^3 + 2y^3 + x * y + one(x)
        dfdx(x, y, _, _) = 3x^2 + y
        dfdy(x, y, _, _) = 6y^2 + x
        lap(x, y, _, _) = 6x + 12y
        u = FFT(Field(g, f))

        @test ddx!(FTField(g), u) ≈ FFT(Field(g, dfdx))
        @test ddy!(FTField(g), u) ≈ FFT(Field(g, dfdy))
        @test laplacian!(FTField(g), u) ≈ FFT(Field(g, lap))
        @test NSEBase.derivative_matrix(g, 1, Val(1), Val(false)) === g.D₁[1]
        @test NSEBase.derivative_matrix(g, 2, Val(2), Val(false)) === g.D₂[2]
        @test NSEBase.derivative_matrix(g, 1, Val(1), Val(false)) ≈ NSEBase.derivative_matrix(g, 2, Val(1), Val(false))

        rectangular = LidDrivenCavityGrid(13, 11; Nt=5, dist=FDGrids.GaussLobattoGrid(), width=5)
        x, y = rectangular.xs
        field, dx, dy, Δ = FTField(rectangular), FTField(rectangular), FTField(rectangular), FTField(rectangular)
        parent(field) .= reshape(x .^ 3, :, 1, 1) .+ reshape(y .^ 3, 1, :, 1)
        ddx!(dx, field)
        ddy!(dy, field)
        inhomogeneous_laplacian!(Δ, field)
        @test parent(dx)[:, 1, 1] ≈ 3 .* x .^ 2 atol=1e-10
        @test parent(dy)[1, :, 1] ≈ 3 .* y .^ 2 atol=1e-10
        @test parent(Δ)[:, :, 1] ≈ reshape(6 .* x, :, 1) .+ reshape(6 .* y, 1, :) atol=1e-9
    end

    @testset verbose=true "Analytical norms and inner products                         " begin
        velocity = (((0, 0, 1 // 1), (1, 0, 1 // 1), (0, 1, 2 // 1), (1, 1, 1 // 1)),
                    ((2, 0, 1 // 1), (0, 1, -1 // 1), (1, 1, 3 // 1)))
        test_velocity = (((0, 0, 2 // 1), (1, 0, -1 // 1), (0, 2, 1 // 1)),
                         ((0, 0, 1 // 1), (1, 0, 2 // 1), (1, 1, -1 // 1)))
        analytic_grid = LidDrivenCavityGrid(17, 17; dist=FDGrids.GaussLobattoGrid(), width=5)
        u = case_analytic_velocity_field(analytic_grid, velocity)
        v = case_analytic_velocity_field(analytic_grid, test_velocity)
        exact = case_analytic_velocity_dot(velocity, test_velocity)
        exact_norm2 = case_analytic_velocity_dot(velocity, velocity)
        @test exact == 77 // 12
        @test exact_norm2 == 1661 // 180
        @test dot(u, v) ≈ Float64(exact) atol=1e-13
        @test norm(u)^2 ≈ Float64(exact_norm2) atol=1e-13

        spectral_grid = LidDrivenCavityGrid(17, 17; Nt=5, width=5)
        a = deterministic_ftfield(spectral_grid, 0.1)
        b = deterministic_ftfield(spectral_grid, 0.7)
        c = deterministic_ftfield(spectral_grid, 1.3)
        α, β = 1.75, -0.25
        combination = FTField(spectral_grid, α .* parent(b) .+ β .* parent(c))
        @test dot(a, b) ≈ case_reference_dot(a, b)
        @test dot(a, b) ≈ dot(b, a)
        @test dot(a, combination) ≈ α * dot(a, b) + β * dot(a, c)
        @test norm(a)^2 ≈ dot(a, a)
        q, p = VectorField(a, b), VectorField(b, c)
        @test dot(q, p) ≈ dot(a, b) + dot(b, c)
        @test norm(q)^2 ≈ dot(q, q)
    end

    @testset verbose=true "Analytical cavity norms                                     " begin
        @testset verbose=true "Two-dimensional cavity norms                                " begin
            g = LidDrivenCavityGrid(11, 11; Nt=9, dist=FDGrids.GaussLobattoGrid(), width=5)
            f(x, y, _, t) = case_poly11(x) * case_poly11(y) * case_periodic_profile(2π*t)
            twice_f(x, y, z, t) = 2f(x, y, z, t)
            physical = Field(g, f)
            spectral = FFT(physical)
            exact_norm2 = CASE_UNIT_BUBBLE_NORM2^2 * CASE_PERIODIC_PROFILE_NORM2

            @test case_physical_dot(g, parent(physical)) ≈ exact_norm2 rtol=1e-12
            @test dot(spectral, spectral) ≈ exact_norm2 rtol=1e-12
            @test norm(spectral) ≈ sqrt(exact_norm2) rtol=1e-12
            @test norm(FFT(VectorField(g, f, twice_f)))^2 ≈ 5exact_norm2 rtol=1e-12
        end

        @testset verbose=true "Bounded three-dimensional cavity norms                      " begin
            g = LidDrivenCavityGrid(11, 11, 11; Nt=9, spanwise=:bounded,
                                    dist=FDGrids.GaussLobattoGrid(), width=5)
            f(x, y, z, t) = case_poly11(x) * case_poly11(y) * case_poly11(z) * case_periodic_profile(2π*t)
            twice_f(x, y, z, t) = 2f(x, y, z, t)
            thrice_f(x, y, z, t) = 3f(x, y, z, t)
            physical = Field(g, f)
            spectral = FFT(physical)
            exact_norm2 = CASE_UNIT_BUBBLE_NORM2^3 * CASE_PERIODIC_PROFILE_NORM2

            @test case_physical_dot(g, parent(physical)) ≈ exact_norm2 rtol=1e-12
            @test dot(spectral, spectral) ≈ exact_norm2 rtol=1e-12
            @test norm(spectral) ≈ sqrt(exact_norm2) rtol=1e-12
            @test norm(FFT(VectorField(g, f, twice_f, thrice_f)))^2 ≈ 14exact_norm2 rtol=1e-12
        end

        @testset verbose=true "Periodic three-dimensional cavity norms                     " begin
            g = LidDrivenCavityGrid(11, 11, 9; Nt=9, spanwise=:periodic, Lz=2,
                                    dist=FDGrids.GaussLobattoGrid(), width=5)
            f(x, y, z, t) = case_poly11(x) * case_poly11(y) * case_periodic_profile(π*z) * case_periodic_profile(2π*t)
            twice_f(x, y, z, t) = 2f(x, y, z, t)
            thrice_f(x, y, z, t) = 3f(x, y, z, t)
            physical = Field(g, f)
            spectral = FFT(physical)
            exact_norm2 = CASE_UNIT_BUBBLE_NORM2^2 * CASE_PERIODIC_PROFILE_NORM2^2

            @test case_physical_dot(g, parent(physical)) ≈ exact_norm2 rtol=1e-12
            @test dot(spectral, spectral) ≈ exact_norm2 rtol=1e-12
            @test norm(spectral) ≈ sqrt(exact_norm2) rtol=1e-12
            @test norm(FFT(VectorField(g, f, twice_f, thrice_f)))^2 ≈ 14exact_norm2 rtol=1e-12
        end
    end

    @testset verbose=true "Weighted adjoints                                           " begin
        g = LidDrivenCavityGrid(17, 17; width=5)
        a = reshape(sin.(1:17^2), 17, 17)
        b = reshape(cos.(1:17^2), 17, 17)
        Da, D⁺b = similar(a), similar(b)
        mul!(Da, g.D₁[1], a, Val(1))
        mul!(D⁺b, g.D₁⁺[1], b, Val(1))
        @test sum(g.w .* Da .* b) ≈ sum(g.w .* a .* D⁺b)
        mul!(Da, g.D₁[2], a, Val(2))
        mul!(D⁺b, g.D₁⁺[2], b, Val(2))
        @test sum(g.w .* Da .* b) ≈ sum(g.w .* a .* D⁺b)

        spectral_grid = LidDrivenCavityGrid(17, 17; Nt=5, width=5)
        u, v = deterministic_ftfield(spectral_grid, 0.2), deterministic_ftfield(spectral_grid, 0.9)
        Dxu, Dx⁺v = ddx!(FTField(spectral_grid), u), ddx!(FTField(spectral_grid), v; adjoint=true)
        Dyu, Dy⁺v = ddy!(FTField(spectral_grid), u), ddy!(FTField(spectral_grid), v; adjoint=true)
        Δu, Δ⁺v = laplacian!(FTField(spectral_grid), u), laplacian!(FTField(spectral_grid), v; adjoint=true)
        @test dot(Dxu, v) ≈ dot(u, Dx⁺v)
        @test dot(Dyu, v) ≈ dot(u, Dy⁺v)
        @test dot(Δu, v) ≈ dot(u, Δ⁺v)

        rectangular = LidDrivenCavityGrid(13, 11; dist=FDGrids.GaussLobattoGrid(), width=5)
        x, y = rectangular.xs
        u = ftfield_from_inhomogeneous(rectangular, x .^ 3 .+ transpose(y .^ 3))
        v = ftfield_from_inhomogeneous(rectangular, ((1 .- x) .* x) * transpose((1 .- y) .* y))
        for derivative! in (ddx!, ddy!)
            Du = derivative!(FTField(rectangular), u)
            D⁺v = derivative!(FTField(rectangular), v; adjoint=true)
            @test dot(Du, v) ≈ dot(u, D⁺v) rtol=1e-12 atol=1e-12
        end
        Δu = inhomogeneous_laplacian!(FTField(rectangular), u)
        Δ⁺v = inhomogeneous_laplacian!(FTField(rectangular), v; adjoint=true)
        @test dot(Δu, v) ≈ dot(u, Δ⁺v) rtol=1e-12 atol=1e-12
    end

    @testset verbose=true "Temporal derivative and flow constructor                    " begin
        temporal = LidDrivenCavityGrid(9, 9; Nt=9, width=3)
        plans = FFTPlans(temporal; flags=FFTW.ESTIMATE, dealias=false)
        physical = Field(temporal, (_, _, _, t) -> case_periodic_profile(2π * t))
        spectral, derivative, result = FTField(temporal), FTField(temporal), Field(temporal)
        plans(spectral, physical)
        ddt!(derivative, spectral)
        plans(result, derivative)
        expected = 2π .* case_periodic_profile_d1(2π .* points(temporal)[3]) .* ones(size(temporal, 1), size(temporal, 2), 1)
        @test parent(result) ≈ expected atol=1e-11

        compact = LidDrivenCavityGrid(13, 11; dist=FDGrids.GaussLobattoGrid(), width=5)
        X, Y, _ = points(compact)
        U = @. 16X^2 * (1 - X)^2 * (3Y^2 - 2Y)
        V = @. -32X * (1 - X) * (1 - 2X) * Y^2 * (Y - 1)
        base = (U, V)
        equations = LidDrivenCavityFlow(compact, 100; base, fftw_flags=FFTW.ESTIMATE, dealias=false)
        @test equations isa ProjectedNSE && equations.base === base

        grid3 = LidDrivenCavityGrid(9, 9, 9; spanwise=:bounded, width=3)
        X3, Y3, _, _ = points(grid3)
        U3 = @. 16X3^2 * (1 - X3)^2 * (3Y3^2 - 2Y3)
        V3 = @. -32X3 * (1 - X3) * (1 - 2X3) * Y3^2 * (Y3 - 1)
        base3 = (U3, V3, nothing)
        equations3 = LidDrivenCavityFlow(grid3, 100; base=base3, fftw_flags=FFTW.ESTIMATE, dealias=false)
        @test equations3 isa ProjectedNSE && equations3.base === base3

        periodic3 = LidDrivenCavityGrid(9, 9, 9; spanwise=:periodic, Lz=2, width=3)
        equations3p = LidDrivenCavityFlow(periodic3, 100; base=base3, fftw_flags=FFTW.ESTIMATE, dealias=false)
        @test equations3p isa ProjectedNSE && equations3p.base === base3
    end
end
