@testset verbose=true "Square-duct case                                            " begin
    N, Nz, Nt = 25, 5, 3
    α = 1.25
    square = SquareDuctGrid(N, Nz, Nt, α; dist=FDGrids.GaussLobattoGrid(), width=9)
    historical = SquareDuctGrid(N, 9, Nz, Nt, α; dist=FDGrids.GaussLobattoGrid())

    @testset verbose=true "Square grid and product quadrature                          " begin
        @test square isa SquareDuctGrid
        @test square isa RectangularGrid{2}
        @test size(square) == (N, N, Nz, Nt)
        @test SQUARE_DUCT_AXES == (1, 2, 3, 4)
        @test SQUARE_DUCT_FFT_ORDER == (3, 4)
        @test SQUARE_DUCT_INHOMOGENEOUS_DIMS == (1, 2)
        @test fft_storage_dims(square) == SQUARE_DUCT_FFT_ORDER
        @test inhomogeneous_storage_dims(square) == SQUARE_DUCT_INHOMOGENEOUS_DIMS
        @test spatial_inhomogeneous_physical_dims(square) == (:x, :y)
        for field in (:xs, :ws, :D₁, :D₂, :D₁⁺, :D₂⁺)
            values = getproperty(square, field)
            @test values[1] === values[2]
        end

        x = square.xs[1]
        w = square.ws[1]
        sine, polynomial = sin.(π .* x), case_poly11(x)
        @test weights(square) === square.w
        @test size(weights(square)) == (N, N)
        @test weights(square) ≈ w * transpose(w) rtol=1e-15
        @test sum(w .* sine .^ 2) ≈ 1 / 2 rtol=1e-12
        @test sum(weights(square) .* (sine * transpose(sine)) .^ 2) ≈ 1 / 4 rtol=1e-12
        @test sum(w .* polynomial) ≈ 1 / 6 rtol=1e-12
        @test sum(weights(square) .* (polynomial * transpose(polynomial))) ≈ 1 / 36 rtol=1e-12
        @test points(historical) == points(square)
    end

    @testset verbose=true "Square cross-section FD contract                            " begin
        test_two_inhomogeneous_fd_contract(square, (63, 3))
    end

    duct = square

    @testset verbose=true "Layout, coordinates, and growth                             " begin
        @test duct isa SquareDuctGrid
        @test size(duct) == (N, N, Nz, Nt)
        @test extrema(duct.xs[1]) == (0.0, 1.0)
        @test extrema(duct.xs[2]) == (0.0, 1.0)
        @test duct.D₁[1] === duct.D₁[2]
        @test weights(duct) ≈ duct.ws[1] * transpose(duct.ws[2]) rtol=1e-15
        @test sum(duct.ws[1]) ≈ 1 rtol=1e-14
        @test sum(duct.ws[2]) ≈ 1 rtol=1e-14

        X, Y, Z, T = points(duct)
        @test map(size, (X, Y, Z, T)) == ((N, 1, 1, 1), (1, N, 1, 1), (1, 1, Nz, 1), (1, 1, 1, Nt))
        @test vec(X) == duct.xs[1]
        @test vec(Y) == duct.xs[2]
        @test vec(Z) ≈ (0:Nz-1) .* (2π / (α * Nz))
        @test vec(T) ≈ (0:Nt-1) ./ Nt
        @test wavenumber_scale(duct, 3) == α
        @test wavenumber_scale(duct, 4) == 2π

        target = (7, 5)
        grown = growto(duct, target)
        @test size(grown) == (N, N, target...)
        @test points(grown) == points(duct, target)
        @test all(grown.xs[i] === duct.xs[i] for i in 1:2)
        @test all(grown.ws[i] === duct.ws[i] for i in 1:2)
        @test all(grown.D₁[i] === duct.D₁[i] && grown.D₂[i] === duct.D₂[i] for i in 1:2)
        @test all(grown.D₁⁺[i] === duct.D₁⁺[i] && grown.D₂⁺[i] === duct.D₂⁺[i] for i in 1:2)
        @test weights(grown) ≈ weights(duct) rtol=1e-15
    end

    @testset verbose=true "Analytic derivatives and adjoints                           " begin
        x, y = duct.xs
        u, dx, dy, lap = FTField(duct), FTField(duct), FTField(duct), FTField(duct)
        parent(u)[:, :, 1, 1] .= reshape(x .^ 3, :, 1) .+ reshape(2 .* y .^ 2, 1, :)
        ddx!(dx, u)
        ddy!(dy, u)
        inhomogeneous_laplacian!(lap, u)
        expected_dx = reshape(3 .* x .^ 2, :, 1) .* ones(1, N)
        expected_dy = ones(N, 1) .* reshape(4 .* y, 1, :)
        expected_lap = reshape(6 .* x, :, 1) .+ 4 .* ones(1, N)
        @test parent(dx)[:, :, 1, 1] ≈ expected_dx atol=2e-11
        @test parent(dy)[:, :, 1, 1] ≈ expected_dy atol=2e-11
        @test parent(lap)[:, :, 1, 1] ≈ expected_lap atol=5e-10
        @test iszero(parent(dx)[:, :, 2:end, :])
        @test iszero(parent(dy)[:, :, 2:end, :])
        @test iszero(parent(lap)[:, :, 2:end, :])

        a, b = FTField(duct), FTField(duct)
        aₓ, aᵧ = (@. 1 + x + x^2), (@. 0.7 - y + 0.2y^2)
        bₓ, bᵧ = (@. 0.3 - x + 0.5x^3), (@. 1 + y + y^2)
        parent(a)[:, :, 1, 1] .= reshape(aₓ, :, 1) .* reshape(aᵧ, 1, :)
        parent(b)[:, :, 1, 1] .= reshape(bₓ, :, 1) .* reshape(bᵧ, 1, :)
        for derivative! in (ddx!, ddy!)
            Da = derivative!(FTField(duct), a)
            D⁺b = derivative!(FTField(duct), b; adjoint=true)
            @test dot(Da, b) ≈ dot(a, D⁺b) atol=2e-12 rtol=2e-12
        end
        Δa = inhomogeneous_laplacian!(FTField(duct), a)
        Δ⁺b = inhomogeneous_laplacian!(FTField(duct), b; adjoint=true)
        @test dot(Δa, b) ≈ dot(a, Δ⁺b) atol=2e-12 rtol=2e-12
    end

    @testset verbose=true "Flow constructors                                           " begin
        x, y = duct.xs
        Wₓ, Wᵧ = (@. x * (1 - x)), (@. y * (1 - y))
        base = (nothing, nothing, reshape(Wₓ, :, 1) .* reshape(Wᵧ, 1, :))
        equations = SquareDuctFlow(duct, 500; base, f=1.25, fftw_flags=FFTW.ESTIMATE, dealias=false)
        compatibility = SquareDuctFlow(square, 500; f=0.75, fftw_flags=FFTW.ESTIMATE, dealias=false)
        @test equations isa ProjectedNSE
        @test compatibility isa ProjectedNSE
        @test equations.base === base
        @test compatibility.base == (nothing, nothing, nothing)
        @test equations.ln isa CartesianPrimitive3DLNSE{AdjointDiscrete}
        @test equations.nl.force isa ConstantBodyForce
        @test equations.ln.force isa ConstantBodyForce
        @test equations.nl.force.value == 1.25
        @test equations.nl.force.component == 3
        @test compatibility.nl.force.value == 0.75
        @test compatibility.nl.force.component == 3
        @test size(equations.nl.pcache[1][1]) == size(duct)
        @test size(compatibility.nl.pcache[1][1]) == size(square)
    end

    @testset verbose=true "Analytical product norm and Parseval identity               " begin
        g = SquareDuctGrid(15, 9, 9, 2π; width=5)
        u(x, y, z, t) = case_poly11(x) * case_poly11(y) * case_periodic_profile(2π*z) * case_periodic_profile(2π*t)
        twice_u(x, y, z, t) = 2u(x, y, z, t)
        thrice_u(x, y, z, t) = 3u(x, y, z, t)
        physical = Field(g, u)
        spectral = FFT(physical)
        exact_norm2 = CASE_UNIT_BUBBLE_NORM2^2 * CASE_PERIODIC_PROFILE_NORM2^2
        @test case_physical_dot(g, parent(physical)) ≈ exact_norm2 rtol=1e-12
        @test dot(spectral, spectral) ≈ exact_norm2 rtol=1e-12
        @test norm(FFT(VectorField(g, u, twice_u, thrice_u)))^2 ≈ 14exact_norm2 rtol=1e-12
    end

    @testset verbose=true "Square 4D velocity norm with an analytical reference        " begin
        g = SquareDuctGrid(25, 9, 9, 2π; dist=FDGrids.GaussLobattoGrid(), width=9)
        u(x, y, z, t) = sin(π*x) * sin(π*y) * case_periodic_profile(2π*z) * case_periodic_profile(2π*t)
        physical = Field(g, u)
        spectral = FFT(physical)
        exact_norm2 = CASE_PERIODIC_PROFILE_NORM2^2 / 4
        @test case_physical_dot(g, parent(physical)) ≈ exact_norm2 rtol=1e-10
        @test norm(spectral)^2 ≈ exact_norm2 rtol=1e-10

        velocity = FFT(VectorField(g, u, u, u))
        @test norm(velocity)^2 ≈ 3exact_norm2 rtol=1e-10
    end
end
