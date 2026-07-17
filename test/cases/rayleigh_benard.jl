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

    @testset verbose=true "Physical and spectral norms                                 " begin
        g = ChannelGrid(9, 17, 9; Nt=9, α=π, β=π, dist=FDGrids.GaussLobattoGrid(), width=3)
        plans = FFTPlans(g; flags=FFTW.ESTIMATE, dealias=false)
        wall_normal_norm = sum(g.ws[1] .* (1 .- g.xs[1] .^ 2) .^ 2)

        y_only, _ = case_physical_to_spectral(g, plans, (X, Y, Z, T) -> @. (1 - Y^2) + 0 * X + 0 * Z + 0 * T)
        @test dot(y_only, y_only) ≈ wall_normal_norm rtol=1e-12
        @test norm(y_only) ≈ sqrt(wall_normal_norm) rtol=1e-12

        yx, _ = case_physical_to_spectral(g, plans, (X, Y, Z, T) -> @. (1 - Y^2) * cos(π * X) + 0 * Z + 0 * T)
        @test dot(yx, yx) ≈ wall_normal_norm / 2 rtol=1e-10
        @test norm(yx) ≈ sqrt(wall_normal_norm / 2) rtol=1e-10

        full, physical = case_physical_to_spectral(g, plans, (X, Y, Z, T) -> @. (1 - Y^2) * cos(π * X) * cos(π * Z) * cos(2π * T))
        Ny, Nx, Nz, Nt = size(g)
        physical_norm = sum(g.ws[1][j] * physical[j, kx, kz, kt]^2
                            for j in 1:Ny, kx in 1:Nx, kz in 1:Nz, kt in 1:Nt) / (Nx * Nz * Nt)
        @test physical_norm ≈ wall_normal_norm / 8 rtol=1e-10
        @test dot(full, full) ≈ physical_norm rtol=1e-10
        @test norm(full) ≈ sqrt(wall_normal_norm / 8) rtol=1e-10
    end
end
