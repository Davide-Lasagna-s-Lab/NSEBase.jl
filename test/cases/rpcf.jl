@testset verbose=true "RPCF case                                                   " begin
    @testset verbose=true "Grid layout and coordinates                                 " begin
        g = ChannelGrid(17, 9; Nt=5, β=0.5, width=3)
        positional = ChannelGrid(17, 9, 5, 3, 0.5)
        y = g.xs[1]
        grid_points = points(g)

        @test g isa AbstractChannel2D3CGrid
        @test g isa AbstractChannelGrid
        @test size(g) == (17, 9, 5)
        @test grid_points == points(positional)
        @test fft_storage_dims(g) == CHANNEL_2D3C_FFT_ORDER
        @test inhomogeneous_storage_dims(g) == CHANNEL_2D3C_INHOMOGENEOUS_DIMS
        @test length(grid_points) == 3
        @test map(size, grid_points) == ((17, 1, 1), (1, 9, 1), (1, 1, 5))
        @test vec(grid_points[1]) ≈ y
        @test vec(grid_points[2]) ≈ range(0, 2π / 0.5 - 2π / (0.5 * 9); length=9)
        @test vec(grid_points[3]) ≈ range(0, 1 - 1 / 5; length=5)
        @test wavenumber_scale(g, 1) == 1.0
        @test wavenumber_scale(g, 2) == 0.5
        @test wavenumber_scale(g, 3) == 2π
        @test rpcf_base(g) ≈ y

        @test_throws ArgumentError ChannelGrid(17, 2; Nt=5, β=0.5, width=3)
        @test_throws ArgumentError ChannelGrid(17, 9; Nt=2, β=0.5, width=3)
    end

    @testset verbose=true "Wall-normal finite-difference contract                      " begin
        g = ChannelGrid(32, 1; Nt=1, β=2π, dist=FDGrids.GaussLobattoGrid(), width=9)
        @test g isa AbstractChannel2D3CGrid
        @test size(g) == (32, 1, 1)
        @test fft_storage_dims(g) == CHANNEL_2D3C_FFT_ORDER
        @test inhomogeneous_storage_dims(g) == CHANNEL_2D3C_INHOMOGENEOUS_DIMS
        test_one_inhomogeneous_fd_contract(g, (63, 3))
    end

    @testset verbose=true "Flow constructors                                           " begin
        g = ChannelGrid(17, 9; Nt=5, β=0.5, width=3)
        no_rotation = PlaneCouetteFlow(g, 500; fftw_flags=FFTW.ESTIMATE, dealias=false)
        with_rotation = PlaneCouetteFlow(g, 500; Ro=0.2, fftw_flags=FFTW.ESTIMATE, dealias=false)
        @test no_rotation isa ProjectedNSE
        @test no_rotation.nl.force isa NoForce
        @test with_rotation isa ProjectedNSE
        @test with_rotation.nl.force isa CoriolisForce
        @test with_rotation.nl.force.components == (3, 1)
        @test with_rotation.base == (nothing, nothing, rpcf_base(g))
    end

    @testset verbose=true "Physical and spectral norms                                 " begin
        g = ChannelGrid(17, 9; Nt=9, β=π, dist=FDGrids.GaussLobattoGrid(), width=3)
        plans = FFTPlans(g; flags=FFTW.ESTIMATE, dealias=false)
        wall_normal_norm = sum(g.ws[1] .* (1 .- g.xs[1] .^ 2) .^ 2)

        y_only, _ = case_physical_to_spectral(g, plans, (Y, Z, _, T) -> @. (1 - Y^2) + 0 * Z + 0 * T)
        @test dot(y_only, y_only) ≈ wall_normal_norm rtol=1e-12
        @test norm(y_only) ≈ sqrt(wall_normal_norm) rtol=1e-12

        yz, _ = case_physical_to_spectral(g, plans, (Y, Z, _, T) -> @. (1 - Y^2) * cos(π * Z) + 0 * T)
        @test dot(yz, yz) ≈ wall_normal_norm / 2 rtol=1e-10
        @test norm(yz) ≈ sqrt(wall_normal_norm / 2) rtol=1e-10

        yzt, physical = case_physical_to_spectral(g, plans, (Y, Z, _, T) -> @. (1 - Y^2) * cos(π * Z) * cos(2π * T))
        Ny, Nz, Nt = size(g)
        physical_norm = sum(g.ws[1][j] * physical[j, k, n]^2 for j in 1:Ny, k in 1:Nz, n in 1:Nt) / (Nz * Nt)
        @test physical_norm ≈ wall_normal_norm / 4 rtol=1e-10
        @test dot(yzt, yzt) ≈ physical_norm rtol=1e-10
        @test norm(yzt) ≈ sqrt(wall_normal_norm / 4) rtol=1e-10
    end
end
