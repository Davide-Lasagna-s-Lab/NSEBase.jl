# FarazmandWeight is an NSEBase coefficient-space metric. The tests below use a
# production grid only to supply ordered wavenumber scales.

@testset verbose=true "FarazmandWeight                                             " begin
    @testset verbose=true "Construction and evaluation                                 " begin
        g = steady_channel_grid(α=1.5, β=2.0)
        from_grid = FarazmandWeight(g)
        explicit = FarazmandWeight(2π, 4π)
        promoted = FarazmandWeight(1, 2.0, 3)

        @test from_grid.scales === (1.5, 2.0)
        @test explicit.scales === (2π, 4π)
        @test promoted isa FarazmandWeight{3, Float64}
        @test promoted.scales === (1.0, 2.0, 3.0)

        A = FarazmandWeight(2.0, 3.0)
        @test A[WaveNumberVector(0, 0)] == 1
        @test A[WaveNumberVector(1, 0)] ≈ 1 / 5
        @test A[WaveNumberVector(0, 1)] ≈ 1 / 10
        @test A[WaveNumberVector(2, 3)] ≈ 1 / (1 + (2 * 2)^2 + (3 * 3)^2)
        @test A[WaveNumberVector(-1, 2)] ≈ A[WaveNumberVector(1, -2)]
        @test A[WaveNumberVector(1, -2)] ≈ A[WaveNumberVector(-1, -2)]
    end

    @testset verbose=true "In-place coefficient weighting                              " begin
        g = steady_channel_grid(Nx=9, Nz=7, α=2π, β=2π)
        modes = test_modes(g; ncomponents=3, nmodes=5)
        a = ProjectedField(g, randn(MersenneTwister(1), ComplexF64, 5, 5, 7), modes)
        before = copy(parent(a))
        A = FarazmandWeight(g)

        @test lmul!(A, a) === a

        expected = copy(before)
        for Ih in CartesianIndices(NSEBase.homogeneous_axes(a))
            weight = A[to_wavenumber_vector(g, Ih)]
            for mode in axes(a, 1)
                expected[mode, Tuple(Ih)...] *= weight
            end
        end
        @test parent(a) ≈ expected
    end

    @testset verbose=true "Weighted inner product                                      " begin
        g = steady_channel_grid(Nx=9, Nz=7, α=2π, β=2π)
        modes = test_modes(g; ncomponents=3, nmodes=4)
        a = ProjectedField(g, randn(MersenneTwister(2), ComplexF64, 4, 5, 7), modes)
        b = ProjectedField(g, randn(MersenneTwister(3), ComplexF64, 4, 5, 7), modes)
        A = FarazmandWeight(g)

        @test dot(a, A, b) ≈ dot(a, lmul!(A, copy(b)))
    end
end
