# Core inner-product tests exercise discrete algebra and container contracts.
# Continuous quadrature accuracy is intentionally tested by the concrete grid
# package, where the geometry and analytical reference functions belong.

@testset verbose=true "Inner products, norms, and distances                        " begin
    @testset verbose=true "FTField algebra                                             " begin
        g = line_grid()
        u = test_ftfield(g; seed=1)
        v = test_ftfield(g; seed=2)
        w = test_ftfield(g; seed=3)
        α, β = 0.3, -1.7
        αvβw = FTField(g, α .* parent(v) .+ β .* parent(w))

        @test dot(u, v) isa Real
        @test dot(u, v) ≈ dot(v, u)
        @test dot(u, u) ≥ 0
        @test dot(zero(u), zero(u)) == 0
        @test dot(u, αvβw) ≈ α * dot(u, v) + β * dot(u, w)
        @test norm(u)^2 ≈ dot(u, u)
    end

    @testset verbose=true "FTField distance identities                                 " begin
        g = line_grid()
        u = test_ftfield(g; seed=4)
        v = test_ftfield(g; seed=5)
        difference = FTField(g, parent(u) .- parent(v))

        @test normdiff(u, v) ≈ norm(difference)
        @test normdiff(u, v)^2 ≈ dot(u, u) + dot(v, v) - 2 * dot(u, v)
        @test normdiff(u, u) == 0
        @test normdiff(u, v, (0.0,)) == normdiff(u, v)
    end

    @testset verbose=true "VectorField component decomposition                         " begin
        g = line_grid()
        u = VectorField(ntuple(n -> test_ftfield(g; seed=n), 3)...)
        v = VectorField(ntuple(n -> test_ftfield(g; seed=n + 3), 3)...)

        @test dot(u, v) ≈ sum(dot(u[n], v[n]) for n in 1:3)
        @test norm(u)^2 ≈ dot(u, u)
        @test normdiff(u, u) == 0
        @test_throws DimensionMismatch normdiff(u, v, (0.0, 0.0))
        @test_throws DimensionMismatch normdiff(u, v, (0.0, 0.0), zero(v[1]))
    end

    @testset verbose=true "Shifted distances                                           " begin
        g = steady_channel_grid()
        u = test_ftfield(g; seed=7)
        v = test_ftfield(g; seed=8)
        shifts = (0.27, -0.41)
        workspace = zero(v)
        explicit = FTField(g, parent(u) .- parent(shift(v, shifts)))

        @test normdiff(u, v, shifts) ≈ norm(explicit)
        @test normdiff(u, v, shifts, workspace) ≈ norm(explicit)
        @test_throws DimensionMismatch normdiff(u, v, (0.0,))
        @test_throws DimensionMismatch normdiff(u, v, (0.0, 0.0, 0.0), workspace)
    end

    @testset verbose=true "Minimum shifted distance                                    " begin
        g = line_grid(scale=2)
        u = test_ftfield(g; seed=9)
        samples = (8,)
        step = 2π / (wavenumber_scale(g, 2) * only(samples))
        expected_shift = 3 * step
        v = shift(u, (-expected_shift,))
        v_before = copy(parent(v))
        workspace = zero(v)

        minimum_distance, minimizing_shift = minnormdiff(u, v, samples, workspace)
        @test minimum_distance ≈ 0 atol=1e-12
        @test only(minimizing_shift) ≈ expected_shift
        @test parent(v) == v_before

        default_distance, default_shift = minnormdiff(u, u)
        @test default_distance ≈ 0 atol=1e-14
        @test default_shift == (0.0,)
        @test_throws DimensionMismatch minnormdiff(u, v, (4, 4))
    end

    @testset verbose=true "ProjectedField discrete metric                              " begin
        g = steady_channel_grid()
        modes = test_modes(g; ncomponents=3, nmodes=4)
        a = ProjectedField(g, randn(MersenneTwister(10), ComplexF64, 4, 5, 7), modes)
        b = ProjectedField(g, randn(MersenneTwister(11), ComplexF64, 4, 5, 7), modes)

        reference = sum((i == 1 ? 1 : 2) * real(conj(parent(a)[m, i, j]) *
                                                  parent(b)[m, i, j])
                        for m in axes(a, 1), i in axes(a, 2), j in axes(a, 3))
        @test dot(a, b) ≈ reference
        @test dot(a, b) ≈ dot(b, a)
        @test norm(a)^2 ≈ dot(a, a)
        @test normdiff(a, a) == 0

        workspace = zero(b)
        shifts = (0.2, -0.3)
        @test normdiff(a, b, shifts, workspace) ≈ normdiff(a, shift(b, shifts))
        @test_throws DimensionMismatch normdiff(a, b, (0.0,), workspace)
    end
end
