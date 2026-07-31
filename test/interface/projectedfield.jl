# Contract tests for modal coefficient storage in `src/projectedfield.jl`.
# The production steady-channel fixture has the existing supported layout:
# one bounded direction and two Fourier directions in canonical storage order.

@testset verbose=true "ProjectedField construction and modal array contract        " begin
    @testset verbose=true "Allocated storage places the mode axis before Fourier axes  " begin
        g = steady_channel_grid(Nx=7, Ny=7, Nz=5)
        basis = test_modes(g; ncomponents=2, nmodes=3, seed=31)
        a = ProjectedField(g, basis)

        @test grid(a) === g
        @test modes(a) === basis
        @test size(a) == (3, 4, 5)
        @test eltype(a) === ComplexF64
        @test all(iszero, a)

        from_ft = ProjectedField(FTField(g), basis)
        from_field = ProjectedField(Field(g), basis)
        from_vector = ProjectedField(VectorField(g; N=2), basis)
        @test size(from_ft) == size(from_field) == size(from_vector) == size(a)
        @test grid(from_ft) === grid(from_field) === grid(from_vector) === g
        @test modes(from_ft) === modes(from_field) === modes(from_vector) === basis

        single_basis = first(basis)
        single = ProjectedField(g, single_basis)
        @test modes(single) isa Tuple{typeof(single_basis)}
        @test only(modes(single)) === single_basis
        @test size(single) == size(a)
    end

    @testset verbose=true "Plain arrays are sanitized and copied by construction       " begin
        g = steady_channel_grid(Nx=7, Ny=7, Nz=5)
        basis = test_modes(g; ncomponents=2, nmodes=3, seed=32)
        rng = MersenneTwister(33)
        raw = randn(rng, ComplexF64, 3, 4, 5)
        a = ProjectedField(g, raw, basis)

        @test parent(a) !== raw
        @test parent(a) == raw
        @test all(iszero, imag.(parent(a)[:, 1, 1]))
        for m in axes(a, 1), kz in 1:(size(a, 3) >> 1)
            @test parent(a)[m, 1, kz + 1] ≈ conj(parent(a)[m, 1, end - kz + 1])
        end
    end

    @testset verbose=true "Linear Cartesian and storage indices expose the parent      " begin
        g = steady_channel_grid(Nx=7, Ny=7, Nz=5)
        a = ProjectedField(g, test_modes(g; ncomponents=2, nmodes=3, seed=34))

        @test Base.IndexStyle(typeof(a)) isa IndexLinear
        @test parent(a) === a.data
        @test axes(a) == axes(parent(a))

        @test setindex!(a, 5 + 6im, 3) == 5 + 6im
        @test a[3] == parent(a)[3] == 5 + 6im
        @test setindex!(a, 3 + 1im, 1, 2, 3) == 3 + 1im
        @test a[1, 2, 3] == parent(a)[1, 2, 3] == 3 + 1im

        I = CartesianIndex(2, 1, 2)
        @test setindex!(a, 7 + 4im, I) == 7 + 4im
        @test a[I] == parent(a)[I] == 7 + 4im
        @test all(a[J] == parent(a)[J] for J in CartesianIndices(a))
        @test_throws BoundsError a[length(a) + 1]
    end

    @testset verbose=true "Copy zero similar and abs preserve modal metadata           " begin
        g = steady_channel_grid(Nx=7, Ny=7, Nz=5)
        basis = test_modes(g; ncomponents=2, nmodes=3, seed=35)
        a = ProjectedField(g, basis)
        a[1, 2, 1] = 1 + 2im

        copied = copy(a)
        @test copied isa typeof(a)
        @test grid(copied) === g
        @test modes(copied) === basis
        @test parent(copied) !== parent(a)
        @test parent(copied) == parent(a)

        z = zero(a)
        @test z isa typeof(a)
        @test grid(z) === g
        @test modes(z) === basis
        @test all(iszero, z)

        s = similar(a)
        @test s isa typeof(a)
        @test grid(s) === g
        @test modes(s) === basis
        @test size(s) == size(a)
        @test all(iszero, s)

        magnitude = abs(a)
        @test magnitude isa typeof(a)
        @test grid(magnitude) === g
        @test modes(magnitude) === basis
        @test parent(magnitude) == complex.(abs.(parent(a)))
        @test parent(magnitude) !== parent(a)
    end

    @testset verbose=true "Wavenumber indexing maintains modal coefficient symmetry    " begin
        g = steady_channel_grid(Nx=7, Ny=7, Nz=5)
        a = ProjectedField(g, test_modes(g; ncomponents=2, nmodes=3, seed=36))

        @test setindex!(a, 1 + 2im, 2, WaveNumberVector(0, 1)) == 1 + 2im
        @test a[2, WaveNumberVector(0, 1)] == 1 + 2im
        @test a[2, WaveNumberVector(0, -1)] == 1 - 2im

        @test setindex!(a, 4 + 8im, 1, WaveNumberVector(0, 0)) == 4 + 0im
        @test a[1, WaveNumberVector(0, 0)] == 4 + 0im

        @test setindex!(a, -3 + 5im, 3, WaveNumberVector(-2, 1)) == -3 + 5im
        @test a[3, WaveNumberVector(-2, 1)] == -3 + 5im
        @test a[3, WaveNumberVector(2, -1)] == -3 - 5im

        parent(a)[1, 3, 1] = 7 + 3im
        @test a[1, WaveNumberVector(2, 0)] == 7 + 3im
        @test a[1, WaveNumberVector(-2, 0)] == 7 - 3im
    end

    @testset verbose=true "Symmetry repair treats every mode and only the DC plane     " begin
        g = steady_channel_grid(Nx=7, Ny=7, Nz=5)
        a = ProjectedField(g, test_modes(g; ncomponents=2, nmodes=4, seed=37))
        rng = MersenneTwister(38)
        parent(a) .= randn(rng, ComplexF64, size(a))
        before = copy(parent(a))

        @test NSEBase.apply_symmetry!(a) === a
        for m in axes(a, 1), kz in 1:(size(a, 3) >> 1)
            @test parent(a)[m, 1, kz + 1] ≈ conj(parent(a)[m, 1, end - kz + 1])
        end
        @test parent(a)[:, 2:end, :] == before[:, 2:end, :]

        repaired = copy(parent(a))
        NSEBase.apply_symmetry!(a)
        @test parent(a) == repaired
    end

    @testset verbose=true "Homogeneous axes omit the mode and bounded directions       " begin
        g = steady_channel_grid(Nx=7, Ny=7, Nz=5)
        a = ProjectedField(g, test_modes(g; ncomponents=2, nmodes=3, seed=39))

        @test NSEBase.homogeneous_axes(a) == (axes(parent(a), 2), axes(parent(a), 3))
        @test length(CartesianIndices(NSEBase.homogeneous_axes(a))) == 20
    end
end
