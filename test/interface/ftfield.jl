# Contract tests for the Fourier-space scalar wrapper in `src/ftfield.jl`.
# `steady_channel_grid` supplies one bounded, one rfft, and one signed-FFT
# direction, which is the smallest production layout covering every invariant.

@testset verbose=true "FTField construction and spectral array contract            " begin
    @testset verbose=true "Constructors own storage and enforce real-field invariants  " begin
        g = steady_channel_grid(Nx=7, Ny=7, Nz=5)
        rng = MersenneTwister(11)
        raw = randn(rng, ComplexF64, transform_size(g))
        u = FTField(g, raw)

        @test grid(u) === g
        @test parent(u) === raw
        @test size(u) == transform_size(g) == (7, 4, 5)
        @test eltype(u) === ComplexF64
        @test all(iszero, imag.(parent(u)[:, 1, 1]))
        for j in axes(u, 1), kz in 1:(size(u, 3) >> 1)
            @test parent(u)[j, 1, kz + 1] ≈ conj(parent(u)[j, 1, end - kz + 1])
        end

        g32 = steady_channel_grid(Nx=7, Ny=7, Nz=5, T=Float32)
        input64 = randn(rng, ComplexF64, transform_size(g32))
        u32 = FTField(g32, input64)
        @test parent(u32) !== input64
        @test eltype(u32) === ComplexF32
        @test eltype(parent(u32)) === ComplexF32

        z = FTField(g)
        @test size(z) == transform_size(g)
        @test all(iszero, z)
    end

    @testset verbose=true "Linear and Cartesian indexing expose raw coefficients       " begin
        g = steady_channel_grid(Nx=7, Ny=7, Nz=5)
        u = FTField(g)

        @test Base.IndexStyle(typeof(u)) isa IndexLinear
        @test parent(u) === u.data
        @test axes(u) == axes(parent(u))

        @test setindex!(u, 2 + 3im, 4) == 2 + 3im
        @test u[4] == parent(u)[4] == 2 + 3im
        I = CartesianIndex(2, 3, 4)
        @test setindex!(u, -1 + 5im, I) == -1 + 5im
        @test u[I] == parent(u)[I] == -1 + 5im
        @test_throws BoundsError u[length(u) + 1]
    end

    @testset verbose=true "Copy zero and similar preserve spectral wrapper semantics   " begin
        g = steady_channel_grid(Nx=7, Ny=7, Nz=5)
        u = test_ftfield(g; seed=12)

        v = copy(u)
        @test v isa typeof(u)
        @test v !== u
        @test grid(v) === g
        @test parent(v) !== parent(u)
        @test parent(v) == parent(u)

        z = zero(u)
        @test z isa typeof(u)
        @test grid(z) === g
        @test all(iszero, z)

        s = similar(u)
        @test s isa typeof(u)
        @test grid(s) === g
        @test size(s) == size(u)
        @test all(iszero, s)

        s32 = similar(u, ComplexF32)
        @test eltype(s32) === ComplexF32
        @test eltype(parent(s32)) === ComplexF32
        @test eltype(grid(s32)) === Float32
        @test size(s32) == size(u)
    end

    @testset verbose=true "Symmetry repair changes only the zero-rfft plane            " begin
        g = steady_channel_grid(Nx=7, Ny=7, Nz=5)
        rng = MersenneTwister(13)
        u = FTField(g)
        parent(u) .= randn(rng, ComplexF64, size(u))
        before = copy(parent(u))

        @test NSEBase.apply_symmetry!(u) === u
        for j in axes(u, 1), kz in 1:(size(u, 3) >> 1)
            @test parent(u)[j, 1, kz + 1] ≈ conj(parent(u)[j, 1, end - kz + 1])
        end
        @test parent(u)[:, 2:end, :] == before[:, 2:end, :]

        repaired = copy(parent(u))
        NSEBase.apply_symmetry!(u)
        @test parent(u) == repaired
    end

    @testset verbose=true "Wavenumber indexing reads and maintains logical symmetry    " begin
        g = steady_channel_grid(Nx=7, Ny=7, Nz=5)
        u = FTField(g)

        @test setindex!(u, 3 + 4im, WaveNumberVector(0, 1), 2) == 3 + 4im
        @test u[WaveNumberVector(0, 1), 2] == 3 + 4im
        @test u[WaveNumberVector(0, -1), 2] == 3 - 4im

        @test setindex!(u, 2 + 9im, WaveNumberVector(0, 0), 1) == 2 + 0im
        @test u[WaveNumberVector(0, 0), 1] == 2 + 0im

        @test setindex!(u, 5 - 7im, WaveNumberVector(-2, 1), 1) == 5 - 7im
        @test u[WaveNumberVector(-2, 1), 1] == 5 - 7im
        @test u[WaveNumberVector(2, -1), 1] == 5 + 7im

        u[WaveNumberVector(2, -1), 4] = 6 + 8im
        raw_negative_view = u[WaveNumberVector(-2, 1)]
        @test raw_negative_view isa SubArray
        @test raw_negative_view[4] == 6 + 8im
        @test u[WaveNumberVector(-2, 1), 4] == 6 - 8im

        positive_view = u[WaveNumberVector(1, 0)]
        positive_view[3] = -2 + 3im
        @test u[WaveNumberVector(1, 0), 3] == -2 + 3im
    end

    @testset verbose=true "Spectral axis helpers partition the production layout       " begin
        g = steady_channel_grid(Nx=7, Ny=7, Nz=5)
        u = FTField(g)
        pu = parent(u)

        @test NSEBase.homogeneous_axes(u) == (axes(pu, 2), axes(pu, 3))
        @test NSEBase.inhomogeneous_axes(u) == (axes(pu, 1),)
        @test NSEBase.homogeneous_axes(pu, Val((2, 3))) == (axes(pu, 2), axes(pu, 3))
        @test NSEBase.inhomogeneous_axes(pu, Val((2, 3))) == (axes(pu, 1),)
        @test length(CartesianIndices(NSEBase.homogeneous_axes(u))) == size(u, 2) * size(u, 3)
    end

    @testset verbose=true "growto embeds coefficients at identical wavenumbers         " begin
        Ny, Nx, Nz = 7, 7, 5
        g = steady_channel_grid(Nx=Nx, Ny=Ny, Nz=Nz)
        u = FTField(g)
        coeff(k, j) = complex(100j + 10k[1] + abs(k[2]), k[1] == 0 ? 3k[2] : 7k[1] + 3k[2])

        for Ih in CartesianIndices(NSEBase.homogeneous_axes(u)), j in 1:Ny
            k = NSEBase.to_wavenumber_vector(g, Ih)
            u[k, j] = coeff(k, j)
        end

        v = growto(u, (11, 9))
        @test size(grid(v)) == (Ny, 11, 9)
        @test size(v) == (Ny, 6, 9)

        for Ih in CartesianIndices(NSEBase.homogeneous_axes(u)), j in 1:Ny
            k = NSEBase.to_wavenumber_vector(g, Ih)
            @test u[k, j] == coeff(k, j)
            @test v[k, j] == u[k, j]
        end

        for Ih in CartesianIndices(NSEBase.homogeneous_axes(v)), j in 1:Ny
            k = NSEBase.to_wavenumber_vector(grid(v), Ih)
            in_source = 0 <= k[1] <= (Nx >> 1) && abs(k[2]) <= (Nz >> 1)
            @test in_source ? v[k, j] == u[k, j] : iszero(v[k, j])
        end

        @test_throws ArgumentError growto(u, (11,))
    end
end
