# Contract tests for field-aware broadcasting in `src/broadcasting.jl`.
# Scalar wrappers retain their metadata, while VectorField expressions are
# unpacked and evaluated independently for every component.

_field_broadcast_axpy!(dest, x, y) = (@. dest = 2x + y / 3)
_field_broadcast_zero!(dest) = (dest .= 0; dest)

@testset verbose=true "Field-aware broadcasting                                    " begin
    @testset verbose=true "Physical and spectral broadcasts preserve scalar wrappers   " begin
        g = line_grid(Nb=7, Nh=7)
        f = Field(g, (a, b) -> a + cos(b))
        h = Field(g, (a, b) -> 2a - sin(b))
        out = @. 2f - h / 3

        @test out isa Field
        @test grid(out) === g
        @test parent(out) ≈ 2 .* parent(f) .- parent(h) ./ 3

        scalar_first = @. 1 + 0.5f
        @test scalar_first isa Field
        @test grid(scalar_first) === g
        @test parent(scalar_first) ≈ 1 .+ 0.5 .* parent(f)

        u = test_ftfield(g; seed=51)
        v = test_ftfield(g; seed=52)
        w = @. u + 0.25v
        @test w isa FTField
        @test grid(w) === g
        @test parent(w) ≈ parent(u) .+ 0.25 .* parent(v)
    end

    @testset verbose=true "Projected broadcasts retain the template grid and basis     " begin
        g = line_grid(Nb=7, Nh=7)
        basis = test_modes(g; ncomponents=2, nmodes=3, seed=53)
        rng = MersenneTwister(54)
        a = ProjectedField(g, randn(rng, ComplexF64, 3, 4), basis)
        b = ProjectedField(g, randn(rng, ComplexF64, 3, 4), basis)
        c = @. 3a - b / 2

        @test c isa ProjectedField
        @test grid(c) === g
        @test modes(c) === basis
        @test parent(c) ≈ 3 .* parent(a) .- parent(b) ./ 2
    end

    @testset verbose=true "VectorField broadcasts operate component by component       " begin
        g = line_grid(Nb=7, Nh=7)
        u = VectorField(test_ftfield(g; seed=55), test_ftfield(g; seed=56), test_ftfield(g; seed=57))
        v = VectorField(test_ftfield(g; seed=58), test_ftfield(g; seed=59), test_ftfield(g; seed=60))
        w = @. 3u - v

        @test w isa VectorField{3, <:FTField}
        for n in eachindex(w)
            @test grid(w[n]) === g
            @test parent(w[n]) ≈ 3 .* parent(u[n]) .- parent(v[n])
        end

        destination = zero(u)
        @test _field_broadcast_axpy!(destination, u, v) === destination
        for n in eachindex(destination)
            @test parent(destination[n]) ≈ 2 .* parent(u[n]) .+ parent(v[n]) ./ 3
        end
    end

    @testset verbose=true "Scalar assignment fills every nested component              " begin
        g = line_grid(Nb=7, Nh=7)
        f = Field(g)
        u = FTField(g)
        q = VectorField(g; N=3)
        basis = test_modes(g; ncomponents=2, nmodes=3, seed=61)
        a = ProjectedField(g, basis)

        @test fill!(f, 2.5) === f
        @test all(==(2.5), f)
        @test fill!(u, -1) === u
        @test all(==(-1 + 0im), u)
        @test fill!(a, 4) === a
        @test all(==(4 + 0im), a)

        @test _field_broadcast_zero!(q) === q
        @test all(all(iszero, component) for component in q)
        q .= 2.5
        @test all(all(==(2.5 + 0im), component) for component in q)
    end

    @testset verbose=true "Warmed in-place broadcasts allocate no heap storage         " begin
        g = line_grid(Nb=7, Nh=7)
        physical = ntuple(seed -> Field(g, (a, b) -> seed * a + b), 3)
        spectral = ntuple(seed -> test_ftfield(g; seed=60 + seed), 3)
        vectors = ntuple(seed -> VectorField(test_ftfield(g; seed=70 + 2seed), test_ftfield(g; seed=71 + 2seed)), 3)
        basis = test_modes(g; ncomponents=2, nmodes=3, seed=80)
        rng = MersenneTwister(81)
        projected = ntuple(_ -> ProjectedField(g, randn(rng, ComplexF64, 3, 4), basis), 3)

        for fields in (physical, spectral, vectors, projected)
            _field_broadcast_axpy!(fields...)
            _field_broadcast_zero!(first(fields))
            @test (@allocated _field_broadcast_axpy!(fields...)) == 0
            @test (@allocated _field_broadcast_zero!(first(fields))) == 0
        end
    end
end
