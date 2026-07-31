# Tests for the JLD2 persistence helpers in `src/io.jl`.
#
# Contract:
#   - `save_grid(g)` writes `g` to disk so that `load_grid(path)` returns an
#     object equal in type and contents to `g`.
#   - `save_field(a)` persists the *coefficient array* of a ProjectedField,
#     not its modes or its grid; `load_field(g, modes, path)` rebuilds the
#     ProjectedField from those three components.

@testset verbose=true "Persistence helpers                                         " begin
    # A unique temporary directory prevents concurrent test processes from
    # racing on the default JLD2 file names and is removed automatically.
    mktempdir() do tmpdir
        @testset verbose=true "Grid round trip                                             " begin
            g = line_grid(Nb=9, Nh=7, scale=3)
            path = joinpath(tmpdir, "grid.jld2")

            @test save_grid(g; path) === nothing
            loaded = load_grid(path)

            @test typeof(loaded) === typeof(g)
            @test all(name -> isequal(getproperty(loaded, name), getproperty(g, name)), propertynames(g))
        end

        @testset verbose=true "Projected-field round trip                                  " begin
            g = line_grid(Nb=9, Nh=7)
            basis = test_modes(g; ncomponents=3, nmodes=4, seed=31)
            coefficients = randn(MersenneTwister(32), ComplexF64, 4, 4)
            a = ProjectedField(g, coefficients, basis)
            path = joinpath(tmpdir, "field.jld2")

            @test save_field(a; path) === nothing
            loaded = load_field(g, basis, path)

            @test loaded isa ProjectedField
            @test parent(loaded) == parent(a)
            @test modes(loaded) === basis
            @test grid(loaded) === g
        end
    end
end
