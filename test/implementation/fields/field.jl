# Tests for src/field.jl.
#
# Contract:
#   - `Field(grid, fun)` calls `fun(x,y,z,t)` and passes `nothing` when absent.
#   - `Field(grid)` allocates the zero field on the physical grid.
#   - `copy`, `zero`, and `similar` preserve the field wrapper and grid.

@testset verbose=true "Field contract                                              " begin
    Ny, Nx = 7, 9
    g = shear_test_grid(Ny, Nx)

    @testset verbose=true "function constructor uses physical coordinate order         " begin
        ypts, xpts = points(g)
        fun(x, y, z, t) = isnothing(z) && isnothing(t) ? 1 + 2x - cos(y) : NaN
        u = Field(g, fun)

        @test grid(u) === g
        @test size(u) == size(g)
        @test parent(u) ≈ fun.(xpts, ypts, nothing, nothing)
    end

    @testset verbose=true "zero, copy, and similar preserve wrapper semantics          " begin
        u = Field(g, (x, y, _, _) -> x + sin(y))

        @test Field(g) isa Field
        @test all(iszero, parent(Field(g)))

        v = copy(u)
        @test v isa Field
        @test v !== u
        @test grid(v) === g
        @test parent(v) == parent(u)

        z = zero(u)
        @test z isa Field
        @test grid(z) === g
        @test all(iszero, parent(z))

        s = similar(u)
        @test s isa Field
        @test grid(s) === g
        @test size(s) == size(u)
    end
end
