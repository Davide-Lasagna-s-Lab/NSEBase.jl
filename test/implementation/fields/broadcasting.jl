# Tests for the broadcast integration in src/broadcasting.jl.
#
# Contract:
#   - Broadcasting scalar fields produces a scalar field of the same wrapper
#     type and grid.
#   - Broadcasting vector fields happens component-wise, not by treating the
#     vector field as a plain length-N array of field objects.
#   - Scalar assignment broadcasts into every component of a VectorField.

@testset verbose=true "Broadcasting                                                " begin

    @testset verbose=true "Field and FTField broadcasts preserve wrappers              " begin
        Nx, Ny = 7, 9
        g = planar_test_grid(Nx, Ny)

        f = Field(g, (x, y, _, _) -> x + cos(y))
        h = Field(g, (x, y, _, _) -> 2x - sin(y))
        out = @. 2f - h / 3

        @test out isa Field
        @test grid(out) === g
        @test parent(out) ≈ 2 .* parent(f) .- parent(h) ./ 3

        û = FTField(g, randn(ComplexF64, Nx, (Ny >> 1) + 1))
        v̂ = FTField(g, randn(ComplexF64, Nx, (Ny >> 1) + 1))
        ŵ = @. û + 0.25v̂

        @test ŵ isa FTField
        @test grid(ŵ) === g
        @test parent(ŵ) ≈ parent(û) .+ 0.25 .* parent(v̂)
    end

    @testset verbose=true "VectorField broadcasts are component-wise                   " begin
        Nx, Ny = 7, 9
        g = planar_test_grid(Nx, Ny)

        u = VectorField([FTField(g, randn(ComplexF64, Nx, (Ny >> 1) + 1)) for _ in 1:3]...)
        v = VectorField([FTField(g, randn(ComplexF64, Nx, (Ny >> 1) + 1)) for _ in 1:3]...)

        w = @. 3u - v
        @test w isa VectorField{3, <:FTField}
        for n in 1:3
            @test grid(w[n]) === g
            @test parent(w[n]) ≈ 3 .* parent(u[n]) .- parent(v[n])
        end
    end

    @testset verbose=true "scalar broadcast assignment fills every vector component    " begin
        Nx, Ny = 7, 9
        g = planar_test_grid(Nx, Ny)
        u = VectorField(g; N=3)

        u .= 2.5
        for n in 1:3
            @test all(parent(u[n]) .== 2.5 + 0im)
        end
    end
end
