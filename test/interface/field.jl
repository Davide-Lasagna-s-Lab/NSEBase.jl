# Contract tests for the physical-space scalar wrapper in `src/field.jl`.
# Production rectangular grids provide both the ordinary and dealiased point
# layouts, so these tests exercise the public behavior without a test grid.

@testset verbose=true "Field construction and array contract                       " begin
    @testset verbose=true "Function constructors follow storage-order points           " begin
        g = line_grid(Nb=9, Nh=7)
        p1, p2 = points(g)
        fun(a, b) = 1 + 2a - cos(b)

        u = Field(g, fun)
        @test grid(u) === g
        @test size(u) == size(g) == (9, 7)
        @test parent(u) == fun.(p1, p2)

        p1d, p2d = points(g; dealias=true)
        ud = Field(g, fun; dealias=true)
        @test size(ud) == (9, 11)
        @test parent(ud) == fun.(p1d, p2d)

        z = Field(g)
        zd = Field(g; dealias=true)
        @test size(z) == (9, 7)
        @test size(zd) == (9, 11)
        @test all(iszero, z)
        @test all(iszero, zd)
    end

    @testset verbose=true "Data constructors preserve or convert parent storage        " begin
        g = line_grid(Nb=7, Nh=9)
        data = reshape(collect(1.0:63.0), size(g))
        u = Field(g, data)

        @test parent(u) === data
        @test eltype(u) === Float64
        @test size(u) == size(data)
        data[2, 3] = -4
        @test u[2, 3] == -4

        integers = reshape(1:12, 3, 4)
        converted = Field(g, integers)
        @test parent(converted) !== integers
        @test eltype(converted) === Float64
        @test parent(converted) == Float64.(integers)
        @test size(converted) == (3, 4)
    end

    @testset verbose=true "AbstractArray indexing delegates to the exact parent        " begin
        g = line_grid(Nb=7, Nh=7)
        u = Field(g)

        @test Base.IndexStyle(typeof(u)) isa IndexLinear
        @test parent(u) === u.data
        @test axes(u) == axes(parent(u))

        @test setindex!(u, 2, 3) == 2
        @test u[3] == parent(u)[3] == 2
        I = CartesianIndex(2, 4)
        @test setindex!(u, 5, I) == 5
        @test u[I] == parent(u)[I] == 5
        @test_throws BoundsError u[length(u) + 1]
    end

    @testset verbose=true "Copy zero and similar preserve wrapper semantics            " begin
        g = line_grid(Nb=7, Nh=7)
        u = Field(g, (a, b) -> a + sin(b))

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

        s32 = similar(u, Float32)
        @test eltype(s32) === Float32
        @test eltype(grid(s32)) === Float32
        @test size(s32) == size(u)
        @test all(iszero, s32)
    end
end
