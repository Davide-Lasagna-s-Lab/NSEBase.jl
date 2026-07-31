# Contract tests for the ordered scalar-field container in `src/vectorfield.jl`.
# The tests distinguish component-container behavior from scalar-field behavior
# and exercise base profiles with one and several bounded directions.

@testset verbose=true "VectorField construction and component contract             " begin
    @testset verbose=true "Direct and allocating constructors preserve component order " begin
        g = line_grid(Nb=9, Nh=7)
        f1 = Field(g, (a, b) -> a + b)
        f2 = Field(g, (a, b) -> a - b)
        q = VectorField(f1, f2)

        @test q isa VectorField{2, <:Field}
        @test parent(q) == (f1, f2)
        @test q[1] === f1
        @test q[2] === f2
        @test grid(q) === g
        @test size(q) == (2,)
        @test length(q) == 2
        @test eltype(q) === typeof(f1)
        @test Base.IndexStyle(typeof(q)) isa IndexLinear

        spectral = VectorField(g)
        @test spectral isa VectorField{3, <:FTField}
        @test all(grid(component) === g for component in spectral)
        @test all(size(component) == transform_size(g) for component in spectral)
        @test all(all(iszero, component) for component in spectral)
        @test parent(spectral[1]) !== parent(spectral[2])

        physical = VectorField(g, Field; N=4)
        @test physical isa VectorField{4, <:Field}
        @test all(size(component) == size(g) for component in physical)
        @test all(all(iszero, component) for component in physical)
    end

    @testset verbose=true "Function constructors forward the dealiasing layout         " begin
        g = line_grid(Nb=9, Nh=7)
        f1(a, b) = a^2 + cos(b)
        f2(a, b) = exp(a) - sin(b)

        q = VectorField(g, f1, f2)
        qd = VectorField(g, f1, f2; dealias=true)
        p1, p2 = points(g)
        p1d, p2d = points(g; dealias=true)

        @test q isa VectorField{2, <:Field}
        @test qd isa VectorField{2, <:Field}
        @test parent(q[1]) == f1.(p1, p2)
        @test parent(q[2]) == f2.(p1, p2)
        @test parent(qd[1]) == f1.(p1d, p2d)
        @test parent(qd[2]) == f2.(p1d, p2d)
        @test size(q[1]) == (9, 7)
        @test size(qd[1]) == (9, 11)
    end

    @testset verbose=true "Component assignment copies values without replacing fields " begin
        g = line_grid(Nb=7, Nh=7)
        q = VectorField(g, Field; N=2)
        destination = q[1]
        source = Field(g, (a, b) -> 2a - b)

        @test setindex!(q, source, 1) === source
        @test q[1] === destination
        @test parent(q[1]) == parent(source)
        parent(source)[1] = 100
        @test parent(q[1])[1] != parent(source)[1]
        @test_throws BoundsError q[3]
    end

    @testset verbose=true "Copy zero and similar act independently per component       " begin
        g = line_grid(Nb=7, Nh=7)
        q = VectorField(g, (a, b) -> a + b, (a, b) -> a - b)

        copied = copy(q)
        @test copied isa typeof(q)
        @test all(grid(copied[n]) === g for n in eachindex(q))
        @test all(parent(copied[n]) == parent(q[n]) for n in eachindex(q))
        @test all(parent(copied[n]) !== parent(q[n]) for n in eachindex(q))

        z = zero(q)
        @test z isa typeof(q)
        @test all(all(iszero, component) for component in z)

        s = similar(q)
        @test s isa typeof(q)
        @test all(all(iszero, component) for component in s)
        @test all(parent(s[n]) !== parent(s[m]) for n in eachindex(s) for m in eachindex(s) if n != m)

        s32 = similar(q, Float32)
        @test s32 isa VectorField{2, <:Field}
        @test all(eltype(component) === Float32 for component in s32)
        @test all(eltype(grid(component)) === Float32 for component in s32)
    end

    @testset verbose=true "Base flow addition selects only homogeneous zero modes      " begin
        g = steady_channel_grid(Nx=7, Ny=7, Nz=5)
        u = VectorField(g; N=3)
        U = collect(range(-1, 1, length=size(g, 1)))
        W = collect(range(2, 3, length=size(g, 1)))

        @test add_base_flow!(u, (U, nothing, W)) === u
        expected_u = zeros(ComplexF64, size(u[1]))
        expected_w = zeros(ComplexF64, size(u[3]))
        expected_u[:, 1, 1] .= U
        expected_w[:, 1, 1] .= W
        @test parent(u[1]) == expected_u
        @test all(iszero, u[2])
        @test parent(u[3]) == expected_w

        gd = square_duct_grid(N=7, Nz=5, Nt=3)
        q = VectorField(gd; N=2)
        profile = [i + 2j for i in axes(q[1], 1), j in axes(q[1], 2)]
        expected = zeros(ComplexF64, size(q[1]))
        expected[:, :, 1, 1] .= profile
        add_base_flow!(q, (profile, nothing))
        @test parent(q[1]) == expected
        @test all(iszero, q[2])
    end

    @testset verbose=true "growto changes each component homogeneous resolution        " begin
        g = line_grid(Nb=7, Nh=7)
        q = VectorField(test_ftfield(g; seed=21), test_ftfield(g; seed=22))
        grown = growto(q, (11,))

        @test grown isa VectorField{2, <:FTField}
        @test size(grid(grown)) == (7, 11)
        @test all(size(component) == (7, 6) for component in grown)
        for n in eachindex(q), j in axes(q[n], 1), k in 0:(size(g, 2) >> 1)
            wavenumber = WaveNumberVector(k)
            @test grown[n][wavenumber, j] == q[n][wavenumber, j]
        end
    end
end
