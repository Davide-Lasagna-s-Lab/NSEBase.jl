# Tests for src/projectedfield.jl.
#
# Contract:
#   - `ProjectedField(grid, modes)` stores modal coefficients as
#     `(mode, fft_dims...)`, independent of the physical grid storage order.
#   - Linear/storage indexing is ordinary array indexing.
#   - WaveNumberVector indexing maintains the same zero-mode and Hermitian
#     symmetry invariants as FTField.

@testset verbose=true "ProjectedField contract             " begin

    function _projected_modes(g, Nm, Ncomp=2)
        Ny, Nx, Nz = size(g)
        ntuple(_ -> randn(ComplexF64, Ny, Nm, (Nx >> 1) + 1, Nz), Ncomp)
    end

    @testset "constructor uses mode axis followed by FFT axes" begin
        Ny, Nx, Nz, Nm = 3, 8, 5, 4
        g = TripleGrid(Ny, Nx, Nz)
        Ψ = _projected_modes(g, Nm)

        a = ProjectedField(g, Ψ)
        @test grid(a) === g
        @test modes(a) === Ψ
        @test size(a) == (Nm, (Nx >> 1) + 1, Nz)
        @test all(iszero, parent(a))

        b = similar(a)
        @test b isa typeof(a)
        @test size(b) == size(a)
        @test modes(b) === Ψ
    end

    @testset "WaveNumberVector indexing preserves modal symmetry" begin
        Ny, Nx, Nz, Nm = 2, 8, 5, 3
        g = TripleGrid(Ny, Nx, Nz)
        a = ProjectedField(g, _projected_modes(g, Nm))

        a[2, WaveNumberVector(0, 1)] = 1 + 2im
        @test a[2, WaveNumberVector(0,  1)] == 1 + 2im
        @test a[2, WaveNumberVector(0, -1)] == 1 - 2im

        a[1, WaveNumberVector(0, 0)] = 4 + 8im
        @test a[1, WaveNumberVector(0, 0)] == 4 + 0im

        a[3, WaveNumberVector(-2, 1)] = -3 + 5im
        @test a[3, WaveNumberVector(-2,  1)] == -3 + 5im
        @test a[3, WaveNumberVector( 2, -1)] == -3 - 5im
    end
end
