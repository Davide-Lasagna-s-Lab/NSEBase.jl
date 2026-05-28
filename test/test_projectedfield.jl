# Tests for src/projectedfield.jl.
#
# Contract:
#   - `ProjectedField(grid, modes)` stores modal coefficients as
#     `(mode, fft_dims...)`, independent of the physical grid storage order.
#   - Linear, CartesianIndex, and storage indexing are ordinary array indexing
#     into the parent array, with no symmetry logic applied.
#   - WaveNumberVector indexing maintains the same zero-mode and Hermitian
#     symmetry invariants as FTField.

@testset verbose=true "ProjectedField contract             " begin

    function _projected_modes(g, Nm, Ncomp=2)
        Ny, Nx, Nz = size(g)
        ntuple(_ -> randn(ComplexF64, Nm, Ny, (Nx >> 1) + 1, Nz), Ncomp)
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

    @testset "linear indexing (a[i]) reads and writes the parent array" begin
        Ny, Nx, Nz, Nm = 2, 4, 3, 2
        g = TripleGrid(Ny, Nx, Nz)
        a = ProjectedField(g, _projected_modes(g, Nm))

        # write via setindex!, read back via getindex
        a[3] = 5 + 6im
        @test a[3] == 5 + 6im
        # parent is updated
        @test parent(a)[3] == 5 + 6im
        # write via parent, read back via a
        parent(a)[5] = 7 + 8im
        @test a[5] == 7 + 8im
    end

    @testset "storage indexing (a[m,i1,i2,...]) matches parent array" begin
        Ny, Nx, Nz, Nm = 2, 4, 3, 2
        g = TripleGrid(Ny, Nx, Nz)
        a = ProjectedField(g, _projected_modes(g, Nm))
        # parent shape: (Nm=2, Nx_half=3, Nz=3)

        a[1, 2, 3] = 3 + 1im
        @test a[1, 2, 3] == 3 + 1im
        @test parent(a)[1, 2, 3] == 3 + 1im

        # write via parent, read via storage index
        parent(a)[2, 1, 2] = 9 + 4im
        @test a[2, 1, 2] == 9 + 4im
    end

    @testset "CartesianIndex indexing matches storage indexing" begin
        Ny, Nx, Nz, Nm = 2, 4, 3, 2
        g = TripleGrid(Ny, Nx, Nz)
        a = ProjectedField(g, _projected_modes(g, Nm))

        a[1, 2, 3] = 3 + 1im
        @test a[CartesianIndex(1, 2, 3)] == a[1, 2, 3]

        # setindex! via CartesianIndex
        a[CartesianIndex(2, 1, 2)] = 5 + 4im
        @test a[2, 1, 2] == 5 + 4im
        @test parent(a)[CartesianIndex(2, 1, 2)] == 5 + 4im

        # CartesianIndices iteration roundtrip
        parent(a) .= reshape(1:length(a), size(a))
        for I in CartesianIndices(a)
            @test a[I] == parent(a)[I]
        end
    end

    @testset "WaveNumberVector indexing preserves modal symmetry" begin
        Ny, Nx, Nz, Nm = 2, 8, 5, 3
        g = TripleGrid(Ny, Nx, Nz)
        a = ProjectedField(g, _projected_modes(g, Nm))

        # Zero-rfft wavenumber: Hermitian symmetry — (0, kz) and (0, -kz) are conjugates
        a[2, WaveNumberVector(0, 1)] = 1 + 2im
        @test a[2, WaveNumberVector(0,  1)] == 1 + 2im
        @test a[2, WaveNumberVector(0, -1)] == 1 - 2im

        # Fully-zero wavenumber forced real (imaginary part discarded on write)
        a[1, WaveNumberVector(0, 0)] = 4 + 8im
        @test a[1, WaveNumberVector(0, 0)] == 4 + 0im

        # Negative rfft wavenumber — storage is at the conjugate location
        a[3, WaveNumberVector(-2, 1)] = -3 + 5im
        @test a[3, WaveNumberVector(-2,  1)] == -3 + 5im
        @test a[3, WaveNumberVector( 2, -1)] == -3 - 5im
    end

    @testset "WaveNumberVector get reads conjugate for negative rfft index" begin
        Ny, Nx, Nz, Nm = 2, 8, 5, 2
        g = TripleGrid(Ny, Nx, Nz)
        a = ProjectedField(g, _projected_modes(g, Nm))

        # Write directly into storage at rfft index 3 (wavenumber kx=2), kz-index 1
        a[1, 3, 1] = 7 + 3im    # storage index (m=1, ix=3, iz=1)
        # Positive wavenumber read — should get the stored value directly
        @test a[1, WaveNumberVector(2, 0)] ≈ 7 + 3im
        # Negative wavenumber read — should return conj of stored value
        @test a[1, WaveNumberVector(-2, 0)] ≈ 7 - 3im
    end

    @testset "copy and zero return independent fields with same grid/modes" begin
        Ny, Nx, Nz, Nm = 2, 4, 3, 2
        g = TripleGrid(Ny, Nx, Nz)
        Ψ = _projected_modes(g, Nm)
        a = ProjectedField(g, Ψ)
        # Use rfft index 2 (kx=1, non-zero) so the value is not forced real by
        # normalise_mean! when copy re-runs the constructor.
        a[1, 2, 1] = 1 + 1im

        b = copy(a)
        @test b[1, 2, 1] == a[1, 2, 1]
        @test grid(b) === grid(a)
        @test modes(b) === modes(a)
        # Independence: modifying a does not affect b
        a[1, 2, 1] = 99 + 0im
        @test b[1, 2, 1] == 1 + 1im

        z = zero(a)
        @test all(iszero, parent(z))
        @test grid(z) === grid(a)
        @test modes(z) === modes(a)
    end

    @testset "abs returns element-wise absolute values" begin
        Ny, Nx, Nz, Nm = 2, 4, 3, 2
        g = TripleGrid(Ny, Nx, Nz)
        a = ProjectedField(g, _projected_modes(g, Nm))
        parent(a) .= randn(ComplexF64, size(parent(a)))

        b = abs(a)
        @test all(real(b[i]) ≈ abs(a[i]) for i in eachindex(a))
        @test all(imag(b[i]) == 0 for i in eachindex(a))
    end

end
