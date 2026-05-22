# Tests for src/ftfield.jl.
#
# Contract:
#   - Construction enforces the real-field Fourier invariants at the mean and
#     Hermitian-symmetric storage locations.
#   - `u[k::WaveNumberVector, I...]` reads signed wavenumbers in FFT order.
#   - Setting coefficients through a `WaveNumberVector` preserves the same
#     Hermitian invariants.

@testset verbose=true "FTField contract                    " begin

    @testset "constructor normalises mean and Hermitian mirror pairs" begin
        Ny, Nx, Nz = 3, 8, 5
        g = TripleGrid(Ny, Nx, Nz)

        raw = randn(ComplexF64, Ny, (Nx >> 1) + 1, Nz)
        u = FTField(g, copy(raw))

        # The full zero-wavenumber line is real because a real physical field
        # cannot have an imaginary mean at any inhomogeneous point.
        @test all(iszero, imag.(parent(u)[:, 1, 1]))

        # On the rfft zero plane, signed z modes are conjugate pairs.
        for j in 1:Ny, kz in 1:(Nz >> 1)
            @test parent(u)[j, 1, kz + 1] ≈ conj(parent(u)[j, 1, Nz - kz + 1])
        end
    end

    @testset "WaveNumberVector getindex and setindex preserve symmetry" begin
        Ny, Nx, Nz = 2, 8, 5
        g = TripleGrid(Ny, Nx, Nz)
        u = FTField(g)

        # Writing at kx=0 must also write the signed-FFT mirror coefficient.
        u[WaveNumberVector(0, 1), 2] = 3 + 4im
        @test u[WaveNumberVector(0,  1), 2] == 3 + 4im
        @test u[WaveNumberVector(0, -1), 2] == 3 - 4im

        # The fully-zero mode is forced to be real.
        u[WaveNumberVector(0, 0), 1] = 2 + 9im
        @test u[WaveNumberVector(0, 0), 1] == 2 + 0im

        # Negative rfft wavenumbers are represented through conjugate storage.
        u[WaveNumberVector(-2, 1), 1] = 5 - 7im
        @test u[WaveNumberVector(-2,  1), 1] == 5 - 7im
        @test u[WaveNumberVector( 2, -1), 1] == 5 + 7im
    end
end
