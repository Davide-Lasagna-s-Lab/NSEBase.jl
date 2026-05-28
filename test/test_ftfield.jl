# Tests for src/ftfield.jl.
#
# Contract:
#   - Construction enforces the real-field Fourier invariants at the mean and
#     Hermitian-symmetric storage locations.
#   - `u[k::WaveNumberVector, I...]` reads signed wavenumbers in FFT order.
#   - Setting coefficients through a `WaveNumberVector` preserves the same
#     Hermitian invariants.

@testset verbose=true "FTField contract                    " begin

    @testset "constructor normalises mean and Hermitian mirror pairs            " begin
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

    @testset "growto copies source coefficients to the same wavenumbers" begin
        Ny, Nx, Nz = 3, 8, 5
        g  = TripleGrid(Ny, Nx, Nz)
        u  = FTField(g)

        # Deterministic coefficients for the whole stored source spectrum.  On
        # the rfft-zero plane this formula respects Hermitian symmetry:
        # coeff(0, -kz, j) == conj(coeff(0, kz, j)).
        coeff(k, j) = complex(100j + 10k[1] + abs(k[2]),
                              k[1] == 0 ? 3k[2] : 7k[1] + 3k[2])

        # Populate every stored source wavenumber and every inhomogeneous point.
        for Ih in CartesianIndices(NSEBase.homogeneous_axes(u)), j in 1:Ny
            k = NSEBase.to_wavenumber_vector(g, Ih)
            u[k, j] = coeff(k, j)
        end

        # Grow to a larger grid: (Nx, Nz) → (16, 11)
        v = NSEBase.growto(u, (16, 11))

        @test size(grid(v)) == (Ny, 16, 11)
        @test size(parent(v)) == (Ny, 9, 11)   # rfft: (16>>1)+1=9, Nz stays 11

        # Exactness: every source wavenumber is copied to the same signed
        # wavenumber in the target, including the negative signed-FFT block.
        for Ih in CartesianIndices(NSEBase.homogeneous_axes(u)), j in 1:Ny
            k = NSEBase.to_wavenumber_vector(g, Ih)
            @test u[k, j] == coeff(k, j)
            @test v[k, j] == u[k, j]
        end

        # Every target wavenumber outside the source resolution stays exactly
        # zero.  This also catches the old bug where source k=(1,-1) was copied
        # to target k=(1,4) by reusing storage indices instead of wavenumbers.
        for Ih in CartesianIndices(NSEBase.homogeneous_axes(v)), j in 1:Ny
            k = NSEBase.to_wavenumber_vector(grid(v), Ih)
            in_source = 0 <= k[1] <= (Nx >> 1) && abs(k[2]) <= (Nz >> 1)
            @test in_source ? v[k, j] == u[k, j] : iszero(v[k, j])
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
