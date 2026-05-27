# Tests for `FarazmandWeight`, its `lmul!` action, and the weighted `dot`.
#
# Contract (from src/weighting.jl):
#
#   w(k) = 1 / (1 + Σⱼ (σⱼ · kⱼ)²)
#
#   where σⱼ are the scales in `FFT_DIMS_ORDER` order and kⱼ is the signed
#   integer wavenumber along the j-th homogeneous dimension.

@testset verbose=true "FarazmandWeight                     " begin

    @testset "constructors and scale storage" begin
        # Grid form: pulls scales from `wavenumber_scale(g, ORDER[k])`.
        Ny, Nx, Nz = 3, 8, 6
        α, β = 1.5, 2.0
        g = TripleGrid(Ny, Nx, Nz; α=α, β=β)

        A = FarazmandWeight(g)
        @test A.scales === (α, β)

        # Varargs form: takes the scales explicitly, in fft_dims order.
        B = FarazmandWeight(2π, 4π)
        @test B.scales === (2π, 4π)

        # Mixed Real types are promoted to a common type.
        C = FarazmandWeight(1, 2.0, 3)
        @test C isa FarazmandWeight{3, Float64}
        @test C.scales === (1.0, 2.0, 3.0)
    end

    @testset "getindex evaluates the formula at a signed wavenumber             " begin
        # w(k) = 1 / (1 + (σ₁ k₁)² + (σ₂ k₂)²).  Verify on a handful of k.
        σ₁, σ₂ = 2.0, 3.0
        A = FarazmandWeight(σ₁, σ₂)

        # k = 0: weight is 1 (no penalty).
        @test A[WaveNumberVector(0, 0)] ≈ 1.0

        # k = (1, 0): weight is 1 / (1 + σ₁²) = 1 / 5.
        @test A[WaveNumberVector(1, 0)] ≈ 1 / (1 + σ₁^2)

        # k = (0, 1): weight is 1 / (1 + σ₂²) = 1 / 10.
        @test A[WaveNumberVector(0, 1)] ≈ 1 / (1 + σ₂^2)

        # k = (2, 3): weight is 1 / (1 + 4·4 + 9·9) = 1 / 98.
        @test A[WaveNumberVector(2, 3)] ≈ 1 / (1 + (σ₁*2)^2 + (σ₂*3)^2)

        # The weight is *invariant under sign* of k (depends only on k²).
        @test A[WaveNumberVector(-1,  2)] ≈ A[WaveNumberVector( 1, -2)]
        @test A[WaveNumberVector( 1, -2)] ≈ A[WaveNumberVector(-1, -2)]
    end

    @testset "lmul! scales every coefficient by A[k] in place" begin
        # Build a ProjectedField, apply lmul!, and check coefficient-by-
        # coefficient that the result equals the original times A[k].
        # The reference must be built from `parent(a)` *after* the
        # ProjectedField constructor has applied its Hermitian-symmetry
        # enforcement — `coeffs` is the raw user input which the
        # constructor mutates.
        Ny, Nx, Nz = 3, 8, 4
        α, β = 2π, 2π
        g = TripleGrid(Ny, Nx, Nz; α=α, β=β)
        Nm = 5
        Ψ = ntuple(_ -> randn(ComplexF64, Ny, Nm, (Nx>>1)+1, Nz), 3)
        a = ProjectedField(g, randn(ComplexF64, Nm, (Nx>>1)+1, Nz), Ψ)
        coeffs = copy(parent(a))         # canonical (post-symmetry) data

        A = FarazmandWeight(g)
        ret = lmul!(A, a)
        @test ret === a                  # in-place: same object returned

        # Build an explicit reference by scaling each (m, _nx, _nz) by
        # A[(nx_signed, nz_signed)].  Iterate over *storage* indices so the
        # Nyquist mode (when Nz is even) is visited exactly once — iterating
        # over signed wavenumbers would map both +Nyquist and -Nyquist to
        # the same storage index and apply the weight twice.
        ref = copy(coeffs)
        for _nz in 1:Nz, _nx in 1:(Nx >> 1) + 1, m in 1:Nm
            nx_signed = _nx - 1                                # rfft: 0..Nx/2
            nz_signed = _nz <= (Nz >> 1) + 1 ? _nz - 1 : _nz - Nz - 1  # signed FFT
            w = 1 / (1 + (α*nx_signed)^2 + (β*nz_signed)^2)
            ref[m, _nx, _nz] *= w
        end
        @test parent(a) ≈ ref atol=1e-14
    end

    @testset "weighted dot(a, A, b) matches manual weighting" begin
        # The documented formula:
        #   ⟨a, b⟩_A = (1/2) Σₖ c_{k₁} w(k) Σₘ Re( conj(a_{m,k}) b_{m,k} )
        # We test that this equals `dot(a, lmul!(A, copy(b)))` (the
        # equivalent of applying A to one side and then taking the regular
        # inner product).  The two are mathematically identical.
        Ny, Nx, Nz = 3, 8, 4
        g = TripleGrid(Ny, Nx, Nz; α=2π, β=2π)
        Nm = 4
        Ψ = ntuple(_ -> randn(ComplexF64, Ny, Nm, (Nx>>1)+1, Nz), 3)
        a = ProjectedField(g, randn(ComplexF64, Nm, (Nx>>1)+1, Nz), Ψ)
        b = ProjectedField(g, randn(ComplexF64, Nm, (Nx>>1)+1, Nz), Ψ)
        A = FarazmandWeight(g)

        @test dot(a, A, b) ≈ dot(a, lmul!(A, copy(b))) atol=1e-13
    end
end
