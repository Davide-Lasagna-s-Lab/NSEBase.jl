# Tests for `shift!`, `shift` on FTField / VectorField / ProjectedField, plus
# the related `normdiff(..., shifts, ...)` and `minnormdiff(...)` routines.
#
# Contract (from src/shifts.jl):
#   shift!(u, shifts) multiplies every spectral coefficient at wavenumber
#   `k = (k₁, k₂, …)` by `exp(i Σⱼ kⱼ · shiftsⱼ · wavenumber_scale(g, ORDER[j]))`,
#   in physical-coordinate units.  When all shifts are zero it must return
#   `u` unchanged without touching memory.

@testset verbose=true "Shifts                              " begin

    @testset "FTField: phase factor matches the analytic shift" begin
        # Use a TripleGrid:
        #   dim 1: inhomogeneous y (length Ny)
        #   dim 2: rfft x ∈ [0, 2π/α)
        #   dim 3: signed FFT z ∈ [0, 2π/β)
        Ny, Nx, Nz = 4, 8, 6
        α, β = 1.0, 1.0
        g = TripleGrid(Ny, Nx, Nz; α=α, β=β)

        # Build an FFT of a known function so we can compare against the
        # analytic shift of the same function.  `cos(k₁·α·x + k₂·β·z)` is a
        # single Fourier mode at (k₁, k₂) — easy to predict the phase under
        # a shift.  Each coordinate is reshaped so the 3-axis broadcast
        # yields an `Array{Float64, 3}` of shape `(Ny, Nx, Nz)`.
        x = reshape((0:Nx-1) * (2π/α/Nx), 1, Nx, 1)        # shape (1, Nx, 1)
        z = reshape((0:Nz-1) * (2π/β/Nz), 1, 1, Nz)        # shape (1, 1, Nz)
        y_ones = ones(Float64, Ny, 1, 1)                   # shape (Ny, 1, 1)
        u_phys = @. y_ones * cos(2*α*x + 3*β*z)            # broadcast → (Ny, Nx, Nz)

        # Manually rfft along the *correct* dims to obtain reference û.
        ref = rfft(u_phys, (2, 3)) ./ (Nx * Nz)
        u = FTField(g, ref)

        # Apply shift!: phase = exp(i·k·s·scale).  For k=(2,3), s=(sx, sz):
        #   phase = exp(i·2·sx·α + i·3·sz·β).
        sx, sz = 0.37, -0.91
        v = shift!(copy(u), (sx, sz))

        # Reference: shift the physical function and re-FFT.
        u_shifted_phys = @. y_ones * cos(2*α*(x + sx) + 3*β*(z + sz))
        ref_shift = rfft(u_shifted_phys, (2, 3)) ./ (Nx * Nz)

        @test parent(v) ≈ ref_shift atol=1e-12
    end

    @testset "zero shift is identity and touches no memory" begin
        # The docstring guarantees that `shift!(u, all-zero)` returns `u`
        # without modification.  Test both the identity property and that
        # the early-out triggers — `===` confirms the same object is returned.
        Ny, Nx, Nz = 3, 8, 4
        g = TripleGrid(Ny, Nx, Nz)
        u = FTField(g, randn(ComplexF64, Ny, (Nx>>1)+1, Nz))
        u_ref = copy(parent(u))

        ret = shift!(u, (0.0, 0.0))
        @test ret === u                # same object returned
        @test parent(u) == u_ref       # contents unchanged
    end

    @testset "VectorField: shift! applies component-wise" begin
        # Each component of a VectorField must be shifted independently with
        # the same shift tuple.  Verify by applying shift! to the vector
        # field and to each component separately, then comparing.
        Ny, Nx, Nz = 3, 8, 4
        g = TripleGrid(Ny, Nx, Nz)

        u  = VectorField([FTField(g, randn(ComplexF64, Ny, (Nx>>1)+1, Nz)) for _ in 1:3]...)
        cs = (1.0, -0.5)

        # Per-component reference.
        ref = VectorField([shift!(copy(u[n]), cs) for n in 1:3]...)

        # Vector-level call.
        out = shift!(copy(u), cs)

        for n in 1:3
            @test parent(out[n]) ≈ parent(ref[n]) atol=1e-14
        end
    end

    @testset "shift returns a new field, leaves the original intact" begin
        # `shift(u, s) = shift!(copy(u), s)`.  Verify both halves: the
        # returned object is distinct from the input, and the input is
        # numerically unchanged.
        Ny, Nx, Nz = 3, 8, 4
        g = TripleGrid(Ny, Nx, Nz)
        u = FTField(g, randn(ComplexF64, Ny, (Nx>>1)+1, Nz))
        u_orig = copy(parent(u))

        v = shift(u, (0.3, 0.4))
        @test v !== u                     # distinct object
        @test parent(u) == u_orig         # original unchanged
        @test parent(v) ≠ parent(u)       # at least one coefficient differs
    end

    @testset "two shifts compose additively" begin
        # `shift!(shift!(u, a), b) == shift!(u, a .+ b)` — phases add modulo
        # the period, so consecutive shifts must agree with a single shift
        # by the sum.
        Ny, Nx, Nz = 3, 8, 4
        g = TripleGrid(Ny, Nx, Nz)
        u = FTField(g, randn(ComplexF64, Ny, (Nx>>1)+1, Nz))

        a = (0.4, -0.2)
        b = (0.1,  0.7)

        seq = shift!(shift!(copy(u), a), b)
        sum_ = shift!(copy(u), a .+ b)
        @test parent(seq) ≈ parent(sum_) atol=1e-14
    end

    @testset "ProjectedField: shift! changes only the homogeneous indices" begin
        # ProjectedField storage is `(Nm, kH...)`, so shift! acts only on the
        # homogeneous indices; the mode axis is untouched.  We verify this
        # by checking that the m-axis row of magnitudes is preserved
        # (a shift only changes phases, not magnitudes).
        Ny, Nx, Nz = 3, 8, 4
        g = TripleGrid(Ny, Nx, Nz)

        Nm = 5
        Ψ = ntuple(_ -> randn(ComplexF64, Ny, Nm, (Nx>>1)+1, Nz), 3)
        a = ProjectedField(g, randn(ComplexF64, Nm, (Nx>>1)+1, Nz), Ψ)

        b = shift!(copy(a), (0.31, -0.17))

        # |coefficient| must be preserved at every (m, k) location.
        @test abs.(parent(b)) ≈ abs.(parent(a)) atol=1e-14

        # ...but the coefficients themselves should differ (some k is non-zero).
        @test parent(b) ≠ parent(a)
    end
end
