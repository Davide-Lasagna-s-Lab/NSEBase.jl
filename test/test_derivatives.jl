# Tests for the spectral-derivative wrappers `ddx_1!`, `ddx_2!`, `ddx_3!`,
# `ddx_4!`, `add_homogeneous_laplacian!`, and `laplacian!`.
#
# Contract (from src/derivatives.jl):
#
#   - `ddx_n!(out, u)` dispatches to `ddx!(out, u, Val(AXES[n]))` and is the
#     spectral derivative along the n-th physical coordinate.  For
#     homogeneous (FFT) directions, the derivative is multiplication by
#     `i · k · wavenumber_scale`.
#
#   - `ddx_n!` on a coordinate whose `AXES[n] === nothing` is a compile-time
#     no-op (returns out unchanged).
#
#   - `add_homogeneous_laplacian!(out, u)`  adds `-Σ_{d∈ORDER} (k_d · σ_d)² · u`
#     into `out` (note the `+=`, not `=`).
#
#   - `laplacian!(out, u)` = `inhomogeneous_laplacian!(out, u)` followed by
#     `add_homogeneous_laplacian!(out, u)`.  On a fully-homogeneous grid the
#     inhomogeneous part is identity-equivalent if the user defines it that
#     way; FakeGrid does so (see fake.jl).

@testset verbose=true "Derivatives                         " begin

    @testset "analytic field derivatives on a mixed inhomogeneous/FFT grid" begin
        # This is the NSEBase version of the analytic derivative tests that
        # used to live naturally in ChannelFlow.  The grid has one polynomial
        # collocation direction y and one periodic rfft direction x.  The
        # y-derivative is supplied by the test grid extension, while x is the
        # generic NSEBase spectral derivative.  The field is a low-degree
        # polynomial in y times resolved Fourier modes in x, so both reference
        # derivatives are known analytically at the collocation points.
        Ny, Nx = 7, 16
        g = PolynomialGrid(range(-1, 1, length=Ny), Nx, 2π)

        u_fun(y, x)      = (1 + y + y^2 - 0.5y^3) * cos(2x) +
                           (0.25 - y^2) * sin(3x)
        dudx_fun(y, x)   = -2 * (1 + y + y^2 - 0.5y^3) * sin(2x) +
                            3 * (0.25 - y^2) * cos(3x)
        dudy_fun(y, x)   = (1 + 2y - 1.5y^2) * cos(2x) -
                            2y * sin(3x)
        lapl_fun(y, x)   = ((2 - 3y) - 4 * (1 + y + y^2 - 0.5y^3)) * cos(2x) +
                           (-2 - 9 * (0.25 - y^2)) * sin(3x)

        u = FFT(Field(g, u_fun))

        # Physical x is logical coordinate 1, so `ddx_1!` should be the
        # spectral derivative.  Physical y is logical coordinate 2, so
        # `ddx_2!` should dispatch to PolynomialGrid's inhomogeneous extension.
        @test parent(NSEBase.ddx_1!(FTField(g), u)) ≈
              parent(FFT(Field(g, dudx_fun))) atol=1e-12
        @test parent(NSEBase.ddx_2!(FTField(g), u)) ≈
              parent(FFT(Field(g, dudy_fun))) atol=1e-12

        # The Laplacian contract is the sum of the inhomogeneous second
        # derivative and the homogeneous spectral contribution.
        @test parent(NSEBase.laplacian!(FTField(g), u)) ≈
              parent(FFT(Field(g, lapl_fun))) atol=1e-11
    end

    @testset "ddx_n! is multiplication by i·k·σ on the homogeneous axis" begin
        # FakeGrid is 2-D: AXES = (1, 2, nothing, nothing), ORDER = (2,).
        # Therefore:
        #   ddx_1!: AXES[1] = 1 → inhomogeneous (custom, defined in fake.jl)
        #   ddx_2!: AXES[2] = 2 → rfft direction; multiplies by i·k·σ.
        #   ddx_3!, ddx_4!: AXES[3] = AXES[4] = nothing → compile-time no-ops.
        Nx, Ny = 8, 12
        L = 2π
        g = FakeGrid(rand(Float64, Nx), Ny, L)
        σ = NSEBase.wavenumber_scale(g, 2)            # 2π / L = 1.0 here

        # Build u with random coefficients so we can probe individual
        # wavenumbers after applying ddx_2!.
        data = randn(ComplexF64, Nx, (Ny >> 1) + 1)
        u = FTField(g, data)
        out = FTField(g)
        NSEBase.ddx_2!(out, u)

        # Each storage column k (1-based) corresponds to signed wavenumber
        # k-1 on the rfft axis.  Expected: out[:, k] == i · (k-1) · σ · u[:, k].
        pu, po = parent(u), parent(out)
        for k in 1:(Ny >> 1) + 1
            n_signed = k - 1
            expected = (im * n_signed * σ) .* pu[:, k]
            @test po[:, k] ≈ expected atol=1e-13
        end

        # ddx_3!, ddx_4! are no-ops on FakeGrid (AXES[3:4] === nothing).
        out2 = FTField(g, copy(data))
        @test parent(NSEBase.ddx_3!(out2, u)) == data
        out2 .= data
        @test parent(NSEBase.ddx_4!(out2, u)) == data
    end

    @testset "add_homogeneous_laplacian! adds −(k σ)² u (not =, +=)" begin
        # The function accumulates into `out`, it does not overwrite.  The
        # added value at wavenumber k is `-(k·σ)² · u[k]` on FakeGrid (only
        # one FFT dim).
        Nx, Ny = 8, 12
        L = 2π
        g = FakeGrid(rand(Float64, Nx), Ny, L)
        σ = NSEBase.wavenumber_scale(g, 2)

        u = FTField(g, randn(ComplexF64, Nx, (Ny >> 1) + 1))

        # Seed `out` with some initial value to verify that
        # add_homogeneous_laplacian! accumulates.  Read the seed back from
        # `parent(out)` *after* FTField has applied its Hermitian-symmetry
        # invariant (so the comparison sees the same starting state).
        out  = FTField(g, randn(ComplexF64, Nx, (Ny >> 1) + 1))
        seed = copy(parent(out))

        NSEBase.add_homogeneous_laplacian!(out, u)

        pu, po = parent(u), parent(out)
        for k in 1:(Ny >> 1) + 1
            n_signed = k - 1
            expected = seed[:, k] .- ((n_signed * σ)^2) .* pu[:, k]
            @test po[:, k] ≈ expected atol=1e-13
        end
    end

    @testset "laplacian! = inhomogeneous_laplacian! + add_homogeneous_laplacian!" begin
        # On FakeGrid, `inhomogeneous_laplacian!(out, u)` is defined in
        # fake.jl as `out .= u`, so `laplacian!(out, u)` = u − (kσ)² u
        # at every wavenumber k.
        Nx, Ny = 8, 12
        L = 2π
        g = FakeGrid(rand(Float64, Nx), Ny, L)
        σ = NSEBase.wavenumber_scale(g, 2)

        u = FTField(g, randn(ComplexF64, Nx, (Ny >> 1) + 1))
        out = NSEBase.laplacian!(FTField(g), u)

        pu, po = parent(u), parent(out)
        for k in 1:(Ny >> 1) + 1
            n_signed = k - 1
            expected = pu[:, k] .- ((n_signed * σ)^2) .* pu[:, k]
            @test po[:, k] ≈ expected atol=1e-13
        end
    end

    @testset "VectorField wrappers apply component-wise" begin
        # Every derivative has a VectorField method that loops over
        # components and calls the single-field method.  Confirm the loop
        # produces the same result as per-component application.
        Nx, Ny = 8, 12
        g = FakeGrid(rand(Float64, Nx), Ny, 2π)

        u = VectorField([FTField(g, randn(ComplexF64, Nx, (Ny>>1)+1)) for _ in 1:3]...)
        out_vec = VectorField(g; N=3)
        NSEBase.ddx_2!(out_vec, u)

        for n in 1:3
            expected = FTField(g)
            NSEBase.ddx_2!(expected, u[n])
            @test parent(out_vec[n]) ≈ parent(expected) atol=1e-14
        end
    end
end
