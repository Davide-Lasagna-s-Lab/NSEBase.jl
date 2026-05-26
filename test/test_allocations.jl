# Allocation contracts for hot in-place kernels.
#
# These tests are deliberately written through small helper functions instead
# of top-level `@allocated` expressions. Top-level closures can allocate because
# they capture non-constant globals; the contract we care about here is whether
# the package kernels allocate once their argument types are known.

function _allocated_after_warmup(f)
    f()
    f()
    return @allocated f()
end

function _spectral_kernel_allocations()
    g = TripleGrid(3, 8, 6)
    u = FTField(g)
    out = zero(u)
    parent(u) .= randn(ComplexF64, size(parent(u)))

    return (
        ddx_1 = _allocated_after_warmup(() -> NSEBase.ddx_1!(out, u)),
        ddx_direct = _allocated_after_warmup(() -> NSEBase.ddx!(out, u, Val(2))),
        homogeneous_laplacian = _allocated_after_warmup(() -> NSEBase.add_homogeneous_laplacian!(out, u)),
    )
end

function _polynomial_kernel_allocations()
    # The inhomogeneous derivative extension in `test_grids.jl` uses a dense
    # matrix multiply. This guards the common downstream contract: packages can
    # implement non-FFT derivatives without allocating temporary matrix products.
    g = PolynomialGrid([-1.0, -0.3, 0.2, 0.7, 1.0], 8)
    u = FTField(g)
    out = zero(u)
    parent(u) .= randn(ComplexF64, size(parent(u)))

    return (
        ddx_2 = _allocated_after_warmup(() -> NSEBase.ddx_2!(out, u)),
        inhomogeneous_laplacian = _allocated_after_warmup(() -> NSEBase.inhomogeneous_laplacian!(out, u)),
    )
end

function _shift_allocations()
    g = TripleGrid(3, 8, 6)
    u = FTField(g)
    parent(u) .= randn(ComplexF64, size(parent(u)))

    Nm = 4
    modes = ntuple(_ -> randn(ComplexF64, 3, Nm, (8 >> 1) + 1, 6), 3)
    a = ProjectedField(g, randn(ComplexF64, Nm, (8 >> 1) + 1, 6), modes)

    shifts = (0.13, -0.21)
    return (
        ftfield = _allocated_after_warmup(() -> NSEBase.shift!(u, shifts)),
        projectedfield = _allocated_after_warmup(() -> NSEBase.shift!(a, shifts)),
    )
end

function _weighting_allocations()
    g = TripleGrid(3, 8, 6; α=1.5, β=2.0)
    Nm = 4
    modes = ntuple(_ -> randn(ComplexF64, 3, Nm, (8 >> 1) + 1, 6), 3)
    a = ProjectedField(g, randn(ComplexF64, Nm, (8 >> 1) + 1, 6), modes)
    b = copy(a)
    A = FarazmandWeight(g)

    return (
        lmul = _allocated_after_warmup(() -> lmul!(A, a)),
        dot = _allocated_after_warmup(() -> dot(a, A, b)),
    )
end

function _broadcast_allocations()
    g = TripleGrid(3, 8, 6)
    u = FTField(g)
    v = FTField(g)
    out = zero(u)
    parent(u) .= randn(ComplexF64, size(parent(u)))
    parent(v) .= randn(ComplexF64, size(parent(v)))

    return (
        scalar_field = _allocated_after_warmup(() -> (out .= 2 .* u .- v)),
    )
end

@testset verbose=true "Allocation contracts              " begin
    @test _spectral_kernel_allocations() == (ddx_1 = 0, ddx_direct = 0, homogeneous_laplacian = 0)
    @test _polynomial_kernel_allocations() == (ddx_2 = 0, inhomogeneous_laplacian = 0)
    @test _shift_allocations() == (ftfield = 0, projectedfield = 0)
    @test _weighting_allocations() == (lmul = 0, dot = 0)
    @test _broadcast_allocations() == (scalar_field = 0,)
end
