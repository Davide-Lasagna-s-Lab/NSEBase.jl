# Spectral differentiation operators for FTField, ProjectedField, and VectorField.
#
# Differentiation in a homogeneous (FFT-transformed) direction is exact in
# spectral space: the derivative of mode k is simply multiplied by
# ±i·n·wavenumber_scale(g, storage_dim), where n is the signed integer
# wavenumber and the sign is +1 for the forward operator and −1 for its L2
# adjoint.
#
# The core primitive is `dd!(out, u, Val(storage_dim))`, implemented with
# CartesianIndices loops.  For the rfft dimension all stored wavenumbers are
# non-negative so a single loop over the full array suffices.  For signed-FFT
# dimensions the index range is split into a positive block [1:(N÷2)+1] and a
# negative block [(N÷2)+2:N] so that the signed wavenumber can be computed
# branch-free from the loop index inside each block.  The two-block
# specialisation (rfft vs signed FFT) and the ntuple range construction are
# eliminated at compile time because the storage-dim type parameter and
# FFT_DIMS_ORDER are type parameters.
#
# `dd!(out, u, physical_dim::Symbol)` is the user-facing generic form;
# `dd!` reads as "directional derivative". The per-direction wrappers
# `ddx!`, `ddy!`, `ddz!`, and `ddt!` are reserved for their literal
# coordinate and never take a `physical_dim` argument.
#
# Both forms translate to storage dimensions with [`storage_dim`](@ref)
# at the low-level kernel boundary. When a physical direction is absent
# from the grid, its storage dimension is `nothing` and the kernel is a
# no-op.
#
# Inhomogeneous (non-FFT) directions are NOT handled here — `dd!` throws
# `NotImplementedError` for those, and downstream packages must extend
# it with a grid-specific method (typically a matrix–vector multiply).
#
# The full Laplacian combines `inhomogeneous_laplacian!` (provided by
# downstream) with `add_homogeneous_laplacian!` (provided here), which
# subtracts the spatial ‖k‖² · u contribution from each spectral coefficient.

"""
    dd!(out, u, ::Val{STORAGE_DIM}; adjoint=false)

Low-level in-place derivative of `u` along the storage dimension
encoded by `Val(STORAGE_DIM)`.

For `STORAGE_DIM in FFT_DIMS_ORDER` the derivative is multiplication by
`±im * n * wavenumber_scale(grid, STORAGE_DIM)` where `n` is the signed
wavenumber. `adjoint=false` (default) gives `+im·n·scale·u`;
`adjoint=true` gives `-im·n·scale·u` (the L2 adjoint of the spectral
derivative for homogeneous directions).

For an inhomogeneous storage dimension the method throws
`NotImplementedError`. Downstream packages should extend this for each
non-homogeneous direction and handle the `adjoint` keyword there too.

For `Val(nothing)` the function reduces to identity.
"""
function dd!(out::F, u::F, ::Val{STORAGE_DIM};
             adjoint::Bool=false) where {
        STORAGE_DIM, T, D, AXES, FFT_DIMS_ORDER,
        G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER},
        F<:Union{FTField{G}, ProjectedField{G}}}

    # Missing physical coordinates translate to Val(nothing), e.g. `:t` on a
    # steady grid. There is no storage axis to process in that case.
    isnothing(STORAGE_DIM) && return out
    # Inhomogeneous directions are not handled here; downstream packages must
    # extend `dd!` with a grid-specific method (e.g. a matrix-vector multiply).
    STORAGE_DIM ∉ FFT_DIMS_ORDER &&
        throw(NotImplementedError(grid(u), Val(STORAGE_DIM)))

    # Hoist the scale and adjoint sign outside the loop so each element update
    # is a single complex multiply with no repeated division or branching.
    scale = wavenumber_scale(grid(u), STORAGE_DIM)
    coeff = adjoint ? -im * T(scale) : im * T(scale)
    Nd    = size(u, STORAGE_DIM)
    pu    = parent(u)
    pout  = parent(out)

    if STORAGE_DIM == FFT_DIMS_ORDER[1]
        # rfft dimension: only non-negative wavenumbers 0 … Nd-1 are stored,
        # so the wavenumber is simply the loop index minus one.  A single
        # CartesianIndices pass over the full array suffices.
        @inbounds for I in CartesianIndices(pu)
            pout[I] = (coeff * (I[STORAGE_DIM] - 1)) * pu[I]
        end
    else
        # Signed-FFT dimension: split into a non-negative block [1:(Nd÷2)+1]
        # and a negative block [(Nd÷2)+2:Nd] so the signed wavenumber can be
        # computed branch-free as a simple offset inside each block.
        # STORAGE_DIM and FFT_DIMS_ORDER are type parameters, so the ntuple
        # ranges and the if-branch are both eliminated at compile time.
        pos_ranges = ntuple(d -> d == STORAGE_DIM ? (1:(Nd >> 1) + 1)  : Base.OneTo(size(pu, d)), Val(D))
        neg_ranges = ntuple(d -> d == STORAGE_DIM ? ((Nd >> 1) + 2:Nd) : Base.OneTo(size(pu, d)), Val(D))
        @inbounds for I in CartesianIndices(pos_ranges)
            pout[I] = (coeff * (I[STORAGE_DIM] - 1)) * pu[I]
        end
        @inbounds for I in CartesianIndices(neg_ranges)
            pout[I] = (coeff * (I[STORAGE_DIM] - Nd - 1)) * pu[I]
        end
    end
    return out
end

"""
    dd!(out, u, physical_dim::Symbol; adjoint=false) -> out

User-facing generic derivative: differentiate `u` along
`physical_dim` (`:x`, `:y`, `:z`, or `:t`).

The symbol is translated to a `Val` of the matching storage dimension
through [`physical_to_storage_dim`](@ref) — fully resolved at compile
time when `physical_dim` is a literal at the call site — and dispatched
to the low-level `Val{STORAGE_DIM}` primitive above.

If the grid does not expose `physical_dim` (e.g. `:t` on a steady grid),
the call is a no-op.
"""
dd!(out, u, physical_dim::Symbol; kwargs...) =
    dd!(out, u, physical_to_storage_dim(grid(u), Val(physical_dim)); kwargs...)

dd!(out::VectorField{N}, u::VectorField{N}, d; kwargs...) where {N} =
    (for n in 1:N; dd!(out[n], u[n], d; kwargs...); end; return out)

"""
    ddx!(out, u; adjoint=false) -> out
    ddy!(out, u; adjoint=false) -> out
    ddz!(out, u; adjoint=false) -> out
    ddt!(out, u; adjoint=false) -> out

Differentiate `u` along the literal physical direction in the function
name, storing the result in `out`. These are thin wrappers around
[`dd!`](@ref) that fix the direction at the call site — `ddx!` is `d/dx`,
`ddy!` is `d/dy`, and so on. They never take a `physical_dim` argument;
use `dd!(out, u, :x)` if the direction is chosen at runtime.

For an absent direction, such as `:z` on a 2D grid, the operation is a no-op.

For inhomogeneous (non-FFT) directions the derivative is the grid-specific
physical derivative along that coordinate.
"""
ddx!(out, u; kwargs...) = dd!(out, u, :x; kwargs...)
ddy!(out, u; kwargs...) = dd!(out, u, :y; kwargs...)
ddz!(out, u; kwargs...) = dd!(out, u, :z; kwargs...)
ddt!(out, u; kwargs...) = dd!(out, u, :t; kwargs...)

"""
    ddx_1!(out, u; adjoint=false) -> out
    ddx_2!(out, u; adjoint=false) -> out
    ddx_3!(out, u; adjoint=false) -> out
    ddx_4!(out, u; adjoint=false) -> out

Compatibility aliases for [`ddx!`](@ref), [`ddy!`](@ref), [`ddz!`](@ref), and
[`ddt!`](@ref), respectively.
"""
ddx_1!(out, u; kwargs...) = ddx!(out, u; kwargs...)
ddx_2!(out, u; kwargs...) = ddy!(out, u; kwargs...)
ddx_3!(out, u; kwargs...) = ddz!(out, u; kwargs...)
ddx_4!(out, u; kwargs...) = ddt!(out, u; kwargs...)

"""
    add_homogeneous_laplacian!(out::FTField, u::FTField)
    add_homogeneous_laplacian!(out::VectorField, u::VectorField)

Add the homogeneous Laplacian contribution of `u` to `out`:

    out[mode] -= (∑_{d∈spatial_fft_storage_dims(g)} (wavenumber_scale(g, d) · n_d)²) · u[mode]

Call after computing the non-homogeneous (e.g. wall-normal) second derivative.
If the grid includes a transformed logical time coordinate, that direction is
not part of the spatial Laplacian.
"""
function add_homogeneous_laplacian!(out::FTField{G}, u::FTField{G}) where {T, D, AXES, FFT_DIMS_ORDER, G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}}
    g = grid(u)

    # `to_wavenumber_vector(g, Ih)` returns k in full FFT_DIMS_ORDER order.
    # Keep `scales` in that same order.  A transformed logical time coordinate
    # is homogeneous, but it is not in `spatial_fft_storage_dims(g)`, so give it zero
    # scale rather than building a second filtered index mapping.
    H = spatial_fft_storage_dims(g)
    scales = map(dim -> dim in H ? T(wavenumber_scale(g, dim)) : zero(T), FFT_DIMS_ORDER)
    pu = parent(u)
    pout = parent(out)

    # The multiplier k² depends only on the homogeneous wavenumber.  Loop over
    # homogeneous modes outside, compute k² once, then apply it to every
    # inhomogeneous/collocation index for that same mode.
    @inbounds for Ih in CartesianIndices(homogeneous_axes(u))
        k² = zero(T)
        k = to_wavenumber_vector(g, Ih)
        for j in eachindex(FFT_DIMS_ORDER)
            k² += (scales[j] * k[j])^2
        end

        # `combine_indices` interleaves inhomogeneous and homogeneous indices
        # according to the grid storage layout, so this works for both
        # `(y, x, z)` and FFT-first layouts.
        for Iinh in CartesianIndices(inhomogeneous_axes(u))
            I = combine_indices(g, Iinh, Ih)
            pout[I...] -= k² * pu[I...]
        end
    end
    return out
end
add_homogeneous_laplacian!(out::VectorField{N}, u::VectorField{N}) where {N} =
    (for n in 1:N; add_homogeneous_laplacian!(out[n], u[n]); end; return out)

"""
    inhomogeneous_laplacian!(out::FTField, u::FTField; kwargs...) -> out

Apply the inhomogeneous (non-FFT) part of the Laplacian of `u` to `out`.

This contribution contains the second derivatives along directions that are
not in [`fft_storage_dims`](@ref), such as wall-normal collocation directions.  The
full spatial Laplacian is the sum of this contribution and the homogeneous
spectral contribution from [`add_homogeneous_laplacian!`](@ref).
"""
inhomogeneous_laplacian!(out::FTField, u::FTField) = throw(NotImplementedError(out, u))

"""
    laplacian!(out::FTField{G}, u::FTField{G}; kwargs...)
    laplacian!(out::VectorField{N}, u::VectorField{N}; kwargs...)

Compute the full Laplacian of `u` in-place, storing the result in `out`:

    out[mode] = (inhomogeneous second derivatives
                 - ∑_{d∈spatial_fft_storage_dims(g)} (wavenumber_scale(g, d) · n_d)²) · u[mode]

Only spatial transformed directions enter the homogeneous sum; a transformed
logical time coordinate is excluded through [`spatial_fft_storage_dims`](@ref).
`kwargs` may be used by grid-specific inhomogeneous directions (e.g. boundary
condition parameters).  The `VectorField` method applies the same scalar
operation independently to each component.
"""
function laplacian!(out::FTField{G}, u::FTField{G}; kwargs...) where {G}
    inhomogeneous_laplacian!(out, u; kwargs...)
    add_homogeneous_laplacian!(out, u)
    return out
end
laplacian!(out::VectorField{N}, u::VectorField{N}; kwargs...) where {N} =
    (for n in 1:N; laplacian!(out[n], u[n]; kwargs...); end; return out)
