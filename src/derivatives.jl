# Spectral differentiation operators for FTField, ProjectedField, and VectorField.
#
# Differentiation in a homogeneous (FFT-transformed) direction is exact in
# spectral space: the derivative of mode k is simply multiplied by
# ±i·n·wavenumber_scale(g, dim), where n is the signed integer wavenumber and
# the sign is +1 for the forward operator and −1 for its L2 adjoint.
#
# The core primitive is `ddx!(out, u, Val{DIM})`, implemented with
# CartesianIndices loops.  For the rfft dimension all stored wavenumbers are
# non-negative so a single loop over the full array suffices.  For signed-FFT
# dimensions the index range is split into a positive block [1:(N÷2)+1] and a
# negative block [(N÷2)+2:N] so that the signed wavenumber can be computed
# branch-free from the loop index inside each block.  The two-block
# specialisation (rfft vs signed FFT) and the ntuple range construction are
# eliminated at compile time because DIM and FFT_DIMS_ORDER are type parameters.
#
# Four named wrappers `ddx_1!`, `ddx_2!`, `ddx_3!`, `ddx_4!` forward to
# `ddx!` with the array dimension read from `AXES[1]`, `AXES[2]`, `AXES[3]`,
# `AXES[4]` respectively.  When `AXES[j] === nothing` (e.g. a steady 3D grid
# has no time dimension), the guard clause turns the call into a no-op.
#
# Inhomogeneous (non-FFT) directions are NOT handled here — `ddx!` throws
# `NotImplementedError` for those, and downstream packages must extend it with
# a grid-specific method (typically a matrix–vector multiply).
#
# The full Laplacian combines `inhomogeneous_laplacian!` (provided by
# downstream) with `add_homogeneous_laplacian!` (provided here), which
# subtracts the spatial ‖k‖² · u contribution from each spectral coefficient.

"""
    ddx!(out::FTField, u::FTField, ::Val{DIM}; adjoint=false)
    ddx!(out::VectorField, u::VectorField, ::Val{DIM}; adjoint=false)

In-place spectral derivative of `u` along array dimension `DIM`, writing into `out`.

For `DIM in FFT_DIMS_ORDER` the derivative is multiplication by
`±im * n * wavenumber_scale(grid, DIM)` where `n` is the signed wavenumber.
`adjoint=false` (default) gives `+im·n·scale·u`; `adjoint=true` gives `-im·n·scale·u`
(the L2 adjoint of the spectral derivative for homogeneous directions).

For `DIM not in FFT_DIMS_ORDER`, the derivative is grid-specific. If no
derivative is defined for that non-homogeneous direction, the method throws
`NotImplementedError`.

For `AXES[DIM] === nothing` the function reduces to a no-op.
"""
function ddx!(out::F, u::F, ::Val{DIM};
              adjoint::Bool=false) where {
        DIM, T, D, AXES, FFT_DIMS_ORDER,
        G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER},
        F<:Union{FTField{G}, ProjectedField{G}}}

    # A missing logical coordinate, e.g. AXES[4] === nothing on a steady grid,
    # is represented by Val(nothing) and should be a no-op.
    (isnothing(DIM) || isnothing(AXES[DIM])) && return out
    # Inhomogeneous directions are not handled here; downstream packages must
    # extend ddx! with a grid-specific method (e.g. a matrix–vector multiply).
    DIM ∉ FFT_DIMS_ORDER && throw(NotImplementedError(grid(u), Val(DIM)))

    # Hoist the scale and adjoint sign outside the loop so each element update
    # is a single complex multiply with no repeated division or branching.
    scale = wavenumber_scale(grid(u), DIM)
    coeff = adjoint ? -im * T(scale) : im * T(scale)
    Nd    = size(u, DIM)
    pu    = parent(u)
    pout  = parent(out)

    if DIM == FFT_DIMS_ORDER[1]
        # rfft dimension: only non-negative wavenumbers 0 … Nd-1 are stored,
        # so the wavenumber is simply the loop index minus one.  A single
        # CartesianIndices pass over the full array suffices.
        @inbounds for I in CartesianIndices(pu)
            pout[I] = (coeff * (I[DIM] - 1)) * pu[I]
        end
    else
        # Signed-FFT dimension: split into a non-negative block [1:(Nd÷2)+1]
        # and a negative block [(Nd÷2)+2:Nd] so the signed wavenumber can be
        # computed branch-free as a simple offset inside each block.
        # DIM and FFT_DIMS_ORDER are type parameters, so the ntuple ranges and
        # the if-branch are both eliminated at compile time.
        pos_ranges = ntuple(d -> d == DIM ? (1:(Nd >> 1) + 1)  : Base.OneTo(size(pu, d)), Val(D))
        neg_ranges = ntuple(d -> d == DIM ? ((Nd >> 1) + 2:Nd) : Base.OneTo(size(pu, d)), Val(D))
        @inbounds for I in CartesianIndices(pos_ranges)
            pout[I] = (coeff * (I[DIM] - 1)) * pu[I]
        end
        @inbounds for I in CartesianIndices(neg_ranges)
            pout[I] = (coeff * (I[DIM] - Nd - 1)) * pu[I]
        end
    end
    return out
end
ddx!(out::VectorField{N}, u::VectorField{N}, d; kwargs...) where {N} =
    (for n in 1:N; ddx!(out[n], u[n], d; kwargs...); end; return out)

"""
    ddx_1!(out, u; adjoint=false) -> out

Differentiate `u` in the first physical coordinate (`x`), storing
the result in `out`.  The grid's `AXES` layout determines which array
dimension represents the first physical coordinate.

Defined for `FTField`, `ProjectedField`, and `VectorField` arguments on any
`AbstractGrid`.
"""
ddx_1!(out::ProjectedField{G}, a::ProjectedField{G}; kwargs...) where {AXES, D, G<:AbstractGrid{<:Any, D, AXES}} =
    ddx!(out, a, Val(AXES[1]); kwargs...)
ddx_1!(out::FTField{G}, u::FTField{G}; kwargs...) where {AXES, D, G<:AbstractGrid{<:Any, D, AXES}} =
    ddx!(out, u, Val(AXES[1]); kwargs...)
ddx_1!(out::VectorField{N}, u::VectorField{N}; kwargs...) where {N} =
    (for n in 1:N; ddx_1!(out[n], u[n]; kwargs...); end; return out)

"""
    ddx_2!(out, u; adjoint=false) -> out

Differentiate `u` in the second physical coordinate (`y`, or `r`), storing
the result in `out`.  The grid's `AXES` layout determines which array
dimension represents the second physical coordinate.

For inhomogeneous (non-FFT) directions the derivative is the grid-specific
physical derivative along that coordinate.
"""
ddx_2!(out::ProjectedField{G}, a::ProjectedField{G}; kwargs...) where {AXES, D, G<:AbstractGrid{<:Any, D, AXES}} =
    ddx!(out, a, Val(AXES[2]); kwargs...)
ddx_2!(out::FTField{G}, u::FTField{G}; kwargs...) where {AXES, D, G<:AbstractGrid{<:Any, D, AXES}} =
    ddx!(out, u, Val(AXES[2]); kwargs...)
ddx_2!(out::VectorField{N}, u::VectorField{N}; kwargs...) where {N} =
    (for n in 1:N; ddx_2!(out[n], u[n]; kwargs...); end; return out)

"""
    ddx_3!(out, u; adjoint=false) -> out

Differentiate `u` in the third physical coordinate (`z`, or `theta`), storing the
result in `out`.  The grid's `AXES` layout determines which array dimension
represents the third physical coordinate.

For grids without a third physical coordinate, `out` is left unchanged.
"""
ddx_3!(out::ProjectedField{G}, a::ProjectedField{G}; kwargs...) where {AXES, D, G<:AbstractGrid{<:Any, D, AXES}} =
    ddx!(out, a, Val(AXES[3]); kwargs...)
ddx_3!(out::FTField{G}, u::FTField{G}; kwargs...) where {AXES, D, G<:AbstractGrid{<:Any, D, AXES}} =
    ddx!(out, u, Val(AXES[3]); kwargs...)
ddx_3!(out::VectorField{N}, u::VectorField{N}; kwargs...) where {N} =
    (for n in 1:N; ddx_3!(out[n], u[n]; kwargs...); end; return out)

"""
    ddx_4!(out, u; adjoint=false) -> out

Differentiate `u` in the fourth physical coordinate (`t`), storing
the result in `out`.  The grid's `AXES` layout determines which array dimension
represents the fourth physical coordinate.  For grids without a fourth physical
coordinate, `out` is left unchanged.
"""
ddx_4!(out::ProjectedField{G}, a::ProjectedField{G}; kwargs...) where {AXES, D, G<:AbstractGrid{<:Any, D, AXES}} =
    ddx!(out, a, Val(AXES[4]); kwargs...)
ddx_4!(out::FTField{G}, u::FTField{G}; kwargs...) where {AXES, D, G<:AbstractGrid{<:Any, D, AXES}} =
    ddx!(out, u, Val(AXES[4]); kwargs...)
ddx_4!(out::VectorField{N}, u::VectorField{N}; kwargs...) where {N} =
    (for n in 1:N; ddx_4!(out[n], u[n]; kwargs...); end; return out)

"""
    add_homogeneous_laplacian!(out::FTField, u::FTField)
    add_homogeneous_laplacian!(out::VectorField, u::VectorField)

Add the homogeneous Laplacian contribution of `u` to `out`:

    out[mode] -= (∑_{d∈spatial_fft_dims(g)} (wavenumber_scale(g, d) · n_d)²) · u[mode]

Call after computing the non-homogeneous (e.g. wall-normal) second derivative.
If the grid includes a transformed logical time coordinate, that direction is
not part of the spatial Laplacian.
"""
function add_homogeneous_laplacian!(out::FTField{G}, u::FTField{G}) where {T, D, AXES, FFT_DIMS_ORDER, G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}}
    g = grid(u)

    # `to_wavenumber_vector(g, Ih)` returns k in full FFT_DIMS_ORDER order.
    # Keep `scales` in that same order.  A transformed logical time coordinate
    # is homogeneous, but it is not in `spatial_fft_dims(g)`, so give it zero
    # scale rather than building a second filtered index mapping.
    H = spatial_fft_dims(g)
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
not in [`fft_dims`](@ref), such as wall-normal collocation directions.  The
full spatial Laplacian is the sum of this contribution and the homogeneous
spectral contribution from [`add_homogeneous_laplacian!`](@ref).
"""
inhomogeneous_laplacian!(out::FTField, u::FTField) = throw(NotImplementedError(out, u))

"""
    laplacian!(out::FTField{G}, u::FTField{G}; kwargs...)
    laplacian!(out::VectorField{N}, u::VectorField{N}; kwargs...)

Compute the full Laplacian of `u` in-place, storing the result in `out`:

    out[mode] = (inhomogeneous second derivatives
                 - ∑_{d∈spatial_fft_dims(g)} (wavenumber_scale(g, d) · n_d)²) · u[mode]

Only spatial transformed directions enter the homogeneous sum; a transformed
logical time coordinate is excluded through [`spatial_fft_dims`](@ref).
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
