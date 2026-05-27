# Type aliases and compile-time axis utilities.
#
# `FieldType` is a Union used by the broadcasting hooks.
#
# `homogeneous_axes` and `inhomogeneous_axes` are @generated functions that
# return axis tuples for the FFT (homogeneous) and non-FFT (inhomogeneous)
# storage dimensions of a field, respectively.  Both are resolved at compile
# time from the grid type parameters and produce zero-allocation calls at
# runtime.  They are used:
#   - `homogeneous_axes` in `shift!` (outer loop over wavenumbers) and in
#     various derivative / indexing helpers;
#   - `inhomogeneous_axes` in `shift!` (inner loop for the scalar per-element
#     update that avoids allocating a temporary slice).

"""
    FieldType

Union of all concrete field types that share a common BroadcastStyle override:
`FTField`, `Field`, `VectorField`, and `ProjectedField`.

Used internally by the broadcasting hooks so that a single `BroadcastStyle`
registration covers all field types.
"""
const FieldType = Union{FTField,Field,VectorField,ProjectedField}

"""
    homogeneous_axes(grid, u) -> Tuple

Return a tuple of `axes(u, d)` for each storage dimension `d` in
`FFT_DIMS_ORDER` — the homogeneous (FFT) dimensions of the grid.

The result is suitable for `CartesianIndices(homogeneous_axes(g, u))` to
iterate over all spectral wavenumber indices without touching the
inhomogeneous (quadrature) dimension(s).
"""
@generated function homogeneous_axes(::AbstractGrid{<:Any,D,AXES,FFT_DIMS_ORDER}, u::Union{ProjectedField,FTField}) where {D,AXES,FFT_DIMS_ORDER}
    return Expr(:tuple, (:(axes(u, $d)) for d in FFT_DIMS_ORDER)...)
end

"""
    inhomogeneous_axes(grid, u) -> Tuple

Return a tuple of `axes(u, d)` for each storage dimension `d` that is
**not** in `FFT_DIMS_ORDER` — the inhomogeneous (non-FFT, e.g. Chebyshev)
dimensions of the grid.  Complementary to [`homogeneous_axes`](@ref).

Used as the inner loop range in [`shift!`](@ref) so that each inhomogeneous
index is updated via a scalar `parent[I...] *= phase` with no temporary
slice allocation.
"""
@generated function inhomogeneous_axes(::AbstractGrid{<:Any,D,AXES,FFT_DIMS_ORDER}, u::Union{ProjectedField,FTField}) where {D,AXES,FFT_DIMS_ORDER}
    inh_dims = [d for d in 1:D if d ∉ FFT_DIMS_ORDER]
    return Expr(:tuple, (:(axes(u, $d)) for d in inh_dims)...)
end