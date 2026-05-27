# This file contains type definitions

"""
    FieldType

Union of all concrete field types that share a common BroadcastStyle override:
`FTField`, `Field`, `VectorField`, and `ProjectedField`.

Used internally by the broadcasting hooks so that a single `BroadcastStyle`
registration covers all field types.
"""
const FieldType = Union{FTField,Field,VectorField,ProjectedField}

"""
    homogeneous_axes(u::FieldType, grid) -> Tuple

Return the tuple of axes corresponding to the homogeneous dimensions of `grid` for a field of
type `FieldType`. This is used to generate CartesianIndices over the homogeneous dimensions.
"""
@generated function homogeneous_axes(::AbstractGrid{<:Any,D,AXES,FFT_DIMS_ORDER}, u::Union{ProjectedField,FTField}) where {D,AXES,FFT_DIMS_ORDER}
    return Expr(:tuple, (:(axes(u, $d)) for d in FFT_DIMS_ORDER)...)
end