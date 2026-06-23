# Broadcasting interface for all field types (Field, FTField, VectorField, ProjectedField).
#
# Julia's broadcasting machinery (AbstractArray + BroadcastStyle) is hooked here so
# that expressions like `u .* v`, `@. f(u)`, `u .= v .+ w` work naturally on all
# field types and produce a result of the same field type rather than a plain Array.
#
# For scalar fields (Field, FTField, ProjectedField) the default IndexLinear
# broadcasting inherited from AbstractArray is sufficient — no override is needed.
#
# For VectorField, broadcasting must propagate component-wise into the N scalar
# subfields rather than treating the VectorField itself as a 1-D array of N
# scalars.  This is done by implementing `copy` and `copyto!` for the VectorField
# BroadcastStyle, which unpack each VectorField argument into its n-th component
# and broadcast the scalar expression independently into each destination
# component.  The helper `unpack` / `_unpack` walks the broadcast argument tree
# recursively, replacing every VectorField node with its n-th element.

"""
    FieldType

Union of all concrete field types that share a common BroadcastStyle override:
`FTField`, `Field`, `VectorField`, and `ProjectedField`.

Used internally by the broadcasting hooks so that a single `BroadcastStyle`
registration covers all field types.
"""
const FieldType = Union{FTField, Field, VectorField, ProjectedField}

# Register a distinct BroadcastStyle for every FieldType subtype.  This is
# necessary so that Julia's broadcast fusion knows to call our custom `copy` and
# `similar` methods when a broadcast expression involves any field.
Base.BroadcastStyle(S::Type{<:FieldType}) = Broadcast.ArrayStyle{S}()

# Allocate the output of a broadcast expression: find the first FieldType in the
# argument tree and call `similar` on it, so the result has the correct grid,
# element type, and field type.
Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{F}}, ::Type{T}) where {F<:FieldType, T} = similar(find_field(bc), T)

"""
    find_field(bc) -> FieldType or nothing

Walk a `Broadcast.Broadcasted` argument tree and return the first `FieldType`
instance encountered, or `nothing` if none is found.

Used by `similar` to determine the template field for output allocation.
"""
find_field(bc::Broadcast.Broadcasted) = find_field(bc.args)
find_field(args::Tuple)               = find_field(find_field(args[1]), Base.tail(args))
find_field(u::FieldType, rest)        = u
find_field(::Any, rest)               = find_field(rest)
find_field(x)                         = x
find_field(::Tuple{})                 = nothing


"""
    copyto!(dest::Union{FTField, Field}, bc::Broadcast.Broadcasted{ArrayStyle{Union{FTField, Field}}}) -> dest

Copy the result of an `FTField` or `Field` broadcast expression into `dest`.

Lowers the broadcast expression to the parent data of the `FTField` or `Field`
such that the correct broadcast machinery is used (such as when using CUDA).
"""
function Base.copyto!(dest::FieldType, bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{F}}) where {F<:FieldType}
    copyto!(parent(dest), unpack_data(bc))
    return dest
end

"""
    unpack_data(bc) -> Broadcasted or scalar

Replace every `FTField` or `Field` node in the broadcast argument tree `bc` with
its parent data, leaving scalars and other array types unchanged.
"""
@inline unpack_data(bc::Broadcast.Broadcasted) = Broadcast.Broadcasted(bc.f, _unpack_data(bc.args))
@inline unpack_data(x::FieldType)              = parent(x)
@inline unpack_data(x::Any)                    = x

"""
    _unpack_data(args::Tuple) -> Tuple

Recursively apply [`unpack_data`](@ref) to every element of the argument tuple,
returning a new tuple of the same length with `FTField` or `Field` nodes replaced
by their parent data.
"""
@inline _unpack_data(args::Tuple)              = (unpack_data(args[1]), _unpack_data(Base.tail(args))...)
@inline _unpack_data(args::Tuple{Any})         = (unpack_data(args[1]),)
@inline _unpack_data(::Tuple{})                = ()


"""
    copy(bc::Broadcast.Broadcasted{ArrayStyle{VectorField{N}}}) -> VectorField

Materialise a broadcast expression whose result is a `VectorField`.

Allocates a fresh destination via `similar`, then delegates to the `copyto!`
method below, which applies the broadcast expression component-wise.
"""
function Base.copy(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{V}}) where {N, V<:VectorField{N}}
    dest = similar(bc, eltype(find_field(bc)[1]))
    for n in 1:N
        copyto!(dest[n], unpack(bc, n))
    end
    return dest
end

"""
    copyto!(dest::VectorField{N}, bc::Broadcast.Broadcasted{ArrayStyle{VectorField{N}}}) -> dest

Copy the result of a VectorField broadcast expression into `dest`.

Iterates over each component index `n ∈ 1:N`, unpacks every VectorField
argument to its n-th component, and materialises the resulting scalar broadcast
expression into `dest[n]`.
"""
function Base.copyto!(dest::VectorField{N}, bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{V}}) where {N, V<:VectorField{N}}
    for n in 1:N
        copyto!(dest[n], unpack(bc, n))
    end
    return dest
end

"""
    copyto!(dest::VectorField{N}, bc::Broadcast.Broadcasted{<:AbstractArrayStyle{0}}) -> dest

Special case: scalar (0-dimensional) broadcast assignment, e.g. `q .= 0.0`.

Fills every component of `dest` with the scalar value, avoiding the unpack
path which would fail for a 0-D broadcasted expression with no field arguments.
"""
Base.copyto!(dest::VectorField{N}, bc::Broadcast.Broadcasted{<:Broadcast.AbstractArrayStyle{0}}) where {N} =
    fill!(dest, bc.args[1][])

Base.fill!(dest::FieldType, x) = (fill!(parent(dest), x); return dest)
function Base.fill!(dest::VectorField{N}, x) where {N}
    for n in 1:N
        fill!(dest[n], x)
    end
    return dest
end

"""
    unpack(bc, n) -> Broadcasted or scalar

Replace every `VectorField` node in the broadcast argument tree `bc` with its
`n`-th component, leaving scalars and other array types unchanged.  Returns a
new `Broadcast.Broadcasted` value that can be materialised into a scalar field.
"""
@inline unpack(bc::Broadcast.Broadcasted, n) = Broadcast.Broadcasted(bc.f, _unpack(n, bc.args))
@inline unpack(x::Any,                    n) = x
@inline unpack(x::VectorField,            n) = x.elements[n]

"""
    _unpack(n, args::Tuple) -> Tuple

Recursively apply [`unpack`](@ref) to every element of the argument tuple,
returning a new tuple of the same length with `VectorField` nodes replaced by
their `n`-th components.
"""
@inline _unpack(n, args::Tuple) = (unpack(args[1], n), _unpack(n, Base.tail(args))...)
@inline _unpack(n, args::Tuple{Any}) = (unpack(args[1], n),)
@inline _unpack(::Any, args::Tuple{}) = ()
