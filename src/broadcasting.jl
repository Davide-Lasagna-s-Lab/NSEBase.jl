# This file contains the interface definitions required to get broadcasting
# as desired for concrete scalar and vector fields. Mainly this is to ensure
# that broadcasting on vector fields propagates into the underlying scalar
# field.

const FieldType = Union{FTField, Field, VectorField, ProjectedField}

Base.BroadcastStyle(S::Type{<:FieldType}) = Broadcast.ArrayStyle{S}()
Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{F}}, ::Type{T}) where {F<:FieldType, T} = similar(find_field(bc), T)

find_field(bc::Broadcast.Broadcasted) = find_field(bc.args)
find_field(args::Tuple)               = find_field(find_field(args[1]), Base.tail(args))
find_field(u::FieldType, rest)        = u
find_field(::Any, rest)               = find_field(rest)
find_field(x)                         = x
find_field(::Tuple{})                 = nothing

# Vector field broadcasting applies the broadcast expression component-wise
# into the underlying scalar fields.
function Base.copy(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{V}}) where {N, V<:VectorField{N}}
    dest = similar(bc, eltype(find_field(bc)[1]))
    for n in 1:N
        copyto!(dest[n], unpack(bc, n))
    end
    return dest
end

function Base.copyto!(dest::VectorField{N}, bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{V}}) where {N, V<:VectorField{N}}
    for n in 1:N
        copyto!(dest[n], unpack(bc, n))
    end
    return dest
end

# Special case: scalar assignment fills every component of the vector field.
function Base.copyto!(dest::VectorField{N}, bc::Broadcast.Broadcasted{<:Broadcast.AbstractArrayStyle{0}}) where {N}
    for n in 1:N
        fill!(dest[n], bc.args[1][])
    end
    return dest
end

@inline unpack(bc::Broadcast.Broadcasted, n) = Broadcast.Broadcasted(bc.f, _unpack(n, bc.args))
@inline unpack(x::Any,                    n) = x
@inline unpack(x::VectorField,            n) = x.elements[n]

@inline _unpack(n, args::Tuple) = (unpack(args[1], n), _unpack(n, Base.tail(args))...)
@inline _unpack(n, args::Tuple{Any}) = (unpack(args[1], n),)
@inline _unpack(::Any, args::Tuple{}) = ()
