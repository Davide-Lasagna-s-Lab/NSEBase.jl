# This file contains the interface definitions required to get broadcasting
# as desired for concrete scalar and vector fields. Mainly this is to ensure
# that broadcasting on vector fields propagates into the underlying scalar
# field.

Base.BroadcastStyle(u::Type{<:AbstractScalarField}) = Broadcast.ArrayStyle{u}()
Base.similar(bc::Base.Broadcast.Broadcasted{Broadcast.ArrayStyle{S}}, ::Type{T}) where {T, S<:AbstractScalarField} = similar(find_field(bc), T)

Base.BroadcastStyle(::Type{<:ProjectedField}) = Broadcast.ArrayStyle{ProjectedField}()
Base.similar(bc::Base.Broadcast.Broadcasted{Broadcast.ArrayStyle{ProjectedField}}, ::Type{T}) where {T} = similar(find_field(bc), T)

Base.BroadcastStyle(::Type{<:VectorField{N}}) where {N} = Broadcast.ArrayStyle{VectorField{N}}()
Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{F}}, ::Type{T}) where {F<:VectorField, T} = similar(find_field(bc), T)

Base.BroadcastStyle(::Broadcast.ArrayStyle{<:VectorField}, ::Broadcast.ArrayStyle{<:AbstractScalarField}) = Broadcast.ArrayStyle{V}()

find_field(bc::Broadcast.Broadcasted)    = find_field(bc.args)
find_field(args::Tuple)                  = find_field(find_field(args[1]), Base.tail(args))
find_field(u::AbstractScalarField, rest) = u
find_field(u::VectorField, rest)         = u
find_field(a::ProjectedField, rest)      = a
find_field(::Any, rest)                  = find_field(rest)
find_field(x)                            = x
find_field(::Tuple{})                    = nothing

# vector field broadcasting into underlying field
function Base.copy(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{V}}) where {N, V<:VectorField{N}}
    dest = similar(bc, eltype(find_field(bc)[1]))
    for i in 1:N
        copyto!(dest[i], unpack(bc, i))
    end
    return dest
end

function Base.copyto!(dest::VectorField{N}, bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{V}}) where {N, V<:VectorField{N}}
    for i in 1:N
        copyto!(dest[i], unpack(bc, i))
    end
    return dest
end

@inline unpack(bc::Broadcast.Broadcasted, i) = Broadcast.Broadcasted(bc.f, _unpack(i, bc.args))
@inline unpack(x::Any,                    i) = x
@inline unpack(x::VectorField,            i) = x.elements[i]

@inline _unpack(i, args::Tuple) = (unpack(args[1], i), _unpack(i, Base.tail(args))...)
@inline _unpack(i, args::Tuple{Any}) = (unpack(args[1], i),)
@inline _unpack(::Any, args::Tuple{}) = ()
