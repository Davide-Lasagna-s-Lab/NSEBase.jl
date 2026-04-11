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
    for n in 1:N
        copyto!(dest[n], unpack(bc, n))
    end
    return dest
end

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
