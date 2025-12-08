# This file contains the abstract type and function definitions required for a
# scalar field.

"""
    AbstractScalarField

A scalar field defined over a finite domain.
"""
abstract type AbstractScalarField{DIM, T<:Number} <: AbstractArray{T, DIM} end


# ! required !
hsize(u::AbstractScalarField) = throw(NotImplementedError(u))
grid_size(u::AbstractScalarField) = throw(NotImplementedError(u)) # required for FFT plans constructor to work

# --------------- #
# array interface #
# --------------- #
# ! required !
Base.parent(u::AbstractScalarField) = throw(NotImplementedError(u))

# ! required !
Base.similar(u::AbstractScalarField, ::Type{T}) where {T} = throw(NotImplementedError(u))

Base.size(u::AbstractScalarField) = size(parent(u))
Base.IndexStyle(::Type{<:AbstractScalarField}) = Base.IndexLinear()
Base.copy(u::AbstractScalarField) = (v = similar(u); v .= u; return v)
Base.zero(u::AbstractScalarField{D, T}) where {D, T} = (v = similar(u); v .= zero(T); return v)
Base.abs(u::AbstractScalarField) = (v = zero(u); v .= abs.(u); return v)

Base.@propagate_inbounds function Base.getindex(u::AbstractScalarField, i)
    @boundscheck checkbounds(parent(u), i)
    @inbounds v = parent(u)[i]
    return v
end

Base.@propagate_inbounds function Base.setindex!(u::AbstractScalarField, v, i)
    @boundscheck checkbounds(parent(u), i)
    @inbounds parent(u)[i] = v
    return v
end


# ------------ #
# norm methods #
# ------------ #
# * optional *
LinearAlgebra.dot(u::S, v::S) where {S<:AbstractScalarField} = throw(NotImplementedError(u, v))
LinearAlgebra.norm(p::AbstractScalarField) = sqrt(dot(p, p))
