# This file contains the concrete implementation of the vector fields based
# based on the abstract scalar field defined elsewhere.

"""
    VectorField{Int, <:AbstractScalarField}([elements])

Subtype of vectors with elements that are subtypes of the AbstractScalarField.
"""
struct VectorField{N, S} <: AbstractVector{S}
    elements::NTuple{N, S}

    function VectorField(elements::Vararg{S, N}) where {S<:AbstractScalarField, N}
        new{N, S}(elements)
    end
end


# ! required !
add_base!(u::VectorField, base) = throw(NotImplementedError(u, base))


# ------------------- #
# constructor methods #
# ------------------- #
VectorField(u::S, ::Type{T}=Float64; N::Int=3) where {S, T} = VectorField([similar(u) for _ in 1:N]...)


# ------------- #
# array methods #
# ------------- #
Base.parent(q::VectorField)                                          = q.elements
Base.IndexStyle(::Type{<:VectorField})                               = Base.IndexLinear()
Base.getindex(q::VectorField, i::Int)                                = parent(q)[i]
Base.setindex!(q::VectorField{N, F}, v::F, i::Int) where {N, F}      = (parent(q)[i] .= v; return v)
Base.size(::VectorField{N}) where {N}                                = (N,)
Base.length(::VectorField{N}) where {N}                              = N
Base.eltype(::VectorField{N, F}) where {N, F}                        = F
Base.similar(q::VectorField{N}, ::Type{T}=eltype(q[1])) where {N, T} = VectorField([similar(q.elements[n], T) for n in 1:N]...)
Base.copy(q::VectorField{N}) where {N}                               = VectorField([copy(q.elements[n]) for n in 1:N]...)
Base.zero(q::VectorField{N}) where {N}                               = VectorField([zero(q.elements[n]) for n in 1:N]...)


# ------------ #
# norm methods #
# ------------ #
LinearAlgebra.dot(q::VectorField{N}, p::VectorField{N}) where {N} = sum(dot(q[i], p[i]) for i in 1:N) # TODO: check output type is same as underlying data type
LinearAlgebra.norm(q::VectorField) = sqrt(dot(q, q))
