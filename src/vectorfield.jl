# This file contains the concrete implementation of the vector fields based
# based on the abstract scalar field defined elsewhere.

"""
    VectorField{N, S} where {S<:Union{FTField, Field}}

A vectorfield made up of an ordered list of scalar fields either
represented using `FTField` or `Field`.

# Fields
- `elements`: scalar field components of the vectorfield
"""
struct VectorField{N, S} <: AbstractVector{S}
    elements::NTuple{N, S}

    function VectorField(elements::Vararg{S, N}) where {S<:Union{FTField, Field}, N}
        new{N, S}(elements)
    end
end

# hsize(u::VectorField) = hsize(u[1]) # ! necessary?

# ! required !
add_base!(u::VectorField, base) = throw(NotImplementedError(u, base))

grid(u::VectorField) = grid(u[1])


# ------------------- #
# constructor methods #
# ------------------- #
VectorField(g::AbstractGrid, ::Type{S}=FTField; N::Int=3, kwargs...) where {S} = VectorField([S(g; kwargs...) for _ in 1:N]...)
VectorField(g::AbstractGrid, funcs...         ; dealias::Bool=false)           = VectorField([Field(g, f; dealias=dealias) for f in funcs]...)


# ------------- #
# array methods #
# ------------- #
Base.IndexStyle(::Type{<:VectorField})                               = Base.IndexLinear()
Base.parent(q::VectorField)                                          = q.elements
Base.getindex(q::VectorField, i::Int)                                = parent(q)[i]
Base.setindex!(q::VectorField{N, F}, v::F, i::Int) where {N, F}      = (parent(q)[i] .= v; return v)
Base.size(::VectorField{N}) where {N}                                = (N,)
Base.eltype(::VectorField{N, F}) where {N, F}                        = F
Base.similar(q::VectorField{N}, ::Type{T}=eltype(q[1])) where {N, T} = VectorField([Base.similar(q.elements[n], T) for n in 1:N]...)
Base.copy(q::VectorField{N}) where {N}                               = VectorField([copy(q.elements[n]) for n in 1:N]...)
Base.zero(q::VectorField{N}) where {N}                               = VectorField([zero(q.elements[n]) for n in 1:N]...)


# ------------ #
# norm methods #
# ------------ #
LinearAlgebra.dot(q::VectorField{N}, p::VectorField{N}) where {N} = sum(dot(q[n], p[n]) for n in 1:N)
LinearAlgebra.norm(q::VectorField)                                = sqrt(dot(q, q))
