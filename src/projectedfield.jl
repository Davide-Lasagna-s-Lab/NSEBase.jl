# Field of modal coefficinets for a vectorfield projected onto a set of modes.

# ---------------------- #
# projected vector field #
# ---------------------- #
struct ProjectedField{S, T, D, A} <: AbstractArray{T, D}
    data::Array{T, D}
    modes::A

    function ProjectedField(::Type{S}, data::Array{T, D}, modes::A) where {S<:AbstractScalarField, T, D, A}
        size(data) == size(modes)[2:end] || throw(ArgumentError("number of modes available not compatible with data"))
        length(size(modes)) - 1 == D || throw(ArgumentError("dimension of data and modes are not compatible"))
        new{S, T, D, A}(data, modes)
    end
end
ProjectedField(u::S,                 modes) where {S}    = ProjectedField(S, zeros(eltype(u),    size(modes, 2), hsize(u)...), modes)
ProjectedField(u::VectorField{N, S}, modes) where {S, N} = ProjectedField(S, zeros(eltype(u[1]), size(modes, 2), hsize(u)...), modes)


# --------------- #
# utility methods #
# --------------- #
modes(a::ProjectedField) = a.modes


# ----------------- #
# interface methods #
# ----------------- #
Base.IndexStyle(::Type{<:ProjectedField})                            = Base.IndexLinear()
Base.parent(a::ProjectedField)                                       = a.data
Base.eltype(::ProjectedField{G, T}) where {G, T}                     = T
Base.size(a::ProjectedField)                                         = size(parent(a))
# ! this method breakes the implicit contract that the eltype of the underlying data is the same as the eltype prescribed by the type `S`
Base.similar(a::ProjectedField{S}, ::Type{T}=eltype(a)) where {S, T} = ProjectedField(S, similar(parent(a), T), modes(a))
Base.copy(a::ProjectedField{S}) where {S}                            = ProjectedField(S, copy(parent(a)), modes(a))
Base.zero(a::ProjectedField{S}) where {S}                            = ProjectedField(S, zero(parent(a)), modes(a))
Base.abs(a::ProjectedField{S}) where {S}                             = (b = zero(a); b .= abs.(a); return b)


# ---------------- #
# indexing methods #
# ---------------- #
# linear indexing
Base.@propagate_inbounds function Base.getindex(u::ProjectedField, i::Int)
    @boundscheck checkbounds(parent(u), i)
    @inbounds val = parent(u)[i]
    return val
end
Base.@propagate_inbounds function Base.setindex!(u::ProjectedField, val, i::Int)
    @boundscheck checkbounds(parent(u), i)
    @inbounds parent(u)[i] = val
    return val
end


# ------------------- #
# dot product methods #
# ------------------- #
LinearAlgebra.dot(a::ProjectedField, b::ProjectedField) = throw(NotImplementedError(a, b))
LinearAlgebra.norm(a::ProjectedField) = sqrt(dot(a, a))


# ---------------- #
# galerkin methods #
# ---------------- #
# ! required !
project!(a::ProjectedField, u::VectorField) = throw(NotImplementedError(a, u))
project(u::VectorField, modes) = project!(ProjectedField(u, modes), u)

# ! required !
expand!(u::VectorField, a::ProjectedField) = throw(NotImplementedError(u, a))


# ------------------ #
# derivative methods #
# ------------------ #
# ! required !
dds!(out::ProjectedField, a::ProjectedField) = throw(NotImplementedError(out, a))
