# Field of modal coefficinets for a vectorfield projected onto a set of modes.

# ---------------------- #
# projected vector field #
# ---------------------- #
"""
    ProjectedField{G<:AbstractGrid, M}

A representation of a vector field derived from the projection
of an `FTField` onto a set of basis `modes`.

# Fields
- `grid`: concrete instance of `AbstractGrid`
- `data`: modal coefficients stored as a multi-dimensional array
- `modes`: set of basis modes

# Constructors
- `ProjectedField(grid::AbstractGrid, M, modes)`
- `ProjectedField(u::Union{FTField, Field, VectorField}, modes)`
- `project(u::VectorField)`
"""
struct ProjectedField{G<:AbstractGrid, M, A<:AbstractArray, T, D} <: AbstractArray{Complex{T}, D}
    grid::G
    data::A
    modes::M

    ProjectedField(grid::G, data::A, modes::M) where {T, D, H, G<:AbstractGrid{T, D, H}, A<:AbstractArray{<:Any, D}, M} =
        new{G, M, A, T, D}(grid, Complex{T}.(normalise_mean!(apply_symmetry!(data, H), H)), modes)
end

ProjectedField(grid::AbstractGrid{T, D, H}, modes) where {T, D, H} =
    ProjectedField(grid, zeros(Complex{T}, no_of_modes(modes), transform_size(grid)[collect(H)]...), modes)
ProjectedField(u::Union{FTField, Field, VectorField}, modes) = ProjectedField(grid(u), modes)

no_of_modes(modes) = throw(NotImplementedError(modes))

# ----------------- #
# interface methods #
# ----------------- #
Base.IndexStyle(::Type{<:ProjectedField})                                    = Base.IndexLinear()
Base.parent(a::ProjectedField)                                               = a.data
Base.eltype(::ProjectedField{<:AbstractGrid{T}}) where {T}                   = Complex{T}
Base.size(a::ProjectedField)                                                 = size(parent(a))
Base.similar(a::ProjectedField{<:AbstractGrid{T}}, ::Type{S}=T) where {S, T} = ProjectedField(convert(real(S), grid(a)), zero(parent(a)), modes(a))
Base.copy(a::ProjectedField)                                                 = ProjectedField(grid(a), copy(parent(a)), modes(a))
Base.zero(a::ProjectedField)                                                 = ProjectedField(grid(a), zero(parent(a)), modes(a))
Base.abs(a::ProjectedField)                                                  = (b = zero(a); parent(b) .= abs.(parent(a)); return b)

modes(a::ProjectedField) = a.modes
grid(a::ProjectedField)  = a.grid


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
# TODO: should be possible to just implement this as an appropriately weighted sum
# TODO: of the dot of the all the elements of each component of the dot product
LinearAlgebra.dot(a::ProjectedField, b::ProjectedField) = throw(NotImplementedError(a, b))
LinearAlgebra.norm(a::ProjectedField) = sqrt(dot(a, a))


# ---------------- #
# galerkin methods #
# ---------------- #
# TODO: there might be a better interface that defines the vectorfield project on top of an FTField project method?
project!(a::ProjectedField, u::VectorField) = throw(NotImplementedError(a, u))
project(u::VectorField, modes) = project!(ProjectedField(grid(u), modes), u)

expand!(u::VectorField, a::ProjectedField) = throw(NotImplementedError(u, a))
expand(a::ProjectedField) = expand!(VectorField(grid(a), FTField), a)


# ------------------ #
# derivative methods #
# ------------------ #
dds!(out::ProjectedField, a::ProjectedField) = throw(NotImplementedError(out, a))
