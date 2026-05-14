# Field of modal coefficinets for a vectorfield projected onto a set of modes.

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
- `project(u::VectorField, modes)`
"""
struct ProjectedField{G<:AbstractGrid, M, A<:AbstractArray, T, D} <: AbstractArray{Complex{T}, D}
    grid::G
    data::A
    modes::M

    ProjectedField(grid::G, data::A, modes::M) where {T, D, G<:AbstractGrid{T, D}, A<:AbstractArray{<:Any, D}, M} = begin
        # ProjectedField storage is `(mode, fft_dims...)`, not physical grid
        # storage.  The rfft dimension is therefore axis 2, followed by the
        # remaining transformed axes in FFT order.
        new{G, M, A, T, D}(grid, Complex{T}.(normalise_mean!(apply_symmetry!(data, fft_dims(grid)), fft_dims(grid))), modes)
    end
end
ProjectedField(grid::AbstractGrid{T}, data::AbstractArray, modes) where {T} = ProjectedField(grid, Complex{T}.(data), modes)

ProjectedField(grid::AbstractGrid{T, D, H}, modes) where {T, D, H} =
    ProjectedField(grid, zeros(Complex{T}, no_of_modes(modes), transform_size(grid)[collect(fft_dims(grid))]...), modes)
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
# Linear indexing — delegates straight to the underlying data array.
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

"""
    dot(a::ProjectedField, b::ProjectedField)

Inner product of two projected fields, exploiting the rfft Hermitian symmetry.
Modes with rfft index `> 1` (wavenumber `nx > 0`) are stored once but represent
both `+nx` and `-nx`, so they contribute with weight 2; the `nx = 0` plane has
weight 1.  The result is divided by 2 to account for the double-counting of
signed-FFT pairs `(nz, nt)` and `(-nz, -nt)` that both appear in storage.
"""
function LinearAlgebra.dot(a::ProjectedField{G}, b::ProjectedField{G}) where {G<:AbstractGrid}
    s = zero(real(eltype(a)))
    for_each_mode(grid(a)) do args...
        w = first(args) == 1 ? 1 : 2
        for m in axes(a, 1)
            @inbounds s += w * real(LinearAlgebra.dot(parent(a)[m, args...], parent(b)[m, args...]))
        end
    end
    return s / 2
end

"""
    norm(a::ProjectedField)

Norm induced from [`dot(a::ProjectedField, b::ProjectedField)`](@ref).
"""
LinearAlgebra.norm(a::ProjectedField) = sqrt(dot(a, a))


# ---------------- #
# galerkin methods #
# ---------------- #
# TODO: there might be a better interface that defines the vectorfield project on top of an FTField project method?
project!(a::ProjectedField, u::VectorField) = throw(NotImplementedError(a, u))
project(u::VectorField, modes) = project!(ProjectedField(grid(u), modes), u)

expand!(u::VectorField, a::ProjectedField) = throw(NotImplementedError(u, a))
expand(a::ProjectedField) = expand!(VectorField(grid(a), FTField), a)
