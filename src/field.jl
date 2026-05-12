# Physical representation of scalar field.

"""
    Field{G} where {G<:AbstractGrid}

Physical representation of a scalar field on a concrete
sub-type of `AbstractGrid`.

# Fields
- `grid`: concrete instance of `AbstractGrid`
- `data`: value of scalar field at collocation points from `points(grid)`
"""
struct Field{G<:AbstractGrid, A<:AbstractArray, T, D} <: AbstractArray{T, D}
    grid::G
    data::A

    Field(grid::G, data::A) where {T, D, H, G<:AbstractGrid{T, D, H}, A<:AbstractArray{T, D}} =
        new{G, A, T, D}(grid, data)
end
Field(grid::AbstractGrid{T}, data::AbstractArray) where {T} = Field(grid, T.(data))

# construct from functions
Field(grid::AbstractGrid{T}, func::Function; dealias=false) where {T} = Field(grid, T.(func.(points(grid, dealias=dealias)...)))
Field(grid::AbstractGrid{T}                ; dealias=false) where {T} = Field(grid, (pts...)->zero(T); dealias=dealias)

Base.IndexStyle(::Type{<:Field})                                    = Base.IndexLinear()
Base.parent(u::Field)                                               = u.data
Base.eltype(::Field{<:AbstractGrid{T}}) where {T}                   = T
Base.similar(u::Field{<:AbstractGrid{T}}, ::Type{S}=T) where {T, S} = Field(convert(S, grid(u)), zero(parent(u)))
Base.size(u::Field)                                                 = Base.size(parent(u))
Base.copy(u::Field)                                                 = (v = Base.similar(u); parent(v) .= parent(u); return v)
Base.zero(u::Field{<:AbstractGrid{T}}) where {T}                    = (v = Base.similar(u); parent(v) .= zero(T)  ; return v)

# linear indexing
Base.@propagate_inbounds function Base.getindex(u::Field, i)
    @boundscheck checkbounds(parent(u), i)
    @inbounds v = parent(u)[i]
    return v
end

Base.@propagate_inbounds function Base.setindex!(u::Field, v, i)
    @boundscheck checkbounds(parent(u), i)
    @inbounds parent(u)[i] = v
    return v
end

grid(u::Field) = u.grid
