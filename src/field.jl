# Physical representation of scalar field.

struct Field{G<:AbstractGrid, A<:AbstractArray, T, N} <: AbstractArray{T, N}
    grid::G
    data::A

    Field(grid::G, data::A) where {D, H, T, N, G<:AbstractGrid{D, H, T}, A<:AbstractArray{<:Any, N}} =
        new{G, A, T, N}(grid, T.(data))
end

Field(grid::AbstractGrid{D, H, T}, func; dealias=false) where {D, H, T} =
    Field(grid, func.(points(grid, dealias)))

Field(grid::AbstractGrid{D, H, T}; dealias=false) where {D, H, T} =
    Field(grid, pt->zero(T); dealias=dealias)

Base.IndexStyle(::Type{<:Field})                                                = Base.IndexLinear()
Base.parent(u::Field)                                                           = u.data
Base.eltype(::Field{<:AbstractGrid{D, H, T}}) where {D, H, T}                   = T
Base.similar(u::Field{<:AbstractGrid{D, H, T}}, ::Type{S}=T) where {D, H, T, S} = Field(similar(grid(u), S), zero(parent(u)))
Base.size(u::Field)                                                             = size(parent(u))
Base.copy(u::Field)                                                             = (v = similar(u); parent(v) .= parent(u); return v)
Base.zero(u::Field)                                                             = (v = similar(u); parent(v) .= zero(T)  ; return v)

# linear indexing
Base.@propagate_inbounds function Base.getindex(u::FTField, i)
    @boundscheck checkbounds(parent(u), i)
    @inbounds v = parent(u)[i]
    return v
end

Base.@propagate_inbounds function Base.setindex!(u::FTField, v, i)
    @boundscheck checkbounds(parent(u), i)
    @inbounds parent(u)[i] = v
    return v
end

grid(u::FTField) = u.grid
