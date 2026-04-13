# Fourier transformed scalar field.

struct FTField{G<:AbstractGrid, A<:AbstractArray, T, N} <: AbstractArray{Complex{T}, N}
    grid::G
    data::A

    FTField(grid::G, data::A) where {D, H, T, N, G<:AbstractGrid{D, H, T}, A<:AbstractArray{<:Any, N}} =
        # TODO: make sure array shape is compatible with grid
        new{G, A, T, N}(grid, Complex{T}.(data))
end

# construct from standard array and sanitise input
FTField(grid::G, data::Array{<:Any, N}) where {D, H, T, G<:AbstractGrid{D, H, T}, N} =
    FTField(grid, normalise_mean!(apply_symmetry!(data)))

# ! not the nicest way to do this, but at least doesn't require any extra machinary to achieve
FTField(grid::AbstractGrid{D, H, T}) where {D, H, T} = FFT(Field(grid; dealias=false))

Base.IndexStyle(::Type{<:FTField})                                                = Base.IndexLinear()
Base.parent(u::FTField)                                                           = u.data
Base.eltype(::FTField{<:AbstractGrid{D, H, T}}) where {D, H, T}                   = Complex{T}
Base.similar(u::FTField{<:AbstractGrid{D, H, T}}, ::Type{S}=T) where {D, H, T, S} = FTField(similar(grid(u), S), zero(parent(u)))
Base.size(u::FTField)                                                             = size(parent(u))
Base.copy(u::FTField)                                                             = (v = similar(u); parent(v) .= parent(u); return v)
Base.zero(u::FTField)                                                             = (v = similar(u); parent(v) .= zero(T)  ; return v)

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


# --------------- #
# utility methods #
# --------------- #
function apply_symmetry!(data)

end

@generated function normalise_mean!(data::Array{<:Any, N}, ::Val{H}) where {N, H}
    indxs = [n ∈ H ? 1 : Colon() for n in 1:N]
    slice = :(data[$(indxs...)])
    return :(@views $slice .= real.($slice); return data)
end
