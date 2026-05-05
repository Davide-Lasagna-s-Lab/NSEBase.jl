# Abstract interface for computational grids that
# FTField and Field use for their construction.

"""
    AbstractGrid{T, D, H} where {T<:Real}

Abstract type that represents a generic computational grid of a
`D` dimensional domain. The variable `H` is a tuple of values
for the directions that are statistically homogeneous and are
transformed using [`FFTPlans`](@ref).

# Required methods
- `points`: collocation points making up the grid
"""
abstract type AbstractGrid{T<:Real, D, H} end

Base.convert(::Type{T}, grid::AbstractGrid{T}) where {T}    = grid
Base.convert(::Type{S}, grid::AbstractGrid{T}) where {S, T} = throw(NotImplementedError(grid, S))
   Base.size(           grid::AbstractGrid)                 = throw(NotImplementedError(grid))
      points(           grid::AbstractGrid; dealias=false)  = throw(NotImplementedError(grid))
    fft_norm(           grid::AbstractGrid)                 = throw(NotImplementedError(grid))

transform_size(grid::AbstractGrid{T, D, H}) where {T, D, H} = (shape = size(grid); ntuple(d->d==H[1] ? (shape[d] >> 1) + 1 : shape[d], length(shape)))
