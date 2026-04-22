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

 similar(grid::AbstractGrid{T}, ::Type{S}=T) where {T, S} = throw(NotImplementedError(grid, S))
    size(grid::AbstractGrid)                              = throw(NotImplementedError(grid))
  points(grid::AbstractGrid; dealias=false)               = throw(NotImplementedError(grid))
fft_norm(grid::AbstractGrid)                              = throw(NotImplementedError(grid))
