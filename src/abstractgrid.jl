# Abstract interface for computational grids that
# FTField and Field use for their construction.

"""
    AbstractGrid{D, H, T} where {T<:AbstractFloat}

Abstract type that represents a generic computational grid of a
`D` dimensional domain. The variable `H` is a tuple of values
for the directions that are statistically homogeneous and are
transformed using [`FFTPlans`](@ref).

# Required methods
- `points`: collocation points making up the grid
"""
abstract type AbstractGrid{D, H, T<:AbstractFloat} end

points(g::AbstractGrid)                                           = throw(NotImplementedError(g))
similar(g::AbstractGrid{D, H, T}, ::Type{S}=T) where {D, H, T, S} = throw(NotImplementedError(g, S))
