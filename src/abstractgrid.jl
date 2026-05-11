# Abstract interface for computational grids that Field, FTField, FFTPlans,
# and the generic differential operators use for construction and dispatch.

# TODO: determine how to specify the order of the transformed dimensions.  
# The current convention is that the first entry of Hs is the rfft dimension, 
# and the remaining entries are ordinary FFT dimensions, including Ht. 
# This is currently hardcoded in FTField and FFTPlans, but it would be nice
# to make it part of the grid metadata.  This may not be the best idea, but
# at it is worth discussing this. I am tempted to hardcode that time is 
# transformed first and then the spatial dimensions, but this may not be
# optimal in terms of performance for some applications. 

"""
    AbstractGrid{T, D, Axes, Hs, Ht} where {T<:Real}

Abstract type that represents a generic computational grid of a
`D`-dimensional domain.

Type parameters:
- `T`: scalar real type used by physical-space fields on this grid.
- `D`: number of array dimensions.
- `Axes`: length-4 tuple `(x_dim, y_dim, z_dim, t_dim)` giving the array
  dimension associated with the streamwise, wall-normal, spanwise, and
  temporal coordinates.
- `Hs`: tuple of spatial homogeneous array dimensions. These are transformed
  by FFTs; `Hs[1]` is the rfft dimension.
- `Ht`: temporal homogeneous array dimension. This is transformed by an
  ordinary complex FFT.

The combined FFT dimensions are `(Hs..., Ht)`.

# Example

A channel-flow grid stored as `(t, x, z, y)` uses:

```julia
abstract type AbstractChannelGrid{S, T} <:
    AbstractGrid{T, 4, (2, 4, 3, 1), (2, 3), 1}
end
```

Here `x` is array dimension 2 and is the rfft direction, `z` is array
dimension 3 and is an ordinary FFT direction, `t` is array dimension 1, and
the inhomogeneous wall-normal direction `y` is array dimension 4.

# Required downstream methods

Concrete grid packages must implement:

- `Base.size(grid)` returning an `NTuple{D, Int}`.
- `points(grid; dealias=false)` returning one coordinate array per dimension,
  in array-dimension order.
- `wavenumber_scale(grid, dim)` for each transformed dimension in
  `fft_dims(grid)`.
- `Base.convert(::Type{S}, grid)` if fields should support changing scalar
  precision via `similar(field, S)`.

# Optional downstream methods

Implement `growto(grid, target_size)` if the package supports changing the
homogeneous resolution.  Implementing grid growth is also required for
`FFT(field, target_size)` and `IFFT(ftfield, target_size)`.

# Provided defaults (override only if needed)

- `Base.size(grid, dim)`: indexes `size(grid)`.
- `fft_norm(grid)`: returns `map(d -> size(grid, d), fft_dims(grid))`, the mode
  counts used to normalise forward transforms.
- `transform_size(grid)`: returns the size of the corresponding `FTField`.
"""
abstract type AbstractGrid{T<:Real, D, Axes, Hs, Ht} end

# ----------------------------------------- #
# compile-time dimension metadata accessors #
# ----------------------------------------- #
"""
    x_dim(grid::AbstractGrid) -> Int

Return the array dimension corresponding to the streamwise coordinate.
"""
x_dim(::AbstractGrid{<:Any, <:Any, Axes})              where {Axes}   = Axes[1]

"""
    y_dim(grid::AbstractGrid) -> Int

Return the array dimension corresponding to the wall-normal coordinate.
"""
y_dim(::AbstractGrid{<:Any, <:Any, Axes})              where {Axes}   = Axes[2]

"""
    z_dim(grid::AbstractGrid) -> Int

Return the array dimension corresponding to the spanwise coordinate.
"""
z_dim(::AbstractGrid{<:Any, <:Any, Axes})              where {Axes}   = Axes[3]

"""
    t_dim(grid::AbstractGrid) -> Int

Return the array dimension corresponding to time.
"""
t_dim(::AbstractGrid{<:Any, <:Any, <:Any, <:Axes})     where {Axes}    = last(Axes)

"""
    spatial_hom_dims(grid::AbstractGrid) -> Tuple

Return the spatial homogeneous array dimensions.  The first entry is the rfft
dimension; remaining entries are ordinary FFT dimensions.
"""
spatial_hom_dims(::AbstractGrid{<:Any, <:Any, <:Any, Hs}) where {Hs} = Hs

"""
    fft_dims(grid::AbstractGrid) -> Tuple

Return all transformed dimensions in FFT order, i.e. `(spatial_hom_dims(grid)...,
t_dim(grid))`.
"""
fft_dims(::AbstractGrid{<:Any, <:Any, <:Any, Hs, Ht}) where {Hs, Ht} = (Hs..., Ht)

# ------------------------------ #
# required downstream interface  #
# ------------------------------ #
"""
    Base.convert(::Type{S}, grid::AbstractGrid)

Convert `grid` to scalar real type `S`.

The identity conversion is provided.  Downstream packages should implement
non-identity conversions when their fields support `similar(field, S)`.
"""
Base.convert(::Type{T}, grid::AbstractGrid{T}) where {T}    = grid
Base.convert(::Type{S}, grid::AbstractGrid{T}) where {S, T} = throw(NotImplementedError(grid, S))

"""
    Base.size(grid::AbstractGrid) -> NTuple{D, Int}

Return the physical-space array size associated with `grid`, in array-dimension
order.  This is required by `Field`, `FTField`, `FFTPlans`, and generated
operators.
"""
Base.size(grid::AbstractGrid) = throw(NotImplementedError(grid))

"""
    Base.size(grid::AbstractGrid, dim::Int) -> Int

Return the size of `grid` along array dimension `dim`.
"""
Base.size(grid::AbstractGrid, dim::Int) = size(grid)[dim]

"""
    points(grid::AbstractGrid; dealias=false) -> Tuple

Return one coordinate array per array dimension, suitable for broadcasting a
function into a `Field`.

When `dealias=true`, transformed dimensions should use the physical padded
sizes expected by `FFTPlans(grid; dealias=true)`.
"""
points(grid::AbstractGrid; dealias=false) = throw(NotImplementedError(grid))

"""
    growto(grid::AbstractGrid, target_size)

Return an equivalent grid with a new homogeneous resolution.

This is optional.  Packages only need to implement it if they support
resolution-changing transforms such as `FFT(u, target_size)` and
`IFFT(û, target_size)`.
"""
growto(grid::AbstractGrid, target_size) = throw(NotImplementedError(grid, target_size))

"""
    wavenumber_scale(grid::AbstractGrid, dim::Int) -> Real

Return the physical wavenumber scaling factor for homogeneous dimension `dim`.
For a spatial direction with period `L` the factor is `2π/L`; for a unit-period
temporal direction, return `1`.

Downstream packages must extend this for each dimension in `fft_dims(grid)`.
"""
wavenumber_scale(grid::AbstractGrid, dim::Int) = throw(NotImplementedError(grid, dim))

# ----------------------------------- #
# derived quantities
# ----------------------------------- #

"""
    fft_norm(g::AbstractGrid)

Number of grid points in each FFT dimension `(Hs..., Ht)`. Used to normalise
forward transforms. Default: reads from `size(g)`.
"""
fft_norm(g::AbstractGrid) = map(d -> size(g, d), fft_dims(g))

"""
    transform_size(g::AbstractGrid)

Size of the spectral (FTField) array: `Hs[1]` (rfft) becomes `N÷2+1`,
all other dimensions are unchanged.
"""
transform_size(g::AbstractGrid{<:Any, D, <:Any, Hs}) where {D, Hs} =
    ntuple(d -> d == Hs[1] ? (size(g, d) >> 1) + 1 : size(g, d), D)

"""
    physical_side(g::AbstractGrid) -> Dims

Return the physical-space array size associated with `g`.
"""
physical_side(g::AbstractGrid) = size(g)
