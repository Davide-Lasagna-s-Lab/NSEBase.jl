# Abstract interface for computational grids that
# FTField and Field use for their construction.

"""
    AbstractGrid{T, D, AXES, ORDER} where {T<:Real}

Abstract type that represents a generic computational grid of a
`D`-dimensional domain.

Type parameters:
- `T`: scalar real type used by physical-space fields on this grid.
- `D`: number of array dimensions.
- `AXES`: length-4 tuple `(x_dim, y_dim, z_dim, t_dim)` giving the array
  dimension associated with the streamwise, wall-normal, spanwise, and
  temporal coordinates. If a direction isn't required, then a nothing
  is given for that direction.
- `ORDER`: tuple of statistically homogeneous array dimensions. These are
  transformed by FFTs; `ORDER[1]` is the rfft dimension.

# Examples

A channel-flow grid stored as `(t, x, z, y)` uses:

```julia
struct ChannelGrid{T} <: AbstractGrid{T, 4, (2, 4, 3, 1), (1, 2, 3)} end
```

A streamwise independent square-duct flow stored as `(y, z, t)` uses:

```julia
struct DuctGrid{T} <: AbstractGrid{T, 3, (nothing, 1, 2, 3), (3,)}
```

Here `t` is the first array dimension and is the rfft direction. The direction
`x` is array dimension 2, `z` is array dimension 3, both of which are ordinary
FFT directions, and the inhomogeneous wall-normal direction `y` is array
dimension 4.

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
abstract type AbstractGrid{T<:Real, D, AXES, ORDER} end



# ---------------------- #
# compile-time accessors #
# ---------------------- #
"""
    x_dim(grid::AbstractGrid) -> Int

Return the array dimension corresponding to the streamwise coordinate.
"""
x_dim(::AbstractGrid{<:Any, <:Any, AXES}) where {AXES} = AXES[1]

"""
    y_dim(grid::AbstractGrid) -> Int

Return the array dimension corresponding to the wall-normal coordinate.
"""
y_dim(::AbstractGrid{<:Any, <:Any, AXES}) where {AXES} = AXES[2]

"""
    z_dim(grid::AbstractGrid) -> Int

Return the array dimension corresponding to the spanwise coordinate.
"""
z_dim(::AbstractGrid{<:Any, <:Any, AXES}) where {AXES} = AXES[3]

"""
    t_dim(grid::AbstractGrid) -> Int

Return the array dimension corresponding to time.
"""
t_dim(::AbstractGrid{<:Any, <:Any, AXES}) where {AXES} = AXES[4]

"""
    fft_dims(grid::AbstractGrid) -> Tuple

Return all transformed dimensions in FFT order, i.e. `(spatial_hom_dims(grid)...,
t_dim(grid))`.
"""
fft_dims(::AbstractGrid{<:Any, <:Any, <:Any, ORDER}) where {ORDER} = ORDER

"""
    inhomogeneous_dims(grid::AbstractGrid) -> Tuple

Return the array dimensions that are NOT transformed by FFTs, i.e. the
complement of `fft_dims(grid)` within `1:D`.  These are the directions over
which quadrature weights are needed for inner products.
"""
inhomogeneous_dims(::AbstractGrid{<:Any, D, <:Any, ORDER}) where {D, ORDER} =
    Tuple(d for d in 1:D if d ∉ ORDER)


# ------------------ #
# required interface #
# ------------------ #
"""
    Base.convert(::Type{S}, grid::AbstractGrid{T}) -> AbstractGrid{S}

Convert `grid` to scalar real type `S`.
"""
Base.convert(::Type{S}, grid::AbstractGrid{T}) where {S, T} = throw(NotImplementedError(grid, S))

"""
    Base.size(grid::AbstractGrid) -> Dims{D}
    Base.size(grid::AbstractGrid, dim::Int) -> Int

Return the physical-space array size associated with `grid`, in array-dimension
order.  This is required by `Field`, `FTField`, `FFTPlans`, and generated
operators. Optionally provide a `dim` argument to extract the size of the grid
in a particular direction.
"""
Base.size(grid::AbstractGrid) = throw(NotImplementedError(grid))
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
    wavenumber_scale(grid::AbstractGrid, dim::Int) -> Real

Return the physical wavenumber scaling factor for homogeneous dimension `dim`.
For a spatial direction with period `L` the factor is `2π/L`; for a unit-period
temporal direction, return `1`.

Downstream packages must extend this for each dimension in `fft_dims(grid)`.
"""
wavenumber_scale(grid::AbstractGrid, dim::Int) = throw(NotImplementedError(grid, dim))

"""
    weights(grid::AbstractGrid) -> AbstractArray

Return the quadrature weights for the inhomogeneous directions of `grid`.
The returned array has one axis per dimension in `inhomogeneous_dims(grid)`,
in ascending array-dimension order: a `Vector` when there is one inhomogeneous
direction, a `Matrix` when there are two, and so on.

Used by `dot(u::FTField, v::FTField)` to weight each inhomogeneous index
combination.
"""
weights(grid::AbstractGrid) = throw(NotImplementedError(grid))


# ------------------ #
# optional interface #
# ------------------ #
"""
    growto(grid::AbstractGrid, target_size)

Return an equivalent grid with a new homogeneous resolution.

This is optional.  Packages should only implement it if they support
resolution-changing transforms such as `FFT(u, target_size)` and
`IFFT(û, target_size)`.
"""
growto(grid::AbstractGrid, target_size) = throw(NotImplementedError(grid, target_size))


# ------------------ #
# derived quantities #
# ------------------ #
"""
    transform_size(g::AbstractGrid)

Size of the spectral (FTField) array: `ORDER[1]` (rfft) becomes `(N÷2)+1`,
all other dimensions are unchanged.
"""
transform_size(g::AbstractGrid{<:Any, D, <:Any, ORDER}) where {D, ORDER} =
    ntuple(d -> d == ORDER[1] ? (size(g, d) >> 1) + 1 : size(g, d), D)

"""
    fft_norm(g::AbstractGrid)

Number of grid points in each FFT dimension `ORDER`. Used to normalise
forward transforms. Default: reads from `size(g)`.
"""
fft_norm(g::AbstractGrid) = map(d -> size(g, d), fft_dims(g))
