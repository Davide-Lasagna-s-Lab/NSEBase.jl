# Abstract interface for all computational grids in NSEBase.
#
# `AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}` is the single point of coupling between
# field types (`FTField`, `Field`, `VectorField`, `ProjectedField`) and the
# underlying geometry.  Every field stores a reference to its grid and
# dispatches size queries, FFT index ranges, quadrature weights, and coordinate
# arrays through this interface.
#
# Downstream packages (e.g. Resolver-ChannelFlow.jl) implement a concrete grid subtype
# by defining a small set of required methods (`size`, `points`,
# `wavenumber_scale`, `weights`).  All other grid-aware behaviour — spectral
# loop generation, transform sizing, base-flow injection, derivative operators —
# is derived automatically from the four compile-time type parameters without
# any runtime overhead.
#
# The four type parameters encode:
#   T              — real scalar type (Float64 by default)
#   D              — number of array dimensions
#   AXES           — 4-tuple mapping logical coordinates (x,y,z,t) to array dims
#   FFT_DIMS_ORDER — ordered tuple of array dimensions that are FFT-transformed;
#                    FFT_DIMS_ORDER[1] is always the rfft dimension

"""
    AbstractGrid{T, D, AXES, FFT_DIMS_ORDER} where {T<:Real}

Abstract type that represents a generic computational grid of a
`D`-dimensional domain.

Type parameters:
- `T`: scalar real type used by physical-space fields on this grid.
- `D`: number of array dimensions.
- `AXES`: four-entry Cartesian axis layout `(x_dim, y_dim, z_dim, t_dim)`.
  NSEBase assumes this tuple always has four entries, one for each logical
  Cartesian coordinate.  Each entry is the array dimension occupied by that
  coordinate, or `nothing` when the coordinate is absent.  The non-`nothing`
  entries should be a permutation of `1:D`.  For example, a three-dimensional
  grid stored as `(y, x, z)` should use `AXES = (2, 1, 3, nothing)`.
- `FFT_DIMS_ORDER`: tuple of statistically homogeneous array dimensions. These are
  transformed by FFTs; `FFT_DIMS_ORDER[1]` is the rfft dimension.

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
abstract type AbstractGrid{T<:Real, D, AXES, FFT_DIMS_ORDER} end

"""
    fft_dims(grid::AbstractGrid) -> Tuple{Int, …}

Return the tuple of FFT-transformed array dimensions in transform order,
i.e. the grid type parameter `FFT_DIMS_ORDER`.

`fft_dims(g)[1]` is the rfft dimension (only non-negative wavenumbers are
stored); the remaining entries are full-spectrum complex FFT dimensions.
"""
fft_dims(::AbstractGrid{<:Any, <:Any, <:Any, FFT_DIMS_ORDER}) where {FFT_DIMS_ORDER} = FFT_DIMS_ORDER

"""
    spatial_fft_dims(grid::AbstractGrid) -> Tuple{Int, …}

Return the FFT-transformed array dimensions that correspond to spatial
coordinates.

For a steady grid this is the same tuple as [`fft_dims`](@ref).  For a
space-time grid whose logical time coordinate is also transformed, the time
dimension is omitted.
"""
@generated function spatial_fft_dims(::AbstractGrid{<:Any, <:Any, AXES, FFT_DIMS_ORDER}) where {AXES, FFT_DIMS_ORDER}
    :($(Tuple(d for d in FFT_DIMS_ORDER if d != AXES[4])))
end

"""
    inhomogeneous_dims(grid::AbstractGrid) -> Tuple

Return the array dimensions that are NOT transformed by FFTs, i.e. the
complement of `fft_dims(grid)` within `1:D`.  These are the directions over
which quadrature weights are needed for inner products.
"""
@generated function inhomogeneous_dims(::AbstractGrid{<:Any, D, <:Any, FFT_DIMS_ORDER}) where {D, FFT_DIMS_ORDER}
    :($(Tuple(d for d in 1:D if d ∉ FFT_DIMS_ORDER)))
end

"""
    spatial_inhomogeneous_dims(grid::AbstractGrid) -> Tuple

Return the inhomogeneous array dimensions that correspond to spatial (not
temporal) coordinates — i.e. dimensions in `inhomogeneous_dims(grid)` that
are not the time axis `AXES[4]`.  These are the dimensions for which a
finite-difference or collocation derivative must be applied.
"""
@generated function spatial_inhomogeneous_dims(::AbstractGrid{<:Any, D, AXES, FFT_DIMS_ORDER}) where {D, AXES, FFT_DIMS_ORDER}
    :($(Tuple(d for d in 1:D if d ∉ FFT_DIMS_ORDER && d != AXES[4])))
end

"""
    rfft_dim(grid::AbstractGrid) -> Int

Return the array dimension that is transformed by the real-to-complex FFT,
i.e. the first entry of `fft_dims(grid)`.

`rfft_dim(g)` is the rfft dimension (only non-negative wavenumbers are
stored); the remaining entries are full-spectrum complex FFT dimensions.
"""
rfft_dim(::AbstractGrid{<:Any,<:Any,<:Any,FFT_DIMS_ORDER}) where {FFT_DIMS_ORDER} = FFT_DIMS_ORDER[1]

"""
    one_or_two(I::CartesianIndex, g::AbstractGrid) -> Int

Return `1` if the rfft storage index in `I` corresponds to the zero
wavenumber, and `2` otherwise.  Used to apply the correct Hermitian 
multiplicity weight in inner products and norms, which are iterated 
with `CartesianIndices`.
"""
one_or_two(I::CartesianIndex, g::AbstractGrid) = I[rfft_dim(g)] == 1 ? 1 : 2

"""
    inhomogeneous_indices(I::CartesianIndex, g::AbstractGrid) -> Tuple

Extract the inhomogeneous indices from the cartesian index `I` according 
to the grid layout and return a tuple.
"""
@generated function inhomogeneous_indices(I::CartesianIndex, ::AbstractGrid{<:Any,D,AXES,FFT_DIMS_ORDER}) where {D,AXES,FFT_DIMS_ORDER}
    idxs = [d for d in 1:D if d ∉ FFT_DIMS_ORDER]
    return Expr(:tuple, (:(I[$idx]) for idx in idxs)...)
end

"""
    homogeneous_indices(I::CartesianIndex, g::AbstractGrid) -> Tuple

Extract the homogeneous indices from the cartesian index `I` according 
to the grid layout and return a tuple.
"""
@generated function homogeneous_indices(I::CartesianIndex, ::AbstractGrid{<:Any,D,AXES,FFT_DIMS_ORDER}) where {D,AXES,FFT_DIMS_ORDER}
    idxs = [d for d in 1:D if d ∈ FFT_DIMS_ORDER]
    return Expr(:tuple, (:(I[$idx]) for idx in idxs)...)
end

"""
    combine_indices(grid, Inh, Ih) -> Tuple

Combine in-homogeneous indices `Inh` and homogeneous indices `Ih` into a single index 
tuple of length `D`, interleaving them according to the constrained dimensions 
`FFT_DIMS_ORDER` defined by `grid`. Dimensions in `FFT_DIMS_ORDER` draw from `Ih`, all 
others draw from `Inh`.

The index layout is fully resolved at compile time via a generated function.
"""
@generated function combine_indices(::AbstractGrid{T,D,AXES,FFT_DIMS_ORDER}, Inh, Ih) where {T,D,AXES,FFT_DIMS_ORDER}
    inds = []
    k = 1
    i = 1
    for d in 1:D
        if d ∈ FFT_DIMS_ORDER
            push!(inds, :(Ih[$k]))
            k += 1
        else
            push!(inds, :(Inh[$i]))
            i += 1
        end
    end
    return :(($(inds...),))
end

"""
    combine_indices(grid, ::Colon, Ih) -> Tuple

Combine homogeneous indices `Ih` with `Colon` for the non-homogeneous dimensions, returning a tuple 
of length `D` suitable for indexing into a `FTField`.  This is mostly used to create views of `FTField`s
that span all inhomogeneous indices at a particular homogeneous index.
"""
@generated function combine_indices(::AbstractGrid{T,D,AXES,FFT_DIMS_ORDER}, ::Colon, Ih) where {T,D,AXES,FFT_DIMS_ORDER}
    inds = []
    k = 1
    for d in 1:D
        if d ∈ FFT_DIMS_ORDER
            push!(inds, :(Ih[$k]))
            k += 1
        else
            push!(inds, :(Colon()))
        end
    end
    return Expr(:tuple, inds...)
end

"""
    to_storage_order(values, grid) -> Tuple

Permute a tuple from logical coordinate order into storage/array-dimension
order.

The axis layout is read from the grid type parameter `AXES`, where `AXES[i]`
is the array dimension occupied by logical coordinate `i`.  `values` must have
the four Cartesian coordinate slots `(X, Y, Z, T)`.  Coordinates whose `AXES`
entry is `nothing` are omitted from the result.  For example, if
`values = (X, Y, Z, T)` and `AXES = (2, 1, 3, 4)`, then
`to_storage_order(values, grid)` returns `(Y, X, Z, T)`.
"""
@generated function to_storage_order(values::Tuple{Vararg{Any, 4}},
                                     ::AbstractGrid{<:Any, D, AXES}) where {D, AXES}
    present = [(coordinate, axis) for (coordinate, axis) in pairs(AXES) if !isnothing(axis)]
    order = first.(sort(present; by=last))
    return Expr(:tuple, (:(values[$coordinate]) for coordinate in order)...)
end

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

`target_size` contains one physical-space size per homogeneous direction, in
`fft_dims(grid)` order. Implementations should preserve inhomogeneous
directions and return a grid whose transformed dimensions match those sizes.

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

Size of the spectral (FTField) array: `FFT_DIMS_ORDER[1]` (rfft) becomes `(N÷2)+1`,
all other dimensions are unchanged.
"""
transform_size(g::AbstractGrid{<:Any, D, <:Any, FFT_DIMS_ORDER}) where {D, FFT_DIMS_ORDER} =
    ntuple(d -> d == FFT_DIMS_ORDER[1] ? (size(g, d) >> 1) + 1 : size(g, d), D)

"""
    fft_norm(g::AbstractGrid)

Number of grid points in each FFT dimension `FFT_DIMS_ORDER`. Used to normalise
forward transforms. Default: reads from `size(g)`.
"""
fft_norm(g::AbstractGrid) = map(d -> size(g, d), fft_dims(g))

# ------------------------------------------------------------------ #
# Hermitian-symmetry enforcement on raw coefficient arrays           #
# ------------------------------------------------------------------ #

"""
    _average_complex(z1, z2) -> z

Return the unique complex number `z` such that `z` and `conj(z)` are the
nearest conjugate pair to `z1` and `z2`:

```
re(z) = (re(z1) + re(z2)) / 2   # real parts should agree → average them
im(z) = (im(z1) - im(z2)) / 2   # imaginary parts should be negatives → split the difference
```

After the call, `data[I] = z` and `data[Ineg] = conj(z)` enforce
`data[I] = conj(data[Ineg])` to floating-point precision.
"""
@inline function _average_complex(z1::T, z2::T) where {T}
    _re = 0.5 * (real(z1) + real(z2))
    _im = 0.5 * (imag(z1) - imag(z2))
    return _re + im * _im
end

"""
    apply_symmetry!(data::AbstractArray, ::Val{FFT_DIMS_ORDER}) -> data
    apply_symmetry!(data::AbstractArray, dims::Tuple)           -> data

Enforce Hermitian symmetry in-place on a raw spectral coefficient array.

# Background

A real-valued physical field `f(x)` has a Fourier transform satisfying
`F̂(-k) = conj(F̂(k))` for every wavenumber vector `k`.  When the transform
is performed with `rfft` along the first homogeneous axis (`FFT_DIMS_ORDER[1]`),
only non-negative wavenumbers are stored along that axis, so there are no
conjugate-pair redundancies there.  The remaining axes in
`FFT_DIMS_ORDER[2:end]` use a full signed FFT and store both positive and
negative wavenumbers.

The constraint `F̂(-k) = conj(F̂(k))` still applies to the signed-FFT axes,
but **only at the zero-frequency slice of the rfft axis** (storage index 1,
i.e. rfft wavenumber = 0).  At any nonzero rfft wavenumber `kx > 0` the
negative-`kx` entry is not stored, so no symmetry constraint can be applied.

# What this function does

1. **Restrict to the DC plane**: loops only over positions where the rfft axis
   index is 1 (storage index for rfft wavenumber = 0), while iterating all
   other axes freely.

2. **Find conjugate partners**: for each such index `I`, constructs `Ineg` by
   flipping each signed-FFT axis using FFTW's wrap-around convention:
   - index 1 (wavenumber 0) maps to itself (self-conjugate),
   - index `i > 1` (wavenumber `n = i - 1 > 0`) maps to `size(data, d) - i + 2`
     (the index storing wavenumber `-n`).
   Axes not in `FFT_DIMS_ORDER` (e.g. wall-normal, mode index) are kept
   identical between `I` and `Ineg` — they are not transformed.

3. **Average each pair once**: the `LI[I] <= LI[Ineg]` guard ensures each
   `{I, Ineg}` pair is visited exactly once.  The `_average_complex` helper
   computes the unique value `a` satisfying `a = (z1 + conj(z2)) / 2` in the
   real part and `a = (z1 - conj(z2)) / 2` in the imaginary part, then writes
   `data[I] = a` and `data[Ineg] = conj(a)`, enforcing `F̂(-k) = conj(F̂(k))`
   exactly.

# Arguments

- `FFT_DIMS_ORDER` (via `Val`): tuple of transformed array dimensions in
  transform order — `FFT_DIMS_ORDER[1]` is the rfft dim, the rest are
  signed-FFT dims.

No-op when `length(FFT_DIMS_ORDER) ≤ 1` (only an rfft dim, no partners).
"""
function apply_symmetry!(data::AbstractArray{<:Any, D}, ::Val{FFT_DIMS_ORDER}) where {D, FFT_DIMS_ORDER}
    # No signed-FFT dimensions: nothing to symmetrize.
    length(FFT_DIMS_ORDER) <= 1 && return data

    # Guard against offset arrays whose index-1 slot may not be the DC bin.
    Base.require_one_based_indexing(data)

    # Split FFT_DIMS_ORDER into the rfft dim and the signed-FFT dims.
    rfft_dim           = FFT_DIMS_ORDER[1]
    secondary_fft_dims = Base.tail(FFT_DIMS_ORDER)

    # Build the loop ranges: the rfft axis is pinned to index 1 (DC plane);
    # all other axes (signed-FFT and untransformed) run their full range.
    ranges = ntuple(Val(D)) do d
        d == rfft_dim ? (1:1) : axes(data, d)
    end

    # LinearIndices lets us guard against visiting each {I, Ineg} pair twice
    # without storing a visited-set: we process the pair only when LI[I] <= LI[Ineg].
    LI = LinearIndices(data)

    @inbounds for I in CartesianIndices(ranges)

        # Find the index of the conjugate partner: flip only the secondary_fft_dims Fourier dimensions.
        Ineg = CartesianIndex(ntuple(Val(D)) do d
            if d in secondary_fft_dims
                # zero-frequency plane: partner is the same index, so no conjugation.
                if I[d] == 1
                    return I[d]
                end
                # negative wavenumber partner index, using FFTW's wrap-around order.
                return size(data, d) - I[d] + 2
            else
                return I[d]
            end
        end)

        # only handle each pair once
        if LI[I] <= LI[Ineg]
            av = _average_complex(data[I], data[Ineg])
            data[I]    = av
            data[Ineg] = conj(av)
        end
    end

    return data
end

# Convenience overload so callers can pass a plain tuple without wrapping in Val.
apply_symmetry!(data::AbstractArray, dims::Tuple) = apply_symmetry!(data, Val(dims))
