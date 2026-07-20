"""
    RectangularGrid{NI, T, S, D, AXES, FFT_DIMS}

A concrete tensor-product Cartesian grid combining finite-difference
directions with periodic Fourier directions. It implements the complete
[`AbstractGrid`](@ref) interface required by `Field`, `FTField`, FFT plans,
weighted inner products, and spatial derivatives.

# Type parameters

- `NI`: number of inhomogeneous finite-difference directions. This is also the
  length of `xs`, `D₁`, `D₂`, `D₁⁺`, `D₂⁺`, and `ws`.
- `T`: real scalar type used by physical fields and all numerical grid data.
- `S`: physical-space array size in **storage order**. `size(g) === S`.
- `D`: number of storage dimensions, equal to `length(S)`.
- `AXES`: four-entry map from physical coordinates `(x, y, z, t)` to storage
  dimensions. An absent coordinate is represented by `nothing`.
- `FFT_DIMS`: homogeneous storage dimensions in transform order. Its first
  entry is the real-to-complex FFT direction.

The remaining type parameters are inferred element types for the stored
`NTuple`s and the concrete product-weight type. When directions use different
operator types or stencil widths, the corresponding element parameter is their
narrow concrete union rather than an abstract matrix type.

# Ordering contract

Direction-dependent tuples are ordered exactly like
[`inhomogeneous_storage_dims`](@ref). Fourier scales are ordered exactly like
[`fft_storage_dims`](@ref). Both are storage-dimension orderings, not physical
`(x, y, z, t)` orderings.

For a channel stored as `(y, x, z, t)`, for example,
`inhomogeneous_storage_dims(g) == (1,)` and `g.xs[1]` contains the `y` points.
For a duct stored as `(x, y, z, t)`, the tuple order is `(x, y)`.

# Fields

- `xs`: collocation vectors for the finite-difference directions.
- `D₁`, `D₂`: first- and second-derivative operators.
- `D₁⁺`, `D₂⁺`: caller-supplied discrete adjoint operators.
- `ws`: one-dimensional quadrature vectors.
- `w`: tensor product of `ws`, returned by [`weights`](@ref).
- `scales`: Fourier wavenumber scales. A periodic direction of length `L` uses
  `2π/L`, so integer mode `k` represents physical wavenumber `k * 2π/L`.

Each derivative operator must support dimension-wise application through
`LinearAlgebra.mul!(out, D, u, Val(dim))`. The finite-difference kernels unwrap
`FTField` objects with `parent` before calling `mul!`, allowing FDGrids and
other compatible matrix implementations to control how an operator acts along
one storage dimension.

The supplied adjoints define the discrete adjoint convention; they are stored
and applied exactly as given. For quadrature weights `w`, callers normally
choose them so that `⟨D₁u, v⟩_w = ⟨u, D₁⁺v⟩_w` and likewise for `D₂`. The
constructor does not recompute or verify these identities.

The product quadrature is separable and never constructs a Kronecker matrix.
For two directions, for example, `w[i, j] = ws[1][i] * ws[2][j]`; the analogous
three-direction product is stored as a three-dimensional array.

The grid stores geometry and numerical operators, not boundary conditions.
Endpoint-inclusive FDGrids distributions make wall points available, but a
projection basis or residual formulation remains responsible for imposing
wall values, including a moving cavity lid.

Users should normally use a case constructor such as [`ChannelGrid`](@ref),
[`StreamwiseInvariantChannelGrid`](@ref), [`TwoDimensionalChannelGrid`](@ref),
[`SquareDuctGrid`](@ref), or [`LidDrivenCavityGrid`](@ref). The explicit invariant-channel names
distinguish which homogeneous physical direction is absent. Use the low-level constructor when
supplying custom collocation data or derivative operators.

See also: [`points`](@ref), [`growto`](@ref), [`wavenumber_scale`](@ref),
[`derivative_matrix`](@ref).
"""
struct RectangularGrid{NI, T, S, D, AXES, FFT_DIMS, XS, D1, D2, A1, A2, WS, WP, NFFT} <: AbstractGrid{T, D, AXES, FFT_DIMS}
    xs     :: NTuple{NI, XS}
    D₁     :: NTuple{NI, D1}
    D₂     :: NTuple{NI, D2}
    D₁⁺    :: NTuple{NI, A1}
    D₂⁺    :: NTuple{NI, A2}
    ws     :: NTuple{NI, WS}
    w      :: WP
    scales :: NTuple{NFFT, T}
end

"""
    RectangularGrid(xs, D₁, D₂, D₁⁺, D₂⁺, ws, scales, grid_size, axes, fft_dims, T=Float64)

Construct a [`RectangularGrid`](@ref) from existing one-dimensional
finite-difference discretisations. The common tuple length of `xs`, `D₁`,
`D₂`, `D₁⁺`, `D₂⁺`, and `ws` becomes `NI`.

# Arguments

- `xs`: collocation vectors, ordered by increasing inhomogeneous storage
  dimension.
- `D₁`, `D₂`: first- and second-derivative matrices. Operator `D₁[i]` and
  `D₂[i]` act along `inhomogeneous_storage_dims(g)[i]`.
- `D₁⁺`, `D₂⁺`: discrete adjoints in the same order. These are trusted and
  stored; the constructor does not derive or replace them.
- `ws`: one-dimensional quadrature vectors corresponding to `xs`. Their
  tensor product is constructed automatically for [`weights`](@ref).
- `scales`: one scale per Fourier direction, in `fft_dims` order.
- `grid_size`: physical-space shape in storage order. The value is embedded in
  the concrete grid type, so changing it produces a different type.
- `axes`: `(x_dim, y_dim, z_dim, t_dim)`, with `nothing` for absent physical
  coordinates.
- `fft_dims`: homogeneous storage dimensions in transform order. The first
  entry is the rFFT direction.
- `T`: target real scalar type; defaults to `Float64`.

# Validation

The constructor requires between one and three inhomogeneous directions, a
positive physical size in every direction, exactly four axis-map entries whose
present values are a permutation of `1:D`, and a nonempty tuple of unique
Fourier dimensions in `1:D`. There must be one scale per Fourier direction and
all Fourier sizes must be odd. When a logical time coordinate is present it must
be Fourier transformed; `NI` therefore always counts spatial FD directions.
For every finite-difference direction, its points and weights must match the
corresponding entry of `grid_size`, and all four operators must be square
matrices of that size.

# Conversion and ownership

Points, weights, scales, and operators are converted to `T` when necessary.
Containers whose element type is already `T`, including custom matrix
operators, are retained rather than copied; others are converted by
broadcasting `T` over their values.
Caller-supplied adjoints remain caller-supplied adjoints. Repeated identical
objects that already have scalar type `T` are retained and remain shared;
converting to a different scalar type produces independent copies.

The operators must implement dimension-wise multiplication via
`LinearAlgebra.mul!(out, D, u, Val(dim))`, where `dim` is the corresponding
inhomogeneous storage dimension. Grids with two or three FD directions also
use `LinearAlgebra.mul!(out, D, u, Val(dim), Val(true))` to accumulate later
Laplacian terms in place. The supplied `D₁⁺` and `D₂⁺` should be the adjoints
under the quadrature defined by `ws`; they are accepted without a numerical
adjoint check.

# Example

The following layout describes `(y, x, z, t)` storage with finite differences
in `y` and Fourier transforms in `(x, z, t)`:

```julia
axes = (2, 1, 3, 4)
fft_dims = (2, 3, 4)
grid_size = (length(y), Nx, Nz, Nt)
g = RectangularGrid((y,), (D₁,), (D₂,), (D₁⁺,), (D₂⁺,), (ws,),
                    (α, β, 2π), grid_size, axes, fft_dims)
```
"""
function RectangularGrid(xs::Tuple{Vararg{Any, NI}},
                         D₁::Tuple{Vararg{Any, NI}}, D₂::Tuple{Vararg{Any, NI}},
                         D₁⁺::Tuple{Vararg{Any, NI}}, D₂⁺::Tuple{Vararg{Any, NI}},
                         ws::Tuple{Vararg{Any, NI}}, scales::Tuple,
                         grid_size::NTuple{D, Int}, axes::Tuple, fft_dims::Tuple,
                         ::Type{T}=Float64) where {NI, D, T<:Real}
    # Validate layout and homogeneous resolutions.
    NI > 0 || throw(ArgumentError("at least one inhomogeneous direction is required"))
    NI <= 3 || throw(ArgumentError("at most three inhomogeneous directions are supported"))
    all(>(0), grid_size) || throw(ArgumentError("grid sizes must be positive"))
    all(values -> values isa AbstractVector, xs) || throw(ArgumentError("collocation data must be vectors"))
    all(values -> values isa AbstractVector, ws) || throw(ArgumentError("quadrature data must be vectors"))
    all(operator -> operator isa AbstractMatrix, (D₁..., D₂..., D₁⁺..., D₂⁺...)) || throw(ArgumentError("derivative operators must be matrices"))
    length(axes) == 4 || throw(ArgumentError("axes must contain the four Cartesian coordinate slots"))
    present_axes = filter(!isnothing, axes)
    all(dim -> dim isa Int, present_axes) && sort!(collect(present_axes)) == collect(1:D) ||
        throw(ArgumentError("nonempty axes must be a permutation of the $D storage dimensions"))
    !isempty(fft_dims) || throw(ArgumentError("at least one Fourier direction is required"))
    all(dim -> dim isa Int && 1 <= dim <= D, fft_dims) || throw(ArgumentError("Fourier dimensions must lie in 1:$D"))
    allunique(fft_dims) || throw(ArgumentError("Fourier dimensions must be unique"))
    isnothing(axes[4]) || axes[4] in fft_dims || throw(ArgumentError("a present time coordinate must be Fourier transformed"))
    length(scales) == length(fft_dims) || throw(ArgumentError("expected one wavenumber scale per Fourier direction"))
    all(isodd(grid_size[dim]) for dim in fft_dims) || throw(ArgumentError("grid must be odd in homogeneous directions"))

    # Validate each finite-difference direction and its quadrature data.
    inhomogeneous_dims = Tuple(dim for dim in 1:D if dim ∉ fft_dims)
    length(inhomogeneous_dims) == NI || throw(ArgumentError("grid layout has $(length(inhomogeneous_dims)) inhomogeneous directions, but data has $NI"))
    for (index, dim) in pairs(inhomogeneous_dims)
        n = grid_size[dim]
        length(xs[index]) == length(ws[index]) == n || throw(ArgumentError("points and weights for storage dimension $dim have incompatible sizes"))
        size(D₁[index]) == size(D₂[index]) == size(D₁⁺[index]) == size(D₂⁺[index]) == (n, n) ||
            throw(ArgumentError("differentiation matrices for storage dimension $dim have incompatible sizes"))
    end

    # Convert numerical data to the target scalar type; same-type data is retained as-is.
    convert_data(x) = eltype(x) === T ? x : T.(x)
    converted_xs = map(convert_data, xs)
    converted_ws = map(convert_data, ws)
    operators = map(values -> map(convert_data, values), (D₁, D₂, D₁⁺, D₂⁺))
    product_weights = _rectangular_product_weights(converted_ws)

    # Store direction count, size, and layout in the type.
    operator_types = map(values -> Union{typeof.(values)...}, operators)
    return RectangularGrid{NI, T, grid_size, D, axes, fft_dims, Union{typeof.(converted_xs)...}, operator_types...,
                           Union{typeof.(converted_ws)...}, typeof(product_weights), length(scales)}(
        converted_xs, operators..., converted_ws, product_weights, T.(scales))
end

"""Construct the separable product quadrature from one-dimensional weights."""
_rectangular_product_weights(ws::NTuple{1}) = ws[1]
_rectangular_product_weights(ws::NTuple{2}) = ws[1] * transpose(ws[2])
_rectangular_product_weights(ws::NTuple{3}) = reshape(ws[1], :, 1, 1) .* reshape(ws[2], 1, :, 1) .* reshape(ws[3], 1, 1, :)

"""
    size(g::RectangularGrid) -> Tuple

Return the physical-space grid shape in storage order. The returned tuple is
the compile-time size parameter `S`; it contains unpadded sizes even when a
dealiased field or FFT plan is constructed from `g`.
"""
Base.size(::RectangularGrid{N, T, S}) where {N, T, S} = S

"""
    weights(g::RectangularGrid) -> AbstractArray

Return the tensor-product quadrature weights for all inhomogeneous directions.
Its axes follow `inhomogeneous_storage_dims(g)` order: a vector for
`RectangularGrid{1}`, a matrix for `RectangularGrid{2}`, and a three-dimensional
array for `RectangularGrid{3}`. Homogeneous directions contribute no stored
weight array because their trapezoidal factors are handled by Fourier normalisation.

More explicitly, the stored values are

```text
NI = 1:  w[i]     = ws[1][i]
NI = 2:  w[i,j]   = ws[1][i] * ws[2][j]
NI = 3:  w[i,j,k] = ws[1][i] * ws[2][j] * ws[3][k]
```

Thus multidimensional weighted inner products remain separable. Applying the
one-dimensional weighted adjoint in each direction gives the adjoint of the
multidimensional operator without constructing a dense Kronecker matrix.
"""
weights(g::RectangularGrid) = g.w

"""
    wavenumber_scale(g::RectangularGrid, storage_dim) -> Real

Return the factor mapping an integer Fourier mode to a physical wavenumber in
`storage_dim`. The argument is a **storage dimension**. If it is the `i`th
entry of `fft_storage_dims(g)`, the result is `g.scales[i]`; an
inhomogeneous dimension returns `one(eltype(g))`.

For a periodic interval of length `L`, store `2π/L`. A unit-period temporal
coordinate therefore normally has scale `2π`.

The scale is used not only by Fourier derivatives, but also by physical-unit
translations in [`shift!`](@ref) and phase alignment in [`minnormdiff`](@ref):
an integer mode `n` acquires phase `n * shift * wavenumber_scale(g, dim)`.
"""
function wavenumber_scale(g::RectangularGrid{NI, T}, storage_dim::Int) where {NI, T}
    index = findfirst(==(storage_dim), fft_storage_dims(g))
    return isnothing(index) ? one(T) : g.scales[index]
end

"""
    points(g::RectangularGrid; dealias=false) -> Tuple

Return one broadcast-compatible coordinate array per storage dimension.
Coordinate `points(g)[dim]` varies only along storage dimension `dim` and has
singleton axes elsewhere. Broadcasting all returned arrays therefore produces
the complete physical grid without materialising full coordinate tensors.

Finite-difference coordinates come directly from `g.xs`. Fourier coordinates
are equally spaced on the half-open interval `[0, 2π/scale)`, excluding the
repeated periodic endpoint. With `dealias=true`, only Fourier dimensions use
the padded physical sizes returned by `get_padded_size`.

# Example

For a channel stored as `(y, x, z, t)`, the shapes are
`(Ny,1,1,1)`, `(1,Nx,1,1)`, `(1,1,Nz,1)`, and `(1,1,1,Nt)`.
"""
function points(g::RectangularGrid; dealias::Bool=false)
    storage_size = dealias ? get_padded_size(size(g), fft_storage_dims(g)) : size(g)
    return points(g, map(dim -> storage_size[dim], fft_storage_dims(g)))
end

"""
    points(g::RectangularGrid, homogeneous_size) -> Tuple

Return broadcast-compatible coordinates using explicit physical sizes for the
Fourier directions. `homogeneous_size` must contain one integer per entry of
`fft_storage_dims(g)`, in that exact order. Finite-difference resolutions and
coordinates are unchanged.

This method is the coordinate counterpart of [`growto`](@ref), and
`points(g, target) == points(growto(g, target))`.
"""
points(g::RectangularGrid, homogeneous_size::NTuple{N, Int}) where {N} = _rectangular_points(g, homogeneous_size, g.xs)

"""
    growto(g::RectangularGrid, homogeneous_size) -> RectangularGrid

Return an equivalent grid with new physical resolutions in the Fourier
directions. `homogeneous_size` follows `fft_storage_dims(g)` order, and every
requested resolution must be positive and odd.

Collocation vectors, finite-difference operators, supplied adjoints,
one-dimensional weights, product quadrature, and wavenumber scales are
retained. Only the size type parameter changes. This method supports
resolution-changing `FFT` and `IFFT` operations, including 3/2-rule padding
and continuation at another Fourier resolution.

The retained data preserve object identity. In particular,
`grown.xs === g.xs` and `grown.w === g.w`, and the same holds for `D₁`, `D₂`,
`D₁⁺`, `D₂⁺`, and `ws`. Only homogeneous resolutions may change;
finite-difference sizes are fixed by their collocation data.
"""
function growto(g::RectangularGrid{NI, T, S, D, AXES, FFT_DIMS, XS, D1, D2, A1, A2, WS, WP, NFFT},
                homogeneous_size::NTuple{N, Int}) where {NI, T, S, D, AXES, FFT_DIMS, XS, D1, D2, A1, A2, WS, WP, NFFT, N}
    foreach(homogeneous_size) do resolution
        @noinline
        resolution > 0 || throw(ArgumentError("grid sizes must be positive"))
        isodd(resolution) || throw(ArgumentError("grid must be odd in homogeneous directions"))
    end
    grid_size = _rectangular_storage_size(g, homogeneous_size)
    return RectangularGrid{NI, T, grid_size, D, AXES, FFT_DIMS, XS, D1, D2, A1, A2, WS, WP, NFFT}(
        g.xs, g.D₁, g.D₂, g.D₁⁺, g.D₂⁺, g.ws, g.w, g.scales)
end

"""
    convert(T, g::RectangularGrid) -> RectangularGrid

Convert all numerical grid data to real scalar type `T` while preserving the
number of directions, layout, and physical size. Points, weights, scales,
derivative operators, and supplied adjoints are converted consistently.

If `g` already has scalar type `T`, return `g` itself. Otherwise a new grid is
constructed; adjoints are converted directly and are not recomputed from the
forward operators.

When several tuple entries refer to the same object and the scalar type
actually changes, each entry is converted independently; the resulting grid
holds equal but distinct copies rather than shared objects.
"""
function Base.convert(::Type{T}, g::RectangularGrid{NI, T0, S, D, AXES, FFT_DIMS}) where {T<:Real, NI, T0, S, D, AXES, FFT_DIMS}
    T === T0 && return g
    return RectangularGrid(g.xs, g.D₁, g.D₂, g.D₁⁺, g.D₂⁺, g.ws, g.scales, S, AXES, FFT_DIMS, T)
end

"""
    derivative_matrix(g::RectangularGrid, storage_dim, Val(order), [Val(adjoint)])

Return the finite-difference operator acting along `storage_dim`.

- `storage_dim` must belong to `inhomogeneous_storage_dims(g)`.
- `order` must be `1` or `2`.
- `adjoint=true` selects `D₁⁺` or `D₂⁺`; otherwise the corresponding forward
  operator is returned.

The lookup follows the inhomogeneous tuple-ordering contract and preserves the
concrete operator type. Decomposed-grid implementations call this method on the
parent rectangular grid to obtain the first- or second-derivative stencil for a
local slab; the grid itself therefore remains independent of partition
metadata.

The `Val` adjoint selector is required for type stability because an FDGrids
forward operator and its weighted adjoint have different concrete types.
Higher-level in-place derivative functions expose an `adjoint::Bool` keyword
and branch to this type-specialized accessor internally. Omitting the final
argument selects the forward operator through the type-stable `Val(false)`
default.
"""
function derivative_matrix(g::RectangularGrid, storage_dim::Int, ::Val{ORDER}, ::Val{ADJOINT}=Val(false)) where {ORDER, ADJOINT}
    index = findfirst(==(storage_dim), inhomogeneous_storage_dims(g))
    isnothing(index) && throw(ArgumentError("storage dimension $storage_dim is not inhomogeneous"))
    ORDER == 1 && return ADJOINT ? g.D₁⁺[index] : g.D₁[index]
    ORDER == 2 && return ADJOINT ? g.D₂⁺[index] : g.D₂[index]
    throw(ArgumentError("only first- and second-derivative operators are available"))
end

"""
    inhomogeneous_dd!(out, u, Val(storage_dim); adjoint=false) -> out

Apply the first finite-difference derivative of `u` along the inhomogeneous
storage dimension `storage_dim`, writing into `out`. With `adjoint=true`, apply
the supplied discrete adjoint operator instead.

The operation unwraps both fields with `parent`, delegates to the matrix's
dimension-aware `mul!` method, and does not allocate a temporary field. `out`
and `u` must use the same concrete grid. If the supplied adjoint was built for
the grid quadrature, the operation obeys
`⟨D₁u, v⟩_w = ⟨u, D₁⁺v⟩_w` up to numerical roundoff.
"""
function inhomogeneous_dd!(out::FTField{G}, u::FTField{G}, ::Val{DIM}; adjoint::Bool=false) where {G<:RectangularGrid, DIM}
    if adjoint
        LinearAlgebra.mul!(parent(out), derivative_matrix(grid(u), DIM, Val(1), Val(true)), parent(u), Val(DIM))
    else
        LinearAlgebra.mul!(parent(out), derivative_matrix(grid(u), DIM, Val(1), Val(false)), parent(u), Val(DIM))
    end
    return out
end

"""
    inhomogeneous_laplacian!(out, u; adjoint=false) -> out

Apply the finite-difference contribution to the spatial Laplacian and write it
into `out`. The method applies `D₂` along every entry of
`spatial_inhomogeneous_storage_dims(grid(u))`; the first term overwrites `out`
and subsequent terms accumulate in place, so no temporary field is required.

The generated implementation supports one, two, or three inhomogeneous
directions and unrolls their matrix products at compile time. It emits separate
forward and adjoint branches so every matrix product has a concrete operator
type without exposing a `Val` argument in this in-place API.

Homogeneous Fourier contributions are deliberately excluded and are added by
[`add_homogeneous_laplacian!`](@ref). With `adjoint=true`, every `D₂` is
replaced by its caller-supplied `D₂⁺`. When those operators are weighted
adjoints under [`weights`](@ref), the complete finite-difference sum satisfies
the corresponding multidimensional weighted-adjoint identity.
"""
@generated function inhomogeneous_laplacian!(out::FTField{G}, u::FTField{G}; adjoint::Bool=false) where {NI, G<:RectangularGrid{NI}}
    1 <= NI <= 3 || return :(throw(ArgumentError("inhomogeneous Laplacian supports one to three directions")))
    return quote
        g = grid(u)
        dims = spatial_inhomogeneous_storage_dims(g)
        if adjoint
            Base.Cartesian.@nexprs $NI i -> begin
                if i == 1
                    LinearAlgebra.mul!(parent(out), g.D₂⁺[i], parent(u), Val(dims[i]))
                else
                    LinearAlgebra.mul!(parent(out), g.D₂⁺[i], parent(u), Val(dims[i]), Val(true))
                end
            end
        else
            Base.Cartesian.@nexprs $NI i -> begin
                if i == 1
                    LinearAlgebra.mul!(parent(out), g.D₂[i], parent(u), Val(dims[i]))
                else
                    LinearAlgebra.mul!(parent(out), g.D₂[i], parent(u), Val(dims[i]), Val(true))
                end
            end
        end
        return out
    end
end

"""
    _equal_points(N, L)

Return `N` equally spaced points of the same scalar type as `L` on the
half-open periodic interval `[0, L)`. The repeated endpoint is excluded; for
example, `_equal_points(4, 1.0)` gives `0`, `1/4`, `1/2`, and `3/4`.
"""
_equal_points(N, L::T) where {T} = range(zero(T), step=L / T(N), length=N)

"""Replace the homogeneous entries of `size(g)` using FFT-dimension order."""
function _rectangular_storage_size(g::AbstractGrid{T, D}, homogeneous_size::NTuple{N, Int}) where {T, D, N}
    fft_dims = fft_storage_dims(g)
    N == length(fft_dims) || throw(ArgumentError("expected $(length(fft_dims)) homogeneous sizes"))
    return ntuple(dim -> dim in fft_dims ? homogeneous_size[findfirst(==(dim), fft_dims)] : size(g, dim), D)
end

"""Build broadcast-compatible rectangular-grid coordinates in storage order."""
function _rectangular_points(g::AbstractGrid{T, D}, homogeneous_size::NTuple{N, Int}, xs::Tuple) where {T, D, N}
    storage_size = _rectangular_storage_size(g, homogeneous_size)
    inhomogeneous_dims = inhomogeneous_storage_dims(g)
    length(xs) == length(inhomogeneous_dims) || throw(ArgumentError("expected one collocation vector per inhomogeneous direction"))
    reshape_dims(dim, len) = ntuple(d -> d == dim ? len : 1, D)

    return ntuple(D) do dim
        index = findfirst(==(dim), inhomogeneous_dims)
        values = isnothing(index) ? _equal_points(storage_size[dim], T(2π) / wavenumber_scale(g, dim)) : xs[index]
        reshape(values, reshape_dims(dim, length(values)))
    end
end
