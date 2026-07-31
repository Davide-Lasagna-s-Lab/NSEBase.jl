# Ordered scalar-field tuples and their AbstractVector interface.
#
# A `VectorField` bundles N scalar fields — either `FTField` (spectral) or
# `Field` (physical) — into a single indexable container. Components have one
# concrete scalar-field type. Grid equality is a caller-side invariant rather
# than a constructor check. The struct acts as a 1-D `AbstractVector{S}` over
# its components so generic algorithms can iterate over them uniformly.
#
# The design separates concern of "multiple components" from "field data":
# VectorField only manages component ordering; all arithmetic is delegated to
# the underlying scalar fields through the broadcasting interface in
# broadcasting.jl.  This makes it easy to extend NSEBase to 2D (N=2) or
# arbitrary-component flows without changing any core logic.

"""
    VectorField{N, S} <: AbstractVector{S}

An ordered tuple of `N` scalar fields with the same concrete type `S`, where
`S` is either a [`Field`](@ref) or [`FTField`](@ref) type.

`VectorField` implements the one-dimensional array interface over components:
`size(q) == (N,)`, `q[n]` returns the stored scalar-field object, and field
broadcasts are evaluated component by component.

The constructor enforces the common component type but does not compare the
components' grid values, shapes, or storage identities. Grid-aware operations
require those properties to be compatible. In particular, [`grid`](@ref) returns
the grid of the first component without checking the others.

# Type parameters

- `N`: number of components
- `S`: concrete component type

# Fields

- `elements`: the `NTuple{N, S}` of scalar field components

The tuple and component objects are stored directly; construction performs no
copy. `parent(q)` returns that exact tuple.
"""
struct VectorField{N, S} <: AbstractVector{S}
    elements::NTuple{N, S}

    function VectorField(elements::Vararg{S, N}) where {S<:Union{FTField, Field}, N}
        new{N, S}(elements)
    end
end

"""
    add_base_flow!(u::VectorField, base::NTuple{N}) -> u

Add `base` to the zero-wavenumber slice of spectral vector field `u` in place
and return `u`.

`base` must have exactly one entry per component. Each entry is either
`nothing`, which leaves that component unchanged, or a value broadcastable to
the component's complete inhomogeneous DC slice. Thus the same method supports
one or several inhomogeneous dimensions as well as scalar offsets. Shape and
grid compatibility are left to normal broadcast assignment checks.

# Index construction

For a field stored as a `D`-dimensional spectral array, the zero-wavenumber
slice is selected by an index pattern derived from the grid's `fft_storage_dims`:

- FFT dimensions (homogeneous, transformed): index `1`, the DC bin.
  FFTW always places the zero-frequency coefficient first, for both the real
  FFT (rfft) and the full complex FFT.
- Inhomogeneous dimensions: `Colon()`, selecting all grid points.

For a 4D channel grid with `CHANNEL_FFT_ORDER = (2, 3, 4)`, this selects
`parent(u[n])[:, 1, 1, 1]`: the full inhomogeneous profile at zero streamwise,
spanwise, and temporal wavenumber.

# Example

```julia
# Channel flow: Couette base U(y)=y, no base in v or w.
add_base_flow!(u, (U, nothing, nothing))

# Equivalent to the original hand-written channel specialisation:
#   u[1][:, 1, 1, 1] .+= U
```
"""
@generated function add_base_flow!(u::VectorField{N, <:FTField{G}},
                                   base::Tuple{Vararg{Any, N}}) where {N, T, D, AXES, FFT_DIMS_ORDER, G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}}
    idx = Any[(d ∈ FFT_DIMS_ORDER ? 1 : :(Colon())) for d in 1:D]

    return quote
        Base.Cartesian.@nexprs $N n -> begin
            if base[n] !== nothing
                Base.@views $(Expr(:ref, :(parent(u[n])), idx...)) .+= base[n]
            end
        end
        return u
    end
end

"""
    grid(u::VectorField) -> AbstractGrid

Return `grid(u[1])`.

No comparison with later components is performed. A nonempty vector field is
therefore required, and callers are responsible for keeping component grids
compatible.
"""
grid(u::VectorField) = grid(u[1])

"""
    growto(u::VectorField, target_size)

Return an equivalent vector field with a new homogeneous resolution.

Each component is grown independently via [`growto(::FTField, target_size)`](@ref),
so the returned field covers the same spectral content as `u` but on a grid
with the physical-space sizes specified by `target_size` (one entry per
homogeneous dimension in `FFT_DIMS_ORDER` order).

This is optional.  Packages should only implement it if they support
resolution-changing transforms such as `FFT(u, target_size)` and
`IFFT(û, target_size)`.
"""
growto(u::VectorField{N, <:FTField}, target_size) where {N} = VectorField([growto(u[n], target_size) for n in 1:N]...)


# ------------------- #
# constructor methods #
# ------------------- #
"""
    VectorField(g::AbstractGrid, ::Type{S}=FTField; N::Int=3, kwargs...) -> VectorField

Allocate `N` independent scalar fields by calling `S(g; kwargs...)` once per
component, then store them in a `VectorField`.

`S` must construct a [`Field`](@ref) or [`FTField`](@ref); it defaults to
`FTField`, and `N` defaults to `3`. The zero-valued and shape guarantees are
those of the selected scalar-field constructor. Every call receives the same
grid object `g`, but each component has its own parent storage.
"""
VectorField(g::AbstractGrid, ::Type{S}=FTField; N::Int=3, kwargs...) where {S} = VectorField([S(g; kwargs...) for _ in 1:N]...)

"""
    VectorField(g::AbstractGrid, funcs...; dealias::Bool=false) -> VectorField

Construct one physical [`Field`](@ref) per function in `funcs`.

The number of components is `length(funcs)`. Each function is evaluated with
the argument ordering, conversion, and `dealias` behavior documented by
`Field(g, func; dealias)`. Components share the supplied grid object but own
independent parent arrays.
"""
VectorField(g::AbstractGrid, funcs...; dealias::Bool=false) = VectorField([Field(g, f; dealias=dealias) for f in funcs]...)


# ------------- #
# array methods #
# ------------- #
"""
    IndexStyle(::Type{<:VectorField})

Declare linear, one-dimensional indexing as the native indexing style.
"""
Base.IndexStyle(::Type{<:VectorField}) = Base.IndexLinear()

"""
    parent(q::VectorField) -> NTuple

Return the exact `NTuple{N,S}` stored by `q`; neither the tuple nor its
components are copied.
"""
Base.parent(q::VectorField) = q.elements

"""
    q[i::Int] -> S

Return the exact `i`-th scalar-field object stored by `q`. Tuple indexing
provides the bounds check.
"""
Base.getindex(q::VectorField, i::Int) = parent(q)[i]

"""
    q[i::Int] = v

Broadcast-copy the values of scalar field `v` into the existing `i`-th
component and return `v`.

This does not replace the tuple element: the destination component, its grid,
and its parent storage retain their identities. `v` must have the component
type `F`; compatible shape and grid semantics are the caller's responsibility.
"""
Base.setindex!(q::VectorField{N, F}, v::F, i::Int) where {N, F} = (parent(q)[i] .= v; return v)

"""
    size(::VectorField{N}) -> (N,)

Return the one-dimensional component shape `(N,)`.
"""
Base.size(::VectorField{N}) where {N} = (N,)

"""
    eltype(::VectorField{N, F}) -> Type{F}

Return the component field type `F`, not the numeric element type stored inside
each component. Use `eltype(q[i])` for the latter.
"""
Base.eltype(::VectorField{N, F}) where {N, F} = F

"""
    similar(q::VectorField[, ::Type{T}]) -> VectorField

Allocate a vector field by calling `similar(q[n], T)` independently for every
component.

Unlike the conventional `AbstractArray` interpretation, `T` here is the
requested numeric scalar type of each component, not the outer vector's
component type. For example, `similar(q, Float32)` changes physical components
to `Float32` and spectral components to `ComplexF32`. It also converts each
component grid according to that scalar field's `similar` contract. Parent
storage is newly allocated and zero-valued for ordinary array backends.
"""
Base.similar(q::VectorField{N}, ::Type{T}=eltype(q[1])) where {N, T} = VectorField([Base.similar(q.elements[n], T) for n in 1:N]...)

"""
    copy(q::VectorField) -> VectorField

Copy every component independently into new parent storage.

Each copied component retains its original grid object. Consequently, grid
sharing is preserved when the input components share a grid, but is not newly
enforced by this operation.
"""
Base.copy(q::VectorField{N}) where {N} = VectorField([copy(q.elements[n]) for n in 1:N]...)

"""
    zero(q::VectorField) -> VectorField

Apply `zero` independently to every component. The result has the same number
and type of components, with each component retaining its original grid.
"""
Base.zero(q::VectorField{N}) where {N} = VectorField([zero(q.elements[n]) for n in 1:N]...)
