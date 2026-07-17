# Physical-space scalar field wrapping a raw Julia array and an AbstractGrid.
#
# A `Field` is the physical-space counterpart of `FTField`: it stores real values
# at the collocation points defined by `points(grid)`.  The type tracks which
# grid it lives on so that transforms, derivatives, and inner products can
# dispatch correctly without the caller carrying separate grid arguments.
#
# Broadcasting delegates to the underlying `data` array through the
# `FieldType` style defined in broadcasting.jl, so `u .* v`, `@. f(u)`, etc.
# all work element-wise.  The constructor accepting a `Function` evaluates the
# function at every grid point via broadcasting, making it easy to initialise
# analytical initial conditions.

"""
    Field{G} where {G<:AbstractGrid}

Physical-space scalar field on a concrete sub-type of `AbstractGrid`.

Values are stored at the collocation points returned by `points(grid)`.
The type parameter `G` encodes the grid geometry, scalar type `T`, array
dimensionality `D`, axis layout `AXES`, and FFT dimension order
`FFT_DIMS_ORDER` — all without any runtime overhead.

# Fields
- `grid`: the grid on which the field is defined
- `data`: the underlying `AbstractArray` holding real values at each collocation point
"""
struct Field{G<:AbstractGrid, A<:AbstractArray, T, D} <: AbstractArray{T, D}
    grid::G
    data::A

    Field(grid::G, data::A) where {T, D, H, G<:AbstractGrid{T, D, H}, A<:AbstractArray{T, D}} =
        new{G, A, T, D}(grid, data)
end

"""
    Field(grid::AbstractGrid, data::AbstractArray)

Construct a `Field` from `grid` and a pre-existing array `data`.  If the
element type of `data` does not match `T` of the grid, the array is converted.
"""
Field(grid::AbstractGrid{T}, data::AbstractArray) where {T} = Field(grid, T.(data))

"""
    Field(grid::AbstractGrid, func::Function; dealias=false)

Construct a `Field` by evaluating `func` at every collocation point.

`func` must accept four arguments in physical `(x, y, z, t)` order and return a
scalar. A coordinate absent from the grid is passed as `nothing`. This calling
convention is independent of storage order: on a channel stored as
`(y, x, z, t)`, for example, use `func(x, y, z, t)`.

The coordinate arrays are obtained via `points(grid; dealias=dealias)`, so
setting `dealias=true` constructs a field on the padded physical grid. This is
the appropriate layout for nonlinear products that will be transformed back to
spectral space.

# Examples

A full channel is stored as `(y, x, z, t)`, but the field function still
receives coordinates in physical `(x, y, z, t)` order:

```julia
g = ChannelGrid(9, 17, 9; Nt=3, α=0.5, β=1, width=5)
u = Field(g, (x, y, z, t) -> (1 - y^2) * cos(x + z - 2π*t))
```

For a two-dimensional lid-driven cavity, the absent `z` coordinate is passed
as `nothing`:

```julia
g = LidDrivenCavityGrid(11, 11; Nt=3, width=5)
ψ = Field(g, (x, y, ::Nothing, t) -> sinpi(x) * sinpi(y) * cospi(2t))
```
"""
function Field(grid::AbstractGrid{T}, func::Function; dealias=false) where {T}
    return Field(grid, T.(func.(_field_args(grid; dealias)...)))
end

# Select one storage-order point array, while preserving `nothing` for a
# coordinate that is absent from the grid.
@inline _field_arg(::Tuple, ::Val{nothing}) = nothing
@inline _field_arg(points::Tuple, ::Val{DIM}) where {DIM} = points[DIM]

"""
    _field_args(grid; dealias=false) -> Tuple

Return the four broadcast arguments used by the function-valued [`Field`](@ref)
constructor in physical `(x, y, z, t)` order. The grid's `AXES` parameter maps
each present coordinate to its storage-order array from [`points`](@ref), while
an absent coordinate becomes `nothing`. The `Val`-based selection keeps the
heterogeneous tuple fully inferred for every concrete grid layout.
"""
function _field_args(grid::AbstractGrid{<:Any, <:Any, AXES}; dealias=false) where {AXES}
    grid_points = points(grid; dealias)
    return (_field_arg(grid_points, Val(AXES[1])), _field_arg(grid_points, Val(AXES[2])),
            _field_arg(grid_points, Val(AXES[3])), _field_arg(grid_points, Val(AXES[4])))
end

"""
    Field(grid::AbstractGrid; dealias=false)

Construct a zero-initialised `Field` on `grid`.  Setting `dealias=true` uses
the padded grid size, matching the layout expected by `FFTPlans` with dealiasing.
"""
Field(grid::AbstractGrid{T}; dealias=false) where {T} = Field(grid, (_...)->zero(T); dealias)

# ---------------------------------- #
# AbstractArray interface             #
# ---------------------------------- #
Base.IndexStyle(::Type{<:Field}) = Base.IndexLinear()

"""
    parent(u::Field) -> AbstractArray

Return the underlying data array of `u`.  Prefer `parent(u)` over `u.data` in
library code so that the abstraction boundary is maintained.
"""
Base.parent(u::Field) = u.data

Base.eltype(::Field{<:AbstractGrid{T}}) where {T} = T

"""
    similar(u::Field[, ::Type{S}]) -> Field

Allocate a new `Field` of the same shape as `u`, optionally with element type
`S`.  If `S` differs from `T` the grid is also converted to scalar type `S`
via `convert(S, grid(u))`.
"""
Base.similar(u::Field{<:AbstractGrid{T}}, ::Type{S}=T) where {T, S} =
    Field(S == T ? grid(u) : convert(S, grid(u)), zero(parent(u)))

Base.size(u::Field) = Base.size(parent(u))

"""
    copy(u::Field) -> Field

Return a deep copy of `u` sharing the same grid but with an independent data
array.
"""
Base.copy(u::Field) = (v = Base.similar(u); parent(v) .= parent(u); return v)

"""
    zero(u::Field) -> Field

Return a zero-valued `Field` on the same grid as `u`.
"""
Base.zero(u::Field{<:AbstractGrid{T}}) where {T} = (v = Base.similar(u); parent(v) .= zero(T); return v)

# Linear indexing — required by IndexLinear.  Bounds checking is delegated to
# the underlying array; @propagate_inbounds propagates any @inbounds annotation
# from the call site into this method body.
Base.@propagate_inbounds function Base.getindex(u::Field, i)
    @boundscheck checkbounds(parent(u), i)
    @inbounds v = parent(u)[i]
    return v
end

Base.@propagate_inbounds function Base.setindex!(u::Field, v, i)
    @boundscheck checkbounds(parent(u), i)
    @inbounds parent(u)[i] = v
    return v
end

"""
    grid(u::Field) -> AbstractGrid

Return the grid on which `u` is defined.
"""
grid(u::Field) = u.grid
