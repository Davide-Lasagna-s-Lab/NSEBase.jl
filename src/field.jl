# Physical-space scalar fields and their AbstractArray interface.
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
# analytical initial conditions. Constructors intentionally leave grid/data
# compatibility beyond scalar type and rank to the caller.

"""
    Field{G, A, T, D} <: AbstractArray{T, D}

Physical-space scalar field on an [`AbstractGrid`](@ref).

`Field` is a thin array wrapper: its array shape and axes are exactly those of
`data`, and scalar indexing reads and writes `data` directly. A field created
from a function normally has the shape of the arrays returned by
`points(grid)`. The data constructor requires `data` to have the same rank `D`
as `grid`, but it does not compare their sizes or axes; callers that provide
storage directly are responsible for that compatibility.

# Type parameters

- `G`: concrete grid type
- `A`: concrete parent-array type
- `T`: grid scalar type and field element type
- `D`: rank shared by the grid and parent array

# Fields

- `grid`: the grid associated with the field
- `data`: the parent array holding the collocation values

The fields are exposed through [`grid`](@ref) and `parent`. `parent(u)` aliases
the stored array; mutating either object changes the same values.
"""
struct Field{G<:AbstractGrid, A<:AbstractArray, T, D} <: AbstractArray{T, D}
    grid::G
    data::A

    Field(grid::G, data::A) where {T, D, H, G<:AbstractGrid{T, D, H}, A<:AbstractArray{T, D}} =
        new{G, A, T, D}(grid, data)
end

"""
    Field(grid::AbstractGrid, data::AbstractArray)

Construct a `Field` from `grid` and pre-existing storage `data`.

If `data` already has element type `T`, where `T` is the grid scalar type, and
the same rank as `grid`, it is stored directly without copying. Otherwise,
`T.(data)` converts the values by broadcasting and the resulting array is
stored. For ordinary Julia arrays that conversion allocates new storage.

The constructor does not check that `size(data)` agrees with the grid's
collocation layout. In particular, `size(Field(grid, data)) == size(data)`;
the caller must provide compatible axes and sizes.
"""
Field(grid::AbstractGrid{T}, data::AbstractArray) where {T} = Field(grid, T.(data))

# TODO: we should change the interface and require func to take x, y, z, and t in physical order, not array order
# TODO: we then also need to change the code in resolver-channelflow to match this, and the code in the tests
"""
    Field(grid::AbstractGrid, func::Function; dealias=false)

Construct a physical field by evaluating `func` at every collocation point.

The call is equivalent to

```julia
Field(grid, T.(func.(points(grid; dealias=dealias)...)))
```

where `T` is the grid scalar type. Consequently, `func` receives one argument
per array dimension in the storage order returned by `points`; this is not a
promise of physical `x, y, z` order. Its return values must be convertible to
`T`. The broadcast allocates the parent array.

With `dealias=true`, `points` supplies the padded collocation layout used for
dealiased nonlinear products.
"""
Field(grid::AbstractGrid{T}, func::Function; dealias=false) where {T} = Field(grid, T.(func.(points(grid, dealias=dealias)...)))

"""
    Field(grid::AbstractGrid; dealias=false)

Construct a zero-valued physical field on `grid`.

The parent shape is determined by `points(grid; dealias=dealias)`. Setting
`dealias=true` therefore uses the padded physical layout expected by dealiased
transform plans.
"""
Field(grid::AbstractGrid{T}                ; dealias=false) where {T} = Field(grid, (pts...)->zero(T); dealias=dealias)

# -----------------------------------------------------------------------------
# AbstractArray interface
# -----------------------------------------------------------------------------

"""
    IndexStyle(::Type{<:Field}) -> IndexLinear()

Declare linear indexing as the native indexing style of `Field`.
"""
Base.IndexStyle(::Type{<:Field}) = Base.IndexLinear()

"""
    parent(u::Field) -> AbstractArray

Return the exact array stored by `u`; no copy is made.
"""
Base.parent(u::Field) = u.data

"""
    eltype(u::Field) -> Type

Return the scalar type of `grid(u)`, which is also the element type of
`parent(u)`.
"""
Base.eltype(::Field{<:AbstractGrid{T}}) where {T} = T

"""
    similar(u::Field[, ::Type{S}]) -> Field

Allocate a zero-valued field using `zero(parent(u))` as its storage.

The result therefore inherits the parent's shape and allocation behavior. If
`S == T`, the result references the same grid object as `u`; otherwise it uses
`convert(S, grid(u))`, so the grid and field precision change together. The
new parent does not alias `parent(u)` for ordinary array implementations.
"""
Base.similar(u::Field{<:AbstractGrid{T}}, ::Type{S}=T) where {T, S} =
    Field(S == T ? grid(u) : convert(S, grid(u)), zero(parent(u)))

"""
    size(u::Field) -> Tuple

Return `size(parent(u))`. The result is determined by the stored array, not by
a separate query of the grid.
"""
Base.size(u::Field) = Base.size(parent(u))

"""
    copy(u::Field) -> Field

Copy the values of `u` into a newly allocated parent array.

The result references the same grid object and, for ordinary arrays, does not
alias `parent(u)`.
"""
Base.copy(u::Field) = (v = Base.similar(u); parent(v) .= parent(u); return v)

"""
    zero(u::Field) -> Field

Return a zero-valued field with the same grid, shape, and scalar type as `u`.
"""
Base.zero(u::Field{<:AbstractGrid{T}}) where {T} = (v = Base.similar(u); parent(v) .= zero(T); return v)

"""
    u[i]

Read `parent(u)[i]`. Bounds and supported single-index forms are those of the
parent array.
"""
Base.@propagate_inbounds function Base.getindex(u::Field, i)
    @boundscheck checkbounds(parent(u), i)
    @inbounds v = parent(u)[i]
    return v
end

"""
    u[i] = value

Assign `value` to `parent(u)[i]` and return `value`, following Julia's array
assignment convention. The parent may convert the value before storing it, so
the returned object need not be identical to the subsequently read value.
"""
Base.@propagate_inbounds function Base.setindex!(u::Field, v, i)
    @boundscheck checkbounds(parent(u), i)
    @inbounds parent(u)[i] = v
    return v
end

"""
    grid(u::Field) -> AbstractGrid

Return the exact grid object stored by `u`.
"""
grid(u::Field) = u.grid
