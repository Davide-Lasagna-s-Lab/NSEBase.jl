# Modal coefficients for a vector field projected onto a set of basis functions.
#
# `ProjectedField` is the reduced-order complement of `FTField`: instead of
# storing full spectral coefficients for every wavenumber and every wall-normal
# point, it stores Nm scalar coefficients per wavenumber, where Nm is the
# number of basis modes.  Each coefficient a[m, k] represents the amplitude of
# basis function φ_m at wavenumber k after projecting a physical velocity field
# onto the basis via an L2 inner product weighted by `weights(grid)`.
#
# Storage layout: the parent array has shape `(Nm, kH1_size, kH2_size, …)`,
# where axis 1 is the mode index and the remaining axes follow `FFT_DIMS_ORDER`
# (rfft dim first).  Non-FFT (inhomogeneous) grid dimensions are NOT present —
# the basis functions absorb the inhomogeneous dependence.
#
# The same Hermitian-symmetry and zero-wavenumber-reality invariants as
# `FTField` are enforced on construction via `apply_symmetry!` and
# `normalise_mean!`.
#
# Three indexing APIs are provided (described in detail below the struct):
#   a[i::Int]                      — linear, for broadcasting / copyto!
#   a[m::Int, i1::Int, …]          — storage-index, for spectral inner loops
#   a[m::Int, k::WaveNumberVector] — wavenumber, public API with symmetry logic

"""
    ProjectedField{G, M, A, T, D}

A representation of a vector field derived from the projection
of an `FTField` onto a set of basis `modes`.

# Type parameters
- `G`: concrete grid type, a subtype of `AbstractGrid{T, _, _, FFT_DIMS_ORDER}`
- `M`: type of the `modes` collection (tuple or vector of arrays)
- `A`: type of the underlying coefficient array (`AbstractArray{Complex{T}, D}`)
- `T`: real floating-point type (e.g. `Float64`); elements are stored as `Complex{T}`
- `D`: dimensionality of the coefficient array — one mode axis plus one axis per
  homogeneous dimension: `D = length(fft_dims(grid)) + 1`

# Fields
- `grid`: concrete instance of `AbstractGrid`
- `data`: modal coefficients stored as a multi-dimensional array of shape
  `(Nm, rfft_size, fft2_size, …)`, where axis 1 is the mode index
- `modes`: set of basis modes

# Constructors
- `ProjectedField(grid::AbstractGrid, modes)` — zero-initialised
- `ProjectedField(grid::AbstractGrid, data, modes)` — wrap existing array
- `ProjectedField(u::Union{FTField, Field, VectorField}, modes)`

# Storage layout

The parent array has axis 1 reserved for the mode index and the spectral
dimensions (from `fft_dims(grid) = FFT_DIMS_ORDER`) occupying the subsequent axes in
`FFT_DIMS_ORDER` order.  Concretely, `parent(a)[m, i_H1, i_H2, …]` gives the coefficient
of mode `m` at the spectral storage index `(i_H1, i_H2, …)`, where `i_H1`
indexes the rfft dimension (`FFT_DIMS_ORDER[1]`) and each subsequent index steps over a
full signed-FFT dimension.  This layout differs from the physical grid layout:
non-FFT (inhomogeneous) dimensions are not present.

# Indexing

Three indexing APIs are available, ordered from lowest to highest level:

- `a[i::Int]` / `a[i] = val` — linear, for broadcasting / `copyto!`; no symmetry logic
- `a[I::CartesianIndex]` / `a[I] = val` — full storage index; no symmetry logic
- `a[m, i1, i2, …]` / `a[m, i1, i2, …] = val` — storage index per axis; no symmetry logic
- `a[m, k::WaveNumberVector]` / `a[m, k] = val` — public API; enforces Hermitian symmetry
  and zero-wavenumber reality (see [`WaveNumberVector`](@ref))
"""
struct ProjectedField{G<:AbstractGrid, M, A<:AbstractArray, T, D} <: AbstractArray{Complex{T}, D}
     grid :: G
     data :: A
    modes :: M

    # bypass constructor for special array types
    ProjectedField(grid::G, data::A, modes::M) where {T, D, G<:AbstractGrid{T, D}, A<:AbstractArray{Complex{T}, D}, M} =
        new{G, M, A, T, D}(grid, data, modes)

    ProjectedField(grid::G, data::A, modes::M) where {T, D, G<:AbstractGrid{T, D}, A<:Array{<:Any, D}, M} =
        # ProjectedField storage is `(mode, fft_dims...)`, not physical grid
        # storage. The rfft dimension is therefore axis 2, followed by the
        # remaining transformed axes in FFT order.
        new{G, M, A, T, D}(grid, Complex{T}.(normalise_mean!(apply_symmetry!(data, fft_dims(grid)), fft_dims(grid))), modes)
end

ProjectedField(grid::AbstractGrid{T}, data::AbstractArray, modes) where {T} = 
    ProjectedField(grid, 
                   Complex{T}.(data), 
                   modes)

"""
    ProjectedField(grid, modes) -> ProjectedField

Allocate a zero-initialised `ProjectedField` over `grid` with basis `modes`.

`modes` must be an indexable collection (tuple or `Vector`) of arrays, one per
velocity component, where each `modes[n]` has shape `(Nm, inh_size..., kH...)`.
The number of modes `Nm` is taken from `modes[1]` at array dimension 1.

As a convenience for single-component projections, a bare `AbstractArray` whose
element type is a `Number` is automatically wrapped in a one-element tuple.
"""
function ProjectedField(grid::AbstractGrid{T, D}, modes) where {T, D}
    Nm   = size(modes[1], 1)
    return ProjectedField(grid,
                          zeros(Complex{T}, Nm,
                                transform_size(grid)[collect(fft_dims(grid))]...),
                          modes)
end

ProjectedField(grid::AbstractGrid, modes::AbstractArray{<:Number}) =
    ProjectedField(grid, (modes,))

ProjectedField(u::Union{FTField, Field, VectorField}, modes) = ProjectedField(grid(u), modes)

# ----------------- #
# interface methods #
# ----------------- #
Base.IndexStyle(::Type{<:ProjectedField}) = Base.IndexLinear()

"""
    parent(a::ProjectedField) -> AbstractArray

Return the underlying coefficient array.  Its shape is
`(Nm, rfft_size, fft2_size, …)` — mode index first, then one axis per
homogeneous dimension in `FFT_DIMS_ORDER` order.
"""
Base.parent(a::ProjectedField) = a.data

Base.eltype(::ProjectedField{<:AbstractGrid{T}}) where {T} = Complex{T}
Base.size(a::ProjectedField) = size(parent(a))

"""
    similar(a::ProjectedField[, ::Type{S}]) -> ProjectedField

Allocate a new zero-initialised `ProjectedField` with the same grid, shape,
and modes as `a`.  Optionally pass element type `S` to change the scalar
precision; the grid is also converted via `convert(real(S), grid(a))`.
"""
Base.similar(a::ProjectedField{<:AbstractGrid{T}}, ::Type{S}=T) where {S, T} =
    ProjectedField(real(S) == T ? grid(a) : convert(real(S), grid(a)), zero(parent(a)), modes(a))

"""
    copy(a::ProjectedField) -> ProjectedField

Return a deep copy of `a` with an independent coefficient array but the same
grid and modes objects.
"""
Base.copy(a::ProjectedField) = ProjectedField(grid(a), copy(parent(a)), modes(a))

"""
    zero(a::ProjectedField) -> ProjectedField

Return a zero-coefficient `ProjectedField` with the same grid and modes as `a`.
"""
Base.zero(a::ProjectedField) = ProjectedField(grid(a), zero(parent(a)), modes(a))

"""
    abs(a::ProjectedField) -> ProjectedField

Return a new `ProjectedField` whose coefficients are the element-wise absolute
values of `a`.  Note: the result does not in general satisfy Hermitian symmetry
and is intended for diagnostics, not further spectral computations.
"""
Base.abs(a::ProjectedField) = (b = zero(a); parent(b) .= abs.(parent(a)); return b)

"""
    modes(a::ProjectedField)

Return the basis-mode collection associated with `a`.  The returned object
is an indexable collection (tuple or `Vector`) with one array per velocity
component; see the Storage layout section of [`ProjectedField`](@ref) for the
required array shape.
"""
modes(a::ProjectedField) = a.modes

"""
    grid(a::ProjectedField) -> AbstractGrid

Return the grid on which `a` is defined.
"""
grid(a::ProjectedField)  = a.grid


# ---------------- #
# indexing methods #
# ---------------- #
#
# Three indexing APIs, listed from lowest to highest level:
#
#   1. a[i::Int]                      — linear (required by IndexLinear)
#   2. a[m::Int, indices::Int...]     — storage-index (for internal loops)
#   3. a[m::Int, k::WaveNumberVector] — wavenumber (public API, enforces symmetry)
#
# Only the WaveNumberVector API maintains Hermitian-symmetry invariants.

"""
    a[i::Int]
    a[i::Int] = val

Linear indexing into the underlying data array. Required by `IndexLinear`;
used by broadcasting and `copyto!`. No symmetry logic applied.
"""
Base.@propagate_inbounds function Base.getindex(u::ProjectedField, i::Int)
    @boundscheck checkbounds(parent(u), i)
    @inbounds val = parent(u)[i]
    return val
end
Base.@propagate_inbounds function Base.setindex!(u::ProjectedField, val, i::Int)
    @boundscheck checkbounds(parent(u), i)
    @inbounds parent(u)[i] = val
    return val
end

"""
    a[m::Int, i1::Int, i2::Int, …]
    a[m::Int, i1::Int, i2::Int, …] = val

Read or write the coefficient for mode `m` at 1-based storage indices
`i1, i2, …` (one index per homogeneous dimension in `FFT_DIMS_ORDER` order;
`M = length(fft_dims(grid(a)))`, one less than the parent array rank `D = M+1`).

`Vararg{Int, M}` forces compile-time specialisation on the number of homogeneous
dimensions so the body can index `parent(a)` with a known-length tuple and
generate no runtime overhead.

No symmetry invariants are checked or maintained.  Use the `WaveNumberVector`
API for writes that must preserve Hermitian symmetry.
"""
Base.@propagate_inbounds function Base.getindex(a::ProjectedField, m::Int, indices::Vararg{Int, M}) where {M}
    @boundscheck checkbounds(parent(a), m, indices...)
    @inbounds parent(a)[m, indices...]
end
Base.@propagate_inbounds function Base.setindex!(a::ProjectedField, val, m::Int, indices::Vararg{Int, M}) where {M}
    @boundscheck checkbounds(parent(a), m, indices...)
    @inbounds parent(a)[m, indices...] = val
end

"""
    a[I::CartesianIndex]
    a[I::CartesianIndex] = val

Read or write the coefficient at the full `D`-dimensional storage index
`I = CartesianIndex(m, i1, i2, …)`, where `m` is the mode index and
`i1, i2, …` are the 1-based storage indices for each homogeneous dimension
in `FFT_DIMS_ORDER` order.

The `CartesianIndex{D}` type parameter ensures this method only dispatches when
the index rank matches the parent array rank (`D = length(fft_dims(grid)) + 1`).
No symmetry logic is applied.  Used internally by loops over `CartesianIndices(a)`,
e.g. in [`shift!`](@ref) for `ProjectedField`.
"""
Base.@propagate_inbounds function Base.getindex(a::ProjectedField{<:Any, <:Any, <:Any, <:Any, D}, I::CartesianIndex{D}) where {D}
    @boundscheck checkbounds(parent(a), I)
    @inbounds parent(a)[I]
end
Base.@propagate_inbounds function Base.setindex!(a::ProjectedField{<:Any, <:Any, <:Any, <:Any, D}, val, I::CartesianIndex{D}) where {D}
    @boundscheck checkbounds(parent(a), I)
    @inbounds parent(a)[I] = val
end

#TODO: can the two functions below be made easier to understand?
"""
    a[m::Int, k::WaveNumberVector]
    a[m::Int, k::WaveNumberVector] = val

Read or write the complex modal coefficient for mode `m` at wavenumber vector
`k`, where `k` follows the order of `fft_dims(grid(a)) = FFT_DIMS_ORDER`.

**Reading** — if the rfft wavenumber `k[1]` is negative, the value is obtained
by conjugate symmetry: the entry stored at `(-k[1], -k[2:N]…)` is returned
conjugated.

**Writing** — two symmetry invariants are maintained automatically:

- *Hermitian symmetry*: when `k[1] == 0`, the mirror entry at `(0, -k[2:N]…)`
  is also updated so the physical field remains real-valued.
- *Zero-wavenumber reality*: the fully-zero wavenumber `k = (0, 0, …)` is
  forced real (imaginary part discarded).

If `k[1] < 0` the write targets the conjugate-symmetric storage location and
`conj(val)` is stored, keeping the representation consistent with reads.
"""
Base.@propagate_inbounds function Base.getindex(a::ProjectedField,
                                                m::Int,
                                                k::WaveNumberVector)
    tpl     = to_homogeneous_indices(grid(a), k)
    do_conj = last(tpl)
    indices = Base.front(tpl)
    @boundscheck checkbounds(a, m, indices...)
    @inbounds val = parent(a)[m, indices...]
    return do_conj ? conj(val) : val
end
Base.@propagate_inbounds function Base.setindex!(a::ProjectedField{G},
                                               val,
                                                 m::Int,
                                                 k::WaveNumberVector{N}) where {T, N, G<:AbstractGrid{T}}
    CT      = Complex{T}
    tpl     = to_homogeneous_indices(grid(a), k)
    do_conj = last(tpl)
    indices = Base.front(tpl)
    i0      = first(indices)      # rfft axis index (axis 2 of ProjectedField)
    rest    = Base.tail(indices)  # signed-fft axis indices (axes 3…)

    # Force the fully-zero wavenumber to be real.
    val = (i0 == 1 && all(==(1), rest)) ? CT(real(val)) : CT(val)

    # Conjugate-symmetric indices for each signed-fft axis.
    sym_rest = ntuple(j -> _fftw_sym_index(rest[j], size(a, j + 2)), Val(N-1))

    @boundscheck checkbounds(a, m, i0, rest...)
    @inbounds parent(a)[m, i0, rest...] = do_conj ? conj(val) : val
    # When the rfft wavenumber is zero, also write the mirror entry so that
    # the Hermitian-symmetry invariant is preserved across all signed dims.
    i0 == 1 && @inbounds parent(a)[m, i0, sym_rest...] = do_conj ? val  : conj(val)
    return val
end

"""
    homogeneous_axes(a::ProjectedField) -> Tuple

Return `(axes(parent(a), 2), axes(parent(a), 3), …)` — one axis range per
homogeneous (FFT) dimension of the grid.

`parent(a)` has shape `(Nm, kH_1_sz, kH_2_sz, …)`: axis 1 is the mode index
and axes 2..Nhom+1 are the kH axes in `FFT_DIMS_ORDER` order.  This function
returns those kH axis ranges so callers can iterate over all wavenumber
combinations without knowing the concrete grid type.
"""
@generated function homogeneous_axes(a::ProjectedField{<:AbstractGrid{<:Any,<:Any,<:Any,FFT_DIMS_ORDER}}) where {FFT_DIMS_ORDER}
    Nhom = length(FFT_DIMS_ORDER)
    return Expr(:tuple, (:(axes(parent(a), $(k+1))) for k in 1:Nhom)...)
end

"""
    apply_symmetry!(a::ProjectedField) -> a

Enforce Hermitian symmetry in-place on the modal coefficient array of `a`.

`parent(a)` has shape `(Nm, kH_1_sz, …, kH_Nhom_sz)`: axis 1 is the mode
index and axes 2..Nhom+1 are the kH axes in `FFT_DIMS_ORDER` order (rfft
first, then signed FFTs).  The symmetry condition `a(-k) = conj(a(k))`
applies at the zero-rfft-wavenumber slice exactly as for `FTField`.

The outer loop is restricted to `rfft_index == 1` (the DC plane of
`parent(a)` axis 2) and iterates all signed-FFT axes fully.  The conjugate
partner is computed by flipping each signed-FFT kH axis via FFTW wrap-around.
Within each `(Ih, Ih_neg)` pair the inner mode loop runs over axis 1 at
stride-1 for cache efficiency, and `_average_complex` enforces
`a[m, -k] = conj(a[m, k])` to floating-point precision.
"""
function apply_symmetry!(a::ProjectedField{<:AbstractGrid{<:Any,<:Any,<:Any,FFT_DIMS_ORDER}}) where {FFT_DIMS_ORDER}
    Nhom = length(FFT_DIMS_ORDER)
    pa   = parent(a)
    LI   = LinearIndices(pa)

    # Restrict the rfft axis (parent axis 2, kH position k==1) to index 1.
    # At kx > 0 the negative-kx partner is not stored, so there is nothing
    # to symmetrize.  All other kH axes (signed-FFT) are iterated fully.
    dc_range = ntuple(k -> k == 1 ? (1:1) : axes(pa, k + 1), Val(Nhom))
    @inbounds for Ih in CartesianIndices(dc_range)

        # Conjugate-symmetric kH partner: k==1 is the rfft axis (self-partner),
        # k>1 are signed-FFT axes (wrap-around, or self at zero wavenumber).
        Ih_neg = CartesianIndex(ntuple(Val(Nhom)) do k
            if k == 1
                return Ih[k]
            elseif Ih[k] == 1
                return Ih[k]
            else
                return size(pa, k + 1) - Ih[k] + 2
            end
        end)

        # Mode axis is innermost for stride-1 memory access.
        for Im in axes(pa, 1)
            I    = CartesianIndex(Im, Ih.I...)
            Ineg = CartesianIndex(Im, Ih_neg.I...)

            if LI[I] <= LI[Ineg]
                av = _average_complex(pa[I], pa[Ineg])
                pa[I]    = av
                pa[Ineg] = conj(av)
            end
        end
    end
    return a
end
