# Field of modal coefficinets for a vectorfield projected onto a set of modes.

"""
    ProjectedField{G<:AbstractGrid, M}

A representation of a vector field derived from the projection
of an `FTField` onto a set of basis `modes`.

# Fields
- `grid`: concrete instance of `AbstractGrid`
- `data`: modal coefficients stored as a multi-dimensional array
- `modes`: set of basis modes

# Constructors
- `ProjectedField(grid::AbstractGrid, M, modes)`
- `ProjectedField(u::Union{FTField, Field, VectorField}, modes)`
- `project(u::VectorField, modes)`

# Storage layout

The parent array has axis 1 reserved for the mode index and the spectral
dimensions (from `fft_dims(grid) = FFT_DIMS_ORDER`) occupying the subsequent axes in
FFT_DIMS_ORDER order.  Concretely, `parent(a)[m, i_H1, i_H2, …]` gives the coefficient
of mode `m` at the spectral storage index `(i_H1, i_H2, …)`, where `i_H1`
indexes the rfft dimension (`FFT_DIMS_ORDER[1]`) and each subsequent index steps over a
full signed-FFT dimension.  This layout differs from the physical grid layout:
non-FFT (inhomogeneous) dimensions are not present.
"""
struct ProjectedField{G<:AbstractGrid, M, A<:AbstractArray, T, D} <: AbstractArray{Complex{T}, D}
     grid :: G
     data :: A
    modes :: M

    ProjectedField(grid::G, data::A, modes::M) where {T, D, G<:AbstractGrid{T, D}, A<:AbstractArray{<:Any, D}, M} = begin
        # ProjectedField storage is `(mode, fft_dims...)`, not physical grid
        # storage.  The rfft dimension is therefore axis 2, followed by the
        # remaining transformed axes in FFT order.
        new{G, M, A, T, D}(grid, Complex{T}.(normalise_mean!(apply_symmetry!(data, fft_dims(grid)), fft_dims(grid))), modes)
    end
end

ProjectedField(grid::AbstractGrid{T}, data::AbstractArray, modes) where {T} = 
    ProjectedField(grid, 
                   Complex{T}.(data), 
                   modes)

"""
    ProjectedField(grid, modes) -> ProjectedField

Allocate a zero-initialised `ProjectedField` over `grid` with basis `modes`.

`modes` must be an indexable collection (tuple or `Vector`) of arrays, one per
velocity component, where each `modes[n]` has shape `(inh_size..., Nm, kH...)`.
The number of modes `Nm` is taken from `modes[1]` at array dimension
`Ninh + 1`.

As a convenience for single-component projections, a bare `AbstractArray` whose
element type is a `Number` is automatically wrapped in a one-element tuple.
"""
function ProjectedField(grid::AbstractGrid{T, D}, modes) where {T, D}
    Ninh = D - length(fft_dims(grid))
    Nm   = size(modes[1], Ninh + 1)
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
Base.IndexStyle(::Type{<:ProjectedField})                                    = Base.IndexLinear()
Base.parent(a::ProjectedField)                                               = a.data
Base.eltype(::ProjectedField{<:AbstractGrid{T}}) where {T}                   = Complex{T}
Base.size(a::ProjectedField)                                                 = size(parent(a))
Base.similar(a::ProjectedField{<:AbstractGrid{T}}, ::Type{S}=T) where {S, T} =
    ProjectedField(real(S) == T ? grid(a) : convert(real(S), grid(a)), zero(parent(a)), modes(a))
Base.copy(a::ProjectedField)                                                 = ProjectedField(grid(a), copy(parent(a)), modes(a))
Base.zero(a::ProjectedField)                                                 = ProjectedField(grid(a), zero(parent(a)), modes(a))
Base.abs(a::ProjectedField)                                                  = (b = zero(a); parent(b) .= abs.(parent(a)); return b)

modes(a::ProjectedField) = a.modes
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
# Use the storage-index API inside for_each_homogeneous_index loops where the
# caller already holds FFTW 1-based storage indices and must not double-write.

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
    a[m::Int, indices::Int...]
    a[m::Int, indices::Int...] = val

Read or write the coefficient for mode `m` at FFTW 1-based storage indices
`indices` (one per homogeneous dimension in `FFT_DIMS_ORDER` order).

No symmetry invariants are checked or maintained — use this form only inside
[`for_each_homogeneous_index`](@ref) loops where storage indices are already
known and the caller controls every storage location explicitly.
"""
Base.@propagate_inbounds function Base.getindex(a::ProjectedField, m::Int, indices::Int...)
    @boundscheck checkbounds(parent(a), m, indices...)
    @inbounds parent(a)[m, indices...]
end
Base.@propagate_inbounds function Base.setindex!(a::ProjectedField, val, m::Int, indices::Int...)
    @boundscheck checkbounds(parent(a), m, indices...)
    @inbounds parent(a)[m, indices...] = val
end

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
    tpl     = to_indices(grid(a), k)
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
    tpl     = to_indices(grid(a), k)
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
