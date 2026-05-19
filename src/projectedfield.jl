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
dimensions (from `fft_dims(grid) = ORDER`) occupying the subsequent axes in
ORDER order.  Concretely, `parent(a)[m, i_H1, i_H2, …]` gives the coefficient
of mode `m` at the spectral storage index `(i_H1, i_H2, …)`, where `i_H1`
indexes the rfft dimension (`ORDER[1]`) and each subsequent index steps over a
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

ProjectedField(grid::AbstractGrid{T, D, H}, modes) where {T, D, H} =
    ProjectedField(grid, 
                   zeros(Complex{T}, no_of_modes(modes), 
                   transform_size(grid)[collect(fft_dims(grid))]...), 
                   modes)

ProjectedField(u::Union{FTField, Field, VectorField}, modes) = ProjectedField(grid(u), modes)

no_of_modes(modes) = throw(NotImplementedError(modes))

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
# Linear indexing — delegates straight to the underlying data array.
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
    a[m, n::ModeNumber]

Return the complex modal coefficient for mode `m` at wavenumber tuple `n`.

The wavenumbers in `n` follow the order of `fft_dims(grid(a)) = ORDER`.
If the first (rfft) wavenumber `n.ns[1]` is negative the coefficient is
obtained by conjugate symmetry: the entry stored at `(-n.ns[1], -n.ns[2:N]…)`
is read and conjugated.
"""
Base.@propagate_inbounds function Base.getindex(a::ProjectedField,
                                                m::Int,
                                                n::ModeNumber)
    tpl     = _modenumber_to_indices(grid(a), n)
    do_conj = last(tpl)
    indices = Base.front(tpl)
    @boundscheck checkbounds(a, m, indices...)
    @inbounds val = parent(a)[m, indices...]
    return do_conj ? conj(val) : val
end

"""
    a[m, n::ModeNumber] = val

Write the complex modal coefficient `val` for mode `m` at wavenumber tuple `n`.

Two symmetry invariants are maintained automatically:

- **Hermitian symmetry** — when the rfft wavenumber `n.ns[1] == 0`, the
  conjugate-symmetric entry at `(0, -n.ns[2:N]…)` is also updated so that
  the physical field remains real-valued.
- **Zero-mode reality** — the fully-zero mode `ModeNumber(0, 0, …)` is
  forced to be real (imaginary part discarded).

If `n.ns[1] < 0` the write targets the conjugate-symmetric storage location
and `conj(val)` is stored, keeping the representation consistent with reads.
"""
Base.@propagate_inbounds function Base.setindex!(a::ProjectedField{G},
                                               val,
                                                 m::Int,
                                                 n::ModeNumber{N}) where {T, N, G<:AbstractGrid{T}}
    CT      = Complex{T}
    tpl     = _modenumber_to_indices(grid(a), n)
    do_conj = last(tpl)
    indices = Base.front(tpl)
    i0      = first(indices)      # rfft axis index (axis 2 of ProjectedField)
    rest    = Base.tail(indices)  # signed-fft axis indices (axes 3…)

    # Force the fully-zero mode to be real.
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

