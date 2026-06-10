# Fourier-transformed scalar field and its low-level spectral symmetry utilities.
#
# `FTField` is the spectral counterpart of `Field`: it stores complex Fourier
# coefficients on the half-spectrum produced by an rfft.  The first axis of
# the underlying array is the rfft axis (non-negative wavenumbers only);
# the remaining FFT-transformed axes store both positive and negative wavenumbers
# in FFTW's standard wrap-around order.  Non-FFT (inhomogeneous) dimensions
# appear in the array at whatever position the grid's `AXES` and `FFT_DIMS_ORDER`
# specify.
#
# FTField enforces two invariants on construction (when given a plain Julia Array):
#   1. Hermitian symmetry — the zero-rfft-wavenumber plane must satisfy û(-k) = conj(û(k)).
#      Enforced by `apply_symmetry!`.
#   2. Reality of the zero wavenumber — the fully-zero mode must be real-valued.
#      Enforced by `normalise_mean!`.
#
# Two indexing APIs are provided beyond the base IndexLinear interface:
#   u[k::WaveNumberVector]         — returns a view over all inhomogeneous indices
#                                    at signed wavenumber k, handling conjugate-
#                                    symmetry for negative rfft wavenumbers.
#   u[k::WaveNumberVector, I...]   — returns (or writes) the single complex
#                                    coefficient at wavenumber k and inhomogeneous
#                                    index I, with full symmetry maintenance on write.
#
# `apply_symmetry!` and `normalise_mean!` are `@generated` utilities defined at
# the bottom of this file and are also used by `ProjectedField` on construction.
#
# `homogeneous_axes(u)` and `inhomogeneous_axes(u)` are @generated helpers that
# return the axis ranges for the FFT and non-FFT storage dimensions of `u`.
# They are resolved at compile time from the grid type embedded in `FTField{G}`
# so no grid argument is needed.  Both are used by `shift!` and `growto`.

"""
    FTField{G} where {G<:AbstractGrid}

Fourier transformed representation of a scalar field on
a concrete sub-type of AbstractGrid.

# Fields
- `grid`: concrete instance of `AbstractGrid`
- `data`: fourier coefficient values of scalar field
"""
struct FTField{G<:AbstractGrid, A<:AbstractArray, T, D} <: AbstractArray{Complex{T}, D}
    grid::G
    data::A

    # generic constructor that bypasses sanitation
    FTField(grid::G, data::A) where {T, D, G<:AbstractGrid{T, D}, A<:AbstractArray{Complex{T}, D}} =
        new{G, typeof(data), T, D}(grid, data)

    # main constructor which sanitises data
    FTField(grid::G, data::A) where {T, D, G<:AbstractGrid{T, D}, A<:Array{Complex{T}, D}} =
        new{G,A,T,D}(grid, apply_symmetry!(normalise_mean!(data, fft_storage_dims(grid)), fft_storage_dims(grid)))
end
FTField(grid::AbstractGrid{T}, data::AbstractArray) where {T} = FTField(grid, Complex{T}.(data))

# construct from standard array and sanitise input
FTField(grid::AbstractGrid{T}) where {T} = FTField(grid, zeros(Complex{T}, transform_size(grid)))

Base.IndexStyle(::Type{<:FTField})                                    = Base.IndexLinear()
Base.parent(u::FTField)                                               = u.data
Base.eltype(::FTField{<:AbstractGrid{T}}) where {T}                   = Complex{T}
Base.similar(u::FTField{<:AbstractGrid{T}}, ::Type{S}=T) where {T, S} =
    FTField(real(S) == T ? grid(u) : convert(real(S), grid(u)), zero(parent(u)))
Base.size(u::FTField)                                                 = Base.size(parent(u))
Base.copy(u::FTField)                                                 = (v = Base.similar(u); parent(v) .= parent(u); return v)
Base.zero(u::FTField{<:AbstractGrid{T}}) where {T}                    = (v = Base.similar(u); parent(v) .= zero(T)  ; return v)

# linear indexing
Base.@propagate_inbounds function Base.getindex(u::FTField, i::Int)
    @boundscheck checkbounds(parent(u), i)
    @inbounds v = parent(u)[i]
    return v
end

Base.@propagate_inbounds function Base.setindex!(u::FTField, v, i::Int)
    @boundscheck checkbounds(parent(u), i)
    @inbounds parent(u)[i] = v
    return v
end

# indexing with a cartesianindex
Base.@propagate_inbounds function Base.getindex(u::FTField, I::CartesianIndex)
    @boundscheck checkbounds(parent(u), I)
    @inbounds v = parent(u)[I]
    return v
end

Base.@propagate_inbounds function Base.setindex!(u::FTField, v, I::CartesianIndex)
    @boundscheck checkbounds(parent(u), I)
    @inbounds parent(u)[I] = v
    return v
end

grid(u::FTField) = u.grid

"""
    growto(u::FTField{G}, target_size::NTuple{N, Int}) where {G<:AbstractGrid, N}

Return an equivalent spectral field with a new homogeneous resolution.

`target_size` contains one physical-space grid size for each homogeneous
direction in `fft_storage_dims(grid(u)) = FFT_DIMS_ORDER`, in `FFT_DIMS_ORDER` order. It must therefore
have length `length(FFT_DIMS_ORDER)`, not the full dimension `D` of `grid(u)`.
Inhomogeneous directions are preserved by the grid-specific
`growto(grid(u), target_size)` method.

The spectral coefficients are embedded using FFTW storage conventions: the
rfft block remains a prefix, while negative modes in each signed FFT dimension
are copied to the high end of the corresponding target dimension. Equivalently,
each source coefficient appears at the same signed wavenumber in the output and
newly introduced wavenumbers are left exactly zero. This requires the target
grid to represent every copied source wavenumber, so this method is primarily
intended for increasing homogeneous resolution.
"""
function growto(u::FTField{G}, target_size::NTuple{N, Int}) where {FFT_DIMS_ORDER, N, G<:AbstractGrid{<:Any, <:Any, <:Any, FFT_DIMS_ORDER}}
    N == length(FFT_DIMS_ORDER) ||
        throw(ArgumentError("target_size has incompatible size: expected a tuple of length $(length(FFT_DIMS_ORDER)), got length $N"))
    out = FTField(growto(grid(u), target_size))
    _copy_to_padded!(parent(out), parent(u), FFT_DIMS_ORDER)
    return out
end

"""
    u[k::WaveNumberVector]

Return a view of the underlying array over all non-transform dimensions for
the wavenumber vector `k`. The wavenumbers in `k` follow the order
of `fft_storage_dims(grid(u)) = FFT_DIMS_ORDER`.

If the first (rfft) wavenumber `k[1]` is negative the view is into the
conjugate-symmetric storage location `(-k[1], -k[2:N]…)`; the caller
is responsible for applying conjugation if needed.

The returned view has one dimension for each axis of `u` not in `FFT_DIMS_ORDER`,
in their original order.
"""
Base.@propagate_inbounds function Base.getindex(u::FTField{G},
                                                k::WaveNumberVector) where {FFT_DIMS_ORDER, G<:AbstractGrid{<:Any, <:Any, <:Any, FFT_DIMS_ORDER}}
    tpl     = to_homogeneous_indices(grid(u), k)
    indices = Base.front(tpl)
    colons  = ntuple(_ -> Colon(), ndims(u) - length(FFT_DIMS_ORDER))
    @boundscheck checkbounds(u, combine_indices(grid(u), colons, indices)...)
    @inbounds return view(parent(u), combine_indices(grid(u), colons, indices)...)
end

"""
    u[k::WaveNumberVector, I...]

Return the complex modal coefficient for index `I` at wavenumber vector `k`.

The wavenumbers in `k` follow the order of `fft_storage_dims(grid(a)) = FFT_DIMS_ORDER`.
If the first (rfft) wavenumber `k[1]` is negative the coefficient is
obtained by conjugate symmetry: the entry stored at `(-k[1], -k[2:N]…)`
is read and conjugated.
"""
Base.@propagate_inbounds function Base.getindex(u::FTField,
                                                k::WaveNumberVector,
                                               i1::Int,
                                                I::Vararg{Int})
    tpl     = to_homogeneous_indices(grid(u), k)
    do_conj = last(tpl)
    indices = Base.front(tpl)
    @boundscheck checkbounds(u, combine_indices(grid(u), (i1, I...), indices)...)
    @inbounds val = parent(u)[combine_indices(grid(u), (i1, I...), indices)...]
    return do_conj ? conj(val) : val
end

"""
    u[k::WaveNumberVector, I...] = val

Write the complex modal coefficient `val` for index `I` at wavenumber vector `k`.

Two symmetry invariants are maintained automatically:

- **Hermitian symmetry** — when the rfft wavenumber `k[1] == 0`, the
  conjugate-symmetric entry at `(0, -k[2:N]…)` is also updated so that
  the physical field remains real-valued.
- **Zero-wavenumber reality** — the fully-zero wavenumber `WaveNumberVector(0, 0, …)` is
  forced to be real (imaginary part discarded).

If `k[1] < 0` the write targets the conjugate-symmetric storage location
and `conj(val)` is stored, keeping the representation consistent with reads.
"""
Base.@propagate_inbounds function Base.setindex!(u::FTField{<:AbstractGrid{T}},
                                               val,
                                                 k::WaveNumberVector{N},
                                                 I::Vararg{Int}) where {T, N}
    CT      = Complex{T}
    tpl     = to_homogeneous_indices(grid(u), k)
    do_conj = last(tpl)
    indices = Base.front(tpl)
    i0      = first(indices)      # rfft axis index (axis 2 of ProjectedField)
    rest    = Base.tail(indices)  # signed-fft axis indices (axes 3…)

    # Force the fully-zero wavenumber to be real.
    val = (i0 == 1 && all(==(1), rest)) ? CT(real(val)) : CT(val)

    # Conjugate-symmetric indices for each signed-fft axis.
    sym_rest = ntuple(j -> _fftw_sym_index(rest[j], size(u, j + 2)), Val(N-1))

    @boundscheck checkbounds(u, combine_indices(grid(u), I, (i0, rest...))...)
    @inbounds parent(u)[combine_indices(grid(u), I, (i0, rest...))...] = do_conj ? conj(val) : val
    # When the rfft wavenumber is zero, also write the mirror entry so that
    # the Hermitian-symmetry invariant is preserved across all signed dims.
    i0 == 1 && @inbounds parent(u)[combine_indices(grid(u), I, (i0, sym_rest...))...] = do_conj ? val  : conj(val)
    return val
end

# --------------- #
# utility methods #
# --------------- #

"""
    apply_symmetry!(u::FTField) -> u

Enforce Hermitian symmetry in-place on the spectral coefficient array of `u`.

The symmetry condition `û(-k) = conj(û(k))` must hold for the signed-FFT
wavenumbers **at the zero-frequency slice of the rfft axis only** — at
non-zero rfft wavenumbers the negative-kx partner is not stored, so there is
nothing to enforce.

The outer loop is restricted to `rfft_index == 1` (the DC plane) and iterates
all signed-FFT axes fully.  For each DC-plane kH index `Ih`, the conjugate
partner `Ih_neg` is computed by flipping the signed-FFT axes:

- rfft axis (k==1 in `FFT_DIMS_ORDER`): `Ih_neg` carries the same index.
- signed-FFT axis at index 1 (wavenumber 0): self-conjugate, `Ih_neg = Ih`.
- signed-FFT axis at index `i > 1`: wrap-around `Ih_neg = size(u, dim) - i + 2`.

The inner inhomogeneous loop (over `inhomogeneous_axes`) then maps each
`(Ih, Ih_neg)` kH pair to a full array index via `combine_indices` and
averages each `(I, Ineg)` pair exactly once (guarded by `LI[I] <= LI[Ineg]`)
using `_average_complex`, which enforces `û(-k) = conj(û(k))` to
floating-point precision.
"""
function apply_symmetry!(u::FTField{<:AbstractGrid{<:Any,<:Any,<:Any,FFT_DIMS_ORDER}}) where {FFT_DIMS_ORDER}
    # we'll use this to avoid doing the same work twice for each conjugate pair
    LI = LinearIndices(parent(u))
    g  = grid(u)

    # Restrict the rfft axis (k==1 in the kH tuple) to index 1 — the DC plane.
    # At kx > 0 the negative-kx partner is not stored, so there is nothing to
    # symmetrize.  All other kH axes (signed-FFT) are iterated fully.
    dc_range = ntuple(Val(length(FFT_DIMS_ORDER))) do k
        k == 1 ? (1:1) : axes(u, FFT_DIMS_ORDER[k])
    end
    
    @inbounds for Ih in CartesianIndices(dc_range)

        Ih_neg = CartesianIndex(ntuple(Val(length(FFT_DIMS_ORDER))) do d
            if FFT_DIMS_ORDER[d] == rfft_storage_dim(g)
                # rfft axis: partner is the same index, no conjugation.
                return Ih[d]
            else
                if Ih[d] == 1
                    # zero wavenumber: self-conjugate.
                    return Ih[d]
                else
                    # negative wavenumber partner using FFTW wrap-around order.
                    return size(u, FFT_DIMS_ORDER[d]) - Ih[d] + 2
                end
            end
        end)

        # loop over the inhomogeneous indices and average the conjugate pair
        for Inh in CartesianIndices(inhomogeneous_axes(u))
            I    = CartesianIndex(combine_indices(g, Inh, Ih))
            Ineg = CartesianIndex(combine_indices(g, Inh, Ih_neg))

            if LI[I] <= LI[Ineg]
                av = _average_complex(parent(u)[I], parent(u)[Ineg])
                parent(u)[I]    = av
                parent(u)[Ineg] = conj(av)
            end
        end
    end
    return u
end

"""
    normalise_mean!(data::Array{<:Any, D}, ::Val{FFT_DIMS_ORDER})

Enforce purely real values at mean component of Fourier transformed
data.
"""
@generated function normalise_mean!(data::Array{<:Any, D}, ::Val{FFT_DIMS_ORDER}) where {D, FFT_DIMS_ORDER}
    indxs = [d ∈ FFT_DIMS_ORDER ? 1 : :(:) for d in 1:D]
    slice = Expr(:ref, :data, indxs...)
    return :(@views $slice .= real.($slice); return data)
end
normalise_mean!(data, FFT_DIMS_ORDER::Dims) = normalise_mean!(data, Val(FFT_DIMS_ORDER))

"""
    homogeneous_axes(u::FTField) -> Tuple

Return a tuple of `axes(u, d)` for each storage dimension `d` in
`FFT_DIMS_ORDER` — the homogeneous (FFT) dimensions of the grid.

The result is suitable for `CartesianIndices(homogeneous_axes(u))` to
iterate over all spectral wavenumber indices of `u` without touching the
inhomogeneous (quadrature) dimension(s).

The grid type is extracted from `u` at compile time, so this call
produces zero allocations at runtime.
"""
@generated function homogeneous_axes(u::FTField{<:AbstractGrid{<:Any, D, <:Any, FFT_DIMS_ORDER}}) where {D, FFT_DIMS_ORDER}
    return Expr(:tuple, (:(axes(u, $d)) for d in FFT_DIMS_ORDER)...)
end

"""
    inhomogeneous_axes(u::FTField) -> Tuple

Return a tuple of `axes(u, d)` for each storage dimension `d` that is
**not** in `FFT_DIMS_ORDER` — the inhomogeneous (non-FFT, e.g. Chebyshev)
dimensions of the grid.  Complementary to [`homogeneous_axes`](@ref).

Used as the inner loop range in [`shift!`](@ref) so that each inhomogeneous
index is updated via a scalar `parent[I...] *= phase` with no temporary
slice allocation.

The grid type is extracted from `u` at compile time, so this call
produces zero allocations at runtime.
"""
@generated function inhomogeneous_axes(u::FTField{<:AbstractGrid{<:Any, D, <:Any, FFT_DIMS_ORDER}}) where {D, FFT_DIMS_ORDER}
    inh_dims = [d for d in 1:D if d ∉ FFT_DIMS_ORDER]
    return Expr(:tuple, (:(axes(u, $d)) for d in inh_dims)...)
end
