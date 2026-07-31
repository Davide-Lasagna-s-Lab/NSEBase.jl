# Fourier-transformed scalar fields and their AbstractArray interface.
#
# `FTField` is the spectral counterpart of `Field`: it stores complex Fourier
# coefficients on the half-spectrum produced by an rfft. The storage dimension
# `fft_storage_dims(grid)[1]` is the rfft dimension (non-negative wavenumbers
# only); the remaining transformed dimensions store signed modes in FFTW's
# wrap-around order. Non-transformed dimensions remain in their physical array
# positions.
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
# The raw-array `apply_symmetry!` utility is defined with the grid interface;
# `normalise_mean!` is defined below. Both are also used when constructing a
# `ProjectedField` from a plain Julia array.
#
# `homogeneous_axes(u)` and `inhomogeneous_axes(u)` are @generated helpers that
# return the axis ranges for the FFT and non-FFT storage dimensions of `u`.
# They are resolved at compile time from the grid type embedded in `FTField{G}`
# so no grid argument is needed.  Both are used by `shift!` and `growto`.

"""
    FTField{G, A, T, D} <: AbstractArray{Complex{T}, D}

Fourier-transformed scalar field on an [`AbstractGrid`](@ref).

The parent array has the same rank `D` as the grid and normally has shape
`transform_size(grid)`. Its dimensions remain in grid storage order. The first
entry of `fft_storage_dims(grid)` identifies the rfft dimension, while later
entries identify full signed-FFT dimensions. Dimensions not listed there hold
the inhomogeneous collocation indices.

The data constructor enforces matching scalar type and rank through dispatch,
but does not compare `size(data)` with `transform_size(grid)`. Callers that
provide storage directly are responsible for that compatibility.

# Type parameters

- `G`: concrete grid type
- `A`: concrete parent-array type
- `T`: real grid scalar type; field elements are `Complex{T}`
- `D`: rank shared by the grid and parent array

# Fields

- `grid`: grid associated with the coefficients
- `data`: complex Fourier coefficient array

# Ownership and sanitation

For a plain `Array{Complex{T},D}`, construction stores the exact array and
mutates it in place to enforce Hermitian symmetry and a real mean mode. An
exact-typed non-`Array` implementation is stored directly without copying or
sanitation, allowing specialized backends to manage those operations
themselves. If value conversion is needed, `Complex{T}.(data)` is materialized
first; whether the converted backend then uses the sanitizing or direct path
depends on its resulting array type.

Direct writes through `parent`, linear indices, or Cartesian indices do not
maintain spectral symmetry. Wavenumber-based assignment is the public write
interface that maintains it.
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

"""
    FTField(grid::AbstractGrid, data::AbstractArray)

Construct a spectral field using coefficient storage `data`.

The array must have the same rank as `grid`. Values are converted to the grid's
complex scalar type with `Complex{T}.` when necessary. A correctly sized
spectral array normally has `size(data) == transform_size(grid)`, but this
constructor does not validate that equality. See [`FTField`](@ref) for the
copying and sanitation rules of plain and specialized arrays.
"""
FTField(grid::AbstractGrid{T}, data::AbstractArray) where {T} = FTField(grid, Complex{T}.(data))

"""
    FTField(grid::AbstractGrid)

Allocate a zero-valued spectral field with element type `Complex{T}` and shape
`transform_size(grid)`, where `T` is the grid scalar type. The parent is a
standard Julia `Array` and the result references the supplied grid object.
"""
FTField(grid::AbstractGrid{T}) where {T} = FTField(grid, zeros(Complex{T}, transform_size(grid)))

# -----------------------------------------------------------------------------
# AbstractArray interface
# -----------------------------------------------------------------------------

"""
    IndexStyle(::Type{<:FTField}) -> IndexLinear()

Declare linear indexing as the native indexing style of `FTField`.
"""
Base.IndexStyle(::Type{<:FTField}) = Base.IndexLinear()

"""
    parent(u::FTField) -> AbstractArray

Return the exact coefficient array stored by `u`; no copy is made.
"""
Base.parent(u::FTField) = u.data

"""
    eltype(u::FTField) -> Type

Return `Complex{T}`, where `T` is the scalar type of `grid(u)`.
"""
Base.eltype(::FTField{<:AbstractGrid{T}}) where {T} = Complex{T}

"""
    similar(u::FTField[, ::Type{S}]) -> FTField

Allocate a zero-valued spectral field with the same shape as `u`.

`S` selects the underlying real precision through `real(S)`: both `Float32`
and `ComplexF32`, for example, request a field with element type
`ComplexF32`. If `real(S) == T`, the result references the same grid object;
otherwise it uses `convert(real(S), grid(u))`. Storage is obtained from
`zero(parent(u))`, followed by any conversion required for the requested
precision, so allocation behavior follows the parent-array implementation.
"""
Base.similar(u::FTField{<:AbstractGrid{T}}, ::Type{S}=T) where {T, S} =
    FTField(real(S) == T ? grid(u) : convert(real(S), grid(u)), zero(parent(u)))

"""
    size(u::FTField) -> Tuple

Return `size(parent(u))`; no independent grid-size check is performed.
"""
Base.size(u::FTField) = Base.size(parent(u))

"""
    copy(u::FTField) -> FTField

Copy all coefficients into newly allocated storage. The result references the
same grid object and, for ordinary arrays, does not alias `parent(u)`.
"""
Base.copy(u::FTField) = (v = Base.similar(u); parent(v) .= parent(u); return v)

"""
    zero(u::FTField) -> FTField

Return a zero-valued field with the same grid, shape, and scalar precision as
`u`.
"""
Base.zero(u::FTField{<:AbstractGrid{T}}) where {T} = (v = Base.similar(u); parent(v) .= zero(T); return v)

"""
    u[i::Int]

Read linear coefficient `parent(u)[i]`. Standard bounds checks are delegated
to the parent array; no symmetry operation is performed.
"""
Base.@propagate_inbounds function Base.getindex(u::FTField, i::Int)
    @boundscheck checkbounds(parent(u), i)
    @inbounds v = parent(u)[i]
    return v
end

"""
    u[i::Int] = value

Assign a linear coefficient and return `value`. The parent may convert the
value before storage. This low-level operation does not maintain Hermitian
symmetry or mean-mode reality.
"""
Base.@propagate_inbounds function Base.setindex!(u::FTField, v, i::Int)
    @boundscheck checkbounds(parent(u), i)
    @inbounds parent(u)[i] = v
    return v
end

"""
    u[I::CartesianIndex]

Read `parent(u)[I]` without applying symmetry logic.
"""
Base.@propagate_inbounds function Base.getindex(u::FTField, I::CartesianIndex)
    @boundscheck checkbounds(parent(u), I)
    @inbounds v = parent(u)[I]
    return v
end

"""
    u[I::CartesianIndex] = value

Assign `parent(u)[I]` and return `value`. This low-level operation does not
maintain spectral symmetry.
"""
Base.@propagate_inbounds function Base.setindex!(u::FTField, v, I::CartesianIndex)
    @boundscheck checkbounds(parent(u), I)
    @inbounds parent(u)[I] = v
    return v
end

"""
    grid(u::FTField) -> AbstractGrid

Return the exact grid object stored by `u`.
"""
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
    @boundscheck checkbounds(u, combine_indices(grid(u), Colon(), indices)...)
    @inbounds return view(parent(u), combine_indices(grid(u), Colon(), indices)...)
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
    @boundscheck checkbounds(u, combine_indices(grid(u), CartesianIndex(i1, I...), CartesianIndex(indices...))...)
    @inbounds val = parent(u)[combine_indices(grid(u), CartesianIndex(i1, I...), CartesianIndex(indices...))...]
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

Assignment returns the logical `Complex{T}` coefficient after conversion and,
for the all-zero wavenumber, after discarding its imaginary part. For a
negative rfft wavenumber, this is the value at the requested logical mode rather
than the conjugated value written to its stored partner.
"""
Base.@propagate_inbounds function Base.setindex!(u::FTField{<:AbstractGrid{T}},
                                               val,
                                                 k::WaveNumberVector{N},
                                                 I::Vararg{Int}) where {T, N}
    CT      = Complex{T}
    tpl     = to_homogeneous_indices(grid(u), k)
    do_conj = last(tpl)
    indices = Base.front(tpl)
    i0      = first(indices)      # index in the rfft storage dimension
    rest    = Base.tail(indices)  # indices in the signed-FFT storage dimensions

    # Force the fully-zero wavenumber to be real.
    val = (i0 == 1 && all(==(1), rest)) ? CT(real(val)) : CT(val)

    # Conjugate-symmetric indices for each signed-fft axis.
    sym_rest = ntuple(j -> _fftw_sym_index(rest[j], size(u, j + 2)), Val(N-1))

    @boundscheck checkbounds(u, combine_indices(grid(u), CartesianIndex(I), CartesianIndex(i0, rest...))...)
    @inbounds parent(u)[combine_indices(grid(u), CartesianIndex(I), CartesianIndex(i0, rest...))...] = do_conj ? conj(val) : val
    # When the rfft wavenumber is zero, also write the mirror entry so that
    # the Hermitian-symmetry invariant is preserved across all signed dims.
    i0 == 1 && @inbounds parent(u)[combine_indices(grid(u), CartesianIndex(I), CartesianIndex(i0, sym_rest...))...] = do_conj ? val  : conj(val)
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
    homogeneous_axes(u::FTField)                              -> Tuple
    homogeneous_axes(u::AbstractArray, ::Val{FFT_DIMS_ORDER}) -> Tuple

Return a tuple of `axes(u, d)` for each dimension `d` in `FFT_DIMS_ORDER`.
Suitable for `CartesianIndices(homogeneous_axes(u))` to iterate over all
spectral wavenumber indices without touching the inhomogeneous dimensions.
`D` is inferred from `ndims(u)`.
"""
@generated function homogeneous_axes(u::AbstractArray, ::Val{FFT_DIMS_ORDER}) where {FFT_DIMS_ORDER}
    return Expr(:block, Expr(:meta, :inline), Expr(:tuple, (:(axes(u, $d)) for d in FFT_DIMS_ORDER)...))
end

@inline homogeneous_axes(u::FTField) = homogeneous_axes(parent(u), Val(fft_storage_dims(grid(u))))

"""
    inhomogeneous_axes(u::FTField)                              -> Tuple
    inhomogeneous_axes(u::AbstractArray, ::Val{FFT_DIMS_ORDER}) -> Tuple

Return a tuple of `axes(u, d)` for each dimension `d` in `1:D` that is
**not** in `FFT_DIMS_ORDER`.  Complementary to [`homogeneous_axes`](@ref).
`D` is inferred from `ndims(u)`.
"""
@generated function inhomogeneous_axes(u::AbstractArray{<:Any, D}, ::Val{FFT_DIMS_ORDER}) where {D, FFT_DIMS_ORDER}
    inh_dims = [d for d in 1:D if d ∉ FFT_DIMS_ORDER]
    return Expr(:block, Expr(:meta, :inline), Expr(:tuple, (:(axes(u, $d)) for d in inh_dims)...))
end

@inline inhomogeneous_axes(u::FTField) = inhomogeneous_axes(parent(u), Val(fft_storage_dims(grid(u))))
