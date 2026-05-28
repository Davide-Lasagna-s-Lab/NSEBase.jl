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
        new{G, A, T, D}(grid, apply_symmetry!(normalise_mean!(data, fft_dims(grid)), fft_dims(grid)))
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

#TODO: document this and check
Base.CartesianIndices(u::FTField) = CartesianIndices(size(u))

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
direction in `fft_dims(grid(u)) = FFT_DIMS_ORDER`, in `FFT_DIMS_ORDER` order. It must therefore
have length `length(FFT_DIMS_ORDER)`, not the full dimension `D` of `grid(u)`.
Inhomogeneous directions are preserved by the grid-specific
`growto(grid(u), target_size)` method.

The spectral coefficients are copied by wavenumber vector: each stored
wavenumber of `u` is converted to a [`WaveNumberVector`](@ref), then copied to
the same wavenumber in the output field. This requires the target grid to
represent every copied source wavenumber, so this method is primarily intended
for increasing homogeneous resolution.

This is optional.  Packages should only implement it if they support
resolution-changing transforms such as `FFT(u, target_size)` and
`IFFT(û, target_size)`.
"""
function growto(u::FTField{G}, target_size::NTuple{N, Int}) where {T, D, AXES, FFT_DIMS_ORDER, N, G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}}
    N == length(FFT_DIMS_ORDER) ||
        throw(ArgumentError("target_size has incompatible size: expected a tuple of length $(length(FFT_DIMS_ORDER)), got length $N"))
    out = FTField(growto(grid(u), target_size))
    for Ih in CartesianIndices(homogeneous_axes(u))
        # Build an index tuple with Colon() for each inhomogeneous dimension and
        # the spectral storage index for each homogeneous dimension, then splat it.
        I = combine_indices(grid(u), Colon(), Ih)
        out[I...] .= u[I...]
    end
    return out
end

"""
    u[k::WaveNumberVector]

Return a view of the underlying array over all non-transform dimensions for
the wavenumber vector `k`. The wavenumbers in `k` follow the order
of `fft_dims(grid(u)) = FFT_DIMS_ORDER`.

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

The wavenumbers in `k` follow the order of `fft_dims(grid(a)) = FFT_DIMS_ORDER`.
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
@inline function _average_complex(z1::T, z2::T) where {T}
    _re = 0.5 * (real(z1) + real(z2))
    _im = 0.5 * (imag(z1) - imag(z2))
    return _re + im * _im
end

"""
    apply_symmetry!(data::AbstractArray{T, D}, ::Val{H})

Enforce Hermitian symmetry on `data` after a multi-dimensional RFFT along
the dimensions listed in `H` (in transform order).

For a real-valued physical field, Fourier coefficients must satisfy
û(-k) = conj(û(k)). After the RFFT along `H[1]`, only non-negative
frequencies along that axis are stored. Along the remaining axes `H[2:end]`
both signs are present, but at the zero-frequency plane of `H[1]` (index 1)
the coefficients must still satisfy the conjugate-symmetry constraint — this
function enforces it by averaging each conjugate pair in-place.

Dimensions not in `H` are untransformed and are iterated over in full.
No-op when only one dimension is transformed (`length(H) == 1`).

# Note on cache efficiency
The DC plane of `H[1]` (the rfft dim) has memory stride `size(data, H[1])`
rather than 1, because the rfft dim is stored as the leading (stride-1)
dimension. The loop iterates the remaining dims in ascending stride order,
which is optimal for the current layout. If this function is a bottleneck,
storing the rfft dim last would give stride-1 access to the DC plane.
"""
@generated function apply_symmetry!(data::AbstractArray{T, D}, ::Val{H}) where {T, D, H}
    length(H) <= 1 && return :(return data)

    # All subset enumeration happens at code-generation time so the compiled
    # hot loop contains only literal index expressions — no closures, no dynamic
    # dispatch, no allocations.
    secondary = H[2:end]
    blocks    = Expr[]

    for mask_int in 1:(1 << length(secondary)) - 1

        # active subset of secondary dims for this pass
        active_dims = Tuple(secondary[k] for k in eachindex(secondary) if Bool((mask_int >> (k-1)) & 1))
        half_dim    = active_dims[end]

        # per-dim range literals (e.g. 1:1, 2:Base.size(data,3)>>1+1, ...)
        range_exprs = ntuple(D) do d
            if d == H[1]
                :(1:1)
            elseif d == half_dim
                :(2:(Base.size(data, $d) >> 1) + 1)
            elseif d in active_dims
                :(2:Base.size(data, $d))
            elseif d in secondary
                :(1:1)
            else
                :(1:Base.size(data, $d))
            end
        end

        # per-dim neg-index literals (e.g. I[1], Base.size(data,2)-I[2]+2, ...)
        neg_exprs = ntuple(d -> d in active_dims ? :(Base.size(data, $d) - I[$d] + 2) : :(I[$d]), D)

        push!(blocks, quote
            for I in CartesianIndices($(Expr(:tuple, range_exprs...)))
                neg       = $(Expr(:call, :CartesianIndex, neg_exprs...))
                _av       = _average_complex(data[I], data[neg])
                data[I]   = _av
                data[neg] = conj(_av)
            end
        end)
    end

    return quote
        $(blocks...)
        return data
    end
end
apply_symmetry!(data, H::Dims) = apply_symmetry!(data, Val(H))

"""
    normalise_mean!(data::Array{<:Any, D}, ::Val{H})

Enforce purely real values at mean component of Fourier transformed
data.
"""
@generated function normalise_mean!(data::Array{<:Any, D}, ::Val{H}) where {D, H}
    indxs = [d ∈ H ? 1 : :(:) for d in 1:D]
    slice = Expr(:ref, :data, indxs...)
    return :(@views $slice .= real.($slice); return data)
end
normalise_mean!(data, H::Dims) = normalise_mean!(data, Val(H))

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
