# Wrappers around paired FFTW rfft/brfft plans for in-place physical<->spectral
# transformations on multi-dimensional grids, with optional 3/2-rule dealiasing.


# ------------------------------------------- #
# Re-export FFTW planner flags and time limit  #
# ------------------------------------------- #
const ESTIMATE    = FFTW.ESTIMATE
const EXHAUSTIVE  = FFTW.EXHAUSTIVE
const MEASURE     = FFTW.MEASURE
const PATIENT     = FFTW.PATIENT
const WISDOM_ONLY = FFTW.WISDOM_ONLY
const NO_TIMELIMIT = FFTW.NO_TIMELIMIT


# ---------------- #
# transform object #
# ---------------- #
"""
    FFTPlans{DEALIAS, D, T, ORDER, PLAN, IPLAN}

Paired FFTW real-to-complex (rfft) and complex-to-real (brfft) plans for
in-place transformations between physical and spectral representations of a
scalar field on a `D`-dimensional grid.

# Type parameters
- `DEALIAS`: `Bool`; when `true`, physical arrays are padded by the 3/2 rule
  in each transformed dimension to eliminate aliasing in nonlinear products
- `D`: number of spatial dimensions
- `T`: real element type (e.g. `Float64`)
- `ORDER`: tuple of transformed dimension indices; `ORDER[1]` is the rfft
  dimension (non-negative frequencies only), `ORDER[2:end]` are full-spectrum
  complex FFT dimensions applied in sequence

# Fields
- `plan`: FFTW rfft plan (physical → spectral)
- `iplan`: FFTW brfft plan (spectral → physical)
- `cache`: scratch array in spectral space, sized for the padded physical grid
  when `DEALIAS == true` and for the standard spectral grid otherwise; used as
  a staging buffer during forward/backward transforms
- `norm`: normalisation factor `1 / prod(shape[ORDER])` applied after each
  forward transform so that coefficients represent the true Fourier amplitudes

# Constructor

    FFTPlans(shape, order, T=Float64; dealias=true, flags=EXHAUSTIVE, timelimit=NO_TIMELIMIT)

## Arguments
- `shape::Dims`: size of the physical domain
- `order::NTuple{H,Int}`: dimensions to transform (first → rfft, rest → fft)
- `T::Type`: real element type (default `Float64`)
- `dealias::Bool`: if `true`, pad each transformed dimension by the 3/2 rule
- `flags::UInt32`: FFTW planner effort flag (`ESTIMATE`, `MEASURE`, `PATIENT`,
  `EXHAUSTIVE`); higher effort finds faster plans but takes longer to compile
- `timelimit::Real`: maximum planner wall-time in seconds (`NO_TIMELIMIT` to
  disable)
"""
struct FFTPlans{DEALIAS, D, T, ORDER, PLAN, IPLAN}
    plan::PLAN
    iplan::IPLAN
    cache::Array{Complex{T}, D}
    norm::T

    function FFTPlans(shape::Dims{D},
                      order::NTuple{H, Int},
                           ::Type{T}=Float64;
                    dealias::Bool   =true,
                      flags::UInt32 =EXHAUSTIVE,
                  timelimit::Real   =NO_TIMELIMIT) where {D, H, T}
        all(1 ≤ d ≤ D for d in order) || throw(ArgumentError("order indices must be in 1:$D, got $order"))
        allunique(order)               || throw(ArgumentError("order indices must be unique, got $order"))

        grid_shape     = dealias ? _get_padded_shape(shape, order) : shape
        spectral_array = zeros(Complex{T}, _get_transform_shape(grid_shape, order[1]))
        physical_array = zeros(T, grid_shape)
        norm           = T(1 / prod(grid_shape[i] for i in order))

        plan  = FFTW.plan_rfft( physical_array,                       order, flags=flags, timelimit=timelimit)
        iplan = FFTW.plan_brfft(spectral_array, grid_shape[order[1]], order, flags=flags, timelimit=timelimit)

        new{dealias, D, T, order, typeof(plan), typeof(iplan)}(plan, iplan, spectral_array, norm)
    end
end

FFTPlans(g::AbstractGrid{T, D, H}; kwargs...) where {T, D, H} = FFTPlans(size(g), H, T; kwargs...)
FFTPlans(u::FTField;               kwargs...)                 = FFTPlans(grid(u); kwargs...)
FFTPlans(u::Field;                 kwargs...)                 = FFTPlans(grid(u); kwargs...)


# ------------------------ #
# in-place transformations #
# ------------------------ #
"""
    (f::FFTPlans)(û, u, add=false)
    (f::FFTPlans)(û, u, add=false, use_cache=false)

Forward in-place transform: physical array `u` → spectral array `û`.

## Arguments
- `û`: output spectral array
- `u`: input physical array (never modified)
- `add`: add the transform result into `û` rather than overwriting it. Useful
  when accumulating multiple physical-space contributions into the same spectral
  array (e.g. nonlinear terms) without allocating a temporary.
- `use_cache`: route the FFTW output through `f.cache` before writing to `û`.
  `unsafe_execute!` bypasses FFTW's buffer checks and assumes the output has
  the same memory layout as the array used during planning; set `use_cache=true`
  when `û` is non-contiguous (e.g. a strided view) to avoid undefined behaviour.
"""
function (f::FFTPlans)(û::VectorField{N, <:FTField},
                       u::VectorField{N, <:Field};
                     add::Bool=false,
               use_cache::Bool=false) where {N}
    for n in 1:N
        f(û[n], u[n]; add=add, use_cache=use_cache)
    end
    return û
end

"""
    (f::FFTPlans)(û::FTField, u::Field; add=false, use_cache=false)

Forward transform for `FTField`/`Field` inputs with keyword arguments.

Unwraps both fields to their underlying `parent` arrays and delegates to the
`AbstractArray` method. `unsafe_execute!` requires a plain contiguous array
whose memory layout matches the array used during planning; `parent` extracts
that storage while this method preserves the `FTField` return type for the caller.
"""
(f::FFTPlans)(û::FTField, u::Field; add::Bool=false, use_cache::Bool=false) =
    (f(parent(û), parent(u), add, use_cache); return û)

"""
    (f::FFTPlans)(û::AbstractArray{Complex{T}}, u::AbstractArray{T}, add::Bool, use_cache::Bool)

Inner dispatch point for the forward transform: all type-level unwrapping is
complete and `_forward_transform!` can be called directly. `add` and `use_cache`
carry no defaults here so that every internal call site is explicit about both
flags rather than silently inheriting a behaviour.
"""
(f::FFTPlans{DEALIAS, D, T, ORDER})(û::AbstractArray{Complex{T}},
                                    u::AbstractArray{        T},
                                  add::Bool,
                            use_cache::Bool) where {DEALIAS, D, T, ORDER} = _forward_transform!(û, u, f, add, use_cache)

"""
    _forward_transform!(û, u, f::FFTPlans{DEALIAS, D, T, ORDER}, add::Bool, use_cache::Bool)

Forward transform of `u` into `û`, optionally accumulating into `û`.

Routes through `f.cache` whenever `DEALIAS`, `use_cache`, or `add` is true
(`through_cache = DEALIAS | use_cache | add`). This is required because:
`unsafe_execute!` writes into whatever buffer it is handed without layout checks,
so a non-contiguous `û` requires a staging buffer (`use_cache`); the 3/2-rule
padded plan outputs a larger spectrum than `û` can hold (`DEALIAS`); and
accumulation (`add`) needs `û` separate from the FFTW output buffer to avoid
reading a partially-overwritten target.

When `through_cache` is false, the plan writes directly into `û` — valid only
when `û` has the same memory layout as the array used during planning.

`_copy_from_padded!` and `_add_from_padded!` handle both the dealiased case
(truncation from a padded cache) and the non-dealiased case (full copy, since
same-size arrays make the truncation a no-op that covers all elements).
"""
function _forward_transform!(û, u, f::FFTPlans{DEALIAS, D, T, ORDER}, add::Bool, use_cache::Bool) where {DEALIAS, D, T, ORDER}
    through_cache = DEALIAS | use_cache | add
    buf = through_cache ? f.cache : û
    FFTW.unsafe_execute!(f.plan, u, buf)
    buf .*= f.norm
    if through_cache
        add ? _add_from_padded!(û, f.cache, D, ORDER) : _copy_from_padded!(û, f.cache, D, ORDER)
    end
    return û
end

"""
    (f::FFTPlans)(u, û, preserve_input=true)
    (f::FFTPlans)(u, û, preserve_input=true, use_cache=false)

Backward in-place transform: spectral array `û` → physical array `u`.

## Arguments
- `u`: output physical array
- `û`: input spectral array
- `preserve_input`: FFTW's C2R (brfft) transform is permitted to overwrite its
  complex input buffer as scratch during computation — this is a fundamental
  property of the FFTW algorithm, not a bug. When `preserve_input=true`
  (default), `û` is copied into `f.cache` first so it is never touched. Set
  `preserve_input=false` only when `û` is no longer needed after the transform,
  saving one full spectral-array copy.
- `use_cache`: when `preserve_input=false` and not dealiasing, set this to
  `true` to still stage through `f.cache` — useful when `û` is not a valid FFTW
  input buffer (non-contiguous layout) but the caller does not want to pay for
  the full safe-copy path.
"""
function (f::FFTPlans)(u::VectorField{N, <:Field},
                       û::VectorField{N, <:FTField};
          preserve_input::Bool=true,
               use_cache::Bool=false) where {N}
    for n in 1:N
        f(u[n], û[n]; preserve_input=preserve_input, use_cache=use_cache)
    end
    return u
end

"""
    (f::FFTPlans)(u::Field, û::FTField; preserve_input=true, use_cache=false)

Backward transform for `Field`/`FTField` inputs with keyword arguments.

Unwraps both fields to their `parent` arrays and delegates to the `AbstractArray`
method. Symmetric to the forward `FTField`/`Field` overload — `parent` unwrapping
is required for the same reason: `unsafe_execute!` needs a plain contiguous array
matching the planning layout.
"""
(f::FFTPlans)(u::Field, û::FTField; preserve_input::Bool=true, use_cache::Bool=false) =
    (f(parent(u), parent(û), preserve_input, use_cache); return u)

"""
    (f::FFTPlans)(u::AbstractArray{T}, û::AbstractArray{Complex{T}}, preserve_input::Bool, use_cache::Bool)

Inner dispatch point for the backward transform: all type-level unwrapping is
complete and `_backward_transform!` can be called directly. `preserve_input` and
`use_cache` carry no defaults here so that every internal call site is explicit
about both flags.
"""
(f::FFTPlans{DEALIAS, D, T, ORDER})(u::AbstractArray{        T},
                                    û::AbstractArray{Complex{T}},
                       preserve_input::Bool,
                            use_cache::Bool) where {DEALIAS, D, T, ORDER} = 
    _backward_transform!(u, û, f, preserve_input, use_cache)

"""
    _backward_transform!(u, û, f::FFTPlans{DEALIAS, D, T, ORDER}, preserve_input::Bool, use_cache::Bool)

Backward transform of `û` into `u`.

Routes through `f.cache` whenever `DEALIAS`, `preserve_input`, or `use_cache`
is true (`through_cache = DEALIAS | preserve_input | use_cache`). This is
required because: brfft is permitted to overwrite its complex input buffer during
computation, so preserving `û` requires a staging copy (`preserve_input`); the
3/2-rule padded plan needs a larger input buffer than `û` provides (`DEALIAS`);
and a non-contiguous `û` cannot serve as a valid FFTW input buffer (`use_cache`).

When `through_cache` is false, the plan runs directly on `û`, which brfft may
silently destroy — the caller accepts this side effect.

`_copy_to_padded!` handles both the dealiased case (embed resolved frequencies
into a zero-padded cache after `_apply_mask!`) and the non-dealiased case (full
copy, since same-size embedding covers all elements and no prior zeroing is needed).
"""
function _backward_transform!(u, û, f::FFTPlans{DEALIAS, D, T, ORDER}, preserve_input::Bool, use_cache::Bool) where {DEALIAS, D, T, ORDER}
    through_cache = DEALIAS | preserve_input | use_cache
    if through_cache
        DEALIAS && _apply_mask!(f.cache)
        _copy_to_padded!(f.cache, û, D, ORDER)
        FFTW.unsafe_execute!(f.iplan, f.cache, u)
    else
        # brfft destroys its input; caller accepts this when preserve_input=false, use_cache=false
        FFTW.unsafe_execute!(f.iplan, û, u)
    end
    return u
end


# --------------------- #
# allocating transforms #
# --------------------- #
"""
    FFT(u::Field) -> FTField

Allocating forward transform: compute and return the Fourier coefficients of `u`
normalised by the grid's `fft_norm`. Equivalent to planning and executing a fresh
`rfft` on `parent(u)`.
"""
function FFT(u::Field{G}) where {T, D, H, G<:AbstractGrid{T, D, H}}
    û = FTField(grid(u))
    parent(û) .= rfft(parent(u), H)
    û .*= 1 / prod(fft_norm(grid(u)))
    return û
end
FFT(u::VectorField{N, <:Field}) where {N} = VectorField([FFT(u[n]) for n in 1:N]...)

"""
    IFFT(û::FTField) -> Field

Allocating backward transform: compute and return the physical-space field
corresponding to the Fourier coefficients `û`. No normalisation is applied
(consistent with the convention that normalisation belongs to the forward
transform). Equivalent to planning and executing a fresh `brfft` on `parent(û)`.
"""
function IFFT(û::FTField{G}) where {T, D, H, G<:AbstractGrid{T, D, H}}
    u = Field(grid(û))
    parent(u) .= brfft(parent(û), size(grid(û))[H[1]], H)
    return u
end
IFFT(û::VectorField{N, <:FTField}) where {N} = VectorField([IFFT(û[n]) for n in 1:N]...)


# ----------------- #
# utility functions #
# ----------------- #
"""
    _get_padded_shape(shape, order) -> Dims

Return the physical grid shape padded for 3/2-rule dealiasing: each dimension
in `order` is rounded up to the nearest odd number ≥ 3s/2 (where `s` is the
unpadded size). Odd sizes avoid Nyquist-frequency ambiguity in the brfft plan.
"""
function _get_padded_shape(shape, order)
    return ntuple(length(shape)) do i
        i ∈ order ? cld(3*shape[i], 2) | 1 : shape[i]
    end
end

"""
    _get_transform_shape(shape, dim) -> Dims

Return the spectral array shape produced by an rfft along dimension `dim`:
size along `dim` shrinks from `n` to `(n >> 1) + 1` (non-negative frequencies
only); all other dimensions are unchanged.
"""
function _get_transform_shape(shape, dim)
    return ntuple(length(shape)) do i
        i == dim ? (shape[i] >> 1) + 1 : shape[i]
    end
end

"""
    _spectral_blocks(u, cache, dim, order) -> Tuple of (blk_u, blk_c) pairs

Return a fixed-size tuple of index-block pairs that map resolved frequencies
between `u` (the compact spectral array) and `cache` (the padded spectral array).
Each pair `(blk_u, blk_c)` is a pair of `NTuple{D, UnitRange{Int}}` index
blocks: `blk_u` selects a region of `u` and `blk_c` selects the corresponding
region of `cache`.

Positive-frequency blocks share the same low-index range in both arrays.
Negative-frequency blocks are at the high-index end of `u` and at the
corresponding high-index end of the larger `cache`. When a dimension has no
negative frequencies, its block uses an empty range and is a no-op in any
subsequent broadcast.

Dispatches on `length(order)` and always returns a statically-sized tuple
(1 pair for `NTuple{1}`, 2 for `NTuple{2}`, 4 for `NTuple{3}`), so callers
incur no heap allocation.
"""
function _spectral_blocks(u, _, dim, ::NTuple{1})
    blk = ntuple(i -> 1:size(u, i), dim)
    return ((blk, blk),)
end

function _spectral_blocks(u, cache, dim, order::NTuple{2})
    npos = (size(u, order[2]) >> 1) + 1
    nneg = size(u, order[2]) - npos

    blk_pos = ntuple(i -> i == order[2] ? (1:npos)                                             : (1:size(u, i)), dim)
    blk_u   = ntuple(i -> i == order[2] ? (npos+1:size(u, order[2]))                           : (1:size(u, i)), dim)
    blk_c   = ntuple(i -> i == order[2] ? (size(cache, order[2])-nneg+1:size(cache, order[2])) : (1:size(u, i)), dim)

    return ((blk_pos, blk_pos), (blk_u, blk_c))
end

function _spectral_blocks(u, cache, dim, order::NTuple{3})
    npos2 = (size(u, order[2]) >> 1) + 1;  nneg2 = size(u, order[2]) - npos2
    npos3 = (size(u, order[3]) >> 1) + 1;  nneg3 = size(u, order[3]) - npos3

    # all-positive block
    blk_pos = ntuple(i -> i ∈ order[2:3] ? (1:(size(u, i)>>1)+1) : (1:size(u, i)), dim)

    # edge block: negative in order[2], positive in order[3]
    blk_u2 = ntuple(dim) do i
        i ∈ order[2:end] ? (i == order[2] ? (npos2+1:size(u, order[2]))                           : (1:(size(u, i)>>1)+1)) : (1:size(u, i))
    end
    blk_c2 = ntuple(dim) do i
        i ∈ order[2:end] ? (i == order[2] ? (size(cache, order[2])-nneg2+1:size(cache, order[2])) : (1:(size(u, i)>>1)+1)) : (1:size(u, i))
    end

    # edge block: positive in order[2], negative in order[3]
    blk_u3 = ntuple(dim) do i
        i ∈ order[2:end] ? (i == order[3] ? (npos3+1:size(u, order[3]))                           : (1:(size(u, i)>>1)+1)) : (1:size(u, i))
    end
    blk_c3 = ntuple(dim) do i
        i ∈ order[2:end] ? (i == order[3] ? (size(cache, order[3])-nneg3+1:size(cache, order[3])) : (1:(size(u, i)>>1)+1)) : (1:size(u, i))
    end

    # corner block: negative in both order[2] and order[3]
    blk_uc = ntuple(dim) do i
        i ∈ order[2:end] ? ((size(u, i)>>1)+2 : size(u, i)) : (1:size(u, i))
    end
    blk_cc = ntuple(dim) do i
        if i ∈ order[2:end]
            nneg_i = size(u, i) - (size(u, i)>>1) - 1
            (size(cache, i) - nneg_i + 1 : size(cache, i))
        else
            (1:size(u, i))
        end
    end

    return ((blk_pos, blk_pos), (blk_u2, blk_c2), (blk_u3, blk_c3), (blk_uc, blk_cc))
end

_spectral_blocks(_, _, _, order::NTuple{N}) where {N} = throw(NotImplementedError(order))

"""
    _copy_from_padded!(u, cache, dim, order)

Copy resolved Fourier coefficients from the padded spectral `cache` into the
compact spectral array `u`, discarding the dealiasing padding. Negative-frequency
blocks (at the high-index end of `cache`) are mapped to the corresponding high-index
end of `u`. Delegates block enumeration to `_spectral_blocks`.
"""
function _copy_from_padded!(u, cache, dim, order)
    for (blk_u, blk_c) in _spectral_blocks(u, cache, dim, order)
        @views u[blk_u...] .= cache[blk_c...]
    end
    return u
end

"""
    _add_from_padded!(u, cache, dim, order)

Accumulate resolved Fourier coefficients from the padded spectral `cache` into
`u` (`u .+= cache[resolved]`). Same block structure as `_copy_from_padded!`.
"""
function _add_from_padded!(u, cache, dim, order)
    for (blk_u, blk_c) in _spectral_blocks(u, cache, dim, order)
        @views u[blk_u...] .+= cache[blk_c...]
    end
    return u
end

"""
    _copy_to_padded!(cache, u, dim, order)

Embed resolved Fourier coefficients from the compact spectral array `u` into
the padded `cache`, placing negative-frequency blocks at the high-index end of
each transformed dimension. The caller must zero `cache` first (via `_apply_mask!`)
so that padding-zone entries are zero before the brfft plan runs.
"""
function _copy_to_padded!(cache, u, dim, order)
    for (blk_u, blk_c) in _spectral_blocks(u, cache, dim, order)
        @views cache[blk_c...] .= u[blk_u...]
    end
    return cache
end

"""
    _apply_mask!(cache::Array{T}) -> cache

Zero all elements of `cache` in-place and return it. Called before
`_copy_to_padded!` to ensure that padding-zone entries in the spectral cache
do not pollute the backward transform output.
"""
_apply_mask!(cache::Array{T}) where {T} = (cache .= zero(T); return cache)
