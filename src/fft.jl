# Wrappers around paired FFTW rfft/brfft plans for in-place physical<->spectral
# transformations on multi-dimensional grids, with optional 3/2-rule dealiasing.


# ------------------------------------------- #
# Re-export FFTW planner flags and time limit  #
# ------------------------------------------- #
const ESTIMATE     = FFTW.ESTIMATE
const EXHAUSTIVE   = FFTW.EXHAUSTIVE
const MEASURE      = FFTW.MEASURE
const PATIENT      = FFTW.PATIENT
const WISDOM_ONLY  = FFTW.WISDOM_ONLY
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
- `DEALIAS`: `Bool`; when `true`, the physical array used for planning is larger
  than the spectral domain — either via the 3/2-rule or a user-supplied size —
  and the forward/backward transforms pad/truncate accordingly
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

    FFTPlans(size, order, T=Float64; dealias=true, padded_size=nothing, flags=EXHAUSTIVE, timelimit=NO_TIMELIMIT)

## Arguments
- `size::Dims`: size of the physical domain
- `order::NTuple{H,Int}`: dimensions to transform (first → rfft, rest → fft)
- `T::Type`: real element type (default `Float64`)
- `dealias::Bool`: if `true`, pad each transformed dimension by the 3/2 rule
- `padded_size::Union{Nothing,Dims}`: explicit physical grid size, overriding the
  3/2-rule padding; must be `≥ size` in each transformed dimension; cannot be
  combined with `dealias=false`
- `flags::UInt32`: FFTW planner effort flag (`ESTIMATE`, `MEASURE`, `PATIENT`,
  `EXHAUSTIVE`); higher effort finds faster plans but takes longer to compile
- `timelimit::Real`: maximum planner wall-time in seconds (`NO_TIMELIMIT` to
  disable)
"""
struct FFTPlans{DEALIAS, D, T, ORDER, PLAN, IPLAN}
     plan :: PLAN
    iplan :: IPLAN
    cache :: Array{Complex{T}, D}
     norm :: T

    function FFTPlans(size::Dims{D},
                     order::NTuple{H, Int},
                          ::Type{T}=Float64;
                   dealias::Bool                =true,
               padded_size::Union{Nothing, Dims}=nothing,
                     flags::UInt32              =EXHAUSTIVE,
                 timelimit::Real                =NO_TIMELIMIT) where {D, H, T}
        all(1 ≤ d ≤ D for d in order) || throw(ArgumentError("order indices must be in 1:$D, got $order"))
        allunique(order)              || throw(ArgumentError("order indices must be unique, got $order"))
        padded_size !== nothing && !dealias &&
            throw(ArgumentError("cannot set padded_size with dealias=false"))

        grid_size = if padded_size !== nothing
            length(padded_size) == D ||
                throw(ArgumentError("padded_size must have $D elements, got $(length(padded_size))"))
            all(padded_size[d] >= size[d] for d in order) ||
                throw(ArgumentError("padded_size must be ≥ size along each transformed dimension"))
            padded_size
        elseif dealias
            _get_padded_size(size, order)
        else
            size
        end

        computed_dealias = any(grid_size[d] != size[d] for d in order)
        spectral_array   = zeros(Complex{T}, _get_transform_size(grid_size, order[1]))
        physical_array   = zeros(T, grid_size)
        norm             = T(1 / prod(grid_size[i] for i in order))

        plan  = FFTW.plan_rfft( physical_array,                      order, flags=flags, timelimit=timelimit)
        iplan = FFTW.plan_brfft(spectral_array, grid_size[order[1]], order, flags=flags, timelimit=timelimit)

        return new{computed_dealias, D, T, order, typeof(plan), typeof(iplan)}(plan, iplan, spectral_array, norm)
    end
end

FFTPlans(g::AbstractGrid{T, <:Any, H}; kwargs...) where {T, H} = FFTPlans(size(g), H, T; kwargs...)
FFTPlans(u::FTField;                   kwargs...)              = FFTPlans(grid(u); kwargs...)
FFTPlans(u::Field;                     kwargs...)              = FFTPlans(grid(u); kwargs...)


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
(f::FFTPlans{<:Any, <:Any, T})(û::AbstractArray{Complex{T}},
                               u::AbstractArray{        T},
                             add::Bool,
                       use_cache::Bool) where {T} = _forward_transform!(û, u, f, add, use_cache)

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
function _forward_transform!(û, u, f::FFTPlans{DEALIAS, D, <:Any, ORDER}, add::Bool, use_cache::Bool) where {DEALIAS, D, ORDER}
    through_cache = DEALIAS | use_cache | add
    buf = through_cache ? f.cache : û
    FFTW.unsafe_execute!(f.plan, u, buf)
    buf .*= f.norm
    if through_cache
        add ? _add_from_padded!(û, f.cache, ORDER) : _copy_from_padded!(û, f.cache, ORDER)
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
(f::FFTPlans{<:Any, <:Any, T})(u::AbstractArray{        T},
                               û::AbstractArray{Complex{T}},
                  preserve_input::Bool,
                       use_cache::Bool) where {T} =
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
function _backward_transform!(u, û, f::FFTPlans{DEALIAS, D, <:Any, ORDER}, preserve_input::Bool, use_cache::Bool) where {DEALIAS, D, ORDER}
    through_cache = DEALIAS | preserve_input | use_cache
    if through_cache
        DEALIAS && _apply_mask!(f.cache)
        _copy_to_padded!(f.cache, û, ORDER)
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
function FFT(u::Field{G}) where {H, G<:AbstractGrid{<:Any, <:Any, H}}
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
function IFFT(û::FTField{G}) where {H, G<:AbstractGrid{<:Any, <:Any, H}}
    u = Field(grid(û))
    parent(u) .= brfft(parent(û), size(grid(û))[H[1]], H)
    return u
end
IFFT(û::VectorField{N, <:FTField}) where {N} = VectorField([IFFT(û[n]) for n in 1:N]...)


# ----------------- #
# utility functions #
# ----------------- #
"""
    _get_padded_size(size, order) -> Dims

Return the physical grid size padded for 3/2-rule dealiasing: each dimension
in `order` is rounded up to the nearest odd number ≥ 3s/2 (where `s` is the
unpadded size). Odd sizes avoid Nyquist-frequency ambiguity in the brfft plan.
"""
function _get_padded_size(size, order)
    return ntuple(length(size)) do i
        i ∈ order ? cld(3*size[i], 2) | 1 : size[i]
    end
end

"""
    _get_transform_size(size, dim) -> Dims

Return the spectral array size produced by an rfft along dimension `dim`:
size along `dim` shrinks from `n` to `(n >> 1) + 1` (non-negative frequencies
only); all other dimensions are unchanged.
"""
function _get_transform_size(size, dim)
    return ntuple(length(size)) do i
        i == dim ? (size[i] >> 1) + 1 : size[i]
    end
end

"""
    _loopblk!(dest, ar, src, br, ::Val{ADD})

Copy (`ADD=false`) or accumulate (`ADD=true`) one contiguous block from `src`
into `dest`. `ar` and `br` are `NTuple{D, UnitRange{Int}}` index-range tuples
that describe the destination and source blocks respectively; `br` may be offset
relative to `ar`.

Using `Val{ADD}` instead of a plain `Bool` lets the compiler specialise and
inline the assignment at each call site, eliminating the branch from the inner
loop. A single method covers all dimensionalities `D` because
`CartesianIndices(ar::NTuple{D,...})` is type-stable and stack-allocated, and
its iterator elements are value-typed `CartesianIndex{D}` — no heap allocation
occurs regardless of `D`.
"""
@inline function _loopblk!(dest, ar::NTuple{D}, src, br::NTuple{D}, ::Val{ADD}) where {D, ADD}
    off = ntuple(k -> first(br[k]) - first(ar[k]), Val(D))
    @inbounds for I in CartesianIndices(ar)
        J = CartesianIndex(ntuple(k -> I[k] + off[k], Val(D)))
        ADD ? (dest[I] += src[J]) : (dest[I] = src[J])
    end
end

"""
    _transfer_padded!(dest, src, order, ::Val{ADD})

Transfer resolved Fourier coefficients between the compact spectral array `u`
and the padded spectral array `cache`.  Either argument may be `dest`; the
function identifies the compact array by comparing rfft-dim sizes.

The compact array's indices are valid in both arrays, and are used to define all
copy blocks:

- `order[1]` (rfft dim): the compact array is a strict prefix of the padded
  array — the same index range `1..nspec` is used for both dest and src.
- `order[2:end]` (full-FFT dims): positive frequencies (`1..npos`) share the
  same index range in both arrays.  Negative frequencies differ: they sit at
  `npos+1..n` in the compact array and at `end-nneg+1..end` in the padded array.
  The `dest === compact` pointer check (a single comparison, outside the loop)
  decides which range goes to dest and which to src.

The three public wrappers call this function with different argument order and
`Val{ADD}` flags:

| caller                  | dest    | src     | ADD   |
|-------------------------|---------|---------|-------|
| `_copy_to_padded!`      | `cache` | `u`     | false |
| `_copy_from_padded!`    | `u`     | `cache` | false |
| `_add_from_padded!`     | `u`     | `cache` | true  |

Dispatches on `length(order)` (1, 2, or 3) so the compiler sees the exact block
count at compile time and can fully inline `_loopblk!`.  `Val(ndims(dest))` keeps
`ntuple` return types concrete (`NTuple{D, UnitRange{Int}}`), ensuring all index
construction stays on the stack with zero heap allocations.
"""
function _transfer_padded!(dest, src, ord::NTuple{1, Int}, vadd::Val)
    # rfft dim only: compact array is a prefix of the padded array, so the same
    # index block (1..size(compact,i) in every dim) is valid in both.
    compact = size(dest, ord[1]) <= size(src, ord[1]) ? dest : src
    vd  = Val(ndims(dest))
    blk = ntuple(i -> 1:size(compact, i), vd)
    _loopblk!(dest, blk, src, blk, vadd)
    return dest
end

function _transfer_padded!(dest, src, ord::NTuple{2, Int}, vadd::Val)
    vd = Val(ndims(dest))
    d  = ord[2]
    compact = size(dest, ord[1]) <= size(src, ord[1]) ? dest : src
    padded  = compact === dest ? src : dest
    npos = (size(compact, d) >> 1) + 1
    nneg = size(compact, d) - npos

    # Positive block: same compact-sized range in both arrays.
    blk = ntuple(i -> i == d ? (1:npos) : (1:size(compact, i)), vd)
    _loopblk!(dest, blk, src, blk, vadd)

    # Negative block: indices differ between compact and padded arrays.
    if nneg > 0
        blk_co = ntuple(i -> i == d ? (npos+1:size(compact, d))                : (1:size(compact, i)), vd)
        blk_pa = ntuple(i -> i == d ? (size(padded, d)-nneg+1:size(padded, d)) : (1:size(compact, i)), vd)
        dest === compact ? _loopblk!(dest, blk_co, src, blk_pa, vadd) :
                           _loopblk!(dest, blk_pa, src, blk_co, vadd)
    end
    return dest
end

function _transfer_padded!(dest, src, ord::NTuple{3, Int}, vadd::Val)
    vd     = Val(ndims(dest))
    d2, d3 = ord[2], ord[3]
    compact = size(dest, ord[1]) <= size(src, ord[1]) ? dest : src
    padded  = compact === dest ? src : dest
    npos2 = (size(compact, d2) >> 1) + 1;  nneg2 = size(compact, d2) - npos2
    npos3 = (size(compact, d3) >> 1) + 1;  nneg3 = size(compact, d3) - npos3

    # Iterate over all four quadrants of the (d2, d3) frequency plane.
    for (rco2, rpa2) in ((1:npos2,                   1:npos2),
                         (npos2+1:size(compact, d2), size(padded, d2)-nneg2+1:size(padded, d2)))
        isempty(rco2) && continue
        for (rco3, rpa3) in ((1:npos3,                   1:npos3),
                             (npos3+1:size(compact, d3), size(padded, d3)-nneg3+1:size(padded, d3)))
            isempty(rco3) && continue
            blk_co = ntuple(i -> i == d2 ? rco2 : i == d3 ? rco3 : (1:size(compact, i)), vd)
            blk_pa = ntuple(i -> i == d2 ? rpa2 : i == d3 ? rpa3 : (1:size(compact, i)), vd)
            dest === compact ? _loopblk!(dest, blk_co, src, blk_pa, vadd) :
                               _loopblk!(dest, blk_pa, src, blk_co, vadd)
        end
    end
    return dest
end

_transfer_padded!(_, _, ord::NTuple, _) = throw(NotImplementedError(ord))

"""
    _copy_to_padded!(cache, u, order)

Embed the resolved Fourier coefficients from the compact spectral array `u` into
the corresponding slots of the padded spectral array `cache`.  The caller must
zero `cache` first (via `_apply_mask!`) so that the padding zone does not carry
stale values into the backward transform.

Delegates to `_transfer_padded!` with `cache` as dest and `Val(false)` (overwrite).
"""
_copy_to_padded!(cache, u, order) = _transfer_padded!(cache, u, order, Val(false))

"""
    _copy_from_padded!(u, cache, order)

Copy the resolved Fourier coefficients from the padded spectral array `cache`
into the compact spectral array `u`, overwriting `u`.  Called after the forward
transform to discard the dealiasing padding and produce the resolved spectrum.

Delegates to `_transfer_padded!` with `u` as dest and `Val(false)` (overwrite).
"""
_copy_from_padded!(u, cache, order) = _transfer_padded!(u, cache, order, Val(false))

"""
    _add_from_padded!(u, cache, order)

Accumulate the resolved Fourier coefficients from the padded spectral array
`cache` into `u` (`u .+= resolved part of cache`).  Called in place of
`_copy_from_padded!` when the forward transform runs with `add=true`, so that
the result is added to an existing spectral field rather than overwriting it.

Delegates to `_transfer_padded!` with `u` as dest and `Val(true)` (accumulate).
"""
_add_from_padded!(u, cache, order) = _transfer_padded!(u, cache, order, Val(true))

"""
    _apply_mask!(cache::Array{T}) -> cache

Zero all elements of `cache` in-place and return it. Called before
`_copy_to_padded!` to ensure that padding-zone entries in the spectral cache
do not pollute the backward transform output.
"""
_apply_mask!(cache::Array{T}) where {T} = (cache .= zero(T); return cache)
