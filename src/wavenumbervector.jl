# Integer wavenumber index type and the FFTW index-arithmetic helpers.
#
# `WaveNumberVector{N}` is a lightweight struct holding N signed integer
# wavenumbers, one per FFT-transformed dimension, ordered to match
# `fft_dims(g) = FFT_DIMS_ORDER`.  It serves as a public, coordinate-space key for
# reading and writing individual spectral coefficients without exposing the
# internal 1-based FFTW storage layout to callers.
#
# The first wavenumber k[1] corresponds to the rfft dimension.  Because rfft
# stores only non-negative wavenumbers, a negative k[1] is valid and triggers
# a conjugate-symmetry lookup: the entry stored at (-k[1], -k[2:N]...) is
# returned (or targeted for writing) with a complex conjugate applied.
#
# Two private helpers translate between the public signed-wavenumber space and
# the internal 1-based FFTW index space:
#   _fftw_index     — signed wavenumber n → 1-based storage index
#   _fftw_sym_index — 1-based index i → index of its Hermitian partner
#
# `FTField[WaveNumberVector]` indexing (which must also interleave non-FFT grid
# dimensions) is defined in ftfield.jl; `ProjectedField[m, WaveNumberVector]`
# indexing is in projectedfield.jl.

"""
    WaveNumberVector(ns::Int...)
    WaveNumberVector{N}(ns::NTuple{N, Int})

Integer wavenumber index for spectral fields.  Holds `N` wavenumbers ordered
to match `fft_dims(g) = FFT_DIMS_ORDER`:

- `k[1]` — wavenumber for the rfft dimension `H[1]` (may be negative; see below).
- `k[2:N]` — signed wavenumbers for the remaining FFT dimensions, in FFTW
  storage order: `0, 1, …, N÷2, -(N÷2-1), …, -1`.

**Conjugate symmetry.**  The rfft dimension stores only non-negative wavenumbers.
Requesting a negative first wavenumber `k[1] < 0` is valid: the implementation
reads (or writes) the entry at `(-k[1], -k[2:N]...)` and applies a complex
conjugate, exploiting the Hermitian symmetry of a real-valued field.
"""
struct WaveNumberVector{N}
    ns::NTuple{N, Int}
end
WaveNumberVector(ns::Int...) = WaveNumberVector(ns)

Base.@propagate_inbounds Base.getindex(k::WaveNumberVector, i) = @inbounds getindex(k.ns, i)
Base.length(::WaveNumberVector{N}) where {N} = N

# ------------------------------------------------------------------ #
# Index arithmetic helpers                                             #
# ------------------------------------------------------------------ #
"""
    _fftw_index(n::Int, N::Int) -> Int

Convert a signed wavenumber `n` to a 1-based FFTW storage index in a dimension
of physical size `N`.  Positive and zero wavenumbers occupy the first half of
the storage array (`n ≥ 0 → n + 1`); negative wavenumbers are packed at the
end in wrap-around order (`n < 0 → N + n + 1`).
"""
_fftw_index(n::Int, N::Int) = n >= 0 ? n + 1 : N + n + 1

"""
    _fftw_sym_index(i::Int, N::Int) -> Int

Given a 1-based FFTW storage index `i` in a dimension of size `N`, return
the index of the Hermitian-conjugate entry (the one at wavenumber `−n`).
Index 1 (zero wavenumber) is self-conjugate and maps to itself.
"""
_fftw_sym_index(i::Int, N::Int) = i == 1 ? 1 : N - i + 2

"""
    to_indices(g::AbstractGrid, k::WaveNumberVector{N}) where {N}

Convert a `WaveNumberVector` to 1-based `ProjectedField` axis indices plus a
conjugate flag.  Returns the `(N+1)`-tuple `(i_H1, i_H2, …, i_HN, do_conj)`
where each `i_Hk` is the 1-based index along axis `k+1` of the
`ProjectedField` array (axes follow the order of `fft_dims(g) = FFT_DIMS_ORDER`).

`do_conj = true` means the stored entry is the complex conjugate of the
requested wavenumber (the rfft axis stores only `n ≥ 0`, so negative wavenumbers
are reached via conjugate symmetry).
"""
function to_indices(g::AbstractGrid, k::WaveNumberVector{N}) where {N}
    H = fft_dims(g)
    if k[1] >= 0
        rest = ntuple(j -> _fftw_index( k[j+1], size(g, H[j+1])), Val(N-1))
        return (k[1] + 1, rest..., false)
    else
        rest = ntuple(j -> _fftw_index(-k[j+1], size(g, H[j+1])), Val(N-1))
        return (-k[1] + 1, rest..., true)
    end
end

"""
    to_wavenumber_vector(g, homogeneous_indices) -> WaveNumberVector

Convert the `N`-tuple of 1-based FFTW storage indices `homogeneous_indices`
(as yielded by `for_each_homogeneous_index`) to a `WaveNumberVector`.

- rfft dimension (k=1): storage index `i` → wavenumber `i - 1` (always ≥ 0).
- Full-FFT dimensions (k≥2): positive block `i ≤ N÷2+1` → `i - 1`;
  negative block `i > N÷2+1` → `i - 1 - N`.
"""
function to_wavenumber_vector(g::AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}, homogeneous_indices::NTuple{N, Int}) where {T, D, AXES, FFT_DIMS_ORDER, N}
    N == length(FFT_DIMS_ORDER) || throw(ArgumentError("Mismatch between homogeneous_indices length and FFT_DIMS_ORDER"))
    ns = ntuple(Val(N)) do k
        i = homogeneous_indices[k]
        # rfft dim: only non-negative wavenumbers stored, so i=1 → k=0, i=2 → k=1, …
        # Full-FFT dims: FFTW packs positive wavenumbers first (i ≤ N/2+1 → k=i-1),
        # then negative (i > N/2+1 → k=i-1-N), matching the two-block loop in for_each_homogeneous_index.
        k == 1 ? i - 1 :
                 (i <= (size(g, FFT_DIMS_ORDER[k]) >> 1) + 1 ? i - 1 : i - 1 - size(g, FFT_DIMS_ORDER[k]))
    end
    return WaveNumberVector(ns)
end
