# WaveNumberVector: integer wavenumber index for spectral fields.

# A WaveNumberVector{N} holds N integer wavenumbers ordered to match
# fft_dims(g) = ORDER.  The first wavenumber (ns[1]) is for the rfft
# dimension ORDER[1]; negative values trigger conjugate-symmetry lookup. All
# others are signed FFT wavenumbers in FFTW order: 0, 1, ..., N/2, -(N/2-1), ..., -1.

# ProjectedField[m, WaveNumberVector] indexing lives in projectedfield.jl.
# FTField[..., WaveNumberVector] is grid-layout-specific and belongs downstream.

"""
    WaveNumberVector(ns::Int...)
    WaveNumberVector{N}(ns::NTuple{N, Int})

Integer wavenumber index for spectral fields.  Holds `N` wavenumbers ordered
to match `fft_dims(g) = ORDER`:

- `ns[1]` — wavenumber for the rfft dimension `H[1]` (may be negative; see below).
- `ns[2:N]` — signed wavenumbers for the remaining FFT dimensions, in FFTW
  storage order: `0, 1, …, N÷2, -(N÷2-1), …, -1`.

**Conjugate symmetry.**  The rfft dimension stores only non-negative wavenumbers.
Requesting a negative first wavenumber `ns[1] < 0` is valid: the implementation
reads (or writes) the entry at `(-ns[1], -ns[2:N]...)` and applies a complex
conjugate, exploiting the Hermitian symmetry of a real-valued field.
"""
struct WaveNumberVector{N}
    ns::NTuple{N, Int}
end
WaveNumberVector(ns::Int...) = WaveNumberVector(ns)

# ------------------------------------------------------------------ #
# Index arithmetic helpers                                             #
# ------------------------------------------------------------------ #
# 1-based FFTW storage index for signed wavenumber n in a dimension of size N.
# Positive and zero wavenumbers occupy the first half of the array; negative
# wavenumbers are stored at the end in wrap-around order.
_fftw_index(n::Int, N::Int) = n >= 0 ? n + 1 : N + n + 1

# Given a 1-based FFTW index i, return the index of the conjugate-symmetric
# entry (wavenumber -n).  Index 1 (n = 0) maps to itself.
_fftw_sym_index(i::Int, N::Int) = i == 1 ? 1 : N - i + 2

"""
    _wavenumber_vector_to_indices(g::AbstractGrid, n::WaveNumberVector{N}) where {N}

Convert a `WaveNumberVector` to 1-based `ProjectedField` axis indices plus a
conjugate flag.  Returns the `(N+1)`-tuple `(i_H1, i_H2, …, i_HN, do_conj)`
where each `i_Hk` is the 1-based index along axis `k+1` of the
`ProjectedField` array (axes follow the order of `fft_dims(g) = ORDER`).

`do_conj = true` means the stored entry is the complex conjugate of the
requested mode (the rfft axis stores only `n ≥ 0`, so negative wavenumbers
are reached via conjugate symmetry).
"""
function _wavenumber_vector_to_indices(g::AbstractGrid, n::WaveNumberVector{N}) where {N}
    H = fft_dims(g)
    if n.ns[1] >= 0
        rest = ntuple(j -> _fftw_index( n.ns[j+1], size(g, H[j+1])), Val(N-1))
        return (n.ns[1] + 1, rest..., false)
    else
        rest = ntuple(j -> _fftw_index(-n.ns[j+1], size(g, H[j+1])), Val(N-1))
        return (-n.ns[1] + 1, rest..., true)
    end
end
