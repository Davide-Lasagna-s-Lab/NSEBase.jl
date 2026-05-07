# Generic Fourier-mode iteration utilities.
#
# AbstractGrid{T, D, H} encodes the transformed dimensions in H:
#   H[1]      — rfft dimension: only non-negative frequencies stored (size N/2+1)
#   H[2:end]  — fft dimensions: both signs stored (size N), packed as
#               [0, +1, …, +N/2, -(N/2-1), …, -1]  (Julia/FFTW convention)

"""
    _signed_wavenumber(i, N) -> Int

Map a 1-based array index `i` in an fft dimension of size `N` to its signed
wavenumber:  1 … N/2+1  →  0 … N/2,   N/2+2 … N  →  -(N/2-1) … -1.
"""
@inline _signed_wavenumber(i::Int, N::Int) = i <= (N >> 1) + 1 ? i - 1 : i - N - 1

"""
    _loop_fft_dims(f, u::FTField)

Call `f(i₁, i₂, …)` for every combination of 1-based array indices across the
fft dimensions `H[2:end]` of `u` (all homogeneous dims except the rfft one).
Each combination is visited exactly once; the caller is responsible for any
Hermitian-symmetry normalisation (e.g. dividing by 2 when summing the full
two-sided spectrum).

## Example
For a field with `H = (2, 3, 4)` the loop is equivalent to
```julia
for _i3 in 1:size(u, 3), _i4 in 1:size(u, 4)
    f(_i3, _i4)
end
```
The closure parameters can be named freely in the `do` block.
"""
@generated function _loop_fft_dims(f::F, u::FTField{<:AbstractGrid{<:Any, D, H}}) where {F, D, H}
    fft_dims = H[2:end]
    isempty(fft_dims) && return :(f(); nothing)

    syms = [Symbol("_i", d) for d in fft_dims]
    body = :(f($(syms...)); nothing)
    for (sym, d) in Iterators.reverse(collect(zip(syms, fft_dims)))
        body = quote
            for $sym in 1:Base.size(u, $d)
                $body
            end
        end
    end
    return Base.remove_linenums!(body)
end

"""
    _loop_all_modes(f, u::FTField)

Call `f(i₁, i₂, …)` for every combination of 1-based array indices across
**all** transformed dimensions `H` of `u`:
- `H[1]` (rfft): indices `1:size(u, H[1])` (already the halved+1 size).
- `H[2:end]` (fft): indices `1:size(u, d)` (full two-sided spectrum).

## Example
For `H = (2, 3, 4)` the loop is equivalent to
```julia
for _i2 in 1:size(u, 2), _i3 in 1:size(u, 3), _i4 in 1:size(u, 4)
    f(_i2, _i3, _i4)
end
```
"""
@generated function _loop_all_modes(f::F, u::FTField{<:AbstractGrid{<:Any, D, H}}) where {F, D, H}
    isempty(H) && return :(f(); nothing)

    syms = [Symbol("_i", d) for d in H]
    body = :(f($(syms...)); nothing)
    for (sym, d) in Iterators.reverse(collect(zip(syms, H)))
        body = quote
            for $sym in 1:Base.size(u, $d)
                $body
            end
        end
    end
    return Base.remove_linenums!(body)
end

"""
    @loop_nznt Nt Nz expr

Loop over **all** (nz, nt) index pairs — positive and negative — injecting
into `expr`:
- `_nz`, `_nt` : 1-based array indices
-  `nz`,  `nt` : signed wavenumbers (via `_signed_wavenumber`)

Useful when the signed wavenumber value is needed directly in the loop body
(e.g. applying a phase shift `exp(im*nz*Δz)`). For inner-product sums that
only need array indices, prefer `_loop_fft_dims`.
"""
macro loop_nznt(Nt, Nz, expr)
    quote
        for $(esc(:_nt)) in 1:($(esc(Nt)) >> 1) + 1
            for $(esc(:_nz)) in 1:($(esc(Nz)) >> 1) + 1
                $(esc(:nz)) = $(esc(:_nz)) - 1
                $(esc(:nt)) = $(esc(:_nt)) - 1
                $(esc(expr))
            end
            for $(esc(:_nz)) in ($(esc(Nz)) >> 1) + 2:$(esc(Nz))
                $(esc(:nz)) = $(esc(:_nz)) - $(esc(Nz)) - 1
                $(esc(:nt)) = $(esc(:_nt)) - 1
                $(esc(expr))
            end
        end
        for $(esc(:_nt)) in ($(esc(Nt)) >> 1) + 2:$(esc(Nt))
            for $(esc(:_nz)) in 1:($(esc(Nz)) >> 1) + 1
                $(esc(:nz)) = $(esc(:_nz)) - 1
                $(esc(:nt)) = $(esc(:_nt)) - $(esc(Nt)) - 1
                $(esc(expr))
            end
            for $(esc(:_nz)) in ($(esc(Nz)) >> 1) + 2:$(esc(Nz))
                $(esc(:nz)) = $(esc(:_nz)) - $(esc(Nz)) - 1
                $(esc(:nt)) = $(esc(:_nt)) - $(esc(Nt)) - 1
                $(esc(expr))
            end
        end
    end
end
