# Continuous phase shifts of spectral fields along homogeneous directions.

"""
    shift!(u::FTField, shifts::NTuple{N, Real}) -> u
    shift!(u::VectorField, shifts::NTuple{N, Real}) -> u

Shift `u` in-place by `shifts`, one entry per homogeneous dimension in
`fft_dims(grid(u)) = ORDER` order.  Each entry is a displacement in physical
units: the phase factor applied to spectral coefficient `(k_1, k_2, …)` is

```math
\\exp\\!\\left(i \\sum_{j} k_j \\cdot \\text{shifts}_j \\cdot \\text{wavenumber\\_scale}(g, \\text{ORDER}_j)\\right).
```

Returns `u` unchanged when all shifts are zero.
"""
function shift!(u::FTField{G}, shifts) where {G<:AbstractGrid{T, D, AXES, ORDER}} where {T, D, AXES, ORDER}
    any(!iszero, shifts) || return u
    g         = grid(u)
    pu        = parent(u)
    N         = length(ORDER)
    inh_dims  = inhomogeneous_dims(g)
    inh_sizes = map(d -> size(g, d), inh_dims)

    for_each_mode(g) do _, spectral...
        # Convert each 1-based storage index to a signed wavenumber.
        # rfft dim (k=1): storage i → wavenumber i-1 (always ≥ 0).
        # Full-FFT dims (k≥2): positive block i ≤ N/2+1 → i-1,
        #                       negative block i >  N/2+1 → i-1-N.
        freqs = ntuple(N) do k
            i = spectral[k]
            k == 1 ? i - 1 :
                     (i <= (size(g, ORDER[k]) >> 1) + 1 ? i - 1 : i - 1 - size(g, ORDER[k]))
        end

        phase = prod(1:N) do k
            iszero(freqs[k]) ? one(Complex{T}) :
                               cis(freqs[k] * shifts[k] * wavenumber_scale(g, ORDER[k]))
        end

        # Apply phase to every inhomogeneous index combination for this mode.
        # CartesianIndices loop is the innermost, matching the column-major
        # layout of dim 1 in the parent array.
        for I in CartesianIndices(inh_sizes)
            idx = _combine_indices(g, Tuple(I), spectral)
            @inbounds pu[idx...] *= phase
        end
    end
    return u
end

shift!(u::VectorField{N}, shifts) where {N} = (for n in 1:N; shift!(u[n], shifts); end; return u)

"""
    shift(u, shifts) -> copy

Return a shifted copy of `u`.  See [`shift!`](@ref).
"""
shift(u, shifts) = shift!(copy(u), shifts)
