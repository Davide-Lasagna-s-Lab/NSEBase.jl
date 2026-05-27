# Continuous phase shifts of spectral fields along homogeneous directions.
#
# A continuous shift by displacement `s` in a homogeneous direction multiplies
# every Fourier coefficient at signed wavenumber `n` by the phase factor
# `exp(i · n · s · wavenumber_scale(g, dim))`.  This is a lossless, exact
# operation in spectral space — no interpolation is needed.
#
# `_shift_phase` computes the complex phase factor for a given wavenumber
# vector and shift tuple.  `shift!(u, shifts)` applies it in-place to every
# element of an FTField or ProjectedField (and, by iteration, to VectorField).
# `shift(u, shifts)` is the allocating wrapper.
#
# Shifts are specified in physical units (same units as the period `L` used to
# define `wavenumber_scale`), with one entry per homogeneous dimension in
# `fft_dims(grid(u)) = FFT_DIMS_ORDER` order.  Passing all-zero shifts is a
# no-op (guarded by an early return).

"""
    _shift_phase(g, shifts, k) -> Complex

Compute the complex phase factor for a continuous shift of `shifts` (one entry
per homogeneous dimension in `FFT_DIMS_ORDER` order) at signed wavenumber vector `k`.

The factor is `∏_j exp(i · k[j] · shifts[j] · wavenumber_scale(g, FFT_DIMS_ORDER[j]))`.
Zero wavenumbers contribute a factor of `1` (exact, no floating-point error).
"""
function _shift_phase(g::AbstractGrid{T, D, AXES, FFT_DIMS_ORDER},
                      shifts::NTuple{N, Real},
                      k::WaveNumberVector{N}) where {T, D, AXES, FFT_DIMS_ORDER, N}
    N == length(FFT_DIMS_ORDER) ||
        throw(DimensionMismatch("shifts must have one entry per homogeneous dimension; got $(N), expected $(length(FFT_DIMS_ORDER))"))
    return prod(1:N) do i
        n = k[i]
        iszero(n) ? one(Complex{T}) :
                    cis(n * shifts[i] * wavenumber_scale(g, FFT_DIMS_ORDER[i]))
    end
end

"""
    shift!(u::FTField, shifts) -> u

Shift `u` in-place by `shifts`, one entry per homogeneous dimension in
`fft_dims(grid(u)) = FFT_DIMS_ORDER` order.  Each entry is a displacement in
physical units (same units as the period `L` used to define `wavenumber_scale`).

The phase factor applied to spectral coefficient at signed wavenumber vector `k` is

```math
\\exp\\!\\left(i \\sum_{j} k_j \\cdot \\text{shifts}_j \\cdot \\text{wavenumber\\_scale}(g, \\text{FFT\\_DIMS\\_ORDER}_j)\\right).
```

Returns `u` unchanged when all shifts are zero.
"""
function shift!(u::FTField{G}, shifts) where {G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}} where {T, D, AXES, FFT_DIMS_ORDER}
    any(!iszero, shifts) || return u
    g         = grid(u)
    pu        = parent(u)
    inh_dims  = inhomogeneous_dims(g)
    inh_sizes = map(d -> size(g, d), inh_dims)

    for_each_homogeneous_index(g) do _, homogeneous_indices
        # Convert storage indices to signed wavenumbers, then accumulate the
        # total phase as a product over all homogeneous directions.
        k     = to_wavenumber_vector(g, homogeneous_indices)
        phase = _shift_phase(g, shifts, k)

        # Apply phase to every inhomogeneous index combination for this wavenumber.
        # CartesianIndices loop is the innermost, matching the column-major
        # layout of dim 1 in the parent array.
        # Merge the current inhomogeneous CartesianIndex with the homogeneous
        # storage indices to get the full D-dimensional parent-array index.
        for I in CartesianIndices(inh_sizes)
            indices = _combine_indices(g, Tuple(I), homogeneous_indices)
            @inbounds pu[indices...] *= phase
        end
    end
    return u
end

"""
    shift!(u::VectorField, shifts) -> u

Shift each component of `u` in-place. See [`shift!`](@ref).
"""
shift!(u::VectorField{N}, shifts) where {N} = (for n in 1:N; shift!(u[n], shifts); end; return u)

"""
    shift!(a::ProjectedField, shifts) -> a

Shift the projected field `a` in-place by `shifts`, one entry per homogeneous
dimension in `fft_dims(grid(a)) = FFT_DIMS_ORDER` order. The same phase factor as for
[`shift!`](@ref) is applied to every modal coefficient at each
spectral index independently of the mode index `m`, since the shift acts on
the underlying physical-space field.

Returns `a` unchanged when all shifts are zero.
"""
function shift!(a::ProjectedField{G}, shifts) where {G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}} where {T, D, AXES, FFT_DIMS_ORDER}
    any(!iszero, shifts) || return a
    g = grid(a)
    for_each_homogeneous_index(g) do _, homogeneous_indices
        # Same phase computation as FTField shift; applied uniformly across
        # all mode indices m at this spectral location.
        k     = to_wavenumber_vector(g, homogeneous_indices)
        phase = _shift_phase(g, shifts, k)
        for m in axes(a, 1)
            @inbounds a[m, homogeneous_indices...] *= phase
        end
    end
    return a
end

"""
    shift(u, shifts) -> copy

Return a shifted copy of `u`.  See [`shift!`](@ref).
"""
shift(u, shifts) = shift!(copy(u), shifts)
