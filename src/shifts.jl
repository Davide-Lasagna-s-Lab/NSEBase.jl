# Continuous phase shifts of spectral fields along homogeneous directions.

"""
    shift!(u::FTField, shifts) -> u

Shift `u` in-place by `shifts`, one entry per homogeneous dimension in
`fft_dims(grid(u)) = FFT_DIMS_ORDER` order.  Each entry is a displacement in physical
units: the phase factor applied to spectral coefficient `(k_1, k_2, …)` is

```math
\\exp\\!\\left(i \\sum_{j} k_j \\cdot \\text{shifts}_j \\cdot \\text{wavenumber\\_scale}(g, \\text{FFT_DIMS_ORDER}_j)\\right).
```

Returns `u` unchanged when all shifts are zero.
"""
function shift!(u::FTField{G}, shifts) where {G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}} where {T, D, AXES, FFT_DIMS_ORDER}
    any(!iszero, shifts) || return u
    g         = grid(u)
    pu        = parent(u)
    N         = length(FFT_DIMS_ORDER)
    inh_dims  = inhomogeneous_dims(g)
    inh_sizes = map(d -> size(g, d), inh_dims)

    for_each_homogeneous_index(g) do _, homogeneous_indices
        # Convert storage indices to signed wavenumbers, then accumulate the
        # total phase as a product over all homogeneous directions.
        k     = to_wavenumber_vector(g, homogeneous_indices)
        phase = prod(1:N) do i
            n = k[i]
            iszero(n) ? one(Complex{T}) :
                        cis(n * shifts[i] * wavenumber_scale(g, FFT_DIMS_ORDER[i]))
        end

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

Shift each component of `u` in-place. See [`shift!(::FTField, ...)`](@ref).
"""
shift!(u::VectorField{N}, shifts) where {N} = (for n in 1:N; shift!(u[n], shifts); end; return u)

"""
    shift!(a::ProjectedField, shifts) -> a

Shift the projected field `a` in-place by `shifts`, one entry per homogeneous
dimension in `fft_dims(grid(a)) = FFT_DIMS_ORDER` order. The same phase factor as for
[`shift!(::FTField, ...)`](@ref) is applied to every modal coefficient at each
spectral index independently of the mode index `m`, since the shift acts on
the underlying physical-space field.

Returns `a` unchanged when all shifts are zero.
"""
function shift!(a::ProjectedField{G}, shifts) where {G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}} where {T, D, AXES, FFT_DIMS_ORDER}
    any(!iszero, shifts) || return a
    g = grid(a)
    N = length(FFT_DIMS_ORDER)
    for_each_homogeneous_index(g) do _, homogeneous_indices
        # Same phase computation as FTField shift; applied uniformly across
        # all mode indices m at this spectral location.
        k     = to_wavenumber_vector(g, homogeneous_indices)
        phase = prod(1:N) do i
            n = k[i]
            iszero(n) ? one(Complex{T}) :
                        cis(n * shifts[i] * wavenumber_scale(g, FFT_DIMS_ORDER[i]))
        end
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
