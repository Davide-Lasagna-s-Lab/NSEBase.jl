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
    _shift_ftfield!(u, shifts)
    return u
end

@generated function _shift_ftfield!(u::FTField{G}, shifts) where {T, D, AXES, FFT_DIMS_ORDER, G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}}
    syms = [Symbol("_i", d) for d in 1:D]

    phase_terms = Expr[]
    for (j, d) in enumerate(FFT_DIMS_ORDER)
        n = d == FFT_DIMS_ORDER[1] ? :($(syms[d]) - 1) :
                                     :($(syms[d]) <= (Base.size(u, $d) >> 1) + 1 ?
                                       $(syms[d]) - 1 :
                                       $(syms[d]) - 1 - Base.size(u, $d))
        push!(phase_terms, quote
            _n = $n
            iszero(_n) || (_phase *= cis(_n * shifts[$j] * wavenumber_scale(grid(u), $d)))
        end)
    end

    body = quote
        _phase = one(Complex{$T})
        $(phase_terms...)
        @inbounds parent(u)[$(syms...)] *= _phase
    end

    for d in 1:D
        body = :(for $(syms[d]) in 1:Base.size(u, $d)
                     $body
                 end)
    end

    return quote
        $body
        return u
    end
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
    _shift_projectedfield!(a, shifts)
    return a
end

@generated function _shift_projectedfield!(a::ProjectedField{G}, shifts) where {T, D, AXES, FFT_DIMS_ORDER, G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}}
    N = length(FFT_DIMS_ORDER)
    syms = [Symbol("_i", j) for j in 1:N]

    phase_terms = Expr[]
    for (j, d) in enumerate(FFT_DIMS_ORDER)
        n = j == 1 ? :($(syms[j]) - 1) :
                     :($(syms[j]) <= (Base.size(grid(a), $d) >> 1) + 1 ?
                       $(syms[j]) - 1 :
                       $(syms[j]) - 1 - Base.size(grid(a), $d))
        push!(phase_terms, quote
            _n = $n
            iszero(_n) || (_phase *= cis(_n * shifts[$j] * wavenumber_scale(grid(a), $d)))
        end)
    end

    body = quote
        _phase = one(Complex{$T})
        $(phase_terms...)
        for m in axes(a, 1)
            @inbounds parent(a)[m, $(syms...)] *= _phase
        end
    end

    for j in 1:N
        body = :(for $(syms[j]) in axes(a, $(j + 1))
                     $body
                 end)
    end

    return quote
        $body
        return a
    end
end

"""
    shift(u, shifts) -> copy

Return a shifted copy of `u`.  See [`shift!`](@ref).
"""
shift(u, shifts) = shift!(copy(u), shifts)
