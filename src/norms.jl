# Inner products, norms, and normdiff for all field types.

# --------- #
# FTField   #
# --------- #
"""
    dot(u::FTField, v::FTField) -> Real

Inner product of two spectral fields on the same grid, defined as

```math
\\langle u, v \\rangle = \\frac{1}{2} \\sum_{\\mathbf{k}} c_{k_1}\\, w(\\mathbf{j})\\,
\\operatorname{Re}\\bigl(\\bar{u}_{\\mathbf{k},\\mathbf{j}}\\, v_{\\mathbf{k},\\mathbf{j}}\\bigr),
```

where the sum is over all stored spectral wavenumbers `k` and inhomogeneous indices
`j`, `c_{k_1}` is `1` for the zero rfft wavenumber and `2` otherwise (Hermitian
symmetry of the rfft), `w(j)` are the quadrature weights returned by
[`weights`](@ref), and the overall factor of `1/2` removes the double-counting
of conjugate-symmetric mode pairs in the signed FFT dimensions.
"""
function LinearAlgebra.dot(u::FTField{G}, v::FTField{G}) where {G<:AbstractGrid}
    g  = grid(u)
    s  = zero(real(eltype(u)))
    pu = parent(u)
    pv = parent(v)
    ws = weights(g)
    # one_or_two: rfft Hermitian weight (each +k mode represents ±k).
    # ws[inhomogeneous_indices...]: quadrature weight for the wall-normal location.
    # indices: full D-dimensional array index for direct access to the parent array.
    for_each_index(g) do one_or_two, inhomogeneous_indices, indices
        @inbounds s += one_or_two * ws[inhomogeneous_indices...] * real(conj(pu[indices...]) * pv[indices...])
    end
    return s / 2
end

"""
    norm(u::FTField) -> Real

Norm induced from [`dot(u::FTField, v::FTField)`](@ref).
"""
LinearAlgebra.norm(u::FTField) = sqrt(dot(u, u))

"""
    normdiff(u::FTField, v::FTField,
             shifts=zeros, tmp=nothing) -> Real
    normdiff(u::VectorField, v::VectorField,
             shifts=zeros, tmp=nothing) -> Real

Return `‖u − shift(v, shifts)‖`, the norm of the difference after optionally
shifting `v` along the homogeneous directions.

`shifts` is a tuple with one entry per homogeneous dimension in
`fft_dims(grid(u)) = FFT_DIMS_ORDER` order, defaulting to all zeros (no shift).

`tmp` is an optional pre-allocated `FTField` workspace.  When `shifts` is
non-zero `tmp` is used to hold the shifted copy of `v`; if `tmp` is `nothing`
a temporary is allocated internally.  When all shifts are zero `tmp` is never
used.
"""
function normdiff(u::FTField{G}, v::FTField{G},
                  shifts=ntuple(Returns(0), length(FFT_DIMS_ORDER)),
                  tmp=nothing) where {G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}} where {T, D, AXES, FFT_DIMS_ORDER}
    if any(!iszero, shifts)
        tmp = tmp === nothing ? zero(v) : tmp
        tmp .= v
        shift!(tmp, shifts)
        v = tmp
    end
    g  = grid(u)
    s  = zero(real(eltype(u)))
    pu = parent(u)
    pv = parent(v)
    ws = weights(g)
    # Same weighting as dot: rfft Hermitian factor and quadrature weight.
    for_each_index(g) do one_or_two, inhomogeneous_indices, indices
        @inbounds s += one_or_two * ws[inhomogeneous_indices...] * abs2(pu[indices...] - pv[indices...])
    end
    return sqrt(s / 2)
end


# ----------- #
# VectorField #
# ----------- #
"""
    dot(q::VectorField, p::VectorField) -> Real

Inner product of two vector fields: sum of [`dot`](@ref) over components.
"""
LinearAlgebra.dot(q::VectorField{N}, p::VectorField{N}) where {N} = sum(dot(q[n], p[n]) for n in 1:N)

"""
    norm(q::VectorField) -> Real

Norm induced from [`dot(q::VectorField, p::VectorField)`](@ref).
"""
LinearAlgebra.norm(q::VectorField) = sqrt(dot(q, q))

"""
    normdiff(u::VectorField, v::VectorField,
             shifts=zeros, tmp=nothing) -> Real

Return `‖u − shift(v, shifts)‖` for vector fields: the Euclidean combination
of per-component [`normdiff`](@ref) values.  See the `FTField` method for the
meaning of `shifts` and `tmp`.
"""
function normdiff(u::VectorField{N, <:FTField{G}}, v::VectorField{N, <:FTField{G}},
                  shifts=ntuple(Returns(0), length(FFT_DIMS_ORDER)),
                  tmp=nothing) where {N, G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}} where {T, D, AXES, FFT_DIMS_ORDER}
    s = zero(real(eltype(u[1])))
    for n in 1:N
        s += normdiff(u[n], v[n], shifts, tmp)^2
    end
    return sqrt(s)
end

"""
    minnormdiff(u, v, N, tmp1=zero(v), tmp2=zero(v)) -> (Real, NTuple)
    minnormdiff(u, v, tmp1=zero(v), tmp2=zero(v)) -> (Real, NTuple)

Return `(min_diff, shifts)`: the minimum of `‖u − shift(v, shifts)‖` over a
regular grid of candidate shifts covering one full period in each transform
dimension.  `N` has one entry per transform dimension in `fft_dims(grid(u))`
order; when omitted, each transform dimension uses 32 samples.

If `fft_dims(grid(u)) == (2, 3)`, for example, then `N = (Nx, Nz)` samples the
first transform dimension with `Nx` shifts and the second transform dimension
with `Nz` shifts.  The returned `shifts` tuple has the same order and length as
`N`.

`tmp1` and `tmp2` are optional pre-allocated workspaces of the same type as `v`.
`tmp1` is used to hold the shifted copy of `v` for each candidate shift;
`tmp2` is forwarded to [`normdiff`](@ref) as scratch space for any additional
temporary work.
"""
function minnormdiff(u::Union{FTField{G}, VectorField{<:Any, <:FTField{G}}},
                     v::Union{FTField{G}, VectorField{<:Any, <:FTField{G}}},
                     N::NTuple{M, Int},
                     tmp1=zero(v),
                     tmp2=zero(v)) where {M, G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}} where {T, D, AXES, FFT_DIMS_ORDER}
    M == length(FFT_DIMS_ORDER) ||
        throw(DimensionMismatch("N must have one entry per transform dimension; got $(M), expected $(length(FFT_DIMS_ORDER))"))

    g        = grid(u)
    min_diff = typemax(real(eltype(u isa VectorField ? u[1] : u)))
    s_min    = ntuple(k -> zero(T), Val(M))

    # The shift increment in each transform dimension is one physical period
    # divided by the number of samples requested in that dimension.
    steps      = ntuple(k -> 2π / (wavenumber_scale(g, FFT_DIMS_ORDER[k]) * N[k]), Val(M))
    zero_shift = ntuple(k -> zero(steps[k]), Val(M))

    # `CartesianIndices(N)` enumerates the whole candidate grid, independently
    # of how many transform dimensions the grid has.  Its entries are 1-based,
    # so subtract one to get the shift count in each direction.
    for I in CartesianIndices(N)
        shift_counts = ntuple(k -> Tuple(I)[k] - 1, Val(M))
        shifts       = ntuple(k -> steps[k] * shift_counts[k], Val(M))

        tmp1 .= v
        shift!(tmp1, shifts)
        diff = normdiff(u, tmp1, zero_shift, tmp2)

        if diff < min_diff
            min_diff = diff
            s_min    = shifts
        end
    end

    return min_diff, s_min
end

function minnormdiff(u::Union{FTField{G}, VectorField{<:Any, <:FTField{G}}},
                     v::Union{FTField{G}, VectorField{<:Any, <:FTField{G}}},
                     tmp1=zero(v),
                     tmp2=zero(v)) where {G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}} where {T, D, AXES, FFT_DIMS_ORDER}
    # Use the historical default resolution, but now size the tuple from the
    # grid's number of transform dimensions instead of assuming a fixed value.
    M = length(FFT_DIMS_ORDER)
    return minnormdiff(u, v, ntuple(Returns(32), Val(M)), tmp1, tmp2)
end


# -------------- #
# ProjectedField #
# -------------- #
"""
    dot(a::ProjectedField, b::ProjectedField)

Inner product of two projected fields, exploiting the rfft Hermitian symmetry.
Wavenumbers with rfft index `> 1` (wavenumber `nx > 0`) are stored once but represent
both `+nx` and `-nx`, so they contribute with weight 2; the `nx = 0` plane has
weight 1.  The result is divided by 2 to account for the double-counting of
signed-FFT pairs `(nz, nt)` and `(-nz, -nt)` that both appear in storage.

The loop uses axis 1 as the mode index `m` (see Storage layout in
[`ProjectedField`](@ref)) and the spectral indices `args...` from
`for_each_homogeneous_index`, which correspond to axes 2 onward in FFT_DIMS_ORDER order.
"""
function LinearAlgebra.dot(a::ProjectedField{G}, b::ProjectedField{G}) where {G<:AbstractGrid}
    # Use a Ref accumulator: a plain scalar mutated inside a closure gets boxed
    # by Julia's closure-capture analysis, causing heap allocations on every
    # inner-loop iteration.  Wrapping it in a Ref keeps the value on the heap
    # once and eliminates repeated boxing.
    s = Ref(zero(real(eltype(a))))
    for_each_homogeneous_index(grid(a)) do one_or_two, homogeneous_indices
        for m in axes(a, 1)
            @inbounds s[] += one_or_two * real(LinearAlgebra.dot(a[m, homogeneous_indices...],
                                                                 b[m, homogeneous_indices...]))
        end
    end
    return s[] / 2
end

"""
    norm(a::ProjectedField)

Norm induced from [`dot(a::ProjectedField, b::ProjectedField)`](@ref).
"""
LinearAlgebra.norm(a::ProjectedField) = sqrt(dot(a, a))

"""
    normdiff(a::ProjectedField, b::ProjectedField) -> Real

Return `‖a − b‖`.

Follows the same rfft/signed-FFT weighting as [`dot`](@ref): rfft wavenumbers
with `nx > 0` contribute with weight 2 (they represent both `±nx`), and the
result is divided by 2 to remove signed-FFT double-counting.
"""
function normdiff(a::ProjectedField{G}, b::ProjectedField{G}) where {G<:AbstractGrid}
    # Ref accumulator: a plain scalar mutated inside a closure is heap-boxed on
    # every iteration by Julia's escape analysis.  A Ref is allocated once.
    s = Ref(zero(real(eltype(a))))
    for_each_homogeneous_index(grid(a)) do one_or_two, homogeneous_indices
        for m in axes(a, 1)
            @inbounds s[] += one_or_two * abs2(a[m, homogeneous_indices...] - b[m, homogeneous_indices...])
        end
    end
    return sqrt(s[] / 2)
end

"""
    normdiff(a::ProjectedField, b::ProjectedField, shifts, tmp::ProjectedField) -> Real

Return `‖a − shift(b, shifts)‖`.

`shifts` is a tuple with one entry per homogeneous dimension in
`fft_dims(grid(a)) = FFT_DIMS_ORDER` order.

`tmp` is a pre-allocated `ProjectedField` workspace of the same type as `a`
and `b`; `b` is copied into `tmp` and (if needed) shifted in place, so `b`
itself is not modified.  Providing `tmp` avoids any allocations: use this form
in performance-critical loops.
"""
function normdiff(a::ProjectedField{G}, b::ProjectedField{G},
                  shifts, tmp::ProjectedField{G}) where {G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}} where {T, D, AXES, FFT_DIMS_ORDER}
    tmp .= b
    any(!iszero, shifts) && shift!(tmp, shifts)
    return normdiff(a, tmp)
end
