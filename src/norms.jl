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
    minnormdiff(u, v, N=(32,32,32), tmp1=zero(v), tmp2=zero(v)) -> (Real, NTuple)

Return `(min_diff, (s1, s2, s3))`: the minimum of `‖u − shift(v, shifts)‖` over
a `N[1]×N[2]×N[3]` grid of shifts covering one full period in each of the three
homogeneous directions (in `fft_dims` order), and the shift at which the minimum
is attained.

`tmp1` and `tmp2` are optional pre-allocated workspaces of the same type as `v`.
"""
function minnormdiff(u::Union{FTField{G}, VectorField{<:Any, <:FTField{G}}},
                     v::Union{FTField{G}, VectorField{<:Any, <:FTField{G}}},
                     N::NTuple{3, Int}=(32, 32, 32),
                     tmp1=zero(v),
                     tmp2=zero(v)) where {G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}} where {T, D, AXES, FFT_DIMS_ORDER}
    g        = grid(u)
    min_diff = typemax(real(eltype(u isa VectorField ? u[1] : u)))
    s_min    = ntuple(k -> zero(T), 3)

    steps = ntuple(k -> 2π / (wavenumber_scale(g, FFT_DIMS_ORDER[k]) * N[k]), 3)
    zero_step = ntuple(k -> zero(steps[1]), 3)

    tmp1 .= v
    for i3 in 0:N[3]-1
        for i2 in 0:N[2]-1
            for i1 in 0:N[1]-1
                diff = normdiff(u, tmp1, ntuple(Returns(zero(T)), 3), tmp2)
                if diff < min_diff
                    min_diff = diff
                    s_min = (steps[1]*i1, steps[2]*i2, steps[3]*i3)
                end
                shift!(tmp1, Base.setindex(zero_step, steps[1], 1))
            end
            shift!(tmp1, Base.setindex(zero_step, steps[2], 2))
        end
        shift!(tmp1, Base.setindex(zero_step, steps[3], 3))
    end

    return min_diff, s_min
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
    s = zero(real(eltype(a)))
    for_each_homogeneous_index(grid(a)) do one_or_two, homogeneous_indices...
        for m in axes(a, 1)
            @inbounds s += one_or_two * real(LinearAlgebra.dot(a[m, homogeneous_indices...], 
                                                               b[m, homogeneous_indices...]))
        end
    end
    return s / 2
end

"""
    norm(a::ProjectedField)

Norm induced from [`dot(a::ProjectedField, b::ProjectedField)`](@ref).
"""
LinearAlgebra.norm(a::ProjectedField) = sqrt(dot(a, a))

"""
    normdiff(a::ProjectedField, b::ProjectedField,
             shifts=zeros, tmp=nothing) -> Real

Return `‖a − shift(b, shifts)‖`, the norm of the difference after optionally
shifting `b` along the homogeneous directions.

`shifts` is a tuple with one entry per homogeneous dimension in
`fft_dims(grid(a)) = FFT_DIMS_ORDER` order, defaulting to all zeros (no shift).

`tmp` is an optional pre-allocated `ProjectedField` workspace used to hold the
shifted copy of `b` when shifts are non-zero; if `nothing` a temporary is
allocated internally.

Follows the same rfft/signed-FFT weighting as [`dot`](@ref): rfft wavenumbers with
`nx > 0` contribute with weight 2 (they represent both `±nx`), and the result
is divided by 2 to remove signed-FFT double-counting.
"""
function normdiff(a::ProjectedField{G}, b::ProjectedField{G},
                  shifts=ntuple(Returns(0), length(FFT_DIMS_ORDER)),
                  tmp=nothing) where {G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}} where {T, D, AXES, FFT_DIMS_ORDER}
    if any(!iszero, shifts)
        tmp = tmp === nothing ? zero(b) : tmp
        tmp .= b
        shift!(tmp, shifts)
        b = tmp
    end
    s = zero(real(eltype(a)))
    for_each_homogeneous_index(grid(a)) do one_or_two, homogeneous_indices...
        for m in axes(a, 1)
            @inbounds s += one_or_two * abs2(a[m, homogeneous_indices...] - b[m, homogeneous_indices...])
        end
    end
    return sqrt(s / 2)
end
