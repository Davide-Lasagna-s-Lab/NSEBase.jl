# Distance norms between spectral fields.

"""
    normdiff(u::FTField, v::FTField,
             shifts=zeros, tmp=nothing) -> Real
    normdiff(u::VectorField, v::VectorField,
             shifts=zeros, tmp=nothing) -> Real

Return `‖u − shift(v, shifts)‖`, the norm of the difference after optionally
shifting `v` along the homogeneous directions.

`shifts` is a tuple with one entry per homogeneous dimension in
`fft_dims(grid(u)) = ORDER` order, defaulting to all zeros (no shift).

`tmp` is an optional pre-allocated `FTField` workspace.  When `shifts` is
non-zero `tmp` is used to hold the shifted copy of `v`; if `tmp` is `nothing`
a temporary is allocated internally.  When all shifts are zero `tmp` is never
used.
"""
function normdiff(u::FTField{G}, v::FTField{G},
                  shifts=ntuple(Returns(0), length(ORDER)),
                  tmp=nothing) where {G<:AbstractGrid{T, D, AXES, ORDER}} where {T, D, AXES, ORDER}
    if any(!iszero, shifts)
        tmp = tmp === nothing ? zero(v) : tmp
        tmp .= v
        shift!(tmp, shifts)
        v = tmp
    end
    s  = zero(real(eltype(u)))
    pu = parent(u)
    pv = parent(v)
    ws = weights(grid(u))
    for_each_point(grid(u)) do one_or_two, inhom, idx
        @inbounds s += one_or_two * ws[inhom...] * abs2(pu[idx...] - pv[idx...])
    end
    return sqrt(s / 2)
end

function normdiff(u::VectorField{N, <:FTField{G}}, v::VectorField{N, <:FTField{G}},
                  shifts=ntuple(Returns(0), length(ORDER)),
                  tmp=nothing) where {N, G<:AbstractGrid{T, D, AXES, ORDER}} where {T, D, AXES, ORDER}
    s = zero(real(eltype(u[1])))
    for n in 1:N
        s += normdiff(u[n], v[n], shifts, tmp)^2
    end
    return sqrt(s)
end
