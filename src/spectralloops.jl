# Generic spectral-mode iteration for FTField / ProjectedField.
# for_each_mode is @generated from the grid type — no hardcoded dimension counts.

"""
    for_each_mode(f, g::AbstractGrid)

Call `f(i_H1, i_H2, …, i_HN)` for every stored spectral mode of a field on
`g`, where `H = fft_dims(g) = ORDER` and each `i_Hk` is a 1-based FFTW
storage index:

- `i_H1` steps over `1:(size(g, ORDER[1]) >> 1) + 1` — the rfft dimension stores
  only non-negative wavenumbers.
- Each `i_Hk` for `k ≥ 2` steps over all of `1:size(g, ORDER[k])`, visited in two
  contiguous blocks (positive-wavenumber indices first, then negative) so no
  per-call sign branch appears inside `f`.

Non-FFT dimensions (e.g. the wall-normal direction) are not iterated; loop
over them inside `f` if needed.  The call order is innermost `ORDER[1]` to
outermost `H[N]`, matching `ProjectedField` storage `(mode, ORDER[1], …, ORDER[N])`
for cache efficiency.

# Example

Inner product of two `ProjectedField`s with rfft Hermitian weighting. Modes
with rfft index `> 1` represent both `+nx` and `-nx` and are counted twice;
the `nx = 0` plane is counted once.  The factor of 2 from the signed-FFT
pairs `(nz, nt)` / `(-nz, -nt)` both appearing in storage is removed by the
final `/ 2`.

```julia
function LinearAlgebra.dot(a::ProjectedField{G}, b::ProjectedField{G}) where {G<:AbstractGrid}
    s = zero(real(eltype(a)))
    for_each_mode(grid(a)) do args...
        w = first(args) == 1 ? 1 : 2       # rfft Hermitian weight
        for m in axes(a, 1)
            @inbounds s += w * real(LinearAlgebra.dot(parent(a)[m, args...],
                                                      parent(b)[m, args...]))
        end
    end
    return s / 2
end
```
"""
@generated function for_each_mode(f, g::AbstractGrid{T, D, AXES, ORDER}) where {T, D, AXES, ORDER}
    N    = length(ORDER)
    syms = [Symbol("_i", k) for k in 1:N]

    # Innermost: rfft dimension — non-negative wavenumbers only, single block.
    body = :(for $(syms[1]) in 1:(Base.size(g, $(ORDER[1])) >> 1) + 1
                 f($(syms...))
             end)

    # Wrap in signed-FFT loops ORDER[2:end] from innermost to outermost.
    # Split each into positive and negative blocks.
    for k in 2:N
        dim  = ORDER[k]
        sym  = syms[k]
        pos  = :(for $sym in 1:(Base.size(g, $dim) >> 1) + 1; $body; end)
        neg  = :(for $sym in (Base.size(g, $dim) >> 1) + 2:Base.size(g, $dim); $body; end)
        body = Expr(:block, pos, neg)
    end

    return body
end
