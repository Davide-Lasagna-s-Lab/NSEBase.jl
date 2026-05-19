# Generic spectral-wavenumber iteration for FTField / ProjectedField.
# for_each_wavenumber is @generated from the grid type — no hardcoded dimension counts.

"""
    for_each_wavenumber(f, g::AbstractGrid)

Call `f(one_or_two, i_H1, i_H2, …, i_HN)` for every stored spectral wavenumber of a
field on `g`, where `ORDER = fft_dims(g)` and each `i_Hk` is a 1-based FFTW
storage index:

- `one_or_two`: `1` if `i_H1 == 1` (zero rfft wavenumber, no conjugate partner),
  `2` otherwise (the stored wavenumber represents both `+k` and `-k` via Hermitian symmetry).
- `i_H1` steps over `1:(size(g, ORDER[1]) >> 1) + 1` — the rfft dimension stores
  only non-negative wavenumbers.
- Each `i_Hk` for `k ≥ 2` steps over all of `1:size(g, ORDER[k])`, visited in two
  contiguous blocks (positive-wavenumber indices first, then negative) so no
  per-call sign branch appears inside `f`.

Non-FFT dimensions (e.g. the wall-normal direction) are not iterated; loop
over them inside `f` if needed.  The call order is innermost `ORDER[1]` to
outermost `ORDER[N]`, matching `ProjectedField` storage `(mode, ORDER[1], …, ORDER[N])`
for cache efficiency.

# Example

Inner product of two `ProjectedField`s. Wavenumbers with `one_or_two == 2` represent
both `+nx` and `-nx` and are counted twice; the zero-wavenumber plane is counted
once.  The factor of 2 from signed-FFT pairs `(nz,nt)` / `(-nz,-nt)` both
appearing in storage is removed by the final `/ 2`.

```julia
function LinearAlgebra.dot(a::ProjectedField{G}, b::ProjectedField{G}) where {G<:AbstractGrid}
    s = zero(real(eltype(a)))
    for_each_wavenumber(grid(a)) do one_or_two, args...
        for m in axes(a, 1)
            @inbounds s += one_or_two * real(LinearAlgebra.dot(parent(a)[m, args...],
                                                               parent(b)[m, args...]))
        end
    end
    return s / 2
end
```
"""
@generated function for_each_wavenumber(f, g::AbstractGrid{T, D, AXES, ORDER}) where {T, D, AXES, ORDER}
    N          = length(ORDER)
    syms       = [Symbol("_i", k) for k in 1:N]
    one_or_two = :($(syms[1]) == 1 ? 1 : 2)

    # Innermost: rfft dimension — non-negative wavenumbers only, single block.
    # one_or_two is passed as the first argument so callers need no branch.
    body = :(for $(syms[1]) in 1:(Base.size(g, $(ORDER[1])) >> 1) + 1
                 f($one_or_two, $(syms...))
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


"""
    for_each_point(f, g::AbstractGrid)

Call `f(one_or_two, inhom, idx)` for every element of an `FTField` on `g`, where:
- `one_or_two`: `1` if the rfft index (dimension `ORDER[1]`) is `1`, else `2`.
- `inhom`: `NTuple` of 1-based indices along `inhomogeneous_dims(g)`, ascending.
- `idx`: `NTuple{D}` of all array indices, suitable for indexing `parent(u)`.

Loops are ordered with array dimension 1 innermost (column-major), regardless
of whether that dimension is spectral or inhomogeneous.  This keeps accesses to
the FTField parent array cache-friendly.  Spectral dimensions in `ORDER[2:end]`
are visited in two contiguous blocks (positive wavenumbers first, then negative)
to match FFTW storage layout; the rfft dimension (`ORDER[1]`) visits only
non-negative wavenumber storage indices.

Use this instead of composing `for_each_wavenumber` with a manual inhomogeneous loop
when the operation accesses the parent array of an `FTField` — notably `dot`.

# Generated code example

For a grid with `D=4`, `ORDER=(1,2,3)` (dims 1–3 spectral, dim 1 the rfft
direction) and one inhomogeneous dimension `4`:

```julia
for _i4 in 1:size(g, 4)                               # inhomogeneous, outermost
    for _i3 in 1:(size(g,3)>>1)+1                     # z: positive wavenumbers
        for _i2 in 1:(size(g,2)>>1)+1                 # x: positive wavenumbers
            for _i1 in 1:(size(g,1)>>1)+1             # t: rfft, innermost
                f(_i1==1 ? 1 : 2, (_i4,), (_i1,_i2,_i3,_i4))
            end
        end
        for _i2 in (size(g,2)>>1)+2:size(g,2)         # x: negative wavenumbers
            for _i1 in 1:(size(g,1)>>1)+1
                f(_i1==1 ? 1 : 2, (_i4,), (_i1,_i2,_i3,_i4))
            end
        end
    end
    for _i3 in (size(g,3)>>1)+2:size(g,3)             # z: negative wavenumbers
        for _i2 in 1:(size(g,2)>>1)+1
            for _i1 in 1:(size(g,1)>>1)+1
                f(_i1==1 ? 1 : 2, (_i4,), (_i1,_i2,_i3,_i4))
            end
        end
        for _i2 in (size(g,2)>>1)+2:size(g,2)
            for _i1 in 1:(size(g,1)>>1)+1
                f(_i1==1 ? 1 : 2, (_i4,), (_i1,_i2,_i3,_i4))
            end
        end
    end
end
```

Dim 1 (the rfft direction) is stride-1 in the `FTField` parent array, so it
is the innermost loop regardless of the grid layout.
"""
@generated function for_each_point(f, g::AbstractGrid{T, D, AXES, ORDER}) where {T, D, AXES, ORDER}
    # One loop variable per array dimension, named _i1 … _iD.
    syms = [Symbol("_i", d) for d in 1:D]

    # Inhomogeneous dims are those not transformed by FFTs.
    inh      = Tuple(d for d in 1:D if d ∉ ORDER)
    inh_syms = [syms[d] for d in inh]

    # ORDER[1] is the rfft dimension: only non-negative wavenumbers are stored,
    # so its index runs 1:(N÷2)+1.  When the index equals 1 (zero wavenumber)
    # the mode has no conjugate partner → weight 1; all others → weight 2.
    rfft_dim   = ORDER[1]
    one_or_two = :($(syms[rfft_dim]) == 1 ? 1 : 2)

    # Innermost call: pass the rfft weight, the inhomogeneous index tuple, and
    # the full D-dimensional index tuple.
    body = :(f($one_or_two, ($(inh_syms...),), ($(syms...),)))

    # Build loops from dim 1 (innermost) to dim D (outermost) by wrapping `body`
    # in ascending dimension order.  Each new wrap becomes the new outermost loop,
    # so after the full iteration dim 1 is innermost — matching column-major layout.
    for d in 1:D
        sym = syms[d]
        if d == rfft_dim
            # rfft dim: storage indices 1:(N÷2)+1 cover non-negative wavenumbers only.
            body = :(for $sym in 1:(Base.size(g, $d) >> 1) + 1; $body; end)
        elseif d ∈ ORDER
            # Other spectral dims: FFTW stores positive wavenumbers first (1:(N÷2)+1),
            # then negative ((N÷2)+2:N).  Visit them in two contiguous blocks so the
            # inner loops always see a contiguous range — avoiding a branch per element.
            pos  = :(for $sym in 1:(Base.size(g, $d) >> 1) + 1; $body; end)
            neg  = :(for $sym in (Base.size(g, $d) >> 1) + 2:Base.size(g, $d); $body; end)
            body = Expr(:block, pos, neg)
        else
            # Inhomogeneous dim: no wavenumber structure, visit every index.
            body = :(for $sym in 1:Base.size(g, $d); $body; end)
        end
    end
    return body
end
