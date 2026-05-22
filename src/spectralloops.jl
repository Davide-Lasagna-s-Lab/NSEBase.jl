# Generic index iteration over homogeneous and inhomogeneous dimensions of
# FTField / ProjectedField.  Both @generated functions specialise on the grid
# type — no hardcoded dimension counts.

"""
    for_each_homogeneous_index(f, g::AbstractGrid)

Call `f(one_or_two, i_H1, i_H2, …, i_HN)` for every stored FFTW index
combination across the `N = length(fft_dims(g))` homogeneous (FFT) dimensions
of `g`, where `FFT_DIMS_ORDER = fft_dims(g)` and each `i_Hk` is a 1-based storage
index for `FFT_DIMS_ORDER[k]`.

## Arguments passed to `f`

- `one_or_two`: Hermitian multiplicity.
  - `1` when `i_H1 == 1`: the zero wavenumber has no conjugate partner in the
    rfft and must be counted once.
  - `2` for all other `i_H1`: the stored coefficient represents both `+k₁`
    and its implicit conjugate `−k₁`.
  - **Nyquist caveat**: when `size(g, FFT_DIMS_ORDER[1])` is even, the last rfft index
    `i_H1 = (size(g, FFT_DIMS_ORDER[1]) >> 1) + 1` is the Nyquist frequency, which is
    also self-conjugate (weight 1), but this function assigns it `2`.  Callers
    that need exact quadrature on non-dealiased even-resolution grids must
    handle this index explicitly.
- `i_H1, …, i_HN`: 1-based FFTW storage indices, one per homogeneous
  dimension in `FFT_DIMS_ORDER` order.
  - `i_H1` ranges over `1:(size(g, FFT_DIMS_ORDER[1]) >> 1) + 1` — the rfft half-
    spectrum stores only non-negative wavenumbers.
  - Each `i_Hk` for `k ≥ 2` ranges over `1:size(g, FFT_DIMS_ORDER[k])`, visited as two
    contiguous blocks: positive wavenumbers `1:(N >> 1) + 1` first, then
    negative `(N >> 1) + 2:N`.  The split avoids a per-call sign branch inside
    `f` and keeps memory accesses contiguous.

The innermost loop is always `FFT_DIMS_ORDER[1]` (rfft dimension), the outermost
`FFT_DIMS_ORDER[N]`, so the call order matches `ProjectedField` storage layout
`(mode, FFT_DIMS_ORDER[1], …, FFT_DIMS_ORDER[N])` for cache efficiency.

Non-homogeneous dimensions are not iterated; loop over them inside `f`.

## Example

`FFT_DIMS_ORDER = (1, 2)`, `size(g, 1) = 4` (rfft → 3 stored indices: wavenumbers
0, +1, +2), `size(g, 2) = 6` (signed FFT → positive block 1:4, wavenumbers
0,+1,+2,+3; negative block 5:6, wavenumbers −2,−1).

```
i_H1: 1  2  3      one_or_two: 1  2  2
        (Nyquist ↑, assigned weight 2 — see caveat above)

i_H2=1 (k₂= 0):  f(1, 1, 1)   f(2, 2, 1)   f(2, 3, 1)
i_H2=2 (k₂=+1):  f(1, 1, 2)   f(2, 2, 2)   f(2, 3, 2)
i_H2=3 (k₂=+2):  f(1, 1, 3)   f(2, 2, 3)   f(2, 3, 3)
i_H2=4 (k₂=+3):  f(1, 1, 4)   f(2, 2, 4)   f(2, 3, 4)
i_H2=5 (k₂=−2):  f(1, 1, 5)   f(2, 2, 5)   f(2, 3, 5)
i_H2=6 (k₂=−1):  f(1, 1, 6)   f(2, 2, 6)   f(2, 3, 6)
```

`i_H1` is the inner (fastest-changing) index; `i_H2` is outer.
"""
@generated function for_each_homogeneous_index(f, g::AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}) where {T, D, AXES, FFT_DIMS_ORDER}
    N          = length(FFT_DIMS_ORDER)
    syms       = [Symbol("_i", k) for k in 1:N]
    one_or_two = :($(syms[1]) == 1 ? 1 : 2)

    # Innermost: rfft dimension — non-negative wavenumbers only, single block.
    # one_or_two is passed as the first argument so callers need no branch.
    body = :(for $(syms[1]) in 1:(Base.size(g, $(FFT_DIMS_ORDER[1])) >> 1) + 1
                 f($one_or_two, $(syms...))
             end)

    # Wrap in signed-FFT loops FFT_DIMS_ORDER[2:end] from innermost to outermost.
    # Split each into positive and negative blocks.
    for k in 2:N
        dim  = FFT_DIMS_ORDER[k]
        sym  = syms[k]
        pos  = :(for $sym in 1:(Base.size(g, $dim) >> 1) + 1; $body; end)
        neg  = :(for $sym in (Base.size(g, $dim) >> 1) + 2:Base.size(g, $dim); $body; end)
        body = Expr(:block, pos, neg)
    end

    return body
end


"""
    for_each_index(f, g::AbstractGrid)

Call `f(one_or_two, inhomogeneous_indices, indices)` for every complex element
of an `FTField` on `g`:

- `one_or_two`: Hermitian multiplicity for the rfft dimension (same definition
  and Nyquist caveat as in [`for_each_homogeneous_index`](@ref)).
- `inhomogeneous_indices`: `NTuple` of 1-based indices along
  `inhomogeneous_dims(g)`, in ascending dimension order.
- `indices`: `NTuple{D}` of all `D` 1-based array indices, suitable for
  indexing `parent(u)` directly.

Loops are ordered with array **dimension 1 innermost** (column-major),
regardless of whether any particular dimension is homogeneous or
inhomogeneous — keeping accesses to the `FTField` parent array cache-
friendly.  The rfft dimension (`FFT_DIMS_ORDER[1]`) is limited to
`1:(size(g, FFT_DIMS_ORDER[1]) >> 1) + 1`; all other dimensions are visited over their
full storage range `1:size(g, d)` in a single contiguous block.

Prefer `for_each_index` over composing `for_each_homogeneous_index` with a
manual inhomogeneous loop when the callback accesses `parent(u)` — for
example, inner products on `FTField`.

## Generated code examples

**`D = 2`, `FFT_DIMS_ORDER = (1, 2)`** — both dimensions homogeneous, no inhomogeneous
dimension:

```julia
for _i2 in 1:size(g,2)                  # dim 2: all indices, outermost
    for _i1 in 1:(size(g,1)>>1)+1       # dim 1: rfft, innermost
        f(_i1==1 ? 1 : 2, (), (_i1,_i2))
    end
end
```

`inhomogeneous_indices = ()` (empty), `indices = (_i1, _i2)`.

**`D = 3`, `FFT_DIMS_ORDER = (1, 3)`** — dims 1 and 3 homogeneous (dim 1 rfft), dim 2
inhomogeneous:

```julia
for _i3 in 1:size(g,3)                  # dim 3: all indices, outermost
    for _i2 in 1:size(g,2)              # dim 2: inhomogeneous
        for _i1 in 1:(size(g,1)>>1)+1   # dim 1: rfft, innermost
            f(_i1==1 ? 1 : 2, (_i2,), (_i1,_i2,_i3))
        end
    end
end
```

`inhomogeneous_indices = (_i2,)`, `indices = (_i1, _i2, _i3)`.

**`D = 3`, `FFT_DIMS_ORDER = (2, 3)`** — dims 2 and 3 homogeneous (dim 2 rfft), dim 1
inhomogeneous.  Note that dim 1 remains innermost (column-major), even though
it is inhomogeneous, and `one_or_two` is now driven by `_i2`:

```julia
for _i3 in 1:size(g,3)                  # dim 3: all indices, outermost
    for _i2 in 1:(size(g,2)>>1)+1       # dim 2: rfft
        for _i1 in 1:size(g,1)          # dim 1: inhomogeneous, innermost
            f(_i2==1 ? 1 : 2, (_i1,), (_i1,_i2,_i3))
        end
    end
end
```

`inhomogeneous_indices = (_i1,)`, `indices = (_i1, _i2, _i3)`.
"""
@generated function for_each_index(f, g::AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}) where {T, D, AXES, FFT_DIMS_ORDER}
    # One loop variable per array dimension, named _i1 … _iD.
    syms = [Symbol("_i", d) for d in 1:D]

    # Inhomogeneous dims are those not transformed by FFTs.
    inh      = Tuple(d for d in 1:D if d ∉ FFT_DIMS_ORDER)
    inh_syms = [syms[d] for d in inh]

    # FFT_DIMS_ORDER[1] is the rfft dimension: only non-negative wavenumbers are stored,
    # so its index runs 1:(N÷2)+1.  When the index equals 1 (zero wavenumber)
    # the mode has no conjugate partner → weight 1; all others → weight 2.
    rfft_dim   = FFT_DIMS_ORDER[1]
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
        elseif d ∈ FFT_DIMS_ORDER
            # Other homogeneous dims: visit every storage index in one contiguous
            # block.  Unlike for_each_homogeneous_index, we do not split positive
            # and negative wavenumbers here because every element is visited
            # regardless of sign — the caller uses `indices` directly, not the
            # wavenumber value.
            body = :(for $sym in 1:Base.size(g, $d); $body; end)
        else
            # Inhomogeneous dim: no wavenumber structure, visit every index.
            body = :(for $sym in 1:Base.size(g, $d); $body; end)
        end
    end
    return body
end
