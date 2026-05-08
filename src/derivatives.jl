# Generic spectral-space derivative methods for FTField.
#
# Storage convention:
#   FFT dimensions are H = (Hs..., Ht).  H[1] is the rfft dimension, so only
#   non-negative wavenumbers are stored there.  H[2:end] are ordinary fft
#   dimensions, stored in FFTW order:
#       0, 1, ..., N/2, -(N/2-1), ..., -1
#
# Performance convention:
#   Generated loops are emitted in Julia cache order: dimension 1 is the
#   innermost loop, then dimension 2, etc.  Signed fft dimensions are split
#   into positive/negative blocks so the hot scalar assignment does not call a
#   signed-wavenumber helper or branch per array element.
#
# ddx!(out, u, Val(d)) - differentiate along array dimension d.
#   d in H: handled here via wavenumber multiplication.
#       adjoint=false: multiply by +im·n·scale
#       adjoint=true:  multiply by -im·n·scale  (conjugate, i.e. -(d/dx))
#   d not in H: throws NotImplementedError; downstream packages must
#       extend for non-homogeneous directions and handle adjoint there too.
#
# ddx_x!, ddx_y!, ddx_z! - physical-direction wrappers.
#   Look up the array dim from the field's grid type via x_dim/y_dim/z_dim
#   and forward all kwargs (including adjoint=) to ddx!.
#
# add_homogeneous_laplacian!(out, u) - ADDS the homogeneous Laplacian contribution
#   -(∑_{d∈(Hs...,Ht)} (wavenumber_scale(g,d)·n_d)²) · u[mode] to an existing out.
#   The caller computes the non-homogeneous part first, then calls this.

"""
    ddx!(out::FTField, u::FTField, ::Val{Dim}; adjoint=false)

In-place spectral derivative of `u` along array dimension `Dim`, writing into `out`.

For `Dim in (Hs..., Ht)` the derivative is multiplication by
`±im * n * wavenumber_scale(grid, Dim)` where `n` is the signed wavenumber.
`adjoint=false` (default) gives `+im·n·scale·u`; `adjoint=true` gives `-im·n·scale·u`
(the L2 adjoint of the spectral derivative for homogeneous directions).

For `Dim not in (Hs..., Ht)` the method throws `NotImplementedError`. Downstream packages
should extend this for each non-homogeneous direction (e.g. matrix multiply
with a differentiation matrix) and handle the `adjoint` keyword there too.

# Generated loop shape

The generated code is arranged for memory order, not physical-coordinate order.
For a 4D spectral array `(a, b, c, d)` with `H = (2, 3, 4)`,
differentiating in the rfft dimension `b` generates the
equivalent of:

```julia
_ddx_scale = wavenumber_scale(grid(u), 2)
_ddx_sign = adjoint ? -1im : 1im
for _i4 in 1:size(u, 4), _i3 in 1:size(u, 3), _i2 in 1:size(u, 2)
    _n2 = _i2 - 1
    for _i1 in 1:size(u, 1)
        out[_i1, _i2, _i3, _i4] =
            _ddx_sign * _n2 * _ddx_scale * u[_i1, _i2, _i3, _i4]
    end
end
```

Differentiating in a signed FFT dimension, e.g. `nz`, splits that dimension
into two blocks so `_n3` is computed branch-free inside each block:

```julia
for _i4 in 1:size(u, 4)
    for _i3 in 1:(size(u, 3) >> 1) + 1
        _n3 = _i3 - 1
        for _i2 in 1:size(u, 2), _i1 in 1:size(u, 1)
            # same scalar assignment
        end
    end
    for _i3 in (size(u, 3) >> 1) + 2:size(u, 3)
        _n3 = _i3 - size(u, 3) - 1
        for _i2 in 1:size(u, 2), _i1 in 1:size(u, 1)
            # same scalar assignment
        end
    end
end
```
"""
@generated function ddx!(out::FTField{G}, u::FTField{G}, ::Val{Dim};
                         adjoint::Bool=false) where {T, D, Axes, Hs, Ht, G<:AbstractGrid{T, D, Axes, Hs, Ht}, Dim}
    H = (Hs..., Ht)
    Dim ∉ H && return :(throw(NotImplementedError(grid(u), Val($Dim))))

    syms  = [Symbol("_i", d) for d in 1:D]
    n_sym = Symbol("_n", Dim)

    # Hot scalar update.  The scale and adjoint sign are hoisted outside the
    # loops; each generated block supplies the signed wavenumber variable.
    assign = quote
        @inbounds parent(out)[$(syms...)] = _ddx_sign * $n_sym * _ddx_scale * parent(u)[$(syms...)]
    end

    # Wrap `body` in loops in increasing dimension order.  Because each wrapper
    # encloses the previous expression, dimension 1 ends up innermost, which is
    # the contiguous-memory direction for Julia arrays.
    function cache_ordered_loop(body, ranges, wavenumbers)
        for d in 1:D
            sym = syms[d]
            rng = ranges[d]
            body = if d == Dim
                :(for $sym in $rng
                      $n_sym = $(wavenumbers[d])
                      $body
                  end)
            else
                :(for $sym in $rng
                      $body
                  end)
            end
        end
        return body
    end

    ranges = [:(1:Base.size(u, $d)) for d in 1:D]
    wnums  = Any[:nothing for _ in 1:D]

    if Dim == H[1]
        # rfft dimension: only non-negative wavenumbers are stored.
        wnums[Dim] = :($(syms[Dim]) - 1)
        body = cache_ordered_loop(assign, ranges, wnums)
    else
        # ordinary fft dimension: split into positive and negative storage
        # blocks so no signed-wavenumber branch appears in the scalar loop.
        pos_ranges = copy(ranges)
        neg_ranges = copy(ranges)
        pos_ranges[Dim] = :(1:(Base.size(u, $Dim) >> 1) + 1)
        neg_ranges[Dim] = :((Base.size(u, $Dim) >> 1) + 2:Base.size(u, $Dim))
        wnums[Dim] = :($(syms[Dim]) - 1)
        pos_body = cache_ordered_loop(assign, pos_ranges, wnums)
        wnums[Dim] = :($(syms[Dim]) - Base.size(u, $Dim) - 1)
        neg_body = cache_ordered_loop(assign, neg_ranges, wnums)
        body = quote
            $pos_body
            $neg_body
        end
    end

    return Base.remove_linenums!(quote
        _ddx_scale = wavenumber_scale(grid(u), $Dim)
        _ddx_sign  = adjoint ? -1im : 1im
        $body
        return out
    end)
end

"""
    add_homogeneous_laplacian!(out::FTField, u::FTField)

Add the homogeneous Laplacian contribution of `u` to `out`:

    out[mode] -= (∑_{d∈(Hs...,Ht)} (wavenumber_scale(g, d) · n_d)²) · u[mode]

Call after computing the non-homogeneous (e.g. wall-normal) second derivative.

# Generated loop shape

For a 4D spectral array `(a, b, c, d)` with `H = (2, 3, 4)`,
the generated code computes:

```julia
k2 = (kx_scale * nx)^2 + (kz_scale * nz)^2 + (kt_scale * nt)^2
out[ny, nx_index, nz_index, nt_index] -= k2 * u[ny, nx_index, nz_index, nt_index]
```

The rfft dimension `nx` is looped once.  Each signed FFT dimension is split
into positive and negative blocks, so the full channel case emits four
`(nz block, nt block)` combinations:

```julia
nz >= 0, nt >= 0
nz <  0, nt >= 0
nz >= 0, nt <  0
nz <  0, nt <  0
```

Inside every block, dimension 1 remains the innermost loop and the signed
wavenumbers are plain loop-local integers.
"""
@generated function add_homogeneous_laplacian!(out::FTField{G}, u::FTField{G}) where {T, D, Axes, Hs, Ht, G<:AbstractGrid{T, D, Axes, Hs, Ht}}
    H    = (Hs..., Ht)
    syms = [Symbol("_i", d) for d in 1:D]

    # Build the symbolic k² expression from per-dimension scale and wavenumber
    # variables.  The loops below define `_n<dim>` for each dimension in H.
    k2_terms = Expr[]
    for d in H
        n_expr = Symbol("_n", d)
        scale = Symbol("_k_scale", d)
        push!(k2_terms, :(($scale * $n_expr)^2))
    end
    k2_expr = isempty(k2_terms) ? :(zero($T))          :
              length(k2_terms) == 1 ? k2_terms[1]       :
              Expr(:call, :+, k2_terms...)

    body = quote
        @inbounds parent(out)[$(syms...)] -= $k2_expr * parent(u)[$(syms...)]
    end

    signed_dims = H[2:end]
    blocks = Expr[]
    for mask in 0:(1 << length(signed_dims)) - 1
        # One generated block for each sign combination of H[2:end].
        # The rfft dimension H[1] is not split because it stores only n >= 0.
        ranges = [:(1:Base.size(u, $d)) for d in 1:D]
        wnums  = Any[:nothing for _ in 1:D]
        wnums[H[1]] = :($(syms[H[1]]) - 1)

        for (j, d) in enumerate(signed_dims)
            if Bool((mask >> (j - 1)) & 1)
                ranges[d] = :((Base.size(u, $d) >> 1) + 2:Base.size(u, $d))
                wnums[d] = :($(syms[d]) - Base.size(u, $d) - 1)
            else
                ranges[d] = :(1:(Base.size(u, $d) >> 1) + 1)
                wnums[d] = :($(syms[d]) - 1)
            end
        end

        block = body
        for d in 1:D
            # As in ddx!, increasing wrapper order makes dimension 1 innermost.
            sym = syms[d]
            rng = ranges[d]
            if d in H
                n_sym = Symbol("_n", d)
                block = :(for $sym in $rng
                              $n_sym = $(wnums[d])
                              $block
                          end)
            else
                block = :(for $sym in $rng
                              $block
                          end)
            end
        end
        push!(blocks, block)
    end

    return Base.remove_linenums!(quote
        $([:($(Symbol("_k_scale", d)) = wavenumber_scale(grid(u), $d)) for d in H]...)
        $(blocks...)
        return out
    end)
end

# -------------------------------------------- #
# Physical-direction wrappers                  #
# -------------------------------------------- #
# These look up the array dim from the field's grid type (via x_dim/y_dim/z_dim)
# and forward all kwargs to ddx!, so callers need not know array dim indices.

ddx_x!(out::FTField, u::FTField; kwargs...) = ddx!(out, u, Val(x_dim(grid(u))); kwargs...)
ddx_y!(out::FTField, u::FTField; kwargs...) = ddx!(out, u, Val(y_dim(grid(u))); kwargs...)
ddx_z!(out::FTField, u::FTField; kwargs...) = ddx!(out, u, Val(z_dim(grid(u))); kwargs...)

# VectorField variants
ddx_x!(out::VectorField{N}, u::VectorField{N}; kwargs...) where {N} =
    (for n in 1:N; ddx_x!(out[n], u[n]; kwargs...); end; return out)
ddx_y!(out::VectorField{N}, u::VectorField{N}; kwargs...) where {N} =
    (for n in 1:N; ddx_y!(out[n], u[n]; kwargs...); end; return out)
ddx_z!(out::VectorField{N}, u::VectorField{N}; kwargs...) where {N} =
    (for n in 1:N; ddx_z!(out[n], u[n]; kwargs...); end; return out)

# VectorField variants for array-dim and homogeneous Laplacian
function ddx!(out::VectorField{N}, u::VectorField{N}, d; kwargs...) where {N}
    for n in 1:N; ddx!(out[n], u[n], d; kwargs...); end
    return out
end

function add_homogeneous_laplacian!(out::VectorField{N}, u::VectorField{N}) where {N}
    for n in 1:N; add_homogeneous_laplacian!(out[n], u[n]); end
    return out
end

# laplacian! — full Laplacian (homogeneous + non-homogeneous parts).
# Downstream packages must extend this for their grid type (the non-homogeneous
# part is not knowable generically); add_homogeneous_laplacian! handles the rest.
laplacian!(out::FTField, u::FTField; kwargs...) = throw(NotImplementedError(out, u))
laplacian!(out::VectorField{N}, u::VectorField{N}; kwargs...) where {N} =
    (for n in 1:N; laplacian!(out[n], u[n]; kwargs...); end; return out)
