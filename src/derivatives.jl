# Generic derivative methods for each dimension of an FTField

"""
    ddx!(out::FTField, u::FTField, ::Val{Dim}; adjoint=false)
    ddx!(out::VectorField, u::VectorField, ::Val{Dim}; adjoint=false)

In-place spectral derivative of `u` along array dimension `Dim`, writing into `out`.

For `Dim in ORDER` the derivative is multiplication by
`±im * n * wavenumber_scale(grid, Dim)` where `n` is the signed wavenumber.
`adjoint=false` (default) gives `+im·n·scale·u`; `adjoint=true` gives `-im·n·scale·u`
(the L2 adjoint of the spectral derivative for homogeneous directions).

For `Dim not in ORDER` the method throws `NotImplementedError`. Downstream packages
should extend this for each non-homogeneous direction (e.g. matrix multiply
with a differentiation matrix) and handle the `adjoint` keyword there too.

For `AXES[Dim] === nothing` the function reduces to identity.

# Generated loop shape

The generated code is arranged for memory order, not physical-coordinate order.
For a 4D spectral array `(a, b, c, d)` with `ORDER = (2, 3, 4)`,
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
                         adjoint::Bool=false) where {T, D, AXES, ORDER, G<:AbstractGrid{T, D, AXES, ORDER}, Dim}
    (isnothing(Dim) || isnothing(AXES[Dim])) && return :(return out)
    Dim ∉ ORDER && return :(throw(NotImplementedError(grid(u), Val($Dim))))

    syms  = [Symbol("_i", d) for d in 1:D]
    n_sym = Symbol("_n", Dim)

    # Hot scalar update. The scale and adjoint sign are hoisted outside the
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

    if Dim == ORDER[1]
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

    return quote
        _ddx_scale = wavenumber_scale(grid(u), $Dim)
        _ddx_sign  = adjoint ? -1im : 1im
        $body
        return out
    end
end
ddx!(out::VectorField{N}, u::VectorField{N}, d; kwargs...) where {N} =
    (for n in 1:N; ddx!(out[n], u[n], d; kwargs...); end; return out)

# These look up the array dim from the field's grid type (via x_dim/y_dim/z_dim)
# and forward all kwargs to ddx!, so callers need not know array dim indices.
ddx_x!(out::FTField, u::FTField; kwargs...) = ddx!(out, u, Val(x_dim(grid(u))); kwargs...)
ddx_x!(out::VectorField{N}, u::VectorField{N}; kwargs...) where {N} =
    (for n in 1:N; ddx_x!(out[n], u[n]; kwargs...); end; return out)

ddx_y!(out::FTField, u::FTField; kwargs...) = ddx!(out, u, Val(y_dim(grid(u))); kwargs...)
ddx_y!(out::VectorField{N}, u::VectorField{N}; kwargs...) where {N} =
    (for n in 1:N; ddx_y!(out[n], u[n]; kwargs...); end; return out)

ddx_z!(out::FTField, u::FTField; kwargs...) = ddx!(out, u, Val(z_dim(grid(u))); kwargs...)
ddx_z!(out::VectorField{N}, u::VectorField{N}; kwargs...) where {N} =
    (for n in 1:N; ddx_z!(out[n], u[n]; kwargs...); end; return out)

# TODO: add time derivative here

"""
    add_homogeneous_laplacian!(out::FTField, u::FTField)
    add_homogeneous_laplacian!(out::VectorField, u::VectorField)

Add the homogeneous Laplacian contribution of `u` to `out`:

    out[mode] -= (∑_{d∈ORDER} (wavenumber_scale(g, d) · n_d)²) · u[mode]

Call after computing the non-homogeneous (e.g. wall-normal) second derivative.

# Generated loop shape

For a 4D spectral array `(a, b, c, d)` with `ORDER = (2, 3, 4)`,
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
@generated function add_homogeneous_laplacian!(out::FTField{G}, u::FTField{G}) where {T, D, AXES, ORDER, G<:AbstractGrid{T, D, AXES, ORDER}}
    syms = [Symbol("_i", d) for d in 1:D]

    # Build the symbolic k² expression from per-dimension scale and wavenumber
    # variables.  The loops below define `_n<dim>` for each dimension in ORDER.
    k2_terms = Expr[]
    for d in ORDER
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

    signed_dims = ORDER[2:end]
    blocks = Expr[]
    for mask in 0:(1 << length(signed_dims)) - 1
        # One generated block for each sign combination of ORDER[2:end].
        # The rfft dimension ORDER[1] is not split because it stores only n >= 0.
        ranges = [:(1:Base.size(u, $d)) for d in 1:D]
        wnums  = Any[:nothing for _ in 1:D]
        wnums[ORDER[1]] = :($(syms[ORDER[1]]) - 1)

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
            if d in ORDER
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

    return quote
        $([:($(Symbol("_k_scale", d)) = wavenumber_scale(grid(u), $d)) for d in ORDER]...)
        $(blocks...)
        return out
    end
end
add_homogeneous_laplacian!(out::VectorField{N}, u::VectorField{N}) where {N} =
    (for n in 1:N; add_homogeneous_laplacian!(out[n], u[n]); end; return out)

# Downstream packages must extend this for their grid type (the inhomogeneous
# part is not knowable generically); add_homogeneous_laplacian! handles the rest.
inhomogeneous_laplacian!(out::FTField, u::FTField) = throw(NotImplementedError(out, u))

"""
    laplacian!(out::FTField{G}, u::FTField{G}; kwargs...)
    laplacian!(out::VectorField{N}, u::VectorField{N}; kwargs...)

Compute the full Laplacian of `u` in-place, storing the result in `out`:

    out[mode] = (∂²/∂y² + ∑_{d∈ORDER} (wavenumber_scale(g, d) · n_d)²) · u[mode]

Combines the inhomogeneous (e.g. wall-normal) second derivative with the
homogeneous (spectral) Laplacian by calling, in order:

    inhomogeneous_laplacian!(out, u; kwargs...)
    add_homogeneous_laplacian!(out, u)

`kwargs` are forwarded to `inhomogeneous_laplacian!` only (e.g. boundary
condition parameters).  The `VectorField` method applies the scalar method
independently to each component `n ∈ 1:N`.

# Example

For a 4D spectral array `(t, x, z, y)` with `ORDER = (1, 2, 3)` the
combined operation is equivalent to:

```julia
# inhomogeneous_laplacian! fills out with the wall-normal contribution:
out[:, nx_index, nz_index, nt_index] = D2 * u[:, nx_index, nz_index, nt_index]

# add_homogeneous_laplacian! then accumulates the spectral part:
k2 = (kx_scale * nx)^2 + (kz_scale * nz)^2 + (kt_scale * nt)^2
out[:, nx_index, nz_index, nt_index] -= k2 * u[:, nx_index, nz_index, nt_index]
```

See [`inhomogeneous_laplacian!`](@ref) and [`add_homogeneous_laplacian!`](@ref)
for the loop s
"""
function laplacian!(out::FTField{G}, u::FTField{G}; kwargs...) where {G}
    inhomogeneous_laplacian!(out, u; kwargs...)
    add_homogeneous_laplacian!(out, u)
    return out
end
laplacian!(out::VectorField{N}, u::VectorField{N}; kwargs...) where {N} =
    (for n in 1:N; laplacian!(out[n], u[n]; kwargs...); end; return out)
