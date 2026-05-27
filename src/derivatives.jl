# Spectral differentiation operators for FTField, ProjectedField, and VectorField.
#
# Differentiation in a homogeneous (FFT-transformed) direction is exact in
# spectral space: the derivative of mode k is simply multiplied by
# ±i·n·wavenumber_scale(g, dim), where n is the signed integer wavenumber and
# the sign is +1 for the forward operator and −1 for its L2 adjoint.
#
# The core primitive is `ddx!(out, u, Val{DIM})`, a `@generated` function that
# emits a fully unrolled loop nest specialised to the grid type.  The generated
# code splits signed-FFT dimensions into two contiguous blocks (positive and
# negative wavenumbers) so that the signed wavenumber value can be computed
# branch-free from the loop variable, and processes dimensions in ascending
# order so that dimension 1 (the column-major stride-1 axis) is always the
# innermost loop.
#
# Four named wrappers `ddx_1!`, `ddx_2!`, `ddx_3!`, `ddx_4!` forward to
# `ddx!` with the array dimension read from `AXES[1]`, `AXES[2]`, `AXES[3]`,
# `AXES[4]` respectively.  When `AXES[j] === nothing` (e.g. a steady 3D grid
# has no time dimension), the generated code is a compile-time no-op.
#
# Inhomogeneous (non-FFT) directions are NOT handled here — `ddx!` throws
# `NotImplementedError` for those, and downstream packages must extend it with
# a grid-specific method (typically a matrix–vector multiply).
#
# The full Laplacian combines `inhomogeneous_laplacian!` (provided by
# downstream) with `add_homogeneous_laplacian!` (provided here), which
# subtracts ‖k‖² · u from each spectral coefficient.

"""
    ddx!(out::FTField, u::FTField, ::Val{DIM}; adjoint=false)
    ddx!(out::VectorField, u::VectorField, ::Val{DIM}; adjoint=false)

In-place spectral derivative of `u` along array dimension `DIM`, writing into `out`.

For `DIM in FFT_DIMS_ORDER` the derivative is multiplication by
`±im * n * wavenumber_scale(grid, DIM)` where `n` is the signed wavenumber.
`adjoint=false` (default) gives `+im·n·scale·u`; `adjoint=true` gives `-im·n·scale·u`
(the L2 adjoint of the spectral derivative for homogeneous directions).

For `DIM not in FFT_DIMS_ORDER` the method throws `NotImplementedError`. Downstream packages
should extend this for each non-homogeneous direction (e.g. matrix multiply
with a differentiation matrix) and handle the `adjoint` keyword there too.

For `AXES[DIM] === nothing` the function reduces to identity.

# Generated loop shape

The generated code is arranged for memory order, not physical-coordinate order.
For a 4D spectral array `(a, b, c, d)` with `FFT_DIMS_ORDER = (2, 3, 4)`,
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
@generated function ddx!(out::F,
                           u::F,
                            ::Val{DIM};
                     adjoint::Bool=false) where {T, D, AXES, FFT_DIMS_ORDER, G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}, F<:Union{FTField{G}, ProjectedField{G}}, DIM}
    # A missing logical coordinate, e.g. AXES[4] === nothing on a steady grid,
    # is represented by Val(nothing) and should be a no-op.
    (isnothing(DIM) || isnothing(AXES[DIM])) && return :(return out)
    DIM ∉ FFT_DIMS_ORDER && return :(throw(NotImplementedError(grid(u), Val($DIM))))

    syms  = [Symbol("_i", d) for d in 1:D]
    n_sym = Symbol("_n", DIM)

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
            body = if d == DIM
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

    if DIM == FFT_DIMS_ORDER[1]
        # rfft dimension: only non-negative wavenumbers are stored.
        wnums[DIM] = :($(syms[DIM]) - 1)
        body = cache_ordered_loop(assign, ranges, wnums)
    else
        # ordinary fft dimension: split into positive and negative storage
        # blocks so no signed-wavenumber branch appears in the scalar loop.
        pos_ranges = copy(ranges)
        neg_ranges = copy(ranges)
        pos_ranges[DIM] = :(1:(Base.size(u, $DIM) >> 1) + 1)
        neg_ranges[DIM] = :((Base.size(u, $DIM) >> 1) + 2:Base.size(u, $DIM))
        wnums[DIM] = :($(syms[DIM]) - 1)
        pos_body = cache_ordered_loop(assign, pos_ranges, wnums)
        wnums[DIM] = :($(syms[DIM]) - Base.size(u, $DIM) - 1)
        neg_body = cache_ordered_loop(assign, neg_ranges, wnums)
        body = quote
            $pos_body
            $neg_body
        end
    end

    return quote
        _ddx_scale = wavenumber_scale(grid(u), $DIM)
        _ddx_sign  = adjoint ? -1im : 1im
        $body
        return out
    end
end
ddx!(out::VectorField{N}, u::VectorField{N}, d; kwargs...) where {N} =
    (for n in 1:N; ddx!(out[n], u[n], d; kwargs...); end; return out)

"""
    ddx_1!(out, u; adjoint=false) -> out

Differentiate `u` in the first physical coordinate (`x`), storing
the result in `out`.  The array dimension is read from `AXES[1]` of the grid
type parameter at compile time and forwarded to [`ddx!`](@ref).

Defined for `FTField`, `ProjectedField`, and `VectorField` arguments on any
`AbstractGrid`.
"""
ddx_1!(out::ProjectedField{G}, a::ProjectedField{G}; kwargs...) where {AXES, D, G<:AbstractGrid{<:Any, D, AXES}} =
    ddx!(out, a, Val(AXES[1]); kwargs...)
ddx_1!(out::FTField{G}, u::FTField{G}; kwargs...) where {AXES, D, G<:AbstractGrid{<:Any, D, AXES}} =
    ddx!(out, u, Val(AXES[1]); kwargs...)
ddx_1!(out::VectorField{N}, u::VectorField{N}; kwargs...) where {N} =
    (for n in 1:N; ddx_1!(out[n], u[n]; kwargs...); end; return out)

"""
    ddx_2!(out, u; adjoint=false) -> out

Differentiate `u` in the second physical coordinate (`y`, or `r`), storing
the result in `out`.  The array dimension is read from `AXES[2]` of the grid
type parameter at compile time and forwarded to [`ddx!`](@ref).

For inhomogeneous (non-FFT) directions this dispatches to the downstream
package's extension of `ddx!` (e.g. a matrix–vector product with the
wall-normal differentiation matrix).
"""
ddx_2!(out::ProjectedField{G}, a::ProjectedField{G}; kwargs...) where {AXES, D, G<:AbstractGrid{<:Any, D, AXES}} =
    ddx!(out, a, Val(AXES[2]); kwargs...)
ddx_2!(out::FTField{G}, u::FTField{G}; kwargs...) where {AXES, D, G<:AbstractGrid{<:Any, D, AXES}} =
    ddx!(out, u, Val(AXES[2]); kwargs...)
ddx_2!(out::VectorField{N}, u::VectorField{N}; kwargs...) where {N} =
    (for n in 1:N; ddx_2!(out[n], u[n]; kwargs...); end; return out)

"""
    ddx_3!(out, u; adjoint=false) -> out

Differentiate `u` in the third physical coordinate (`z`, or `theta`), storing the
result in `out`.  The array dimension is read from `AXES[3]` of the grid type
parameter at compile time and forwarded to [`ddx!`](@ref).

For 2D grids `AXES[3] === nothing`, so this call is a compile-time no-op that
leaves `out` unchanged.
"""
ddx_3!(out::ProjectedField{G}, a::ProjectedField{G}; kwargs...) where {AXES, D, G<:AbstractGrid{<:Any, D, AXES}} =
    ddx!(out, a, Val(AXES[3]); kwargs...)
ddx_3!(out::FTField{G}, u::FTField{G}; kwargs...) where {AXES, D, G<:AbstractGrid{<:Any, D, AXES}} =
    ddx!(out, u, Val(AXES[3]); kwargs...)
ddx_3!(out::VectorField{N}, u::VectorField{N}; kwargs...) where {N} =
    (for n in 1:N; ddx_3!(out[n], u[n]; kwargs...); end; return out)

"""
    ddx_4!(out, u; adjoint=false) -> out

Differentiate `u` in the fourth physical coordinate (`t`), storing
the result in `out`.  The array dimension is read from `AXES[4]` of the grid
type parameter at compile time and forwarded to [`ddx!`](@ref).
"""
ddx_4!(out::ProjectedField{G}, a::ProjectedField{G}; kwargs...) where {AXES, D, G<:AbstractGrid{<:Any, D, AXES}} =
    ddx!(out, a, Val(AXES[4]); kwargs...)
ddx_4!(out::FTField{G}, u::FTField{G}; kwargs...) where {AXES, D, G<:AbstractGrid{<:Any, D, AXES}} =
    ddx!(out, u, Val(AXES[4]); kwargs...)
ddx_4!(out::VectorField{N}, u::VectorField{N}; kwargs...) where {N} =
    (for n in 1:N; ddx_4!(out[n], u[n]; kwargs...); end; return out)

"""
    add_homogeneous_laplacian!(out::FTField, u::FTField)
    add_homogeneous_laplacian!(out::VectorField, u::VectorField)

Add the homogeneous Laplacian contribution of `u` to `out`:

    out[mode] -= (∑_{d∈FFT_DIMS_ORDER} (wavenumber_scale(g, d) · n_d)²) · u[mode]

Call after computing the non-homogeneous (e.g. wall-normal) second derivative.

# Generated loop shape

For a 4D spectral array `(a, b, c, d)` with `FFT_DIMS_ORDER = (2, 3, 4)`,
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
@generated function add_homogeneous_laplacian!(out::FTField{G}, u::FTField{G}) where {T, D, AXES, FFT_DIMS_ORDER, G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}}
    # The Laplacian includes spatial homogeneous directions only.  If the
    # fourth logical coordinate is time, AXES[4] is filtered out here.
    H = filter(d->d!=AXES[4], FFT_DIMS_ORDER)
    syms = [Symbol("_i", d) for d in 1:D]

    # Build the symbolic k² expression from per-dimension scale and wavenumber
    # variables.  The loops below define `_n<dim>` for each dimension in FFT_DIMS_ORDER.
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
        # One generated block for each sign combination of FFT_DIMS_ORDER[2:end].
        # The rfft dimension FFT_DIMS_ORDER[1] is not split because it stores only n >= 0.
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
add_homogeneous_laplacian!(out::VectorField{N}, u::VectorField{N}) where {N} =
    (for n in 1:N; add_homogeneous_laplacian!(out[n], u[n]); end; return out)

# Downstream packages must extend this for their grid type (the inhomogeneous
# part is not knowable generically); add_homogeneous_laplacian! handles the rest.
inhomogeneous_laplacian!(out::FTField, u::FTField) = throw(NotImplementedError(out, u))

"""
    laplacian!(out::FTField{G}, u::FTField{G}; kwargs...)
    laplacian!(out::VectorField{N}, u::VectorField{N}; kwargs...)

Compute the full Laplacian of `u` in-place, storing the result in `out`:

    out[mode] = (inhomogeneous second derivatives
                 - ∑_{d∈FFT_DIMS_ORDER} (wavenumber_scale(g, d) · n_d)²) · u[mode]

Combines the grid-specific non-FFT contribution with the homogeneous
(spectral) Laplacian by calling, in order:

    inhomogeneous_laplacian!(out, u; kwargs...)
    add_homogeneous_laplacian!(out, u)

`kwargs` are forwarded to `inhomogeneous_laplacian!` only (e.g. boundary
condition parameters).  The `VectorField` method applies the scalar method
independently to each component `n ∈ 1:N`.

# Example

For a 4D spectral array stored as `(y, x, z, t)` with
`FFT_DIMS_ORDER = (2, 3, 4)` and one inhomogeneous dimension `y`, the combined
operation is equivalent to:

```julia
# inhomogeneous_laplacian! fills out with the grid-specific contribution:
out[:, nx_index, nz_index, nt_index] = D2 * u[:, nx_index, nz_index, nt_index]

# add_homogeneous_laplacian! then accumulates the spectral part:
k2 = (kx_scale * nx)^2 + (kz_scale * nz)^2 + (kt_scale * nt)^2
out[:, nx_index, nz_index, nt_index] -= k2 * u[:, nx_index, nz_index, nt_index]
```

See [`inhomogeneous_laplacian!`](@ref) and [`add_homogeneous_laplacian!`](@ref)
for the two pieces of the operation.
"""
function laplacian!(out::FTField{G}, u::FTField{G}; kwargs...) where {G}
    inhomogeneous_laplacian!(out, u; kwargs...)
    add_homogeneous_laplacian!(out, u)
    return out
end
laplacian!(out::VectorField{N}, u::VectorField{N}; kwargs...) where {N} =
    (for n in 1:N; laplacian!(out[n], u[n]; kwargs...); end; return out)
