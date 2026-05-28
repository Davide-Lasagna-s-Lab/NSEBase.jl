# Spectral differentiation operators for FTField, ProjectedField, and VectorField.
#
# Differentiation in a homogeneous (FFT-transformed) direction is exact in
# spectral space: the derivative of mode k is simply multiplied by
# ±i·n·wavenumber_scale(g, dim), where n is the signed integer wavenumber and
# the sign is +1 for the forward operator and −1 for its L2 adjoint.
#
# The core primitive is `ddx!(out, u, Val{DIM})`, implemented with
# CartesianIndices loops.  For the rfft dimension all stored wavenumbers are
# non-negative so a single loop over the full array suffices.  For signed-FFT
# dimensions the index range is split into a positive block [1:(N÷2)+1] and a
# negative block [(N÷2)+2:N] so that the signed wavenumber can be computed
# branch-free from the loop index inside each block.  The two-block
# specialisation (rfft vs signed FFT) and the ntuple range construction are
# eliminated at compile time because DIM and FFT_DIMS_ORDER are type parameters.
#
# Four named wrappers `ddx_1!`, `ddx_2!`, `ddx_3!`, `ddx_4!` forward to
# `ddx!` with the array dimension read from `AXES[1]`, `AXES[2]`, `AXES[3]`,
# `AXES[4]` respectively.  When `AXES[j] === nothing` (e.g. a steady 3D grid
# has no time dimension), the guard clause turns the call into a no-op.
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

For `AXES[DIM] === nothing` the function reduces to a no-op.

# Loop structure

For the rfft dimension all stored wavenumbers are non-negative, so a single
`CartesianIndices` pass over the full parent array is used.  For signed-FFT
dimensions the index range is split into two contiguous blocks — positive
wavenumbers `[1:(N÷2)+1]` and negative wavenumbers `[(N÷2)+2:N]` — so the
signed wavenumber is computed branch-free inside each block.

Both the rfft/signed-FFT branch and the `ntuple` range construction are
eliminated at compile time because `DIM` and `FFT_DIMS_ORDER` are type
parameters.
"""
function ddx!(out::F, u::F, ::Val{DIM};
              adjoint::Bool=false) where {
        DIM, T, D, AXES, FFT_DIMS_ORDER,
        G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER},
        F<:Union{FTField{G}, ProjectedField{G}}}

    # A missing logical coordinate, e.g. AXES[4] === nothing on a steady grid,
    # is represented by Val(nothing) and should be a no-op.
    (isnothing(DIM) || isnothing(AXES[DIM])) && return out
    # Inhomogeneous directions are not handled here; downstream packages must
    # extend ddx! with a grid-specific method (e.g. a matrix–vector multiply).
    DIM ∉ FFT_DIMS_ORDER && throw(NotImplementedError(grid(u), Val(DIM)))

    # Hoist the scale and adjoint sign outside the loop so each element update
    # is a single complex multiply with no repeated division or branching.
    scale = wavenumber_scale(grid(u), DIM)
    coeff = adjoint ? -im * T(scale) : im * T(scale)
    Nd    = size(u, DIM)
    pu    = parent(u)
    pout  = parent(out)

    if DIM == FFT_DIMS_ORDER[1]
        # rfft dimension: only non-negative wavenumbers 0 … Nd-1 are stored,
        # so the wavenumber is simply the loop index minus one.  A single
        # CartesianIndices pass over the full array suffices.
        @inbounds for I in CartesianIndices(pu)
            pout[I] = (coeff * (I[DIM] - 1)) * pu[I]
        end
    else
        # Signed-FFT dimension: split into a non-negative block [1:(Nd÷2)+1]
        # and a negative block [(Nd÷2)+2:Nd] so the signed wavenumber can be
        # computed branch-free as a simple offset inside each block.
        # DIM and FFT_DIMS_ORDER are type parameters, so the ntuple ranges and
        # the if-branch are both eliminated at compile time.
        pos_ranges = ntuple(d -> d == DIM ? (1:(Nd >> 1) + 1)  : Base.OneTo(size(pu, d)), Val(D))
        neg_ranges = ntuple(d -> d == DIM ? ((Nd >> 1) + 2:Nd) : Base.OneTo(size(pu, d)), Val(D))
        @inbounds for I in CartesianIndices(pos_ranges)
            pout[I] = (coeff * (I[DIM] - 1)) * pu[I]
        end
        @inbounds for I in CartesianIndices(neg_ranges)
            pout[I] = (coeff * (I[DIM] - Nd - 1)) * pu[I]
        end
    end
    return out
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

"""
    inhomogeneous_laplacian!(out::FTField, u::FTField; kwargs...) -> out

Apply the inhomogeneous (non-FFT) part of the Laplacian of `u` to `out`.

This is a required interface method: downstream packages must extend it for
each concrete grid type, typically as a matrix–vector multiply with a
wall-normal differentiation matrix.  NSEBase does not implement the
inhomogeneous part — it only provides the spectral complement via
[`add_homogeneous_laplacian!`](@ref).

The full Laplacian [`laplacian!`](@ref) calls this first, then accumulates
the homogeneous wavenumber contribution via [`add_homogeneous_laplacian!`](@ref).
"""
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
