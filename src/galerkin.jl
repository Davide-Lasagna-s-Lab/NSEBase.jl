# Galerkin projection and reconstruction for reduced-order spectral bases.
#
# `project!(a, u)` computes the L2 inner product of each basis mode with the
# spectral velocity field `u` and writes the resulting modal amplitudes into `a`.
# `expand!(u, a)` reverses the operation: it reconstructs `u` from the modal
# amplitudes in `a` by summing weighted basis modes at each wavenumber.
#
# Two algorithm implementations are provided and selected via a tag argument:
#
# ------------------------------------------------------------------ #
# Required mode-array layout                                         #
# ------------------------------------------------------------------ #
#
# For a grid with `D` dimensions, `length(FFT_DIMS_ORDER)` homogeneous (FFT) dimensions
# and `Ninh = D - length(FFT_DIMS_ORDER)` inhomogeneous dimensions, the mode tuple
# `modes(a)` must be an `NTuple{N, AbstractArray}` (one array per velocity
# component) where each `modes(a)[n]` has **exactly** the shape
#
#   `(inh_sz..., Nm, kH_sz...)`
#
# with axes laid out in this order, listed innermost (axis 1) to outermost:
#
#   axes 1 … Ninh                : inhomogeneous-dimension sizes, in
#                                  ascending grid-dimension order
#                                  (same order returned by
#                                  `inhomogeneous_dims(grid)`)
#   axis  Ninh + 1                : mode index, 1:Nm
#   axes Ninh + 2 … Ninh + 1 + Nhom : homogeneous-dimension sizes, listed
#                                  in `FFT_DIMS_ORDER` order — i.e. the **rfft**
#                                  axis `FFT_DIMS_ORDER[1]` is first and runs over
#                                  the half-spectrum `1 : (size÷2) + 1`,
#                                  followed by the signed-FFT axes
#                                  `FFT_DIMS_ORDER[2:end]` in full storage order.
#
# Example — ChannelFlow-style grid with `D = 4`, `AXES = (2,1,3,4)`,
# `FFT_DIMS_ORDER = (2,3,4)` (so `inh_dims = (1,)` is the wall-normal direction,
# rfft on x, signed FFTs on z and t):
#
#   `modes(a)[n] :: Array{ComplexF64, 5}`,
#   `size(modes(a)[n]) == (Ny, Nm, (Nx ÷ 2) + 1, Nz, Nt)`.
#
# Rationale for this layout:
#
#   - axis 1 = inhomogeneous matches `parent(u[n])`'s leading axis under the
#     ChannelFlow `AXES` convention, so column-major traversal is shared
#     between `pu` and `modes_n` in the inner inh-loop.
#   - axis Ninh + 1 = mode index is contiguous in memory across `m`, which
#     makes the innermost m-loop a unit-stride pass through `modes_n`.
#   - placing the homogeneous axes last lets `GemmGalerkin` reshape modes to
#     `(Ninh, Nm, NkH)` with a single zero-cost `reshape`.
#
# Any other layout will either fail with a dimension mismatch, give incorrect
# results, or pessimise cache behaviour in the inner loops.
#
# ------------------------------------------------------------------ #
# Algorithms                                                         #
# ------------------------------------------------------------------ #
#
#   `LoopGalerkin()` — fully unrolled nested loops over every grid dimension,
#                      emitted by a `@generated` function.  No scratch
#                      allocation.  Preferred when `Nm` is small.
#
#   `GemmGalerkin()` — flattens all homogeneous wavenumbers into one column
#                      dimension and accumulates the contraction with
#                      broadcasted strided slices, hitting every wavenumber
#                      simultaneously.  Preferred when `Nm` is large.
#
# Both implementations hoist the velocity-component loop outside the spectral
# loop so each wavenumber is touched once per component.  The default is
# `LoopGalerkin()`.

"""
    LoopGalerkin()

Algorithm selector for [`project!`](@ref) and [`expand!`](@ref).

Emits a fully unrolled nested loop over every grid dimension via a
`@generated` function — no closures, no `CartesianIndices`, no index-tuple
construction at runtime.  Allocates no scratch storage.  Preferred when the
number of basis modes `Nm` is small (typical wall-bounded Galerkin bases).
"""
struct LoopGalerkin end

"""
    GemmGalerkin()

Algorithm selector for [`project!`](@ref) and [`expand!`](@ref).

Flattens all homogeneous (FFT) wavenumbers into a single column dimension and
accumulates the contraction with broadcasted strided slices, hitting every
wavenumber simultaneously.  Preferred when `Nm` is large.

Requires `parent(u[n])` to lay out the inhomogeneous dimensions first, so
that `reshape(parent(u[n]), Ninh, :)` is a contiguous `(Ninh, NkH)` matrix.
"""
struct GemmGalerkin end


# ------------------------------------------------------------------ #
# Generated inner-loop kernels (LoopGalerkin)                         #
# ------------------------------------------------------------------ #
# Each `@generated` function emits a fully unrolled loop nest at compile
# time, specialised to the grid type parameters `D`, `AXES`, `FFT_DIMS_ORDER`.  At
# runtime there are no closures, no `CartesianIndices`, no `view`s, and no
# index-tuple construction — every array access is a direct N-D indexing
# expression using local loop variables.
#
# Loop order (outermost → innermost):
#
#   n (velocity component)
#       └─ FFT_DIMS_ORDER[end] (signed-FFT, outermost homogeneous)
#               └─ … → FFT_DIMS_ORDER[1] (rfft, innermost homogeneous)
#                       └─ inh_dims[end] → … → inh_dims[1] (innermost inh)
#                               └─ m (mode index, innermost)
#
# Rationale:
#   - `m` innermost matches `pa`'s leading axis (column-major friendly).
#   - The lowest-indexed inhomogeneous dim is next, matching `pu`'s leading
#     axis and `w`'s only axis.
#   - The rfft dim runs over its half-spectrum `1:(N÷2)+1`; the other
#     homogeneous dims run over their full storage range `1:size(g, d)`.
#   - Component pointers (`pu`, `modes_n`) are extracted once per `n`,
#     outside every spatial loop.

@generated function _loop_project!(pa,
                                   u::VectorField{N},
                                   a::ProjectedField,
                                   w,
                                   g::AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}
                                   ) where {N, T, D, AXES, FFT_DIMS_ORDER}
    Nhom     = length(FFT_DIMS_ORDER)
    inh_dims = [d for d in 1:D if d ∉ FFT_DIMS_ORDER]

    syms = [Symbol("_i", d) for d in 1:D]

    # Direct N-D index expressions — no tuple splatting, no views.
    pa_ref = Expr(:ref, :pa,      :m, (syms[d] for d in FFT_DIMS_ORDER)...)
    pu_ref = Expr(:ref, :pu,      (syms[d] for d in 1:D)...)
    w_ref  = Expr(:ref, :w,       (syms[d] for d in inh_dims)...)
    mn_ref = Expr(:ref, :modes_n,
                  (syms[d] for d in inh_dims)...,
                  :m,
                  (syms[d] for d in FFT_DIMS_ORDER)...)

    # Innermost loop: accumulate one (m, k) coefficient of pa.
    body = Expr(:for, Expr(:(=), :m, :(1:Nm)),
                :($pa_ref += conj($mn_ref) * wuj))

    # Hoist the quadrature-weighted velocity value `wuj` out of the m loop.
    body = Expr(:block, :(wuj = $w_ref * $pu_ref), body)

    # Inhomogeneous-dimension loops (forward order ⇒ smallest dim innermost).
    for d in inh_dims
        body = Expr(:for, Expr(:(=), syms[d], :(1:Base.size(g, $d))), body)
    end

    # Homogeneous-dimension loops, FFT_DIMS_ORDER forward.  FFT_DIMS_ORDER[1] is the rfft
    # dimension and runs over its half-spectrum; the rest run over the
    # full storage range.
    for k in 1:Nhom
        d   = FFT_DIMS_ORDER[k]
        rng = k == 1 ? :(1:(Base.size(g, $d) >> 1) + 1) : :(1:Base.size(g, $d))
        body = Expr(:for, Expr(:(=), syms[d], rng), body)
    end

    # Extract per-component pointers once per n, outside all spatial loops.
    body = Expr(:block,
                :(pu      = parent(u[n])),
                :(modes_n = modes(a)[n]),
                body)

    # Outermost loop: velocity components.
    body = Expr(:for, Expr(:(=), :n, :(1:$N)), body)

    return quote
        Nm = Base.size(pa, 1)
        @inbounds $body
    end
end

@generated function _loop_expand!(u::VectorField{N},
                                  pa,
                                  a::ProjectedField,
                                  g::AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}
                                  ) where {N, T, D, AXES, FFT_DIMS_ORDER}
    Nhom     = length(FFT_DIMS_ORDER)
    inh_dims = [d for d in 1:D if d ∉ FFT_DIMS_ORDER]

    syms = [Symbol("_i", d) for d in 1:D]

    pa_ref = Expr(:ref, :pa,      :m, (syms[d] for d in FFT_DIMS_ORDER)...)
    pu_ref = Expr(:ref, :pu,      (syms[d] for d in 1:D)...)
    mn_ref = Expr(:ref, :modes_n,
                  (syms[d] for d in inh_dims)...,
                  :m,
                  (syms[d] for d in FFT_DIMS_ORDER)...)

    # Innermost loop: accumulate the mode sum into a scalar.
    body = Expr(:for, Expr(:(=), :m, :(1:Nm)),
                :(acc += $mn_ref * $pa_ref))

    # Initialise the accumulator, then scatter the result to pu.
    body = Expr(:block,
                :(acc = zero(eltype(pa))),
                body,
                :($pu_ref = acc))

    for d in inh_dims
        body = Expr(:for, Expr(:(=), syms[d], :(1:Base.size(g, $d))), body)
    end

    for k in 1:Nhom
        d   = FFT_DIMS_ORDER[k]
        rng = k == 1 ? :(1:(Base.size(g, $d) >> 1) + 1) : :(1:Base.size(g, $d))
        body = Expr(:for, Expr(:(=), syms[d], rng), body)
    end

    body = Expr(:block,
                :(pu      = parent(u[n])),
                :(modes_n = modes(a)[n]),
                body)

    body = Expr(:for, Expr(:(=), :n, :(1:$N)), body)

    return quote
        Nm = Base.size(pa, 1)
        @inbounds $body
    end
end


# ------------------------------------------------------------------ #
# project!                                                            #
# ------------------------------------------------------------------ #

raw"""
    project!(a, u, alg=LoopGalerkin()) -> a

Project the spectral vector field `u` onto the basis stored by `a` and write
the modal coefficients into `a`:

```math
a_{m,\mathbf{k}} = \sum_n \sum_\mathbf{j} w_\mathbf{j}
    \overline{\phi_{n,m,\mathbf{j},\mathbf{k}}}\, u_{n,\mathbf{j},\mathbf{k}}
```

where `φ_{n,m,j,k} = modes(a)[n][j..., m, k...]` and `w` is
[`weights`](@ref)`(grid(u))`.

Pass [`LoopGalerkin`](@ref) (default) or [`GemmGalerkin`](@ref) to select the
implementation.  See also [`project`](@ref) for the allocating form.
"""
function project!(a::ProjectedField{G},
                  u::VectorField{N, <:FTField{G}},
                  ::LoopGalerkin=LoopGalerkin()
                  ) where {N, T, D, AXES, FFT_DIMS_ORDER, G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}}
    a .= zero(Complex{T})
    _loop_project!(parent(a), u, a, weights(grid(u)), grid(u))
    return a
end

"""
    project(u::VectorField, modes, alg=LoopGalerkin()) -> ProjectedField

Allocate a `ProjectedField` over `modes` and project `u` onto it.
See [`project!`](@ref) for the in-place form.
"""
project(u::VectorField, modes, alg=LoopGalerkin()) =
    project!(ProjectedField(grid(u), modes), u, alg)


# ------------------------------------------------------------------ #
# expand!                                                             #
# ------------------------------------------------------------------ #

raw"""
    expand!(u, a, alg=LoopGalerkin()) -> u

Reconstruct the spectral velocity field `u` from the modal coefficients `a`:

```math
u_n[\mathbf{j}, \mathbf{k}] = \sum_m a_{m,\mathbf{k}}\, \phi_{n,m,\mathbf{j},\mathbf{k}}
```

where `φ_{n,m,j,k} = modes(a)[n][j..., m, k...]`.

Pass [`LoopGalerkin`](@ref) (default) or [`GemmGalerkin`](@ref) to select the
implementation.  See also [`expand`](@ref) for the allocating form.
"""
function expand!(u::VectorField{N, <:FTField{G}},
                 a::ProjectedField{G},
                 ::LoopGalerkin=LoopGalerkin()
                 ) where {N, T, D, AXES, FFT_DIMS_ORDER, G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}}
    # No pre-zeroing: the inner kernel writes every (j, k) of `parent(u[n])`
    # exactly once with `=` (not `+=`), so the previous contents are
    # unconditionally overwritten.
    _loop_expand!(u, parent(a), a, grid(u))
    return u
end

"""
    expand(a::ProjectedField, alg=LoopGalerkin()) -> VectorField

Allocate a `VectorField` and reconstruct it from the modal coefficients `a`.
See [`expand!`](@ref) for the in-place form.
"""
expand(a::ProjectedField, alg=LoopGalerkin()) =
    expand!(VectorField(grid(a), FTField), a, alg)


# ------------------------------------------------------------------ #
# GemmGalerkin                                                        #
# ------------------------------------------------------------------ #
# Because each homogeneous wavenumber carries its own basis matrix, a single
# dense gemm over all wavenumbers is not possible.  Instead we reshape the
# modes to `(Ninh, Nm, NkH)` and the velocity to `(Ninh, NkH)`, then
# accumulate one inhomogeneous-row contribution at a time using broadcasted
# strided slices over all wavenumbers simultaneously.

function project!(a::ProjectedField{G},
                  u::VectorField{N, <:FTField{G}},
                  ::GemmGalerkin) where {N, G<:AbstractGrid{T}} where {T}
    a .= zero(Complex{T})
    g       = grid(u)
    inh_sz  = map(d -> size(g, d), inhomogeneous_dims(g))
    Ninh    = prod(inh_sz)
    # Flatten the kH... dimensions of pa into one column dimension.
    pa      = reshape(parent(a), size(parent(a), 1), :)   # (Nm, NkH)
    Nm, NkH = size(pa)
    wv      = reshape(weights(g), Ninh)                   # (Ninh,)

    for n in 1:N
        modes3 = reshape(modes(a)[n], Ninh, Nm, :)        # (Ninh, Nm, NkH)
        size(modes3, 3) == NkH ||
            throw(DimensionMismatch("modes must have $(NkH) homogeneous slices, got $(size(modes3, 3))"))
        U = reshape(parent(u[n]), Ninh, NkH)              # (Ninh, NkH)

        # For each inhomogeneous row j, accumulate the weighted contribution
        # over all wavenumbers k:
        #   pa[m, k] += conj(modes3[j, m, k]) * w[j] * U[j, k]
        @views for j in 1:Ninh
            mode_j = reshape(modes3[j, :, :], Nm, NkH)
            u_j    = reshape(U[j, :],         1,  NkH)
            @. pa += conj(mode_j) * (wv[j] * u_j)
        end
    end
    return a
end

function expand!(u::VectorField{N, <:FTField{G}},
                 a::ProjectedField{G},
                 ::GemmGalerkin) where {N, G<:AbstractGrid{T}} where {T}
    u .= zero(Complex{T})
    g       = grid(a)
    inh_sz  = map(d -> size(g, d), inhomogeneous_dims(g))
    Ninh    = prod(inh_sz)
    pa      = reshape(parent(a), size(parent(a), 1), :)   # (Nm, NkH)
    Nm, NkH = size(pa)

    for n in 1:N
        modes3 = reshape(modes(a)[n], Ninh, Nm, :)        # (Ninh, Nm, NkH)
        size(modes3, 3) == NkH ||
            throw(DimensionMismatch("modes must have $(NkH) homogeneous slices, got $(size(modes3, 3))"))
        U = reshape(parent(u[n]), Ninh, NkH)              # (Ninh, NkH)

        # For each mode m, scatter its contribution across every (j, k):
        #   U[j, k] += modes3[j, m, k] * pa[m, k]
        @views for m in 1:Nm
            mode_m = reshape(modes3[:, m, :], Ninh, NkH)
            a_m    = reshape(pa[m, :],        1,    NkH)
            @. U += mode_m * a_m
        end
    end
    return u
end
