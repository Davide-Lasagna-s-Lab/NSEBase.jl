# Galerkin projection and reconstruction for reduced-order spectral bases.
#
# `project!(a, u)` computes the L2 inner product of each basis mode with the
# spectral velocity field `u` and writes the resulting modal amplitudes into `a`.
# `expand!(u, a)` reverses the operation: it reconstructs `u` from the modal
# amplitudes in `a` by summing weighted basis modes at each wavenumber.
#
# Both LoopGalerkin methods embed the loop body directly — there are no private
# helper functions.
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
#   `(Nm, inh_sz..., kH_sz...)`
#
# with axes laid out in this order, listed innermost (axis 1) to outermost:
#
#   axis  1                          : mode index, 1:Nm
#   axes  2 … Ninh + 1               : inhomogeneous-dimension sizes, in
#                                      ascending grid-dimension order
#                                      (same order returned by
#                                      `inhomogeneous_storage_dims(grid)`)
#   axes  Ninh + 2 … Ninh + 1 + Nhom : homogeneous-dimension sizes, listed
#                                      in `FFT_DIMS_ORDER` order — i.e. the
#                                      **rfft** axis `FFT_DIMS_ORDER[1]` is
#                                      first and runs over the half-spectrum
#                                      `1 : (size÷2) + 1`, followed by the
#                                      signed-FFT axes `FFT_DIMS_ORDER[2:end]`
#                                      in full storage order.
#
# Example — ChannelFlow-style grid with `D = 4`, `AXES = (2,1,3,4)`,
# `FFT_DIMS_ORDER = (2,3,4)` (so `inh_dims = (1,)` is the wall-normal direction,
# rfft on x, signed FFTs on z and t):
#
#   `modes(a)[n] :: Array{ComplexF64, 5}`,
#   `size(modes(a)[n]) == (Nm, Ny, (Nx ÷ 2) + 1, Nz, Nt)`.
#
# Rationale for this layout:
#
#   - axis 1 = mode index is contiguous in memory across `m`, which makes
#     the innermost m-loop a unit-stride pass through both `pa` and `modes_n`,
#     maximising cache efficiency.
#   - the homogeneous axes last lets `GemmGalerkin` reshape modes to
#     `(Nm, Ninh, NkH)` with a single zero-cost `reshape`.
#
# Any other layout will either fail with a dimension mismatch, give incorrect
# results, or pessimise cache behaviour in the inner loops.
#
# ------------------------------------------------------------------ #
# Algorithms                                                         #
# ------------------------------------------------------------------ #
#
#   `LoopGalerkin()` — nested CartesianIndices loops over the homogeneous and
#                      inhomogeneous dimensions.  No scratch allocation.
#                      Preferred when `Nm` is small.
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

Uses nested `CartesianIndices` loops over the homogeneous (FFT) and
inhomogeneous dimensions — readable, allocation-free, and specialised to the
grid type at compile time via `homogeneous_axes` / `inhomogeneous_axes`.
Preferred when the number of basis modes `Nm` is small (typical wall-bounded
Galerkin bases).
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
# project!                                                            #
# ------------------------------------------------------------------ #

"""
    project!(a, u, alg=LoopGalerkin()) -> a

Project the spectral vector field `u` onto the basis stored by `a` and write
the modal coefficients into `a`:

```math
a_{m,\\mathbf{k}} = \\sum_n \\sum_{\\mathbf{j}} w_{\\mathbf{j}}\\,
    \\overline{\\phi_{n,m,\\mathbf{j},\\mathbf{k}}}\\, u_{n,\\mathbf{j},\\mathbf{k}}
```

where `φ_{n,m,j,k} = modes(a)[n][m, j..., k...]` and `w` is
[`weights`](@ref)`(grid(u))`.

Pass [`LoopGalerkin`](@ref) (default) or [`GemmGalerkin`](@ref) to select the
implementation.  See also [`project`](@ref) for the allocating form.
"""
function project!(a::ProjectedField{G},
                  u::VectorField{N, <:FTField{G}},
                  ::LoopGalerkin=LoopGalerkin()
                  ) where {N, G<:AbstractGrid{T}} where {T}
    a .= zero(Complex{T})
    pa = parent(a)
    g  = grid(u)
    w  = weights(g)
    Nm = size(pa, 1)
    @inbounds for n in 1:N
        pu      = parent(u[n])
        modes_n = modes(a)[n]
        for Ih in CartesianIndices(homogeneous_axes(u[n]))
            for Iinh in CartesianIndices(inhomogeneous_axes(u[n]))
                wuj = w[Iinh] * pu[combine_indices(g, Iinh, Ih)...]
                for m in 1:Nm
                    pa[m, Ih] += conj(modes_n[m, Iinh, Ih]) * wuj
                end
            end
        end
    end
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

"""
    expand!(u, a, alg=LoopGalerkin()) -> u

Reconstruct the spectral velocity field `u` from the modal coefficients `a`:

```math
u_n[\\mathbf{j}, \\mathbf{k}] = \\sum_m a_{m,\\mathbf{k}}\\, \\phi_{n,m,\\mathbf{j},\\mathbf{k}}
```

where `φ_{n,m,j,k} = modes(a)[n][m, j..., k...]`.

No pre-zeroing is needed: every `(j, k)` entry of `parent(u[n])` is written
exactly once with `=` (not `+=`), so previous contents are unconditionally
overwritten.

Pass [`LoopGalerkin`](@ref) (default) or [`GemmGalerkin`](@ref) to select the
implementation.  See also [`expand`](@ref) for the allocating form.
"""
function expand!(u::VectorField{N, <:FTField{G}},
                 a::ProjectedField{G},
                 ::LoopGalerkin=LoopGalerkin()
                 ) where {N, G<:AbstractGrid{T}} where {T}
    pa = parent(a)
    g  = grid(u)
    Nm = size(pa, 1)
    @inbounds for n in 1:N
        pu      = parent(u[n])
        modes_n = modes(a)[n]
        for Ih in CartesianIndices(homogeneous_axes(u[n]))
            for Iinh in CartesianIndices(inhomogeneous_axes(u[n]))
                acc = zero(eltype(pa))
                for m in 1:Nm
                    acc += modes_n[m, Iinh, Ih] * pa[m, Ih]
                end
                pu[combine_indices(g, Iinh, Ih)...] = acc
            end
        end
    end
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
# modes to `(Nm, Ninh, NkH)` and the velocity to `(Ninh, NkH)`, then
# accumulate one inhomogeneous-row contribution at a time using broadcasted
# strided slices over all wavenumbers simultaneously.

function project!(a::ProjectedField{G},
                  u::VectorField{N, <:FTField{G}},
                  ::GemmGalerkin) where {N, G<:AbstractGrid{T}} where {T}
    a .= zero(Complex{T})
    g       = grid(u)
    inh_sz  = map(d -> size(g, d), inhomogeneous_storage_dims(g))
    Ninh    = prod(inh_sz)
    # Flatten the kH... dimensions of pa into one column dimension.
    pa      = reshape(parent(a), size(parent(a), 1), :)   # (Nm, NkH)
    Nm, NkH = size(pa)
    wv      = reshape(weights(g), Ninh)                   # (Ninh,)

    for n in 1:N
        modes3 = reshape(modes(a)[n], Nm, Ninh, :)        # (Nm, Ninh, NkH)
        size(modes3, 3) == NkH ||
            throw(DimensionMismatch("modes must have $(NkH) homogeneous slices, got $(size(modes3, 3))"))
        U = reshape(parent(u[n]), Ninh, NkH)              # (Ninh, NkH)

        # For each inhomogeneous row j, accumulate the weighted contribution
        # over all wavenumbers k:
        #   pa[m, k] += conj(modes3[m, j, k]) * w[j] * U[j, k]
        @views for j in 1:Ninh
            mode_j = reshape(modes3[:, j, :], Nm, NkH)
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
    inh_sz  = map(d -> size(g, d), inhomogeneous_storage_dims(g))
    Ninh    = prod(inh_sz)
    pa      = reshape(parent(a), size(parent(a), 1), :)   # (Nm, NkH)
    Nm, NkH = size(pa)

    for n in 1:N
        modes3 = reshape(modes(a)[n], Nm, Ninh, :)        # (Nm, Ninh, NkH)
        size(modes3, 3) == NkH ||
            throw(DimensionMismatch("modes must have $(NkH) homogeneous slices, got $(size(modes3, 3))"))
        U = reshape(parent(u[n]), Ninh, NkH)              # (Ninh, NkH)

        # For each mode m, scatter its contribution across every (j, k):
        #   U[j, k] += modes3[m, j, k] * pa[m, k]
        @views for m in 1:Nm
            mode_m = reshape(modes3[m, :, :], Ninh, NkH)
            a_m    = reshape(pa[m, :],        1,    NkH)
            @. U += mode_m * a_m
        end
    end
    return u
end
