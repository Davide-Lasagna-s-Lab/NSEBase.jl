# ProjectedNSE: reduced-order NSE operator for wall-bounded variational problems.
#
# `ProjectedNSE` wraps a pair of NSE operators (nonlinear and linearised) and
# adds the expand→operate→project sandwich that maps between the modal
# coefficient space of a `ProjectedField` and the full spectral velocity space
# of a `VectorField{N, FTField}`.
#
# Two pre-allocated `VectorField` caches (`cache1`, `cache2`) are held by the
# struct so that no allocations occur during the operator calls. The wrapped
# nonlinear and linearised operators also share their own derivative caches:
# evaluating the nonlinear action prepares those caches for the subsequent
# linearised action.

"""
    ProjectedNSE{EQ, LEQ, B, N, S1, S2}

Reduced-order Navier-Stokes operator operating in the modal coefficient space
of a [`ProjectedField`](@ref).

Wraps a nonlinear operator `nl` and a linearised operator `ln` (both acting on
full spectral `VectorField`s) with expand/project steps so that the combined
operator maps `ProjectedField → ProjectedField` without exposing the full
spectral representation to the caller.

Use [`construct_equations`](@ref) to build a `ProjectedNSE` with correctly
sized caches and FFTW plans rather than constructing it directly.

# Fields
- `nl`: nonlinear NSE operator; called as `nl(t, u, N_u)`
- `ln`: linearised NSE operator; called as `ln(t, v, M_uv)`
- `base`: laminar base flow tuple, passed to [`add_base_flow!`](@ref)
- `cache1`, `cache2`: pre-allocated full spectral `VectorField` workspaces

The object is stateful because `nl` and `ln` share workspaces. In particular,
the three-argument linearised call must immediately follow the nonlinear call
at the same coefficient state; see the call-method documentation below.
"""
struct ProjectedNSE{EQ, LEQ, B, N, S1, S2}
        nl::EQ
        ln::LEQ
      base::B
    cache1::VectorField{N, S1}
    cache2::VectorField{N, S2}

    ProjectedNSE{N}(nl::EQ, ln::LEQ, base::B, caches) where {EQ, LEQ, B, N} =
        new{EQ, LEQ, B, N, eltype(caches[1]), eltype(caches[2])}(nl, ln, base, caches...)
end

ProjectedNSE(grid::AbstractGrid, N::Int, nl, ln, base) =
    ProjectedNSE{N}(nl, ln, base, ntuple(_->VectorField(grid, FTField, N=N), 2))

"""
    (eq::ProjectedNSE)(out::ProjectedField, a::ProjectedField) -> out

Nonlinear action: expand `a` to a full spectral velocity field, add the laminar
base flow, apply the nonlinear NSE operator, and project the result back onto
the basis, writing modal coefficients into `out`.

Besides producing `out`, this call prepares the shared physical-field and
derivative caches consumed by the three-argument linearised action. Therefore,
call `(eq)(out, a)` before `(eq)(out, a, b)` for every new `a`.
"""
function (eq::ProjectedNSE)(out::ProjectedField, a::ProjectedField)
    # aliases
    u   = eq.cache1
    N_u = eq.cache2

    # expand coefficients into spectral field
    expand!(u, a)
    add_base_flow!(u, eq.base)

    # operator action
    eq.nl(0, u, N_u)

    # project result back onto basis
    project!(out, N_u)

    return out
end

"""
    (eq::ProjectedNSE)(out::ProjectedField,
                       cached_state::ProjectedField,
                       b::ProjectedField) -> out

Apply the linearised or adjoint operator at `cached_state` to the perturbation
`b`, writing the projected result into `out`.

This is a cache-reuse interface. A preceding call to
`eq(nonlinear_out, cached_state)` expands the state, adds `eq.base`, and leaves
the physical state and its derivatives in workspaces shared by `eq.nl` and
`eq.ln`. This method therefore expands only `b`; `cached_state` is a dependency
marker and is deliberately not read again. Avoiding that second expansion and
the repeated derivatives is important in optimisation algorithms such as
L-BFGS, which evaluate an objective and then its gradient at the same state.

The calls must be paired and ordered:

```julia
eq(nonlinear_out, a)       # prepares the caches for a
eq(linearised_out, a, b)   # consumes those caches
```

Passing a different second argument, calling this method before the nonlinear
form, or interleaving a call at another state uses stale cache contents. This
contract is not checked at runtime because comparing or re-expanding the state
would defeat cache reuse. Consequently, a `ProjectedNSE` instance is neither
reentrant nor safe for concurrent calls.
"""
function (eq::ProjectedNSE)(out::ProjectedField, _cached_state::ProjectedField, b::ProjectedField)
    # `_cached_state` is intentionally unused; it names the state whose caches
    # the preceding nonlinear call prepared, as specified above.
    # aliases
    v    = eq.cache1
    M_uv = eq.cache2

    # expand coefficients into spectral fields
    expand!(v, b)

    # operator action
    eq.ln(0, v, M_uv)

    # project result back onto basis
    project!(out, M_uv)

    return out
end
