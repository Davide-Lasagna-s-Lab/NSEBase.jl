# ProjectedNSE: reduced-order NSE operator for wall-bounded variational problems.
#
# `ProjectedNSE` wraps a pair of NSE operators (nonlinear and linearised) and
# adds the expand→operate→project sandwich that maps between the modal
# coefficient space of a `ProjectedField` and the full spectral velocity space
# of a `VectorField{N, FTField}`.
#
# Two pre-allocated `VectorField` caches (`cache1`, `cache2`) are held by the
# struct so that no allocations occur during the operator calls.

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
"""
function (eq::ProjectedNSE)(out::ProjectedField,
                              a::ProjectedField)
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

# TODO: TOM, this interface with a second unused argument does not make much sense to me. It needs better documentation
"""
    (eq::ProjectedNSE)(out::ProjectedField, ::ProjectedField, b::ProjectedField) -> out

Linearised action: expand `b` to a full spectral velocity field, apply the
linearised NSE operator (which uses the base flow cached from a preceding
nonlinear call), and project the result back onto the basis.
"""
function (eq::ProjectedNSE)(out::ProjectedField,
                               ::ProjectedField,
                              b::ProjectedField)
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
