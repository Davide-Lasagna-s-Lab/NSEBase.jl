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


# ------------------------------------------------------------------ #
# construct_equations factory                                        #
# ------------------------------------------------------------------ #
# The operator type itself selects the formulation. `construct_equations` asks
# the operator type for its component count, builds the requested linearised
# operator on the large cache pool, then asks the type to rebuild a nonlinear
# operator sharing those plans and caches.

operator_ncomponents(::Type{OP}) where {OP<:AbstractNSE} =
    throw(ArgumentError("operator_ncomponents is not implemented for $OP"))

default_linearised_mode(::Type{OP}) where {OP<:AbstractNSE} = AdjointDiscrete()

shared_nonlinear_operator(::Type{OP}, visc, form, ln, force) where {OP<:AbstractNSE} =
    throw(ArgumentError("shared nonlinear construction is not implemented for $OP"))

"""
    construct_equations(grid, Re, base, op_type=CartesianPrimitive3DNSE;
                        visc=1/Re, form=Advective(), mode=AdjointDiscrete(),
                        force=NoForce(), flags=FFTW.EXHAUSTIVE, dealias=true)

Build a [`ProjectedNSE`](@ref) for an NSE operator type. The operator type fixes
the component count and coordinate system. The nonlinear and linearised
operators share one cache pool (sized for the linearised operator).

`form` selects the advection form; `mode` the adjoint flavour of the linearised
operator; `visc` the Laplacian coefficient (scalar `ν=1/Re`, or an `NCOMP`-tuple
for a per-component diffusivity). `base` is the laminar base flow, one entry per
component (or `nothing`), passed to [`add_base_flow!`](@ref).
"""
function construct_equations(grid::AbstractGrid{T}, Re, base,
                             op_type::Type{<:AbstractNSE}=CartesianPrimitive3DNSE;
                             visc                =inv(convert(T, Re)),
                             form::AdvectionForm =Advective(),
                             mode::Mode          =default_linearised_mode(op_type),
                             force               =NoForce(),
                             flags               =FFTW.EXHAUSTIVE,
                             dealias             =true) where {T}
    mode isa LinearisedMode ||
        throw(ArgumentError("projected linearised operator has to use a linearised mode"))
    # build the linearised operator (allocates the larger, shared cache pool),
    # then build the nonlinear operator sharing its plans and caches.
    ln = op_type(grid, Re; mode=mode, visc=visc, form=form,
                 force=force, flags=flags, dealias=dealias)
    nl = shared_nonlinear_operator(op_type, visc, form, ln, force)
    NCOMP = operator_ncomponents(op_type)
    return ProjectedNSE(grid, NCOMP, nl, ln, base)
end
