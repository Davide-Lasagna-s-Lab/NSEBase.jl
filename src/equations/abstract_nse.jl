# Shared operator skeletons for velocity-only primitive NSE/LNSE families where
# every component diffuses at 1/Re.
#
# Cartesian, cylindrical, or other coordinate families can share this wrapper
# when their geometry-specific pieces are supplied by `laplacian!` and the
# form-dispatched advection helpers. The surrounding skeleton (viscous term →
# advection helper → body force) stays identical.
#
#   _nse_advection!(out, u, eq, form)          — nonlinear advection
#   _lnse_setup!(u, eq, form)                  — base-flow caching for the LNSE
#   _linearised_advection!(out, v, eq, form)   — linearised advection
#   _adjcont_advection!(out, v, eq, form)            — continuous-adjoint advection
#   adjoint_discrete_advection!(out, v, eq, form)    — discrete-adjoint advection
#
# The Boussinesq family is NOT part of this hierarchy: its temperature component
# diffuses at 1/(Re·Pr) and it adds a buoyancy term, so it keeps its own
# skeleton.

"""
    AbstractNSE{FORM<:AdvectionForm}

Supertype for nonlinear velocity-only primitive NSE operators. Concrete
subtypes carry the same `(Re, plans, scache, pcache, force)` fields and supply
`_nse_advection!(out, u, eq, FORM())`; the call skeleton is inherited.
"""
abstract type AbstractNSE{FORM<:AdvectionForm} end

"""
    AbstractLNSE{MODE<:Mode, FORM<:AdvectionForm}

Supertype for linearised velocity-only primitive NSE operators, parameterised on
the adjoint `MODE` and the advection `FORM`. Concrete subtypes supply
`_lnse_setup!`, `_linearised_advection!`, `_adjcont_advection!`, and
`adjoint_discrete_advection!`; the four call skeletons are inherited.
"""
abstract type AbstractLNSE{MODE<:Mode, FORM<:AdvectionForm} end


# ----------------------- nonlinear NSE ----------------------- #
function (eq::AbstractNSE{FORM})(::Real, u, out) where {FORM}
    laplacian!(out, u); out .*= 1/eq.Re
    _nse_advection!(out, u, eq, FORM())
    eq.force(out, u, Forward())
    return out
end

# ----------------------- linearised NSE ---------------------- #
# 3-arg: cache the base-flow quantities the form needs, then delegate to 2-arg.
function (eq::AbstractLNSE{MODE, FORM})(::Real, u, v, out) where {MODE, FORM}
    _lnse_setup!(u, eq, FORM())
    eq(0, v, out)
    return out
end

function (eq::AbstractLNSE{Forward, FORM})(::Real, v, out) where {FORM}
    laplacian!(out, v); out .*= 1/eq.Re
    _linearised_advection!(out, v, eq, FORM())
    eq.force(out, v, Forward())
    return out
end

function (eq::AbstractLNSE{AdjointContinuous, FORM})(::Real, v, out) where {FORM}
    laplacian!(out, v); out .*= 1/eq.Re
    _adjcont_advection!(out, v, eq, FORM())
    eq.force(out, v, AdjointContinuous())
    return out
end

function (eq::AbstractLNSE{AdjointDiscrete, FORM})(::Real, v, out) where {FORM}
    laplacian!(out, v, adjoint=true); out .*= 1/eq.Re
    adjoint_discrete_advection!(out, v, eq, FORM())
    eq.force(out, v, AdjointDiscrete())
    return out
end

# The exact discrete adjoint has to be derived for each discrete form. Until a
# form supplies that method explicitly, fail loudly rather than silently applying
# the wrong transpose.
adjoint_discrete_advection!(out, v, eq, form::AdvectionForm) =
    throw(ArgumentError("discrete adjoint is implemented for the advective form only; " *
                        "got $(typeof(form)). Use AdjointContinuous for divergence/rotational."))
