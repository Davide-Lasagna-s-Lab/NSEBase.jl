# Equation layer: the abstract Navier-Stokes operator and its call skeletons.
#
# One operator type covers the nonlinear equation and every linearised variant,
# selected by the `MODE` parameter (`NonLinear`, `Forward`, `AdjointContinuous`,
# `AdjointDiscrete`). A concrete operator (Cartesian, cylindrical, …) supplies
# its fields `(visc, plans, scache, pcache, force)` and the form-dispatched
# advection helpers; the surrounding skeleton — viscous term → advection → body
# force — is inherited from here.
#
# `visc` (= ν = 1/Re) may be a scalar (uniform) or an NCOMP-tuple (per-component,
# e.g. Boussinesq, where the temperature diffuses at 1/(Re·Pr)).

"""
    AbstractNSE{MODE<:Mode, FORM<:AdvectionForm}

Supertype for primitive Navier-Stokes operators. Subtypes carry
`(visc, plans, scache, pcache, force)` and supply the form-dispatched advection
helpers: `advection!` for `NonLinear`, and `lnse_setup!`, `linearised_advection!`,
`adjoint_advection!` for the linearised modes. The call skeletons are inherited.
"""
abstract type AbstractNSE{MODE<:Mode, FORM<:AdvectionForm} end

# Apply the viscous coefficient: a scalar scales every component uniformly; an
# NCOMP-tuple gives a per-component diffusivity.
scale_diffusion!(out, visc::Number) = (out .*= visc; out)
scale_diffusion!(out::VectorField{N}, visc::NTuple{N}) where {N} =
    (for n in 1:N; out[n] .*= visc[n]; end; out)

function diffusion!(out, u, eq::AbstractNSE; adjoint::Bool=false)
    laplacian!(out, u; adjoint=adjoint)
    scale_diffusion!(out, eq.visc)
    return out
end


# ------------------------- nonlinear ------------------------- #
function (eq::AbstractNSE{NonLinear, FORM})(::Real, u, out) where {FORM}
    diffusion!(out, u, eq)
    advection!(out, u, eq, FORM())
    eq.force(out, u, Forward())
    return out
end

# ------------------------- linearised ------------------------ #
# 3-arg: cache the base-flow quantities the form needs, then delegate to 2-arg.
function (eq::AbstractNSE{MODE, FORM})(::Real, u, v, out) where {MODE<:LinearisedMode, FORM}
    lnse_setup!(u, eq, FORM())
    eq(0, v, out)
    return out
end

function (eq::AbstractNSE{Forward, FORM})(::Real, v, out) where {FORM}
    diffusion!(out, v, eq)
    linearised_advection!(out, v, eq, FORM())
    eq.force(out, v, Forward())
    return out
end

function (eq::AbstractNSE{MODE, FORM})(::Real, v, out) where {MODE<:AdjointMode, FORM}
    diffusion!(out, v, eq; adjoint=MODE() isa AdjointDiscrete ? true : false)
    adjoint_advection!(out, v, eq, FORM(), MODE())
    eq.force(out, v, MODE())
    return out
end

# The exact discrete adjoint has to be derived per discrete form; until a form
# supplies that method, fail loudly rather than apply the wrong transpose.
adjoint_advection!(out, v, eq, form::AdvectionForm, ::AdjointDiscrete) =
    throw(ArgumentError("discrete adjoint is implemented for the advective form only; " *
                        "got $(typeof(form)). Use AdjointContinuous for divergence/rotational."))
