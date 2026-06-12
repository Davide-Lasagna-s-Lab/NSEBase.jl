# Equation tag and forcing types.
#
# These are deliberately tiny, mostly zero-field types. NSE/LNSE operators carry
# them as type parameters so the compiler specialises on mode and advection form
# without runtime branches in the operator kernels.


# ============================================================================ #
# Mode Tags                                                                    #
# ============================================================================ #
"""
    Mode

Abstract supertype for equation-mode tags.  Concrete subtypes select which
variant of the Navier-Stokes operator is applied: the nonlinear equation
([`NonLinear`](@ref)) or one of the linearised variants ([`Forward`](@ref),
[`AdjointContinuous`](@ref), [`AdjointDiscrete`](@ref)).
"""
abstract type Mode end

"""
    NonLinear <: Mode

Tag selecting the nonlinear Navier-Stokes operator `N(u) = ν·Δu − (u·∇)u + force`.
A single operator type covers the nonlinear and linearised equations; this mode
selects the nonlinear branch.
"""
struct NonLinear         <: Mode end

"""
    Forward <: Mode

Tag selecting the forward linearised operator L·v, i.e. the linearisation of
the NSE around the current base flow in the forward-time direction.
"""
struct Forward           <: Mode end

"""
    AdjointDiscrete <: Mode

Tag selecting the discrete adjoint of the forward linearised operator.
The discrete adjoint is derived by transposing the discrete operator exactly —
it satisfies ⟨L·v, w⟩ = ⟨v, L*·w⟩ with respect to the discrete inner product
used in [`dot`](@ref).
"""
struct AdjointDiscrete   <: Mode end

"""
    AdjointContinuous <: Mode

Tag selecting the continuous adjoint of the linearised operator.  The
continuous adjoint is derived by integration-by-parts before discretisation,
which gives a different operator from the discrete adjoint for finite
resolution.
"""
struct AdjointContinuous <: Mode end

const LinearisedMode = Union{Forward, AdjointContinuous, AdjointDiscrete}
const AdjointMode = Union{AdjointContinuous, AdjointDiscrete}


# ============================================================================ #
# Advection-Form Tags                                                          #
# ============================================================================ #
# The nonlinear advection term of the Navier-Stokes equations can be written in
# several mathematically equivalent forms that differ in transform count and in
# their discrete aliasing / conservation properties. In a pseudo-spectral /
# FD-hybrid solver the FFTs dominate, so the transform count is often the useful
# figure of merit.
#
#   Advective    N_i = u_j ∂_j u_i
#   Divergence   N_i = ∂_j (u_i u_j)
#   Rotational   N   = ω × u  (+ ∇(½|u|²) dropped)
#
# The rotational form applies only to velocity self-advection. In the 2D-3C
# case, the out-of-plane component is still transported conservatively.
"""
    AdvectionForm

Abstract supertype for advection-form tags. Concrete subtypes ([`Advective`](@ref),
[`Divergence`](@ref), [`Rotational`](@ref)) select how the nonlinear advection
term is evaluated. The choice is a type parameter on each NSE/LNSE operator, so
the form is fixed at compile time with no runtime dispatch in the hot loop.
"""
abstract type AdvectionForm end

"""
    Advective <: AdvectionForm

Convective form of the advection term, `N_i = u_j ∂_j u_i`. This is the textbook
form and the default. It transforms the velocity and all of its first
derivatives to physical space.
"""
struct Advective <: AdvectionForm end

"""
    Divergence <: AdvectionForm

Conservative form of the advection term, `N_i = ∂_j(u_i u_j)`. The velocity is
transformed to physical space once, the quadratic products are formed there and
transformed back, and the divergence is taken with the cheap spectral / FD
derivatives. Fewer transforms than [`Advective`](@ref); valid for
divergence-free fields.
"""
struct Divergence <: AdvectionForm end

"""
    Rotational <: AdvectionForm

Rotational (Lamb) form of the advection term, `N = ω × u`, using the identity
`(u·∇)u = ω × u + ∇(½|u|²)` and dropping the gradient term (removed by the
projection). Fewer transforms than [`Advective`](@ref). Applies to the velocity
self-advection block only; the 2D-3C out-of-plane component uses conservative
transport.
"""
struct Rotational <: AdvectionForm end
