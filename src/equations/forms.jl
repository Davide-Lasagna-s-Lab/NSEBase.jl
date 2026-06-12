# Advection-form tags.
#
# The nonlinear advection term of the Navier-Stokes equations can be written in
# several mathematically equivalent forms that differ in how many FFTs they cost
# and in their discrete aliasing / conservation properties. In a pseudo-spectral
# / FD-hybrid solver the FFTs dominate, so the transform count is the figure of
# merit. Each concrete NSE/LNSE operator carries one of these tags as a type
# parameter (default `Advective`), and the advection step dispatches on it while
# the viscous / forcing skeleton stays shared.
#
#   Advective    N_i = u_j ∂_j u_i                 (the textbook convective form)
#   Divergence   N_i = ∂_j (u_i u_j)               (conservative; fewer transforms)
#   Rotational   N   = ω × u  (+ ∇(½|u|²) dropped)  (Lamb form; fewer transforms)
#
# The divergence form also applies to passively-advected scalars (e.g. the
# Boussinesq temperature) because ∂_j u_j = 0. The rotational form applies only
# to the velocity self-advection block; scalar / out-of-plane components fall
# back to the divergence treatment inside the rotational advection helper.

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
divergence-free fields and for passively-advected scalars.
"""
struct Divergence <: AdvectionForm end

"""
    Rotational <: AdvectionForm

Rotational (Lamb) form of the advection term, `N = ω × u`, using the identity
`(u·∇)u = ω × u + ∇(½|u|²)` and dropping the gradient term (removed by the
projection). Fewer transforms than [`Advective`](@ref). Applies to the velocity
self-advection block only; scalar / out-of-plane components use the divergence
treatment.
"""
struct Rotational <: AdvectionForm end
